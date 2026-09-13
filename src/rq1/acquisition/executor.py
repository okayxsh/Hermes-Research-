"""Real train-only RQ1 acquisition executor on the harness-owned episode driver.

One queue unit is one fresh Hermes/ALFWorld session limited to the frozen action
budget.  Only a successful episode asks the same experimental model for one
post-success candidate; the candidate is validated deterministically and, when
accepted, is committed inside the unit's result row.  Nothing else mutates the
skill pool, so infrastructure failures cannot create or lose skills.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from rq1.acquisition.protocol import (
    ACQUISITION_ACTION_BUDGET,
    ACQUISITION_ENVIRONMENT_SEED,
    ACQUISITION_POLICY_VERSION,
    ACQUISITION_PROFILE,
    ACQUISITION_SPLIT,
    ACQUISITION_TEMPERATURE,
    SKILL_GENERATION_ATTEMPTS,
    SKILL_GENERATION_PROTOCOL,
)
from rq1.acquisition.real_executor import run_acquisition_episode
from rq1.acquisition.skill_creation import (
    LEARNING_PROMPT,
    VALIDATION_PROMPT,
    parse_skill_response,
    prompt_text,
    render_skill_prompt,
    validate_skill,
)
from rq1.acquisition.skill_pool import (
    SNAPSHOT_NAME,
    PoolSkill,
    SkillPoolError,
    pool_hash,
    rebuild_pool,
    skill_id_for,
    verify_snapshot,
    write_snapshot,
)
from rq1.experiment.models import ExperimentUnit, RunExecutionContext, RunFailure, RunOutcome
from rq1.experiment.persistence import ExperimentStore
from rq1.hermes.episode_driver import OUTPUT_CAP_REJECTION, reached_output_cap
from rq1.retrieval.text import build_skill_text
from rq1.skills.library import TASK_FAMILIES
from rq1.utils.hashing import sha256_text
from rq1.utils.time import utc_now


def unit_lineage(unit: ExperimentUnit) -> dict[str, Any]:
    """Parent run and logical combined position of a continuation unit; empty otherwise."""
    if unit.payload.get("parent_run_id") is None:
        return {}
    return {
        "parent_run_id": unit.payload["parent_run_id"],
        "logical_acquisition_index": unit.payload["logical_acquisition_index"],
    }


class AcquisitionEpisodeHarness:
    """``AcquisitionHarness`` for one unit: a fresh session and an optional candidate."""

    def __init__(
        self,
        driver: Any,
        *,
        root: Path,
        unit: ExperimentUnit,
        context: RunExecutionContext,
        pool: tuple[PoolSkill, ...],
        scientific: bool,
    ) -> None:
        self.driver = driver
        self.root = root
        self.unit = unit
        self.context = context
        self.pool = pool
        self.pool_after = pool
        self.scientific = scientific
        self.stage = "not_started"
        self.evidence: dict[str, Any] = {}

    def run_episode(self, task_id: str, split: str, seed: int, action_limit: int) -> Mapping[str, Any]:
        if split != ACQUISITION_SPLIT:
            raise ValueError("acquisition accepts the train split only")
        self.stage = "episode"
        output = self.context.output_dir / "episode"
        with self.driver.session(
            output_dir=output,
            run_id=self.context.experiment_id,
            attempt_id=self.context.attempt_id,
            profile=ACQUISITION_PROFILE,
        ) as session:
            session.start(task_id, split, seed, action_limit)
            records = session.run_model_loop(action_limit, phase="acquisition")
            state = session.state or {}
            success = state.get("done") is True and state.get("success") is True
            if success:
                termination = "success"
            elif session.selection_failures:
                termination = "action_selection_exhausted"
            elif len(records) >= action_limit:
                termination = "action_budget_exhausted"
            else:
                termination = "environment_terminated_without_success"
            candidate: dict[str, Any] = {"status": "not_generated_episode_unsuccessful"}
            accepted: PoolSkill | None = None
            if success:
                self.stage = "post_success_skill_generation"
                candidate, accepted = self._create_skill(session, task_id, records)
            actions = [record.action for record in records]
            self.evidence = {
                "task_goal": session.task_goal,
                "termination_reason": termination,
                "episode_actions": actions,
                "invalid_action_selections": session.invalid_model_actions,
                "selection_rejections": dict(session.rejection_counts),
                "skill_candidate": candidate,
                "episode_events_log": str(output / "episode-events.jsonl"),
            }
            invalid = session.invalid_model_actions
        self.stage = "completed"
        if accepted is not None:
            self.pool_after = (*self.pool, accepted)
        return {
            "success": success,
            "steps": len(records),
            "actions": len(records),
            "invalid_actions": invalid,
            "skill_candidate": (
                {"skill_id": accepted.skill_id, "title": accepted.title, "body": accepted.body}
                if accepted is not None else None
            ),
        }

    def _create_skill(self, session: Any, task_id: str, records: Sequence[Any]) -> tuple[dict[str, Any], PoolSkill | None]:
        actions = [record.action for record in records]
        prompt = render_skill_prompt(
            learning_instruction=prompt_text(self.root, LEARNING_PROMPT),
            validation_rules=prompt_text(self.root, VALIDATION_PROMPT),
            task_goal=session.task_goal or "",
            actions=actions,
            final_observation=str((session.state or {}).get("observation", "")),
        )
        response = session.complete_post_success_learning(prompt)
        generation = {
            "generation_protocol": SKILL_GENERATION_PROTOCOL,
            "generation_attempts": SKILL_GENERATION_ATTEMPTS,
            "model": self.driver.model_name,
            "inference_seed": self.driver.inference_seed,
            "temperature": ACQUISITION_TEMPERATURE,
            "prompt_sha256": sha256_text(prompt),
            "response_sha256": sha256_text(response),
        }
        base = {**generation, "response": response, "rejection_reasons": []}
        if getattr(response, "done_reason", None) is not None:
            base["done_reason"] = response.done_reason
        if reached_output_cap(response):
            # Decision 010: a candidate cut off at the output cap is an incomplete
            # model answer, rejected without parsing; never an infrastructure failure.
            return {**base, "status": "rejected", "rejection_reasons": [OUTPUT_CAP_REJECTION]}, None
        parsed = parse_skill_response(response)
        if parsed is None:
            return {**base, "status": "rejected", "rejection_reasons": ["format_invalid"]}, None
        if parsed.declined:
            return {**base, "status": "declined"}, None
        title = " ".join((parsed.title or "").split())
        body = " ".join((parsed.body or "").split())
        reasons = validate_skill(
            title=title,
            body=body,
            source_task_id=task_id,
            executed_actions=actions,
            existing_texts={skill.text for skill in self.pool},
        )
        if reasons:
            return {**base, "status": "rejected", "rejection_reasons": reasons, "title": title, "body": body}, None
        text = build_skill_text(title=title, body=body)
        skill = PoolSkill(
            pool_index=len(self.pool) + 1,
            skill_id=skill_id_for(task_id, title, body),
            title=title,
            body=body,
            text=text,
            text_sha256=sha256_text(text),
            task_family=str(self.unit.payload["task_family"]),
            source_task_id=task_id,
            source_task_index=self.unit.task_index,
            source_run_key=self.unit.run_key,
            source_attempt_id=self.context.attempt_id,
            created_at=utc_now(),
            provenance={
                **generation,
                "operation": "create",
                "policy_version": ACQUISITION_POLICY_VERSION,
                "experiment_id": self.context.experiment_id,
                "scientific_evidence": self.scientific,
                **unit_lineage(self.unit),
            },
        )
        return {**base, "status": "accepted", "skill": skill.to_dict()}, skill

    def post_run_library_hash(self) -> str:
        return pool_hash(self.pool_after)

    def post_run_library_size(self) -> int:
        return len(self.pool_after)


class RealAcquisitionExecutor:
    """Durable-runner executor plus its resume preflight and checkpoint state."""

    def __init__(
        self,
        root: Path,
        store: ExperimentStore,
        driver: Any,
        *,
        scientific: bool,
        queue_sha256: str | None,
        action_budget: int = ACQUISITION_ACTION_BUDGET,
        environment_seed: int = ACQUISITION_ENVIRONMENT_SEED,
        parent_pool: Sequence[PoolSkill] = (),
        parent_run_id: str | None = None,
    ) -> None:
        if parent_pool and parent_run_id is None:
            raise ValueError("a starting skill pool requires its parent run ID")
        self.root = root
        self.store = store
        self.driver = driver
        self.scientific = scientific
        self.queue_sha256 = queue_sha256
        self.action_budget = action_budget
        self.environment_seed = environment_seed
        # The exact final pool of a completed parent run; read-only here.
        self.parent_pool = tuple(parent_pool)
        self.parent_run_id = parent_run_id

    def committed_pool(self) -> tuple[PoolSkill, ...]:
        return rebuild_pool(self.store.terminal_results(phase="acquisition", repair_tail=False).values(), base=self.parent_pool)

    def preflight(self, units: Sequence[ExperimentUnit], latest: Mapping[str, Mapping[str, Any]]) -> None:
        for unit in units:
            if unit.payload.get("task_family") not in TASK_FAMILIES:
                raise SkillPoolError(f"queue unit {unit.task_index} lacks a frozen task family")
            if unit.payload.get("parent_run_id") != self.parent_run_id:
                raise SkillPoolError(f"queue unit {unit.task_index} parent lineage differs from the executor starting pool")
        verify_snapshot(self.store.directory, rebuild_pool(latest.values(), base=self.parent_pool))

    def checkpoint_state(self, units: Sequence[ExperimentUnit], latest: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
        skills = rebuild_pool(latest.values(), base=self.parent_pool)
        write_snapshot(self.store.directory, skills)
        families = {
            family: {"planned": 0, "completed": 0, "successful": 0, "scientific_failures": 0, "infrastructure_failures": 0, "accepted_skills": 0}
            for family in TASK_FAMILIES
        }
        completed: list[str] = []
        failed: list[str] = []
        for unit in units:
            counts = families[str(unit.payload.get("task_family"))]
            counts["planned"] += 1
            record = latest.get(unit.run_key)
            if record is None:
                continue
            if record.get("status") == "completed":
                counts["completed"] += 1
                counts["successful" if record.get("success") is True else "scientific_failures"] += 1
                completed.append(unit.run_key)
            elif record.get("status") == "failed":
                counts["infrastructure_failures"] += 1
                failed.append(unit.run_key)
        # accepted_skills counts the skills this run appended, never the starting pool.
        for skill in skills[len(self.parent_pool):]:
            families[skill.task_family]["accepted_skills"] += 1
        pending = [unit for unit in units if unit.run_key not in latest]
        continuation = {} if self.parent_run_id is None else {
            "continuation": {
                "parent_run_id": self.parent_run_id,
                "starting_pool_size": len(self.parent_pool),
                "starting_pool_hash": pool_hash(self.parent_pool),
                "appended_skills": len(skills) - len(self.parent_pool),
                "pool_per_family": {family: sum(skill.task_family == family for skill in skills) for family in TASK_FAMILIES},
            }
        }
        return {
            **continuation,
            "acquisition_policy_version": ACQUISITION_POLICY_VERSION,
            "scientific_evidence": self.scientific,
            "queue_sha256": self.queue_sha256,
            "action_budget": self.action_budget,
            "next_queue_index": pending[0].task_index if pending else None,
            "completed_run_keys": completed,
            "failed_run_keys": failed,
            "per_family": families,
            "skill_pool": {
                "size": len(skills),
                "hash": pool_hash(skills),
                "snapshot": SNAPSHOT_NAME,
                "skills": [
                    {
                        "pool_index": skill.pool_index,
                        "skill_id": skill.skill_id,
                        "task_family": skill.task_family,
                        "source_task_id": skill.source_task_id,
                        "source_task_index": skill.source_task_index,
                        "source_run_key": skill.source_run_key,
                        "source_attempt_id": skill.source_attempt_id,
                    }
                    for skill in skills
                ],
            },
        }

    def __call__(self, unit: ExperimentUnit, context: RunExecutionContext) -> RunOutcome:
        pool = self.committed_pool()
        family = str(unit.payload.get("task_family"))
        harness = AcquisitionEpisodeHarness(
            self.driver, root=self.root, unit=unit, context=context, pool=pool, scientific=self.scientific,
        )
        try:
            result = run_acquisition_episode(
                harness,
                task_id=unit.task_id,
                task_family=family,
                attempt_id=context.attempt_id,
                log_dir=context.output_dir / "result",
                split=ACQUISITION_SPLIT,
                seed=self.environment_seed,
                action_limit=self.action_budget,
            )
        except Exception as exc:
            # Only genuine execution failures reach this point: invalid or capped
            # model output is handled inside the episode (Decision 010).  The pool
            # changes only through committed results, so the mutation state is
            # known and no skill can come from this failed attempt.
            raise RunFailure(
                f"acquisition infrastructure failure during {harness.stage}: {type(exc).__name__}: {exc}",
                safe_to_continue=True,
                mutation_state_known=True,
                details={"stage": harness.stage, "error_type": type(exc).__name__, "skill_candidates": 0, "failure_class": "infrastructure"},
            ) from exc
        evidence = dict(harness.evidence)
        events_log = evidence.pop("episode_events_log")
        measurements = {
            **evidence,
            "task_family": family,
            "scientific_evidence": self.scientific,
            "scientific_retrieval_count": result.retrieval_count,
            "acquisition_policy_version": ACQUISITION_POLICY_VERSION,
            "acquisition_action_budget": self.action_budget,
            "queue_sha256": self.queue_sha256,
            "skill_pool_size_before": len(pool),
            "skill_pool_hash_before": pool_hash(pool),
            **unit_lineage(unit),
        }
        return replace(result.outcome, measurements=measurements, log_paths=(*result.log_paths, events_log))
