"""Durable-runner executor for one amended controlled-recovery evaluation unit.

Order per unit (fresh session): reset -> replay the frozen checkpoint prefix -> the
frozen controlled detour (digest-verified) -> failure context -> exactly one
Sentence-BERT top-3 retrieval (none for NoLib) -> one structured recovery-memory
block -> model recovery actions within the frozen budget.  The model never sees
the condition label, cosine scores, or any oracle/reference action.  Evaluation
never writes skills.
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from rq1.acquisition.environment import SBERT_MODEL, SBERT_REVISION
from rq1.evaluation.amended_libraries import AmendedLibrary
from rq1.evaluation.amended_matrix import EvaluationTask
from rq1.evaluation.amended_protocol import EVALUATION_POLICY_VERSION, EVALUATION_PROFILE, MODEL_DIGEST, RETRIEVAL_TOP_K
from rq1.evaluation.recovery_executor import RecoveryEpisodeSpec, run_recovery_episode
from rq1.experiment.models import ExperimentUnit, RunExecutionContext, RunFailure, RunOutcome, canonical_hash
from rq1.pilot.real_runtime.harnesses import RealRecoveryHarness
from rq1.recovery.controlled_failure import ControlledFailureError
from rq1.retrieval.controller import build_retrieval_boundary

RESULT_SCHEMA = "rq1-evaluation-result-v1"


def _events(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def recovery_timing(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Wall-clock recovery latency from the recovery-memory injection to the successful step."""
    injected = next((event["timestamp"] for event in events if event.get("event") == "recovery_memory_injected"), None)
    success = next((event["timestamp"] for event in events
                    if event.get("event") == "tool_result" and injected is not None and event["timestamp"] >= injected
                    and (event.get("payload") or {}).get("tool") == "alfworld_step"
                    and ((event.get("payload") or {}).get("response") or {}).get("done") is True
                    and ((event.get("payload") or {}).get("response") or {}).get("success") is True), None)
    return {
        "recovery_memory_injected_at": injected,
        "recovery_success_at": success,
        "recovery_latency_seconds": round(success - injected, 3) if injected is not None and success is not None else None,
    }


class AmendedEvaluationExecutor:
    def __init__(
        self,
        driver: Any,
        *,
        tasks: Mapping[str, EvaluationTask],
        libraries: Mapping[str, AmendedLibrary],
        embedder: Any,
        scientific: bool,
        split: str,
        provenance: Mapping[str, Any],
    ) -> None:
        self.driver = driver
        self.tasks = dict(tasks)
        self.libraries = dict(libraries)
        self.embedder = embedder
        self.scientific = scientific
        self.split = split
        self.provenance = dict(provenance)

    def __call__(self, unit: ExperimentUnit, context: RunExecutionContext) -> RunOutcome:
        task = self.tasks[unit.task_id]
        library = self.libraries[unit.condition]
        seed = int(unit.seed)
        # Decision 012: the replicate seed is the inference seed, identical across conditions.
        self.driver.inference_seed = seed
        harness = RealRecoveryHarness(
            self.driver, output_dir=context.output_dir / "episode", run_id=context.experiment_id, attempt_id=context.attempt_id,
            reference_actions=task.reference_actions, total_action_limit=len(task.prefix_actions) + 1 + task.recovery_action_budget,
            allowed_split=self.split, profile_prefix=EVALUATION_PROFILE, frozen_perturbation=task.frozen_perturbation(),
        )
        boundary = build_retrieval_boundary(
            library.retrieval_documents(), self.embedder, embedding_model=SBERT_MODEL, top_k=RETRIEVAL_TOP_K,
            embedding_model_revision=SBERT_REVISION,
        )
        spec = RecoveryEpisodeSpec(
            run_id=context.experiment_id, attempt_id=context.attempt_id, task_id=task.task_id, task_family=task.task_family,
            split=self.split, seed=seed, condition=unit.condition, library_name=unit.condition, library_size=library.size,
            library_hash=library.content_sha256, checkpoint_id=task.checkpoint_id, prefix_actions=task.prefix_actions,
            action_budget=task.recovery_action_budget, episode_id=unit.run_key,
        )
        stage = "episode"
        try:
            result = run_recovery_episode(harness, spec, boundary, log_dir=context.output_dir / "result")
            stage = "measurement"
            session = harness.session
            assert session is not None
            recovery_records = [record for record in session.records if record.phase == "recovery"]
            rejections = dict(session.rejection_counts)
            exhausted = bool(session.selection_failures)
            success = harness.recovery_succeeded()
            events_log = context.output_dir / "episode" / "episode-events.jsonl"
        except ControlledFailureError as exc:
            raise RunFailure(
                f"evaluation unit could not reproduce the frozen controlled failure: {exc.code}: {exc}",
                safe_to_continue=True, mutation_state_known=True,
                details={"stage": "controlled_failure", "code": exc.code, "failure_class": "infrastructure", "eligible": False},
            ) from exc
        except Exception as exc:
            raise RunFailure(
                f"evaluation infrastructure failure during {stage}: {type(exc).__name__}: {exc}",
                safe_to_continue=True, mutation_state_known=True,
                details={"stage": stage, "error_type": type(exc).__name__, "failure_class": "infrastructure", "eligible": False},
            ) from exc
        finally:
            harness.close()
        if len(recovery_records) > task.recovery_action_budget:
            raise RunFailure("recovery exceeded the frozen action budget", safe_to_continue=False, mutation_state_known=True,
                             details={"stage": "measurement", "failure_class": "protocol"})
        retrieval = result.retrieval_event
        top = [{"rank": rank, "skill_id": item["skill_id"], "score": item["score"]} for rank, item in enumerate(retrieval.get("top") or [], 1)]
        if library.size and (result.no_retrieval or len(top) != min(RETRIEVAL_TOP_K, library.size)):
            raise RunFailure("library condition did not retrieve exactly the top-3 skills", safe_to_continue=False, mutation_state_known=True,
                             details={"stage": "retrieval", "failure_class": "protocol"})
        if not library.size and (not result.no_retrieval or top):
            raise RunFailure("NoLib performed a retrieval", safe_to_continue=False, mutation_state_known=True,
                             details={"stage": "retrieval", "failure_class": "protocol"})
        if success:
            termination = "success"
        elif exhausted:
            termination = "action_selection_exhausted"
        elif len(recovery_records) >= task.recovery_action_budget:
            termination = "recovery_budget_exhausted"
        else:
            termination = "environment_terminated_without_success"
        timing = recovery_timing(_events(events_log))
        measurements = {
            "result_schema": RESULT_SCHEMA,
            "scientific_evidence": self.scientific,
            "evaluation_policy_version": EVALUATION_POLICY_VERSION,
            **self.provenance,
            "task_id": task.task_id,
            "task_family": task.task_family,
            "split": self.split,
            "seed": seed,
            "inference_seed": seed,
            "condition": unit.condition,
            "library_size": library.size,
            "library_hash": library.content_sha256,
            "library_core_sha256": library.core_sha256 if library.size else None,
            "model": self.driver.model_name,
            "model_digest": MODEL_DIGEST,
            "matrix_unit": dict(unit.payload),
            "checkpoint_id": task.checkpoint_id,
            "checkpoint_prefix_length": len(task.prefix_actions),
            "checkpoint_digest": harness.checkpoint_digest,
            "perturbation": result.failure,
            "perturbation_digest": result.failure.get("post_state_digest"),
            "failure_context": result.failure_context,
            "failure_context_sha256": canonical_hash(result.failure_context),
            "retrieval": {"performed": not result.no_retrieval, "event_id": retrieval.get("event_id"), "query_text_hash": retrieval.get("query_text_hash"),
                          "top": top, "ranking_size": len(retrieval.get("ranking") or [])},
            "retrieval_events": 0 if result.no_retrieval else 1,
            "no_retrieval": result.no_retrieval,
            "recovery_memory_sha256": canonical_hash(result.recovery_memory),
            "eligible": True,
            "post_failure_budget_complete": True,
            "recovery_action_budget": task.recovery_action_budget,
            "post_failure_actions": len(recovery_records),
            "recovery_success": success,
            "task_completed": success,
            "termination_reason": termination,
            "recovery_latency_actions": len(recovery_records) if success else None,
            **timing,
            "invalid_action_selections": sum(rejections.values()),
            "selection_rejections": rejections,
            "retries": sum(rejections.values()),
            "selection_exhausted": exhausted,
            "recovery_actions": [record.action for record in recovery_records],
            "skill_writes": False,
        }
        return replace(result.outcome, measurements=measurements, log_paths=(*result.log_paths, str(events_log)))
