"""Real controlled-recovery evaluation episode in the frozen authoritative order.

Order (must not change):

    task start -> replay/reach frozen checkpoint -> apply controlled failure ->
    validate failure + solvability -> build frozen failure context ->
    Sentence-BERT retrieval EXACTLY ONCE -> inject structured recovery memory ->
    recovery actions -> persist result/retrieval/recovery logs.

The retrieval boundary is injected and enforces exactly one crossing per
episode. NoLib uses the same harness and the same boundary, but the library is
empty so the boundary returns ``no_retrieval = true``. The executor never
retrieves at individual ALFWorld steps.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from rq1.experiment.models import RunOutcome
from rq1.recovery.controlled_failure import (
    CANONICAL_FAILURE_MESSAGE,
    FailureContext,
    FailureEnvironment,
    FailureTrajectory,
    apply_controlled_failure,
)
from rq1.recovery.models import RecoveryState
from rq1.retrieval.controller import RetrievalContext, SingleRetrievalBoundary
from rq1.retrieval.logging import RetrievalEventLog
from rq1.retrieval.models import RetrievalQuery


class RecoveryHarness(Protocol):
    """Adapter to the real Hermes + ALFWorld path (injected so it can be faked)."""

    def start_and_replay(
        self, task_id: str, split: str, seed: int, prefix_actions: Sequence[str]
    ) -> RecoveryState: ...
    def current_state(self) -> RecoveryState: ...
    def failure_environment(self) -> FailureEnvironment: ...
    def inject_recovery_memory(self, message: Mapping[str, Any]) -> None: ...
    def run_recovery(self, action_budget: int) -> Sequence[Mapping[str, Any]]: ...
    def recovery_succeeded(self) -> bool: ...
    def failure_trajectory(self) -> FailureTrajectory: ...


@dataclass(frozen=True)
class RecoveryEpisodeSpec:
    run_id: str
    attempt_id: str
    task_id: str
    task_family: str
    split: str
    seed: int
    condition: str
    library_name: str
    library_size: int
    library_hash: str | None
    checkpoint_id: str
    prefix_actions: tuple[str, ...]
    action_budget: int
    episode_id: str | None = None


@dataclass(frozen=True)
class RecoveryEpisodeResult:
    outcome: RunOutcome
    retrieval_event: dict[str, Any]
    failure_context: dict[str, Any]
    recovery_memory: dict[str, Any]
    failure: dict[str, Any]
    trajectory: dict[str, Any]
    recovery_steps: tuple[dict[str, Any], ...]
    retrieval_count: int
    no_retrieval: bool
    log_paths: tuple[str, ...]


def run_recovery_episode(
    harness: RecoveryHarness,
    spec: RecoveryEpisodeSpec,
    boundary: SingleRetrievalBoundary,
    *,
    log_dir: Path,
    failure_message: str = CANONICAL_FAILURE_MESSAGE,
) -> RecoveryEpisodeResult:
    started = time.monotonic()

    # 1-2. Task start and frozen checkpoint replay.
    harness.start_and_replay(spec.task_id, spec.split, spec.seed, spec.prefix_actions)

    # 3-4. The real harness performs a reversible action detour and proves it
    # solvable before retrieval.  The legacy fake harness remains only for the
    # hermetic controlled-relocation unit tests.
    action_perturbation = getattr(harness, "apply_controlled_action_perturbation", None)
    if callable(action_perturbation):
        failure = action_perturbation(spec.checkpoint_id, failure_message)
    else:
        failure = apply_controlled_failure(
            harness.failure_environment(), checkpoint_id=spec.checkpoint_id, failure_message=failure_message
        )
    state = harness.current_state()

    # 5. Frozen failure context.
    failure_context = FailureContext(
        task_instruction=state.instruction,
        observation=state.observation,
        inventory=tuple(state.inventory),
        failure_message=failure.failure_message,
    )

    # 6. Sentence-BERT retrieval exactly once.
    query = RetrievalQuery(
        task_instruction=failure_context.task_instruction,
        observation=failure_context.observation,
        inventory=failure_context.inventory,
        failure_message=failure_context.failure_message,
    )
    retrieval_context = RetrievalContext(
        run_id=spec.run_id,
        attempt_id=spec.attempt_id,
        task_id=spec.task_id,
        condition=spec.condition,
        library_name=spec.library_name,
        library_size=spec.library_size,
        library_hash=spec.library_hash,
        episode_id=spec.episode_id,
    )
    retrieval_outcome = boundary.retrieve(retrieval_context, query)

    # 7. Structured recovery memory injection (rank + id + text only).
    harness.inject_recovery_memory(retrieval_outcome.recovery_memory.to_dict())

    # 8. Recovery actions.
    steps = tuple(dict(step) for step in harness.run_recovery(spec.action_budget))
    success = bool(harness.recovery_succeeded())
    trajectory = harness.failure_trajectory()

    # 9. Persist retrieval/failure/recovery logs.
    log_dir.mkdir(parents=True, exist_ok=True)
    retrieval_log = log_dir / "retrieval.jsonl"
    RetrievalEventLog(retrieval_log).append(retrieval_outcome.event)
    failure_log = log_dir / "failure.json"
    failure_log.write_text(
        json.dumps(
            {
                "failure_context": failure_context.to_dict(),
                "failure": failure.to_dict(),
                "trajectory": trajectory.to_dict(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    steps_log = log_dir / "recovery_steps.jsonl"
    with steps_log.open("w", encoding="utf-8") as handle:
        for step in steps:
            handle.write(json.dumps(step, sort_keys=True) + "\n")

    retrieval_event = retrieval_outcome.event.to_dict()
    action_count = sum(1 for step in steps if step.get("action"))
    invalid_count = sum(1 for step in steps if step.get("action_valid") is False)
    latency_ms = round((time.monotonic() - started) * 1000, 3)
    outcome = RunOutcome(
        success=success,
        recovery_success=success,
        retrieved_skill_ids=tuple(candidate.skill_id for candidate in retrieval_outcome.result.top),
        retrieval_similarity_scores=tuple(candidate.score for candidate in retrieval_outcome.result.top),
        actions=action_count,
        steps=len(steps),
        invalid_actions=invalid_count,
        latency_ms=latency_ms,
        log_paths=(str(retrieval_log), str(failure_log), str(steps_log)),
    )
    return RecoveryEpisodeResult(
        outcome=outcome,
        retrieval_event=retrieval_event,
        failure_context=failure_context.to_dict(),
        recovery_memory=retrieval_outcome.recovery_memory.to_dict(),
        failure=failure.to_dict(),
        trajectory=trajectory.to_dict(),
        recovery_steps=steps,
        retrieval_count=1,
        no_retrieval=not retrieval_outcome.result.retrieved,
        log_paths=outcome.log_paths,
    )
