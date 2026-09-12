"""Real train-only acquisition episode.

This path deliberately performs ZERO scientific retrieval. It never constructs
or imports the Sentence-BERT retrieval boundary. A successful episode may
produce one skill candidate carrying its task family and provenance; a failed
episode may not produce a positive skill. The candidate feeds the named-library
assembler via :func:`rq1.skills.library.acquired_skill_from_candidate`.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

from rq1.experiment.models import RunOutcome
from rq1.utils.hashing import sha256_text


class AcquisitionHarness(Protocol):
    """Adapter to the real Hermes + ALFWorld train path (injected so it can be faked)."""

    def run_episode(self, task_id: str, split: str, seed: int, action_limit: int) -> Mapping[str, Any]:
        """Run one real train episode.

        Returns a mapping with at least ``success`` (bool), and optionally
        ``steps``, ``actions``, ``invalid_actions``, and ``skill_candidate``.
        Must raise on infrastructure errors rather than returning a failure.
        """
    def post_run_library_hash(self) -> str: ...
    def post_run_library_size(self) -> int: ...


@dataclass(frozen=True)
class AcquisitionEpisodeResult:
    outcome: RunOutcome
    skill_candidate: dict[str, Any] | None
    log_paths: tuple[str, ...]
    retrieval_count: int  # invariant: always 0


def _normalise_candidate(
    candidate: Mapping[str, Any], *, task_id: str, task_family: str, attempt_id: str
) -> dict[str, Any]:
    value = dict(candidate)
    value["task_family"] = task_family
    value["source_task_id"] = task_id
    value["source_attempt_id"] = attempt_id
    if not value.get("skill_id"):
        value["skill_id"] = "skill_" + sha256_text(
            f"{task_id}\0{value.get('title', '')}\0{value.get('body', '')}"
        )[:16]
    return value


def run_acquisition_episode(
    harness: AcquisitionHarness,
    *,
    task_id: str,
    task_family: str,
    attempt_id: str,
    log_dir: Path,
    split: str = "train",
    seed: int = 0,
    action_limit: int = 50,
) -> AcquisitionEpisodeResult:
    if split != "train":
        raise ValueError("acquisition accepts the train split only")
    started = time.monotonic()
    result = dict(harness.run_episode(task_id, split, seed, action_limit))
    success = bool(result.get("success"))

    candidate: dict[str, Any] | None = None
    if success and result.get("skill_candidate") is not None:
        candidate = _normalise_candidate(
            result["skill_candidate"], task_id=task_id, task_family=task_family, attempt_id=attempt_id
        )
    # A failed episode may not create a positive skill.
    if not success:
        candidate = None

    log_dir.mkdir(parents=True, exist_ok=True)
    log = log_dir / "acquisition_episode.jsonl"
    with log.open("w", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "task_id": task_id,
                    "task_family": task_family,
                    "attempt_id": attempt_id,
                    "split": split,
                    "success": success,
                    "steps": result.get("steps"),
                    "actions": result.get("actions"),
                    "invalid_actions": result.get("invalid_actions"),
                    "skill_candidate": candidate,
                    "retrieval_count": 0,
                },
                sort_keys=True,
            )
            + "\n"
        )

    latency_ms = round((time.monotonic() - started) * 1000, 3)
    outcome = RunOutcome(
        success=success,
        actions=result.get("actions"),
        steps=result.get("steps"),
        invalid_actions=result.get("invalid_actions"),
        latency_ms=latency_ms,
        # Acquisition never retrieves: both retrieval fields stay empty.
        retrieved_skill_ids=(),
        retrieval_similarity_scores=(),
        log_paths=(str(log),),
        skill_library_hash_after=harness.post_run_library_hash(),
        library_size_after=harness.post_run_library_size(),
    )
    return AcquisitionEpisodeResult(
        outcome=outcome, skill_candidate=candidate, log_paths=outcome.log_paths, retrieval_count=0
    )
