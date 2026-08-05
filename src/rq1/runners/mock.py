from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rq1.analysis.metrics import invalid_action_rate, retrieval_noise_rate, task_success_rate
from rq1.logging.run_registry import Run, RunRegistry
from rq1.utils.ids import deterministic_run_id
from rq1.utils.time import utc_now


@dataclass
class MockRunOutput:
    result: dict[str, Any]
    steps: list[dict[str, Any]]
    skill_events: list[dict[str, Any]]


def deterministic_episode(run_id: str, relevant: bool = True) -> MockRunOutput:
    steps = [
        {"run_id": run_id, "step": 1, "observation": "You are in a room.", "selected_action": "look", "action_valid": True},
        {"run_id": run_id, "step": 2, "observation": "A target is visible.", "selected_action": "take target", "action_valid": True},
    ]
    events = [{"run_id": run_id, "step": 1, "skill_id": "skill_0001", "event": "skill_view", "relevant": relevant}]
    return MockRunOutput({"run_id": run_id, "success": True, "status": "completed", "completed_at": utc_now()}, steps, events)


def run_mock_workflow(root: Path) -> dict[str, Any]:
    registry = RunRegistry(root / "state" / "run_registry.sqlite")
    run_id = deterministic_run_id("L0", "mock_001", 1)
    try:
        registry.plan(Run(run_id, "mock_001", "valid_seen", "L0", "rq1-mock", 1, "planned"))
    except Exception:
        pass  # Idempotent: already planned/completed mock run is retained.
    claimed = registry.claim_next("mock-machine")
    if claimed is None:
        return {"status": "already_complete", "metrics": {}}
    registry.transition(claimed.run_id, "claimed", "running", started_at=utc_now())
    episode = deterministic_episode(claimed.run_id)
    registry.transition(claimed.run_id, "running", "completed", completed_at=utc_now(), result_path=f"runs/pilot/{claimed.run_id}.json")
    return {
        "status": "completed",
        "result": episode.result,
        "metrics": {
            "success_rate": task_success_rate([episode.result]),
            "invalid_action_rate": invalid_action_rate(episode.steps),
            "retrieval_noise_rate": retrieval_noise_rate(episode.skill_events),
        },
    }
