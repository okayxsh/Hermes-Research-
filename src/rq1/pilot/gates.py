"""Selection, split, prerequisite, and evidence gates."""
from __future__ import annotations

from typing import Any

from rq1.pilot.models import EVIDENCE_RANK, EvidenceLevel, PilotMode, PilotStatus, PilotTestSpec


def validate_task_manifest(payload: dict[str, Any], *, allowed_splits: set[str]) -> list[str]:
    errors: list[str] = []
    split = payload.get("split")
    if split not in allowed_splits:
        errors.append(f"split must be one of: {', '.join(sorted(allowed_splits))}")
    tasks = payload.get("tasks")
    if not isinstance(tasks, list):
        errors.append("tasks must be an array")
    elif any(not isinstance(item, dict) or not isinstance(item.get("task_id"), str) for item in tasks):
        errors.append("every task must be an object with a task_id string")
    if split == "valid_unseen":
        errors.append("valid_unseen is forbidden during Phase 6")
    return errors


def unmet_prerequisites(spec: PilotTestSpec, states: dict[str, Any]) -> tuple[str, ...]:
    return tuple(
        prerequisite for prerequisite in spec.prerequisites
        if states.get(prerequisite, {}).get("status") != PilotStatus.PASSED.value
    )


def evidence_satisfies(mode: PilotMode, spec: PilotTestSpec, actual: EvidenceLevel) -> bool:
    required = spec.fake_evidence if mode == PilotMode.FAKE else spec.real_evidence
    return EVIDENCE_RANK[actual] >= EVIDENCE_RANK[required]
