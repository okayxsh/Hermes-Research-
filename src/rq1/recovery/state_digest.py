"""Canonical observable-state hashing; it does not assert internal-state equality."""
from __future__ import annotations

import hashlib
import json
from typing import Any

from rq1.recovery.models import RecoveryState

DIGEST_SCHEMA_VERSION = 1


def canonical_json(value: dict[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def observable_payload(state: RecoveryState, *, include_admissible_actions: bool = True) -> dict[str, Any]:
    value: dict[str, Any] = {"schema_version": DIGEST_SCHEMA_VERSION, "task_id": state.task_id,
        "task_family": state.task_family, "observation": state.observation, "inventory": list(state.inventory),
        "step_number": state.step_number, "done": state.done, "success": state.success}
    if include_admissible_actions:
        value["admissible_actions"] = list(state.admissible_actions)
    return value


def digest_payload(value: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def observable_digest(state: RecoveryState, *, include_admissible_actions: bool = True) -> str:
    return digest_payload(observable_payload(state, include_admissible_actions=include_admissible_actions))


def internal_digest(state: RecoveryState) -> str | None:
    if state.internal_state is None:
        return None
    return digest_payload({"schema_version": DIGEST_SCHEMA_VERSION, "internal_state": state.internal_state})
