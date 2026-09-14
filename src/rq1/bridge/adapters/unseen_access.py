"""Explicit, logged authorization for any valid_unseen access.

valid_unseen stays unavailable by default.  Two named purposes may open it, and
only when the operator sets ``RQ1_VALID_UNSEEN_ACCESS`` to that purpose:

- ``evaluation-task-preparation``: task-selection metadata, hand-coded expert
  routes, and oracle validation of the frozen controlled failure.  No model is
  called and no agent outcome exists.
- ``scientific-evaluation``: the approved, frozen evaluation run itself.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

UNSEEN_ACCESS_ENV = "RQ1_VALID_UNSEEN_ACCESS"
TASK_PREPARATION = "evaluation-task-preparation"
SCIENTIFIC_EVALUATION = "scientific-evaluation"
PURPOSES = (TASK_PREPARATION, SCIENTIFIC_EVALUATION)
BASE_REAL_SPLITS = frozenset({"train", "valid_seen"})


class UnseenAccessError(PermissionError):
    pass


def authorized_purpose() -> str | None:
    value = os.environ.get(UNSEEN_ACCESS_ENV)
    return value if value in PURPOSES else None


def allowed_real_splits() -> frozenset[str]:
    return BASE_REAL_SPLITS | ({"valid_unseen"} if authorized_purpose() else set())


def require_unseen_access(*purposes: str) -> str:
    purpose = authorized_purpose()
    if purpose is None or (purposes and purpose not in purposes):
        wanted = " or ".join(purposes or PURPOSES)
        raise UnseenAccessError(f"valid_unseen access requires {UNSEEN_ACCESS_ENV}={wanted}")
    return purpose


def log_unseen_access(path: Path, *, operation: str, task_id: str | None = None, detail: str | None = None) -> None:
    from rq1.utils.time import utc_now

    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"timestamp": utc_now(), "purpose": authorized_purpose(), "operation": operation, "task_id": task_id, "detail": detail}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
