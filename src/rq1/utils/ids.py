from __future__ import annotations

from uuid import uuid4


def new_attempt_id() -> str:
    return str(uuid4())


def deterministic_run_id(snapshot: str, task_id: str, repetition: int) -> str:
    return f"{snapshot}_{task_id}_r{repetition}"
