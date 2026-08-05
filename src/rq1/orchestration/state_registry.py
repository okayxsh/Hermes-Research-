from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from rq1.orchestration.stages import STAGES, STAGE_MAP, VALID_STAGE_STATUSES, next_stage
from rq1.utils.time import utc_now


class StageTransitionError(RuntimeError):
    pass


@dataclass
class StageState:
    status: str = "not_started"
    completed_at: str | None = None
    report: str | None = None
    attempt_id: str | None = None


class StageRegistry:
    def __init__(self, path: Path) -> None:
        self.path = path

    def initialize(self) -> None:
        if self.path.exists():
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._save({stage.name: StageState() for stage in STAGES})

    def _load(self) -> dict[str, StageState]:
        self.initialize()
        raw: dict[str, Any] = json.loads(self.path.read_text(encoding="utf-8"))
        stages = raw.get("stages", {})
        return {name: StageState(**stages.get(name, {})) for name in STAGE_MAP}

    def _save(self, states: dict[str, StageState]) -> None:
        payload = {"schema_version": 1, "stages": {name: asdict(value) for name, value in states.items()}}
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(self.path)

    def status(self) -> dict[str, StageState]:
        return self._load()

    def can_start(self, name: str) -> tuple[bool, str | None]:
        if name not in STAGE_MAP:
            return False, f"Unknown stage: {name}"
        states = self._load()
        blocked = [item for item in STAGE_MAP[name].prerequisites if states[item].status != "passed"]
        return (not blocked, None if not blocked else f"Prerequisites not passed: {', '.join(blocked)}")

    def mark_running(self, name: str, attempt_id: str) -> None:
        allowed, reason = self.can_start(name)
        if not allowed:
            raise StageTransitionError(reason or "Stage cannot start")
        states = self._load()
        if states[name].status == "passed":
            raise StageTransitionError(f"{name} already passed; use an explicit invalidation workflow.")
        states[name] = StageState(status="running", attempt_id=attempt_id)
        self._save(states)

    def finish(self, name: str, status: str, report: str) -> str | None:
        if status not in VALID_STAGE_STATUSES - {"not_started", "running"}:
            raise StageTransitionError(f"Invalid terminal stage status: {status}")
        states = self._load()
        if states[name].status != "running":
            raise StageTransitionError(f"{name} is not running")
        states[name].status = status
        states[name].completed_at = utc_now()
        states[name].report = report
        self._save(states)
        return next_stage(name) if status == "passed" else None
