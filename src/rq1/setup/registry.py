from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from rq1.setup.models import SETUP_STAGES, SETUP_STAGE_MAP, SetupState


class SetupRegistry:
    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> dict[str, SetupState]:
        if not self.path.exists():
            return {stage.name: SetupState() for stage in SETUP_STAGES}
        try:
            raw: dict[str, Any] = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict) or not isinstance(raw.get("stages", {}), dict):
                raise ValueError("root and stages must be JSON objects")
            values = raw.get("stages", {})
            states = {name: SetupState(**values.get(name, {})) for name in SETUP_STAGE_MAP}
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Invalid setup state at {self.path}; move it aside and rerun setup with --resume"
            ) from exc
        valid_statuses = {"pending", "running", "passed", "failed", "blocked", "skipped"}
        if any(state.status not in valid_statuses for state in states.values()):
            raise RuntimeError(
                f"Invalid setup state at {self.path}; move it aside and rerun setup with --resume"
            )
        return states

    def save(self, states: dict[str, SetupState]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "stages": {name: asdict(state) for name, state in states.items()},
        }
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(self.path)

    def invalidate_from(self, stage_name: str) -> list[str]:
        if stage_name not in SETUP_STAGE_MAP:
            raise ValueError(f"Unknown setup stage: {stage_name}")
        states = self.load()
        names = [stage.name for stage in SETUP_STAGES]
        invalidated = names[names.index(stage_name) :]
        for name in invalidated:
            states[name] = SetupState()
        self.save(states)
        return invalidated
