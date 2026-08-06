"""Atomic, resumable state for Phase 6 pilot runs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rq1.pilot.catalog import PILOT_TESTS
from rq1.pilot.models import PilotStatus
from rq1.utils.time import utc_now


class PilotRegistryError(RuntimeError):
    pass


class PilotRegistry:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.directory = root / "state" / "pilot_runs"
        self.latest_path = root / "state" / "pilot_latest.json"

    def path(self, run_id: str) -> Path:
        return self.directory / f"{run_id}.json"

    def create(self, run_id: str, mode: str, fingerprint: str, selected: list[str], candidate_model: str) -> dict[str, Any]:
        path = self.path(run_id)
        if path.exists():
            raise PilotRegistryError(f"Pilot run already exists: {run_id}")
        payload = {
            "schema_version": 1,
            "run_id": run_id,
            "mode": mode,
            "status": "running",
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "input_fingerprint": fingerprint,
            "candidate_model": candidate_model,
            "selected_tests": selected,
            "tests": {
                spec.test_id: {
                    "status": PilotStatus.NOT_STARTED.value,
                    "attempts": [],
                    "latest_attempt": None,
                    "evidence_level": None,
                    "manual_evidence": [],
                }
                for spec in PILOT_TESTS
            },
        }
        self.save(payload)
        self._write_latest(run_id)
        return payload

    def load(self, run_id: str) -> dict[str, Any]:
        path = self.path(run_id)
        if not path.exists():
            raise PilotRegistryError(f"Unknown pilot run: {run_id}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise PilotRegistryError(f"Invalid pilot state: {path}") from exc
        if payload.get("run_id") != run_id or not isinstance(payload.get("tests"), dict):
            raise PilotRegistryError(f"Invalid pilot state: {path}")
        return payload

    def save(self, payload: dict[str, Any]) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        payload["updated_at"] = utc_now()
        path = self.path(str(payload["run_id"]))
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(path)

    def transition(
        self, run_id: str, test_id: str, target: PilotStatus, *, attempt_id: str | None = None,
        attempt_path: str | None = None, evidence_level: str | None = None,
    ) -> dict[str, Any]:
        payload = self.load(run_id)
        if test_id not in payload["tests"]:
            raise PilotRegistryError(f"Unknown pilot test: {test_id}")
        state = payload["tests"][test_id]
        current = PilotStatus(state["status"])
        allowed = {
            PilotStatus.NOT_STARTED: {PilotStatus.RUNNING, PilotStatus.BLOCKED, PilotStatus.SKIPPED},
            PilotStatus.RUNNING: {PilotStatus.PASSED, PilotStatus.FAILED, PilotStatus.BLOCKED, PilotStatus.INTERRUPTED},
            PilotStatus.FAILED: {PilotStatus.RUNNING, PilotStatus.INVALIDATED},
            PilotStatus.BLOCKED: {PilotStatus.RUNNING, PilotStatus.INVALIDATED},
            PilotStatus.INTERRUPTED: {PilotStatus.RUNNING, PilotStatus.INVALIDATED},
            PilotStatus.INVALIDATED: {PilotStatus.RUNNING, PilotStatus.BLOCKED},
            PilotStatus.SKIPPED: {PilotStatus.INVALIDATED, PilotStatus.RUNNING},
            PilotStatus.PASSED: {PilotStatus.INVALIDATED},
        }
        if target not in allowed[current]:
            raise PilotRegistryError(f"Invalid pilot transition {test_id}: {current.value} -> {target.value}")
        state["status"] = target.value
        if attempt_id:
            state["latest_attempt"] = attempt_id
        if attempt_path:
            state["attempts"].append(attempt_path)
        if evidence_level:
            state["evidence_level"] = evidence_level
        self.save(payload)
        return payload

    def mark_stale_running_interrupted(self, run_id: str) -> list[str]:
        payload = self.load(run_id)
        interrupted: list[str] = []
        for test_id, state in payload["tests"].items():
            if state["status"] == PilotStatus.RUNNING.value:
                state["status"] = PilotStatus.INTERRUPTED.value
                interrupted.append(test_id)
        if interrupted:
            self.save(payload)
        return interrupted

    def invalidate_all_passed(self, run_id: str) -> list[str]:
        payload = self.load(run_id)
        invalidated: list[str] = []
        for test_id, state in payload["tests"].items():
            if state["status"] == PilotStatus.PASSED.value:
                state["status"] = PilotStatus.INVALIDATED.value
                invalidated.append(test_id)
        if invalidated:
            self.save(payload)
        return invalidated

    def add_manual_evidence(self, run_id: str, test_id: str, evidence: dict[str, Any]) -> None:
        payload = self.load(run_id)
        if test_id not in payload["tests"]:
            raise PilotRegistryError(f"Unknown pilot test: {test_id}")
        payload["tests"][test_id]["manual_evidence"].append(evidence)
        self.save(payload)

    def latest(self) -> str | None:
        if not self.latest_path.exists(): return None
        try: return str(json.loads(self.latest_path.read_text(encoding="utf-8"))["run_id"])
        except (OSError, ValueError, KeyError, json.JSONDecodeError): return None

    def _write_latest(self, run_id: str) -> None:
        self.latest_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.latest_path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"run_id": run_id, "updated_at": utc_now()}, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.latest_path)
