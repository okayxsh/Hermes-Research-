"""Shared evidence, capability, and fail-closed helpers for real pilot handlers."""
from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rq1.bridge.environment import real_adapter_capability
from rq1.hermes.capabilities import probe_hermes_capabilities
from rq1.pilot.models import EvidenceLevel, PilotStatus, RuntimeExecution
from rq1.recovery.verification import real_recovery_capabilities
from rq1.utils.time import utc_now


@dataclass(frozen=True)
class CapabilitySnapshot:
    generated_at: str
    alfworld: dict[str, Any]
    hermes: dict[str, Any]
    recovery: dict[str, Any]
    platform: dict[str, Any]

    def to_dict(self) -> dict[str, Any]: return asdict(self)


@dataclass(frozen=True)
class RealExecutionContext:
    root: Path
    run_id: str
    attempt_id: str
    output_dir: Path
    candidate_model: str = "hermes3:8b"

    def snapshot(self) -> CapabilitySnapshot:
        alfworld = real_adapter_capability().to_dict()
        hermes = probe_hermes_capabilities(project_root=self.root).to_dict()
        if hermes.get("executable"):
            hermes["executable"] = Path(str(hermes["executable"])).name
        return CapabilitySnapshot(utc_now(), alfworld, hermes, real_recovery_capabilities(), {
            "system": platform.system(), "machine": platform.machine(), "release": platform.release(),
        })


def passed(level: EvidenceLevel, details: dict[str, Any]) -> RuntimeExecution:
    return RuntimeExecution(PilotStatus.PASSED, level, details)


def blocked(code: str, message: str, remediation: str, details: dict[str, Any] | None = None) -> RuntimeExecution:
    value = {"operation_executed": False, "block_code": code, **(details or {})}
    return RuntimeExecution(PilotStatus.BLOCKED, EvidenceLevel.STATIC, value, message, remediation)


def failed(code: str, message: str, details: dict[str, Any] | None = None) -> RuntimeExecution:
    return RuntimeExecution(PilotStatus.FAILED, EvidenceLevel.REAL_COMPONENT, {"operation_executed": True, "failure_code": code, **(details or {})}, message, "Inspect this immutable attempt and start a new attempt after remediation.")


def write_handler_evidence(context: RealExecutionContext, handler: str, payload: dict[str, Any]) -> str:
    path = context.output_dir / "handler-evidence.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema_version": 1, "handler": handler, "capability_snapshot": context.snapshot().to_dict(), **payload}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return str(path.relative_to(context.root))
