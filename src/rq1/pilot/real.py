"""Real Phase 7 runtime router; every pilot test has an explicit capability-gated handler."""
from __future__ import annotations

from pathlib import Path

from rq1.pilot.models import PilotTestSpec, RuntimeExecution
from rq1.pilot.real_runtime.base import RealExecutionContext, write_handler_evidence
from rq1.pilot.real_runtime.router import build_handlers


class RealPilotRuntime:
    simulated = False

    def __init__(self, root: Path, handlers: dict[int, object] | None = None) -> None:
        self.root = root.resolve()
        self.handlers = handlers or build_handlers()

    def execute(self, spec: PilotTestSpec, *, run_id: str, attempt_id: str, output_dir: Path) -> RuntimeExecution:
        index = int(spec.test_id.split("_")[1])
        context = RealExecutionContext(self.root, run_id, attempt_id, output_dir)
        execution = self.handlers[index](context)
        details = dict(execution.details)
        details.setdefault("handler", f"pilot_{index:02d}")
        details.setdefault("real_operation_executed", bool(details.get("operation_executed", False)))
        details["real_execution_phase"] = 7
        details["capability_snapshot_path"] = write_handler_evidence(context, str(details["handler"]), {"execution": {"status": execution.status.value, "evidence_level": execution.evidence_level.value, "details": details}})
        return RuntimeExecution(execution.status, execution.evidence_level, details, execution.error, execution.remediation)
