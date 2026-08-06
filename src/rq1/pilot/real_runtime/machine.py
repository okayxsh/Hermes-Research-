from __future__ import annotations
import platform
from rq1.pilot.models import EvidenceLevel
from rq1.pilot.real_runtime.base import RealExecutionContext, blocked, passed

def run(context: RealExecutionContext):
    supported = platform.system() == "Linux" and platform.machine().lower() in {"x86_64", "amd64"}
    details = {"handler": "machine", "system": platform.system(), "machine": platform.machine(), "supported_university_platform": supported, "operation_executed": True}
    return passed(EvidenceLevel.INSTALLED, details) if supported else blocked("unsupported_machine", "Real pilot requires supported x86_64 Ubuntu/WSL2.", "Run Phase 7 on the approved university machine.", details)
