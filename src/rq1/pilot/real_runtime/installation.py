from __future__ import annotations
import json
from rq1.pilot.models import EvidenceLevel
from rq1.pilot.real_runtime.base import RealExecutionContext, blocked, passed

def run(context: RealExecutionContext):
    path = context.root / "artifacts" / "stage_reports" / "installation.json"
    try: payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError): payload = {}
    details = {"handler": "installation", "installation_report": "artifacts/stage_reports/installation.json", "installation_ready": payload.get("installation_ready") is True, "operation_executed": True}
    return passed(EvidenceLevel.INSTALLED, details) if details["installation_ready"] else blocked("installation_not_ready", "A fresh passing installation report is unavailable.", "Run setup verification on the university machine; do not install from the pilot runner.", details)
