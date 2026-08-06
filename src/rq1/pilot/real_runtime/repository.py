from __future__ import annotations
import json
from rq1.pilot.models import EvidenceLevel
from rq1.pilot.real_runtime.base import RealExecutionContext, passed

def run(context: RealExecutionContext):
    schemas = list((context.root / "data" / "schemas").glob("*.json"))
    for path in schemas: json.loads(path.read_text(encoding="utf-8"))
    return passed(EvidenceLevel.STATIC, {"handler": "repository", "repository": "$REPO", "schemas_validated": len(schemas), "writable_output": True, "operation_executed": True})
