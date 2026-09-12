from __future__ import annotations
from rq1.hermes.capabilities import probe_hermes_capabilities
from rq1.hermes.verification import verify_real_hermes_integration
from rq1.pilot.models import EvidenceLevel
from rq1.pilot.real_runtime.base import RealExecutionContext, blocked, passed


def _verification(context: RealExecutionContext) -> dict[str, object]:
    """Use the sole real registry exercise; never fall back to a direct bridge adapter."""
    return verify_real_hermes_integration(context.root)


def discovery(context: RealExecutionContext):
    report = probe_hermes_capabilities(project_root=context.root)
    if not (report.installed and report.plugin_supported and report.executable):
        return blocked("hermes_plugin_unavailable", report.details, "Install no software from this runner; resolve the installed Hermes capability report.", {"handler": "hermes_discovery", "capabilities": report.to_dict()})
    evidence = _verification(context)
    if evidence.get("real_plugin_loading") is True:
        return passed(EvidenceLevel.REAL_COMPONENT, {"handler": "hermes_discovery", "operation_executed": True, "plugin_found": True, "verification": evidence})
    return blocked("project_plugin_not_discovered", str(evidence.get("reason", "Hermes did not discover the trusted project plugin.")), "Inspect the isolated registry evidence; do not alter personal profiles.", {"handler": "hermes_discovery", "verification": evidence})

def dispatch(context: RealExecutionContext):
    evidence = _verification(context)
    if evidence.get("real_tool_dispatch") is True:
        return passed(EvidenceLevel.REAL_INTEGRATED, {"handler": "hermes_dispatch", "operation_executed": True, "verification": evidence})
    return blocked("hermes_registry_dispatch_failed", str(evidence.get("reason", "Installed Hermes registry dispatch did not produce complete evidence.")), "Inspect the installed registry, plugin hooks, and local bridge logs; do not substitute a direct bridge adapter.", {"handler": "hermes_dispatch", "verification": evidence})

def skills(context: RealExecutionContext):
    evidence = _verification(context)
    if evidence.get("native_skill_event_capture") is True:
        return passed(EvidenceLevel.REAL_INTEGRATED, {"handler": "hermes_skills", "operation_executed": True, "verification": evidence})
    return blocked("native_skill_events_unobservable", str(evidence.get("reason", "Native Hermes lifecycle evidence was not captured.")), "Capture a genuine Hermes skill operation and its on_skill_lifecycle evidence before measuring retrieval noise.", {"handler": "hermes_skills", "verification": evidence})
