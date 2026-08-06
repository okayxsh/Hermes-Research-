from __future__ import annotations
import os, subprocess
from rq1.hermes.capabilities import probe_hermes_capabilities
from rq1.pilot.models import EvidenceLevel
from rq1.pilot.real_runtime.base import RealExecutionContext, blocked, passed

def discovery(context: RealExecutionContext):
    report = probe_hermes_capabilities(project_root=context.root)
    if not (report.installed and report.plugin_supported and report.executable):
        return blocked("hermes_plugin_unavailable", report.details, "Install no software from this runner; resolve the installed Hermes capability report.", {"handler": "hermes_discovery", "capabilities": report.to_dict()})
    environment = os.environ.copy(); environment.update({"HERMES_HOME": str(context.output_dir / "hermes-home"), "HERMES_ENABLE_PROJECT_PLUGINS": "1"})
    try:
        completed = subprocess.run((report.executable, "plugins", "list"), cwd=context.root, env=environment, capture_output=True, text=True, timeout=30, check=False)
    except OSError as exc:
        return blocked("hermes_command_failed", str(exc), "Inspect the capability report and observed command shape.", {"handler": "hermes_discovery"})
    found = completed.returncode == 0 and "alfworld-experiment" in (completed.stdout + completed.stderr)
    return passed(EvidenceLevel.REAL_COMPONENT, {"handler": "hermes_discovery", "operation_executed": True, "plugin_found": found}) if found else blocked("project_plugin_not_discovered", "Hermes did not discover the trusted project plugin.", "Inspect the temporary-home plugin evidence; do not alter personal profiles.", {"handler": "hermes_discovery", "returncode": completed.returncode})

def dispatch(context: RealExecutionContext):
    return blocked("hermes_dispatch_surface_unobserved", "No installed-version Hermes registry/session dispatch command has been observed.", "Capture the exact supported command through capability evidence before dispatching tools.", {"handler": "hermes_dispatch"})

def skills(context: RealExecutionContext):
    return blocked("native_skill_events_unobservable", "Native Hermes skill retrieval/load events are not observable through the installed adapter.", "Capture native event evidence before measuring retrieval noise.", {"handler": "hermes_skills"})
