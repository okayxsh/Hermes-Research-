from __future__ import annotations
from rq1.profiles.lifecycle import base_profile_plans, real_profile_lifecycle
from rq1.pilot.models import EvidenceLevel
from rq1.pilot.real_runtime.base import RealExecutionContext, blocked, passed

def profile(context: RealExecutionContext):
    try:
        lifecycle = real_profile_lifecycle(context.root)
        plan = base_profile_plans(context.root)[0]
        inspection = lifecycle.inspect(plan.name)
        manifest = lifecycle.validate(plan) if inspection.exists else lifecycle.create(plan)
        if not manifest.validation_result["valid"]: return blocked("profile_contaminated", "rq1-pilot failed isolation validation.", "Inspect the profile manifest; never overwrite a contaminated profile.", {"handler": "profiles", "manifest": manifest.to_dict()})
        return passed(EvidenceLevel.REAL_COMPONENT, {"handler": "profiles", "operation_executed": True, "profile": plan.name, "manifest": manifest.to_dict()})
    except Exception as exc:
        return blocked("profile_capability_unavailable", str(exc), "Use a capability-confirmed Hermes CLI; do not modify default or personal profiles.", {"handler": "profiles"})

def isolation(context: RealExecutionContext):
    return blocked("temporary_profile_isolation_unobserved", "The installed Hermes cleanup/isolation command surface is not fully observed.", "Capture supported temporary-profile command evidence before real isolation testing.", {"handler": "profiles_isolation", "operation_executed": False})

def write_protection(context: RealExecutionContext):
    return blocked("evaluation_profile_uninstantiated", "A frozen real recovery profile/snapshot does not exist yet.", "Do not create one during this pilot handler; first validate the required recovery gate.", {"handler": "profiles_write_protection"})
