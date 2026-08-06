from __future__ import annotations
from rq1.recovery.verification import real_recovery_capabilities
from rq1.pilot.real_runtime.base import RealExecutionContext, blocked

def run(context: RealExecutionContext, stage: str):
    capabilities = real_recovery_capabilities()
    if not capabilities.get("reset_replay_supported"):
        return blocked("real_replay_unverified", "Real reset-and-replay equality is not yet established.", "Complete a real Fix 1 replay-equality operation before recovery execution.", {"handler": "recovery_" + stage, "recovery_capabilities": capabilities})
    if stage in {"perturbation", "solvability", "context", "conditions", "reconcile"} and not capabilities.get("perturbation_supported"):
        return blocked("canonical_perturbation_unsupported", "ALFWorld target relocation/state mutation is unsupported by observed capabilities.", "Do not substitute a perturbation. Obtain manual protocol approval after capability evidence.", {"handler": "recovery_" + stage, "recovery_capabilities": capabilities})
    return blocked("real_recovery_handler_pending_observed_surface", "A required real recovery surface is capability-present but has no observed operational adapter.", "Capture installed API evidence and add only that version-specific operation.", {"handler": "recovery_" + stage, "recovery_capabilities": capabilities})
