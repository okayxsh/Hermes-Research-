"""Solvability evidence; real ALFWorld validation remains capability-gated."""
from __future__ import annotations
from rq1.recovery.fake import FakeRecoveryEnvironment
from rq1.recovery.models import SolvabilityResult

# Candidate frozen solvability methods. The exact method is selected and frozen
# only after the non-scientific recovery pilot demonstrates it on real ALFWorld.
EXPERT_ROUTE = "expert_route"
SCRIPTED_VALIDATOR = "scripted_validator"
MANUAL_PILOT_ROUTE = "manual_pilot_route"
SOLVABILITY_METHODS = (EXPERT_ROUTE, SCRIPTED_VALIDATOR, MANUAL_PILOT_ROUTE)

def validate_fake_solvability(environment: FakeRecoveryEnvironment) -> SolvabilityResult:
    state = environment.state()
    if not state.internal_state or state.internal_state.get("target_location") != "pantry 1":
        return SolvabilityResult("invalid", False, "fake_known_route", "Relocated target is not at the deterministic reachable location.")
    return SolvabilityResult("validated", True, "fake_known_route", "Known deterministic route remains available after relocation.")

def validate_real_solvability(*_args: object, **_kwargs: object) -> SolvabilityResult:
    return SolvabilityResult("unavailable", False, "unverified_real_adapter",
        "Real solvability validation requires a pilot-demonstrated method "
        f"(one of {', '.join(SOLVABILITY_METHODS)}); it remains unverified.")
