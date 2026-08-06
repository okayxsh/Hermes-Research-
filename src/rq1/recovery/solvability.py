"""Solvability evidence; real ALFWorld validation remains unavailable."""
from __future__ import annotations
from rq1.recovery.fake import FakeRecoveryEnvironment
from rq1.recovery.models import SolvabilityResult

def validate_fake_solvability(environment: FakeRecoveryEnvironment) -> SolvabilityResult:
    state = environment.state()
    if not state.internal_state or state.internal_state.get("target_location") != "pantry 1":
        return SolvabilityResult("invalid", False, "fake_known_route", "Relocated target is not at the deterministic reachable location.")
    return SolvabilityResult("validated", True, "fake_known_route", "Known deterministic route remains available after relocation.")

def validate_real_solvability() -> SolvabilityResult:
    return SolvabilityResult("unavailable", False, "unverified_real_adapter", "TO_BE_VERIFIED_BY_RECOVERY_PILOT: real solvability validation is unavailable.")
