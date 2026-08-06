"""Stored fake perturbations and a fail-closed real capability boundary."""
from __future__ import annotations

from rq1.recovery.fake import FakeRecoveryEnvironment
from rq1.recovery.models import PerturbationManifest, RecoveryState
from rq1.recovery.state_digest import internal_digest, observable_digest
from rq1.utils.time import utc_now

class RecoveryCapabilityUnavailable(RuntimeError): pass

def fake_target_relocation(environment: FakeRecoveryEnvironment, checkpoint_id: str, perturbation_id: str) -> tuple[RecoveryState, PerturbationManifest]:
    before = environment.state(); after = environment.relocate_target()
    return after, PerturbationManifest(1, perturbation_id, checkpoint_id, "target_object_relocation", "target", "countertop 1", "pantry 1", "deterministic_fake", internal_digest(after), observable_digest(after), True, "fake_known_route", after.observation, utc_now())

def real_target_relocation(*_args: object, **_kwargs: object) -> None:
    raise RecoveryCapabilityUnavailable("TO_BE_VERIFIED_BY_RECOVERY_PILOT: real ALFWorld target relocation capability is unavailable.")
