"""Stored fake perturbations and a fail-closed real capability boundary.

The canonical failure message is frozen in ``rq1.retrieval.query``; both the
fake and real perturbation paths must emit exactly this message so post-failure
queries are reproducible across conditions.
"""
from __future__ import annotations

from rq1.recovery.fake import FakeRecoveryEnvironment
from rq1.recovery.models import PerturbationManifest, RecoveryState
from rq1.recovery.state_digest import internal_digest, observable_digest
from rq1.retrieval.query import CANONICAL_FAILURE_MESSAGE
from rq1.utils.time import utc_now

class RecoveryCapabilityUnavailable(RuntimeError):
    def __init__(self, message: str, *, capability: str | None = None, remediation: str | None = None) -> None:
        super().__init__(message)
        self.capability = capability
        self.remediation = remediation

def fake_target_relocation(environment: FakeRecoveryEnvironment, checkpoint_id: str, perturbation_id: str) -> tuple[RecoveryState, PerturbationManifest]:
    after = environment.relocate_target()
    return after, PerturbationManifest(1, perturbation_id, checkpoint_id, "target_object_relocation", "target", "countertop 1", "pantry 1", "deterministic_fake", internal_digest(after), observable_digest(after), True, "fake_known_route", CANONICAL_FAILURE_MESSAGE, utc_now())

def real_target_relocation(*_args: object, **_kwargs: object) -> None:
    """Fail closed until a real relocation is observed during the recovery pilot.

    The real implementation must relocate the required target object to another
    reachable receptacle, preserve task solvability, emit exactly the canonical
    failure message, and must not reveal the new location.
    """
    raise RecoveryCapabilityUnavailable(
        "Real ALFWorld target-object relocation is not yet verified.",
        capability="alfworld_target_relocation",
        remediation=(
            "Demonstrate a real relocation that keeps the perturbed task solvable "
            "during the non-scientific recovery pilot before enabling final evaluation."
        ),
    )
