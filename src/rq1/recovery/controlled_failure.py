"""Deterministic controlled failure injection for RQ1 recovery evaluation.

The failure relocates the required target object to another reachable location,
informs the agent only that it is no longer where expected, never reveals the
new location, and fails closed when no valid deterministic relocation exists.
The destination is chosen deterministically (sorted-first reachable location
that is not the original), never based on agent behavior.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from rq1.retrieval.query import CANONICAL_FAILURE_MESSAGE


class ControlledFailureError(RuntimeError):
    def __init__(self, message: str, *, code: str = "controlled_failure_unavailable") -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class FailureContext:
    """The frozen post-failure state observed by the agent.

    Contains only the task instruction, the current observation, the current
    inventory, and the canonical failure message. It never contains the new
    object location or any future information.
    """

    task_instruction: str
    observation: str
    inventory: tuple[str, ...]
    failure_message: str = CANONICAL_FAILURE_MESSAGE

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_instruction": self.task_instruction,
            "observation": self.observation,
            "inventory": list(self.inventory),
            "failure_message": self.failure_message,
        }


@dataclass(frozen=True)
class ControlledFailure:
    checkpoint_id: str
    object_id: str
    original_location: str
    new_location: str
    failure_message: str
    post_state_digest: str
    solvable: bool
    selection_rule: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "object_id": self.object_id,
            "original_location": self.original_location,
            "new_location": self.new_location,
            "failure_message": self.failure_message,
            "post_state_digest": self.post_state_digest,
            "solvable": self.solvable,
            "selection_rule": self.selection_rule,
        }


@dataclass(frozen=True)
class FailureTrajectory:
    """Audit-only record of the action/observation trajectory around the failure.

    This is logged for analysis and never injected into the agent context.
    """

    pre_failure_actions: tuple[str, ...] = ()
    pre_failure_observations: tuple[str, ...] = ()
    post_failure_actions: tuple[str, ...] = ()
    post_failure_observations: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "pre_failure_actions": list(self.pre_failure_actions),
            "pre_failure_observations": list(self.pre_failure_observations),
            "post_failure_actions": list(self.post_failure_actions),
            "post_failure_observations": list(self.post_failure_observations),
        }


class FailureEnvironment(Protocol):
    """Minimal surface a real/fake environment must expose for the failure."""

    def required_object_id(self) -> str: ...
    def object_location(self, object_id: str) -> str: ...
    def reachable_locations(self) -> Sequence[str]: ...
    def relocate_object(self, object_id: str, destination: str) -> str: ...
    def is_solvable(self) -> bool: ...


def select_relocation_destination(original_location: str, reachable_locations: Sequence[str]) -> str:
    """Deterministic destination: sorted-first reachable location != original."""
    candidates = sorted({location for location in reachable_locations if location != original_location})
    if not candidates:
        raise ControlledFailureError(
            "no reachable relocation destination exists", code="no_reachable_destination"
        )
    return candidates[0]


def apply_controlled_failure(
    environment: FailureEnvironment,
    *,
    checkpoint_id: str,
    failure_message: str = CANONICAL_FAILURE_MESSAGE,
) -> ControlledFailure:
    """Relocate the required object and verify the task remains solvable.

    Fails closed if the object is not at its expected location, if no reachable
    destination exists, or if the perturbed task is not solvable.
    """
    object_id = environment.required_object_id()
    original = environment.object_location(object_id)
    destination = select_relocation_destination(original, environment.reachable_locations())
    post_state_digest = environment.relocate_object(object_id, destination)
    if not environment.is_solvable():
        raise ControlledFailureError(
            "perturbed task is not solvable; refusing to inject the failure",
            code="perturbation_unsolvable",
        )
    return ControlledFailure(
        checkpoint_id=checkpoint_id,
        object_id=object_id,
        original_location=original,
        new_location=destination,
        failure_message=failure_message,
        post_state_digest=post_state_digest,
        solvable=True,
        selection_rule="sorted_first_reachable_non_original",
    )
