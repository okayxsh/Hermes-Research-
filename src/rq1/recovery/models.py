"""Typed, versioned contracts for controlled recovery evidence."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class RecoveryState:
    task_id: str; split: str; task_family: str; instruction: str; observation: str
    inventory: tuple[str, ...]; admissible_actions: tuple[str, ...]; step_number: int
    done: bool = False; success: bool = False; action_valid: bool | None = None
    internal_state: dict[str, Any] | None = None


@dataclass(frozen=True)
class CheckpointPolicy:
    kind: Literal["prefix_length", "action_index", "trajectory_fraction", "frozen_prefix"]
    value: int | float | None = None
    frozen_prefix: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReferenceTrajectory:
    task_id: str; split: str; task_family: str; source: str; actions: tuple[str, ...]
    observations: tuple[str, ...]; inventories: tuple[tuple[str, ...], ...]
    admissible_actions: tuple[tuple[str, ...], ...]; action_validity: tuple[bool, ...]
    checkpoint_index: int | None = None; created_at: str | None = None; git_commit: str | None = None
    alfworld_version: str | None = None; model_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CheckpointManifest:
    schema_version: int; checkpoint_id: str; task_id: str; split: str; task_family: str
    trajectory_source: str; prefix_actions: tuple[str, ...]; prefix_length: int
    checkpoint_observation: str; checkpoint_inventory: tuple[str, ...]
    checkpoint_admissible_actions: tuple[str, ...]; environment_step_number: int
    state_digest: str | None; observable_state_digest: str; created_at: str
    validation_result: str; validation_attempts: int; failure_reason: str | None = None
    git_commit: str | None = None; alfworld_version: str | None = None


@dataclass(frozen=True)
class ReplayResult:
    valid: bool; reset_performed: bool; replayed_actions: tuple[str, ...]
    expected_observable_digest: str; actual_observable_digest: str | None
    expected_internal_digest: str | None; actual_internal_digest: str | None
    failure_reason: str | None = None


@dataclass(frozen=True)
class PerturbationManifest:
    schema_version: int; perturbation_id: str; checkpoint_id: str; type: str
    object_id: str; original_location: str; new_location: str; selection_rule: str
    post_state_digest: str | None; observable_post_state_digest: str; solvable: bool | None
    validation_method: str; visible_message: str; created_at: str; failure_reason: str | None = None


@dataclass(frozen=True)
class SolvabilityResult:
    status: Literal["validated", "invalid", "unavailable", "manual_pilot_required"]
    valid: bool; method: str; details: str


@dataclass(frozen=True)
class RecoveryStartContext:
    run_id: str; attempt_id: str; profile: str; snapshot: str | None; checkpoint_id: str
    perturbation_id: str; instruction: str; perturbation_message: str; observation: str
    inventory: tuple[str, ...]; admissible_actions: tuple[str, ...]; action_budget: int
    prompt_metadata: dict[str, Any]; model_metadata: dict[str, Any]


@dataclass(frozen=True)
class RecoveryEvent:
    schema_version: int; phase: Literal["pre_failure", "post_failure"]; event: str
    run_id: str; attempt_id: str; timestamp: str; payload: dict[str, Any]


def to_dict(value: Any) -> dict[str, Any]:
    result = asdict(value)
    return result
