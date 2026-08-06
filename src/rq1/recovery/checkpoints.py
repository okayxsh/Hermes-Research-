"""Checkpoint policy resolution and manifest construction."""
from __future__ import annotations

from rq1.recovery.models import CheckpointManifest, CheckpointPolicy, RecoveryState, ReferenceTrajectory
from rq1.recovery.state_digest import internal_digest, observable_digest
from rq1.utils.time import utc_now


class CheckpointError(ValueError): pass


def select_prefix(trajectory: ReferenceTrajectory, policy: CheckpointPolicy) -> tuple[str, ...]:
    length = len(trajectory.actions)
    if length < 2 or not all(trajectory.action_validity) or len(trajectory.action_validity) != length:
        raise CheckpointError("Reference trajectory must contain at least two valid actions.")
    if policy.kind == "frozen_prefix":
        prefix = policy.frozen_prefix
        if not prefix or tuple(trajectory.actions[:len(prefix)]) != prefix:
            raise CheckpointError("Frozen prefix must be a non-empty prefix of the reference trajectory.")
        index = len(prefix)
    elif policy.kind in {"prefix_length", "action_index"}:
        if not isinstance(policy.value, int): raise CheckpointError("Checkpoint index must be an integer.")
        index = policy.value
    elif policy.kind == "trajectory_fraction":
        if not isinstance(policy.value, (int, float)) or not 0 < float(policy.value) < 1:
            raise CheckpointError("Trajectory fraction must be strictly between zero and one.")
        index = int(length * float(policy.value))
    else: raise CheckpointError("Unsupported checkpoint policy.")
    if index <= 0 or index >= length:
        raise CheckpointError("Checkpoint must be after at least one action and before task completion.")
    return tuple(trajectory.actions[:index])


def create_manifest(trajectory: ReferenceTrajectory, policy: CheckpointPolicy, state: RecoveryState, checkpoint_id: str) -> CheckpointManifest:
    prefix = select_prefix(trajectory, policy)
    if state.done: raise CheckpointError("Checkpoint state is terminal and cannot leave meaningful recovery work.")
    return CheckpointManifest(1, checkpoint_id, trajectory.task_id, trajectory.split, trajectory.task_family,
        trajectory.source, prefix, len(prefix), state.observation, state.inventory, state.admissible_actions,
        state.step_number, internal_digest(state), observable_digest(state), utc_now(), "validated", 1,
        None, trajectory.git_commit, trajectory.alfworld_version)
