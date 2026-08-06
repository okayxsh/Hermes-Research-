"""Explicit reset-and-replay controller."""
from __future__ import annotations

from typing import Protocol
from rq1.recovery.models import CheckpointManifest, RecoveryState, ReplayResult
from rq1.recovery.state_digest import internal_digest, observable_digest

class ReplayEnvironment(Protocol):
    def reset(self) -> RecoveryState: ...
    def step(self, action: str) -> RecoveryState: ...

def replay_checkpoint(environment: ReplayEnvironment, checkpoint: CheckpointManifest) -> tuple[RecoveryState | None, ReplayResult]:
    try: state = environment.reset()
    except Exception as exc: return None, ReplayResult(False, False, (), checkpoint.observable_state_digest, None, checkpoint.state_digest, None, f"reset failed: {type(exc).__name__}")
    actions: list[str] = []
    for action in checkpoint.prefix_actions:
        state = environment.step(action); actions.append(action)
        if state.action_valid is not True or state.done:
            return state, ReplayResult(False, True, tuple(actions), checkpoint.observable_state_digest, observable_digest(state), checkpoint.state_digest, internal_digest(state), "replay action was invalid or terminal")
    observed, internal = observable_digest(state), internal_digest(state)
    valid = observed == checkpoint.observable_state_digest and (checkpoint.state_digest is None or checkpoint.state_digest == internal)
    return state, ReplayResult(valid, True, tuple(actions), checkpoint.observable_state_digest, observed, checkpoint.state_digest, internal, None if valid else "state digest mismatch")
