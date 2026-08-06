"""Identical recovery-start context construction for paired future conditions."""
from __future__ import annotations
from rq1.recovery.models import CheckpointManifest, PerturbationManifest, RecoveryStartContext, RecoveryState

def build_recovery_context(state: RecoveryState, checkpoint: CheckpointManifest, perturbation: PerturbationManifest, *, run_id: str, attempt_id: str, profile: str, snapshot: str | None, action_budget: int, prompt_metadata: dict[str, object] | None = None, model_metadata: dict[str, object] | None = None) -> RecoveryStartContext:
    if action_budget < 1: raise ValueError("Recovery action budget must be positive.")
    return RecoveryStartContext(run_id, attempt_id, profile, snapshot, checkpoint.checkpoint_id, perturbation.perturbation_id,
        state.instruction, perturbation.visible_message, state.observation, state.inventory, state.admissible_actions, action_budget,
        prompt_metadata or {}, model_metadata or {})
