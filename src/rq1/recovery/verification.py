"""End-to-end fake recovery verification and real capability report."""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
from rq1.recovery.checkpoints import create_manifest
from rq1.recovery.context import build_recovery_context
from rq1.recovery.fake import FakeRecoveryEnvironment
from rq1.recovery.logging import append_event, write_json
from rq1.recovery.models import CheckpointPolicy, RecoveryEvent, to_dict
from rq1.recovery.perturbations import fake_target_relocation
from rq1.recovery.replay import replay_checkpoint
from rq1.recovery.solvability import validate_fake_solvability, validate_real_solvability
from rq1.utils.ids import new_attempt_id
from rq1.utils.time import utc_now

def real_recovery_capabilities() -> dict[str, object]:
    from rq1.bridge.environment import real_adapter_capability
    capability = real_adapter_capability()
    return {"real_adapter_available": capability.real_adapter_ready,
        "reset_replay_supported": capability.real_reset_supported and capability.deterministic_replay_candidate,
        "state_observation_supported": capability.admissible_actions_observable,
        "internal_state_supported": False, "perturbation_supported": capability.target_relocation_supported,
        "status": "TO_BE_VERIFIED_BY_RECOVERY_PILOT", "alfworld_capabilities": capability.to_dict()}

def verify_fake_recovery(root: Path) -> dict[str, object]:
    env = FakeRecoveryEnvironment(); trajectory = env.reference_trajectory()
    env.reset(); checkpoint_state = env.step("go to countertop 1")
    checkpoint = create_manifest(trajectory, CheckpointPolicy("prefix_length", 1), checkpoint_state, "cp-fake-001")
    replay_state, replay = replay_checkpoint(env, checkpoint)
    assert replay_state is not None
    perturbed, perturbation = fake_target_relocation(env, checkpoint.checkpoint_id, "pert-fake-001")
    solvability = validate_fake_solvability(env)
    attempt = new_attempt_id(); run_id = "phase5-fake-" + attempt
    context = build_recovery_context(perturbed, checkpoint, perturbation, run_id=run_id, attempt_id=attempt, profile="rq1-pilot", snapshot=None, action_budget=12)
    log = root / "artifacts" / "phase5" / attempt / "recovery-events.jsonl"
    for phase, event, payload in (("pre_failure", "checkpoint_validated", to_dict(checkpoint)), ("post_failure", "perturbation_applied", to_dict(perturbation)), ("post_failure", "recovery_context_created", to_dict(context))):
        append_event(log, RecoveryEvent(1, phase, event, run_id, attempt, utc_now(), payload))
    report = {"schema_version": 1, "generated_at": utc_now(), "mode": "fake", "mock_recovery": replay.valid and solvability.valid,
        "checkpoint_replay_valid": replay.valid, "perturbation_valid": perturbation.solvable is True, "solvability": asdict(solvability),
        "recovery_context": to_dict(context), "artifacts": {"events": str(log.relative_to(root))},
        "real_capabilities": real_recovery_capabilities(), "real_compatibility": False, "phase6_blocked": True,
        "unverified": ["Real ALFWorld state serialization, mutation, replay equality, and solvability remain TO_BE_VERIFIED_BY_RECOVERY_PILOT."]}
    write_json(root / "artifacts" / "stage_reports" / "phase5-recovery.json", report)
    return report
