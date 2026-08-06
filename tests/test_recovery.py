from __future__ import annotations
import json, sys, tempfile, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rq1.recovery.checkpoints import CheckpointError, create_manifest, select_prefix
from rq1.recovery.context import build_recovery_context
from rq1.recovery.fake import FakeRecoveryEnvironment
from rq1.recovery.models import CheckpointPolicy
from rq1.recovery.perturbations import RecoveryCapabilityUnavailable, fake_target_relocation, real_target_relocation
from rq1.recovery.replay import replay_checkpoint
from rq1.recovery.solvability import validate_fake_solvability, validate_real_solvability
from rq1.recovery.state_digest import observable_digest
from rq1.recovery.validation import validate_checkpoint_payload, validate_perturbation_payload
from rq1.recovery.verification import verify_fake_recovery

class RecoveryTests(unittest.TestCase):
    def setUp(self): self.env = FakeRecoveryEnvironment(); self.trajectory = self.env.reference_trajectory()
    def checkpoint(self):
        self.env.reset(); state = self.env.step("go to countertop 1")
        return create_manifest(self.trajectory, CheckpointPolicy("prefix_length", 1), state, "cp-test")
    def test_policy_selection_and_invalid_or_terminal_checkpoints(self):
        self.assertEqual(("go to countertop 1",), select_prefix(self.trajectory, CheckpointPolicy("trajectory_fraction", .5)))
        self.assertEqual(("go to countertop 1",), select_prefix(self.trajectory, CheckpointPolicy("frozen_prefix", frozen_prefix=("go to countertop 1",))))
        with self.assertRaises(CheckpointError): select_prefix(self.trajectory, CheckpointPolicy("prefix_length", 3))
    def test_digest_is_stable_and_replay_equality_detects_mismatch(self):
        checkpoint = self.checkpoint(); self.assertEqual(checkpoint.observable_state_digest, observable_digest(self.env.state()))
        _, replay = replay_checkpoint(self.env, checkpoint); self.assertTrue(replay.valid)
        altered = checkpoint.__class__(**{**checkpoint.__dict__, "observable_state_digest": "0" * 64})
        _, mismatch = replay_checkpoint(self.env, altered); self.assertFalse(mismatch.valid); self.assertEqual("state digest mismatch", mismatch.failure_reason)
    def test_fake_perturbation_solvability_and_context(self):
        checkpoint = self.checkpoint(); state, perturbation = fake_target_relocation(self.env, checkpoint.checkpoint_id, "pert-test")
        self.assertTrue(validate_fake_solvability(self.env).valid)
        context = build_recovery_context(state, checkpoint, perturbation, run_id="r", attempt_id="a", profile="rq1-pilot", snapshot="L0", action_budget=3)
        self.assertIn("no longer", state.observation)
        self.assertEqual("pert-test", context.perturbation_id)
        self.assertEqual([], validate_checkpoint_payload(checkpoint.__dict__))
        self.assertEqual([], validate_perturbation_payload(perturbation.__dict__))
    def test_real_capabilities_fail_closed(self):
        with self.assertRaises(RecoveryCapabilityUnavailable): real_target_relocation()
        self.assertEqual("unavailable", validate_real_solvability().status)
    def test_fake_verification_creates_ordered_jsonl_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); report = verify_fake_recovery(root)
            self.assertTrue(report["mock_recovery"]); self.assertFalse(report["real_compatibility"])
            events = (root / report["artifacts"]["events"]).read_text(encoding="utf-8").splitlines()
            self.assertEqual(["pre_failure", "post_failure", "post_failure"], [json.loads(line)["phase"] for line in events])
