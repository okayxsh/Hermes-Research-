from __future__ import annotations

import copy
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from rq1.university_gpu import engineering_scale as scale
from rq1.university_gpu.authorization import AuthorizationError, approval_digest, verify_engineering_authorization, verify_frozen_manifest_anchor
from tests.test_engineering_scale import frozen


class OwnerReviewTests(unittest.TestCase):
    def setUp(self):
        self.manifest = frozen(acceptance=True)
        self.approval = self.manifest["engineering_approval"]
        self.approval.update(approver_id="fixture-owner", approved_by="fixture-owner",
                             owner_reviewer_id="fixture-owner", review_mode="owner_reviewed",
                             independent_review=False, approval_provenance="owner-reviewed by fixture-owner",
                             test_only=False, authorization_type="engineering_gpu", profile_id=scale.PROFILE_ID,
                             model_tag=scale.MODEL_TAG, model_digest=scale.MODEL_DIGEST,
                             model_quantization=scale.MODEL_QUANTIZATION, total_action_budget=200,
                             run_id=self.manifest["run_id"], conditions=["D0", "D18"],
                             access_end_epoch=time.time() + 10000)
        self.manifest["manifest_sha256"] = scale._manifest_digest(self.manifest)
        self.env = {"RQ1_ENGINEERING_AUTHORITY_MODE": "owner_reviewed",
                    "RQ1_OWNER_REVIEWED_ENGINEERING_APPROVAL_SHA256": approval_digest(self.approval),
                    "RQ1_OWNER_REVIEWED_ENGINEERING_MANIFEST_SHA256": self.manifest["manifest_sha256"]}

    def test_owner_approval_and_manifest_are_distinct_pins(self):
        with patch.dict(os.environ, self.env):
            scale._validate_engineering_approval(self.manifest, self.approval, yes=True)
            for key in ("RQ1_OWNER_REVIEWED_ENGINEERING_APPROVAL_SHA256", "RQ1_OWNER_REVIEWED_ENGINEERING_MANIFEST_SHA256"):
                with patch.dict(os.environ, {key: ""}):
                    with self.assertRaises(scale.FeasibilityError):
                        scale._validate_engineering_approval(self.manifest, self.approval, yes=True)

    def test_provenance_and_scope_cannot_claim_independent_or_scientific(self):
        for changes in ({"independent_review": True}, {"owner_reviewer_id": "another person"},
                        {"approval_provenance": "reviewed"}, {"scientific_run": True},
                        {"test_only": True}):
            m = copy.deepcopy(self.manifest)
            m["engineering_approval"].update(changes)
            m["manifest_sha256"] = scale._manifest_digest(m)
            env = dict(self.env, RQ1_OWNER_REVIEWED_ENGINEERING_APPROVAL_SHA256=approval_digest(m["engineering_approval"]),
                       RQ1_OWNER_REVIEWED_ENGINEERING_MANIFEST_SHA256=m["manifest_sha256"])
            with patch.dict(os.environ, env), self.assertRaises(scale.FeasibilityError):
                scale._validate_engineering_approval(m, m["engineering_approval"], yes=True)
        with patch.dict(os.environ, {"RQ1_ENGINEERING_AUTHORITY_MODE": "independent",
                                    "RQ1_TRUSTED_ENGINEERING_MANIFEST_SHA256": self.manifest["manifest_sha256"]}):
            with self.assertRaises(AuthorizationError):
                verify_frozen_manifest_anchor(self.manifest)

    def test_owner_mode_cannot_authorize_other_profiles(self):
        with patch.dict(os.environ, self.env), self.assertRaises(AuthorizationError):
            verify_frozen_manifest_anchor({"profile_id": "scientific", "manifest_sha256": "a" * 64})

    def test_changed_rehashed_manifest_rejected_by_original_integrity_pin(self):
        changed = copy.deepcopy(self.manifest)
        changed["review_note"] = "changed after confirmation"
        changed["manifest_sha256"] = scale._manifest_digest(changed)
        with patch.dict(os.environ, self.env), self.assertRaises(AuthorizationError):
            verify_frozen_manifest_anchor(changed)

    def test_launch_rejects_invalid_approval_changed_manifest_and_source(self):
        root = Path(__file__).resolve().parents[1]
        config = {"worker_count": 4}
        for mutation in ("approval", "manifest", "source"):
            m, approval = copy.deepcopy(self.manifest), copy.deepcopy(self.approval)
            if mutation == "approval":
                approval["approved_by"] = "other"
            elif mutation == "manifest":
                m["maximum_attempts"] += 1
            else:
                approval["source_commit"] = "b" * 40
            with patch.dict(os.environ, self.env), self.assertRaises(scale.FeasibilityError):
                scale.run_feasibility_launch(root, config, m, approval=approval, yes=True,
                                             cutoff_epoch=time.time() + 6000)

    def test_launch_blocks_missing_backup_before_any_hardware(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "source"; root.mkdir()
            storage = Path(tmp) / "runs"; storage.mkdir()
            m = copy.deepcopy(self.manifest)
            m["backup_destination"] = str(Path(tmp) / "missing")
            m["engineering_approval"]["backup_destination"] = m["backup_destination"]
            m["manifest_sha256"] = scale._manifest_digest(m)
            env = dict(self.env, RQ1_OWNER_REVIEWED_ENGINEERING_APPROVAL_SHA256=approval_digest(m["engineering_approval"]),
                       RQ1_OWNER_REVIEWED_ENGINEERING_MANIFEST_SHA256=m["manifest_sha256"])
            with patch.dict(os.environ, env), patch.object(scale, "require_posix_process_management"), \
                    patch("rq1.university_gpu.hardware.machine_report") as hardware:
                with self.assertRaisesRegex(scale.FeasibilityError, "existing approved storage lacks"):
                    scale.run_feasibility_launch(root, {"worker_count": 4, "run_output_dir": str(storage)}, m,
                                                 approval=m["engineering_approval"], yes=True,
                                                 cutoff_epoch=time.time()+6000)
                hardware.assert_not_called()
