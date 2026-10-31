from __future__ import annotations

import os
from pathlib import Path
import unittest
from unittest.mock import patch

from rq1.university_gpu.authorization import (
    AuthorizationError, TRUSTED_APPROVAL_SHA256_ENV, TRUSTED_MANIFEST_SHA256_ENV,
    approval_digest, verify_engineering_authorization, verify_frozen_manifest_anchor,
)
from rq1.university_gpu.feasibility import FeasibilityError, _validate_engineering_approval
from rq1.university_gpu.ops import OperatorError, run_design_launch


class ExternalAuthorizationTrustTests(unittest.TestCase):
    def setUp(self) -> None:
        self.approval = {
            "schema": "rq1-engineering-feasibility-approval-v1", "test_only": False,
            "authorization_type": "engineering_gpu", "approval_scope": "engineering_feasibility_only",
            "gpu_inference": True, "scientific_run": False, "not_for_scientific_claims": True,
            "candidate_manifest_sha256": "a" * 64, "profile_id": "rq1-feasibility-200-v1",
            "model_tag": "gemma4:12b", "model_digest": "b" * 64, "total_action_budget": 200,
            "model_quantization": "Q4_K_M", "run_id": "rq1-feasibility-200-v1-test",
            "conditions": ["D0", "D18"], "prior_pick_place_exposure_acknowledged": True,
            "approver_id": "fixture-author", "approved_by": "fixture-author", "approval_reference": "fixture-record-1",
            "approval_provenance": "synthetic test fixture; not production authorization",
        }
        self.kwargs = {"candidate_sha256": "a" * 64, "model_tag": "gemma4:12b", "model_digest": "b" * 64,
                       "profile_id": "rq1-feasibility-200-v1", "action_budget": 200,
                       "run_id": "rq1-feasibility-200-v1-test", "model_quantization": "Q4_K_M"}

    def test_missing_external_anchor_fails_closed(self) -> None:
        with patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(AuthorizationError, "not configured"):
            verify_engineering_authorization(self.approval, **self.kwargs)

    def test_independently_pinned_fixture_digest_accepts_exact_scope(self) -> None:
        with patch.dict(os.environ, {TRUSTED_APPROVAL_SHA256_ENV: approval_digest(self.approval)}, clear=False):
            self.assertEqual(verify_engineering_authorization(self.approval, **self.kwargs), approval_digest(self.approval))

    def test_modified_or_wrong_candidate_approval_fails(self) -> None:
        with patch.dict(os.environ, {TRUSTED_APPROVAL_SHA256_ENV: approval_digest(self.approval)}, clear=False):
            changed = dict(self.approval, approval_provenance="different")
            with self.assertRaisesRegex(AuthorizationError, "does not match"):
                verify_engineering_authorization(changed, **self.kwargs)
            with self.assertRaisesRegex(AuthorizationError, "scope is invalid"):
                verify_engineering_authorization(self.approval, **dict(self.kwargs, candidate_sha256="c" * 64))

    def test_test_only_and_scientific_scope_are_rejected(self) -> None:
        with patch.dict(os.environ, {TRUSTED_APPROVAL_SHA256_ENV: approval_digest(self.approval)}, clear=False):
            with self.assertRaises(AuthorizationError):
                verify_engineering_authorization(dict(self.approval, test_only=True), **self.kwargs)
            with self.assertRaises(AuthorizationError):
                verify_engineering_authorization(dict(self.approval, scientific_run=True), **self.kwargs)

    def test_frozen_manifest_needs_separate_exact_operator_pin(self) -> None:
        manifest = {"manifest_sha256": "c" * 64}
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(AuthorizationError):
            verify_frozen_manifest_anchor(manifest)
        with patch.dict(os.environ, {TRUSTED_MANIFEST_SHA256_ENV: "d" * 64}, clear=True):
            with self.assertRaisesRegex(AuthorizationError, "differs"):
                verify_frozen_manifest_anchor(manifest)
        with patch.dict(os.environ, {TRUSTED_MANIFEST_SHA256_ENV: "c" * 64}, clear=True):
            self.assertEqual(verify_frozen_manifest_anchor(manifest), "c" * 64)

    def test_gpu_gate_requires_both_external_pins_and_operator_confirmation(self) -> None:
        manifest = {"engineering_approval": self.approval, "candidate_manifest_sha256": "a" * 64,
                    "model": {"tag": "gemma4:12b", "digest": "b" * 64},
                    "run_id": "rq1-feasibility-200-v1-test", "manifest_sha256": "c" * 64}
        pins = {TRUSTED_APPROVAL_SHA256_ENV: approval_digest(self.approval),
                TRUSTED_MANIFEST_SHA256_ENV: "c" * 64}
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(FeasibilityError):
            _validate_engineering_approval(manifest, self.approval, yes=True)
        with patch.dict(os.environ, pins, clear=True):
            with self.assertRaises(FeasibilityError):
                _validate_engineering_approval(manifest, self.approval, yes=False)
            _validate_engineering_approval(manifest, self.approval, yes=True)
            with self.assertRaises(FeasibilityError):
                _validate_engineering_approval(dict(manifest, manifest_sha256="d" * 64), self.approval, yes=True)
            forged = dict(self.approval, approval_provenance="forged with a new self hash")
            with self.assertRaises(FeasibilityError):
                _validate_engineering_approval(dict(manifest, engineering_approval=forged), forged, yes=True)
        with self.assertRaises(OperatorError):
            run_design_launch(Path.cwd(), {}, {}, approval=self.approval, yes=True)


if __name__ == "__main__":
    unittest.main()
