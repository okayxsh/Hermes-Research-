"""Focused A01 oracle ancestry regression using synthetic, test-only records."""
from __future__ import annotations

import copy
import unittest

from rq1.university_gpu import feasibility as f
from tests import test_feasibility_provenance_regression as fixtures


class OracleBindingTask2Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = fixtures.ProvenanceRegressionTests("test_candidate_reference_freeze_preflight_and_resume_drift")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.candidate = self.fixture.candidate
        self.oracle = self.fixture.oracle
        self.frozen = self.fixture.freeze()

    def test_genuine_candidate_oracle_binds_to_frozen_candidate_identity(self) -> None:
        self.assertEqual(self.oracle["manifest_sha256"], self.candidate["manifest_sha256"])
        self.assertEqual(self.frozen["candidate_manifest_sha256"], self.candidate["manifest_sha256"])
        self.assertNotEqual(self.frozen["manifest_sha256"], self.candidate["manifest_sha256"])
        self.assertEqual(f.validate_oracle_report(self.oracle, self.frozen), [])
        self.assertEqual(f.validate_feasibility_manifest(self.frozen, require_frozen=True), [])

    def test_candidate_and_oracle_substitutions_are_rejected(self) -> None:
        altered_candidate = copy.deepcopy(self.candidate)
        altered_candidate["manifest_sha256"] = "f" * 64
        self.assertTrue(f.validate_oracle_report(self.oracle, altered_candidate))

        substitutions = (
            ("task_id", "substituted-task"),
            ("checkpoint_digest", "f" * 64),
            ("post_detour_digest", "e" * 64),
            ("detour_action", "go north"),
            ("action_sequence_sha256", "d" * 64),
        )
        for key, value in substitutions:
            with self.subTest(field=key):
                altered = copy.deepcopy(self.oracle)
                altered["tasks"][0][key] = value
                altered["report_sha256"] = f.content_hash({k: v for k, v in altered.items() if k != "report_sha256"})
                self.assertTrue(f.validate_oracle_report(altered, self.candidate))

    def test_frozen_manifest_rejects_wrong_candidate_ancestry(self) -> None:
        altered = copy.deepcopy(self.frozen)
        altered["candidate_manifest_sha256"] = "a" * 64
        altered["manifest_sha256"] = f._manifest_digest(altered)
        problems = f.validate_feasibility_manifest(altered, require_frozen=True)
        self.assertTrue(any("candidate ancestry" in problem for problem in problems))


if __name__ == "__main__":
    unittest.main()
