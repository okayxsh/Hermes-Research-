"""Synthetic evidence; no model or human authorization is involved."""
from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from tests import test_feasibility_evidence_regression as fixtures
from rq1.university_gpu.evidence import inventory_attempt
from rq1.university_gpu.evidence_anchor import EvidenceAnchorError, create_anchors, verify_anchor_dir
from rq1.university_gpu.models import content_hash
from rq1.university_gpu.scheduler import SchedulerError
from rq1.university_gpu.snapshot import export_progress, verify_progress


class EvidenceAnchorTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ)
        self.env.start()
        self.addCleanup(self.env.stop)
        os.environ.pop("RQ1_EVIDENCE_ANCHOR_DIR", None)
        self.fixture = fixtures.EvidenceRegressionTests("test_missing_or_mutated_log_blocks_completed_reads_and_export")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.f = self.fixture
        self.result_path = self.f.publish()
        self.anchor_dir = self.f.base / "separate-anchor"

    def _create(self):
        return create_anchors(self.f.store.root, self.f.store.manifest_path, self.anchor_dir)

    def test_coherent_tampering_detected_after_external_anchor(self):
        self._create()
        trace = self.f.attempt / "result/retrieval.jsonl"
        row = json.loads(trace.read_text())
        row["synthetic_extra"] = "changed after completion"
        trace.write_text(json.dumps(row) + "\n")
        envelope = json.loads(self.result_path.read_text())
        envelope["result"]["raw_evidence"] = inventory_attempt(self.f.attempt, envelope["result"])
        envelope["result_sha256"] = content_hash(envelope["result"])
        self.result_path.write_text(json.dumps(envelope))
        self.assertEqual(len(self.f.store.completed()), 1)  # Internally consistent without a pin.
        os.environ["RQ1_EVIDENCE_ANCHOR_DIR"] = str(self.anchor_dir)
        with self.assertRaisesRegex(SchedulerError, "external evidence anchor"):
            self.f.store.completed()
        with self.assertRaises(EvidenceAnchorError):
            verify_anchor_dir(self.f.store.root, self.f.store.manifest_path, self.anchor_dir)

    def test_anchor_identity_resume_and_missing_result(self):
        self._create()
        self.assertEqual(self._create()["created"], 0)
        os.environ["RQ1_EVIDENCE_ANCHOR_DIR"] = str(self.anchor_dir)
        self.assertEqual(len(self.f.store.completed()), 1)
        original = self.result_path.read_bytes()
        self.result_path.unlink()
        with self.assertRaises(SchedulerError):
            self.f.store.completed()
        self.result_path.write_bytes(original)
        anchor = next(self.anchor_dir.glob("*.json"))
        data = json.loads(anchor.read_text())
        data["attempt_id"] = "otherattempt"
        anchor.write_text(json.dumps(data))
        with self.assertRaises(SchedulerError):
            self.f.store.completed()

    def test_missing_and_truncated_logs_are_rejected(self):
        trace = self.f.attempt / "result/retrieval.jsonl"
        original = trace.read_bytes()
        trace.unlink()
        with self.assertRaises(SchedulerError):
            self.f.store.completed()
        trace.write_bytes(original[:1])
        with self.assertRaises(SchedulerError):
            self.f.store.completed()

    def test_partial_export_reports_coverage_and_external_comparison(self):
        plain = self.f.base / "plain-export"
        result = export_progress(fixtures.ROOT, self.f.store.root, self.f.store.manifest_path, plain)
        self.assertEqual(result["evidence_anchor_status"]["classification"], "internally_validated")
        self.assertEqual(result["evidence_anchor_status"]["unanchored_units"], [self.f.unit["unit_key"]])
        self._create()
        os.environ["RQ1_EVIDENCE_ANCHOR_DIR"] = str(self.anchor_dir)
        dest = self.f.base / "anchored-export"
        result = export_progress(fixtures.ROOT, self.f.store.root, self.f.store.manifest_path, dest)
        self.assertEqual(result["evidence_anchor_status"]["unanchored_units"], [])
        self.assertEqual(result["independent_copy_status"], "not_verified")
        self.assertEqual(verify_progress(dest)["external_anchor_comparison"], "performed")
        os.environ.pop("RQ1_EVIDENCE_ANCHOR_DIR")
        self.assertEqual(verify_progress(dest)["external_anchor_comparison"], "not_performed")

    def test_missing_anchor_directory_and_swapped_anchor_fail(self):
        self._create()
        anchor = next(self.anchor_dir.glob("*.json"))
        anchor.rename(self.anchor_dir / "wrong.json")
        with self.assertRaises(EvidenceAnchorError):
            verify_anchor_dir(self.f.store.root, self.f.store.manifest_path, self.anchor_dir)
        os.environ["RQ1_EVIDENCE_ANCHOR_DIR"] = str(self.f.base / "missing")
        with self.assertRaises(SchedulerError):
            self.f.store.completed()

    def test_removed_anchor_is_reported_as_unanchored(self):
        self._create()
        os.environ["RQ1_EVIDENCE_ANCHOR_DIR"] = str(self.anchor_dir)
        next(self.anchor_dir.glob("*.json")).unlink()
        completed = self.f.store.completed()
        self.assertEqual(self.f.store.evidence_anchor_status(completed)["unanchored_units"],
                         [self.f.unit["unit_key"]])


if __name__ == "__main__":
    unittest.main()
