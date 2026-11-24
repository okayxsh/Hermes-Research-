"""Five-arm CLI smoke tests (preparation-only commands; SYNTHETIC inputs; outputs in a temporary directory)."""
from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from five_arm_fixtures import SKILL_TEXTS
from rq1.five_arm import cli


def run(argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = cli.main(argv)
    return code, json.loads(buf.getvalue())


class CliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_decisions_workload_readiness(self):
        code, reg = run(["decisions"])
        self.assertEqual(code, 0)
        self.assertTrue(all(d["approved"] is False for d in reg["decisions"].values()))
        code, w = run(["workload", "--tasks", "79", "--skills", "18", "--candidates-per-task", "3"])
        self.assertEqual((w["full_library"]["judgments"], w["candidate_set"]["judgments"]), (2844, 474))
        state = self.root / "state.json"
        state.write_text(json.dumps({"local_tests_passed": True}), encoding="utf-8")
        code, report = run(["readiness", "--state", str(state)])
        self.assertEqual(report["levels"]["confirmatory"]["status"], "BLOCKED")

    def test_rating_export_writes_new_directory_only_once(self):
        spec = self.root / "items.json"
        spec.write_text(json.dumps({"tasks": [{"task_id": "valid_seen:t/trial_T1", "task_goal": "g", "recovery_context": "c"}],
                                    "skills": SKILL_TEXTS, "design": "full_library", "salt": "fixture-salt-0123456789",
                                    "raters": ["A", "B"]}), encoding="utf-8")
        args = ["rating-export", "--items-input", str(spec), "--output-root", str(self.root / "out"), "--label", "r1"]
        code, out = run(args)
        self.assertEqual((code, out["items"]), (0, 18))
        self.assertEqual(sorted(p.name for p in (self.root / "out" / "rating-export-r1").iterdir()),
                         ["coordinator-key.json", "export-A.csv", "export-B.csv"])
        self.assertEqual(run(args)[0], 3)                                  # versioned directory never reused


if __name__ == "__main__":
    unittest.main()
