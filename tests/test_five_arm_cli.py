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


class UniversityCliTests(unittest.TestCase):
    """CPU-only university commands, exercised with an injected scripted driver and synthetic embedder."""

    def setUp(self):
        import five_arm_e2e_fixtures as E
        from rq1.five_arm import university_cli as U
        self.E, self.U = E, U
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.saved = (U.DRIVER_FACTORY, U.EMBEDDER_FACTORY)
        U.DRIVER_FACTORY = lambda _config: E.ScriptedDriver(E.solver)
        U.EMBEDDER_FACTORY = lambda _config: E.HashEmbedder()
        task = E.freeze_task(E.ScriptedDriver(E.solver), "valid_seen:alpha/trial_T1", self.root, 12)
        spec = task.to_dict()
        config = {"split": "valid_seen", "tasks": {task.task_id: spec},
                  "library": {"skills": [dict(s) for s in E.library().skills]}}
        (self.root / "cpu.json").write_text(json.dumps(config), encoding="utf-8")
        (self.root / "state.json").write_text(json.dumps({"local_tests_passed": True, "code_commit": "a" * 40}), encoding="utf-8")
        (self.root / "blocked.json").write_text(json.dumps({"local_tests_passed": True}), encoding="utf-8")

    def tearDown(self):
        self.U.DRIVER_FACTORY, self.U.EMBEDDER_FACTORY = self.saved
        self.tmp.cleanup()

    def cli(self, argv):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = self.U.main(argv)
        return code, json.loads(buf.getvalue())

    def common(self, command, label, state="state.json"):
        return [command, "--state", str(self.root / state), "--output-root", str(self.root / "out"), "--label", label,
                "--config", str(self.root / "cpu.json"), "--work-dir", str(self.root / f"work-{label}")]

    def test_cpu_commands_write_versioned_outputs_and_never_call_a_model(self):
        code, out = self.cli(self.common("cpu-rankings", "r1") + ["--cap", "12"])
        self.assertEqual((code, out["model_calls"]), (0, 0))
        rankings = json.loads(Path(out["output"]).read_text(encoding="utf-8"))
        self.assertEqual(len(rankings["tasks"][0]["ranking"]), 18)
        self.assertFalse(rankings["approved"])
        code, out = self.cli(self.common("oracle-replay", "o1") + ["--cap", "12"])
        self.assertTrue(json.loads(Path(out["output"]).read_text(encoding="utf-8"))["all_validated"])
        code, out = self.cli(self.common("cap-invariance", "c1") + ["--caps", "8", "12"])
        self.assertTrue(json.loads(Path(out["output"]).read_text(encoding="utf-8"))["all_hold"])
        self.assertEqual(self.cli(self.common("oracle-replay", "o1") + ["--cap", "12"])[0], 3)   # work dir / output never reused

    def test_cpu_commands_require_cpu_readiness(self):
        code, out = self.cli(self.common("oracle-replay", "o2", state="blocked.json") + ["--cap", "12"])
        self.assertEqual((code, out["status"]), (2, "blocked"))
        self.assertFalse((self.root / "out").exists())


if __name__ == "__main__":
    unittest.main()
