import json
import os
import tempfile
import unittest
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from rq1.hermes.runtime import HermesRuntimeError, resolve_hermes_python, validate_hermes_python
from rq1.university_gpu.models import atomic_write_bytes
from rq1.university_gpu.pilot_readiness import budget_problems, package_candidates
from rq1.university_gpu.models import sha256_file
from rq1.skills.library import TASK_FAMILIES


class PilotReadinessTests(unittest.TestCase):
    def test_long_routes_fail_even_with_positive_recovery_budget(self):
        for length, prefix in [(92, 48), (109, 74)]:
            task = {"reference_actions": ["go to room"] * length,
                    "prefix_actions": ["go to room"] * prefix, "recovery_action_budget": 49-prefix}
            self.assertTrue(budget_problems(task))

    def test_boundary_includes_detour(self):
        task = {"reference_actions": ["go to room"] * 49, "prefix_actions": ["go to room"] * 24,
                "recovery_action_budget": 25}
        self.assertEqual([], budget_problems(task))
        task["reference_actions"].append("go to room")
        self.assertTrue(budget_problems(task))

    def test_explicit_missing_or_base_python_fails_closed(self):
        with patch.dict(os.environ, {"RQ1_HERMES_PYTHON": "/missing/python"}):
            with self.assertRaises(HermesRuntimeError):
                resolve_hermes_python()

    def test_application_layout_and_ambiguity(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {}, clear=True):
            root = Path(folder)
            for name in ("a", "b"):
                venv = root / f".hermes/installs/{name}/environments/test/venv"
                (venv / "bin").mkdir(parents=True)
                (venv / "pyvenv.cfg").write_text("home=x")
                (venv / "bin/python").touch()
            with patch("rq1.hermes.runtime.Path.is_file", autospec=True, side_effect=lambda p: p.exists()):
                with self.assertRaises(HermesRuntimeError):
                    resolve_hermes_python(home=root)
            chosen = root / ".hermes/installs/a/environments/test/venv/bin/python"
            with patch.dict(os.environ, {"RQ1_HERMES_PYTHON": str(chosen)}):
                self.assertEqual(chosen, resolve_hermes_python(home=root))

    def test_required_import_failure_blocks(self):
        result = type("Result", (), {"returncode": 1, "stdout": "", "stderr": "missing PluginContext"})()
        with patch("rq1.hermes.runtime.subprocess.run", return_value=result):
            with self.assertRaises(HermesRuntimeError):
                validate_hermes_python(Path("/venv/bin/python"))

    def test_rating_materials_refuse_missing_context_before_writing(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            payloads = {
                "tasks": {"status": "frozen", "tasks": [{"task_id": "fixture"}]},
                "failures": {"tasks": [{"task_id": "fixture", "task_family": "fixture", "oracle": {"validated": True},
                    "prefix_actions": ["go to room"], "reference_actions": ["go to room", "go to other"],
                    "recovery_action_budget": 48, "post_detour_observation": "fixture observation"}]},
                "library": {"Accum-18": {"skills": [{"skill_id": str(i), "text": "fixture"} for i in range(18)]}}}
            args = []
            for name, value in payloads.items():
                path = root / (name + ".json"); path.write_text(json.dumps(value))
                args.extend(["--" + name, str(path)])
            output = root / "ratings"
            script = Path(__file__).resolve().parents[1] / "university-gpu/make_rating_templates.py"
            result = subprocess.run([sys.executable, str(script), *args, "--output", str(output)], capture_output=True, text=True)
            self.assertNotEqual(0, result.returncode)
            self.assertIn("complete task instructions", result.stderr)
            self.assertFalse(output.exists())

    def test_missing_evidence_never_becomes_approval(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "source.json"
            source.write_text(json.dumps({"tasks": []}))
            output = root / "candidate.json"
            result = package_candidates(source, root, output)
            self.assertEqual("BLOCKED", result["status"])
            self.assertFalse(result["approved"])
            with self.assertRaises(FileExistsError):
                package_candidates(source, root, output)

    def test_immutable_publication_has_one_winner(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "authority.json"
            def write(index):
                try:
                    atomic_write_bytes(path, str(index).encode(), overwrite=False)
                    return index
                except FileExistsError:
                    return None
            with ThreadPoolExecutor(max_workers=8) as pool:
                winners = [value for value in pool.map(write, range(16)) if value is not None]
            self.assertEqual(1, len(winners))
            self.assertEqual(str(winners[0]), path.read_text())

    def test_candidate_evidence_rejects_large_budget_and_corruption(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source_task = root / "task-source"
            source_task.write_text("FAKE TEST TASK")
            tasks = []
            for index, family in enumerate(TASK_FAMILIES):
                task = {"task_id": f"valid_seen:{family}:fixture", "split": "valid_seen", "task_family": family,
                        "checkpoint_id": f"fixture-{index}", "checkpoint_digest": "1"*64, "post_detour_digest": "2"*64,
                        "prefix_actions": ["go to room"], "reference_actions": ["go to room", "go to other"],
                        "detour_action": "go to third", "recovery_action_budget": 48,
                        "source_evidence": [{"path": source_task.name, "sha256": sha256_file(source_task)}]}
                oracle = root / f"oracle-{index}.json"
                oracle.write_text(json.dumps({"task_id": task["task_id"], "checkpoint_digest": "1"*64,
                    "post_detour_digest": "2"*64, "validated": True, "success": True,
                    "total_actions": 3, "total_action_limit": 50}))
                task["oracle_evidence"] = {"path": oracle.name, "sha256": sha256_file(oracle)}
                tasks.append(task)
            source = root / "definitions.json"
            source.write_text(json.dumps({"tasks": tasks}))
            self.assertEqual("CANDIDATE_REVIEW_REQUIRED", package_candidates(source, root, root/"pass.json")["status"])
            oracle = root / tasks[0]["oracle_evidence"]["path"]
            record = json.loads(oracle.read_text()); record["total_action_limit"] = 200
            oracle.write_text(json.dumps(record)); tasks[0]["oracle_evidence"]["sha256"] = sha256_file(oracle)
            source.write_text(json.dumps({"tasks": tasks}))
            self.assertEqual("BLOCKED", package_candidates(source, root, root/"large.json")["status"])
            source_task.write_text("CORRUPTED")
            result = package_candidates(source, root, root/"corrupt.json")
            self.assertTrue(all(r["problems"] for r in result["tasks"]))


if __name__ == "__main__":
    unittest.main()
