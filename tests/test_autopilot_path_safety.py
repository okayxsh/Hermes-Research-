from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
from uuid import UUID

from rq1.autopilot import state
from rq1.autopilot.executor import Autopilot
from rq1.autopilot.models import RunMode


class AutopilotPathSafetyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "repo"
        self.root.mkdir()
        self.outside = Path(self.temp.name) / "outside"
        self.outside.mkdir()
        self.sentinel = self.outside / "contingencies.jsonl"
        self.sentinel.write_text("preserve me\n", encoding="utf-8")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_run_id_rejects_traversal_and_absolute_injection(self) -> None:
        for run_id in ("../outside", "..", ".", str(self.outside), "C:\\outside", "run/child", "A" * 36):
            with self.subTest(run_id=run_id), self.assertRaises(ValueError):
                state.run_dir(self.root, run_id)
        self.assertEqual("preserve me\n", self.sentinel.read_text(encoding="utf-8"))

    def test_generated_uuid_is_accepted_and_state_lifecycle_stays_contained(self) -> None:
        with patch("rq1.autopilot.executor.git_state", return_value=("commit", True, None)):
            autopilot = Autopilot(self.root)
            plan = autopilot.plan(RunMode.BOOTSTRAP)
            self.assertEqual(plan.run_plan_id, str(UUID(plan.run_plan_id)))
            autopilot.create(plan)
            self.assertEqual(plan.run_plan_id, autopilot.status(plan.run_plan_id)["run_id"])
            stopped = autopilot.stop(plan.run_plan_id)
            self.assertTrue(stopped["stop_requested"])
            self.assertEqual("STOPPED_BY_USER", autopilot.run(plan.run_plan_id)["top_status"])
            self.assertTrue(state.state_path(self.root, plan.run_plan_id).is_file())

    def test_plan_output_directory_rejects_escape_and_absolute_paths(self) -> None:
        with patch("rq1.autopilot.executor.git_state", return_value=("commit", True, None)):
            autopilot = Autopilot(self.root)
            for output in ("../outside", "results/../../outside", str(self.outside), "C:\\outside", "\\\\server\\share"):
                with self.subTest(output=output), self.assertRaises(ValueError):
                    autopilot.plan(RunMode.BOOTSTRAP, output)
            plan = autopilot.plan(RunMode.BOOTSTRAP, "results/final")
            self.assertEqual("results/final", plan.output_directory)

    def test_create_revalidates_forged_plan_run_id_and_output_directory(self) -> None:
        with patch("rq1.autopilot.executor.git_state", return_value=("commit", True, None)):
            autopilot = Autopilot(self.root)
            plan = autopilot.plan(RunMode.BOOTSTRAP)
            with self.assertRaises(ValueError):
                autopilot.create(replace(plan, run_plan_id="../outside"))
            with self.assertRaises(ValueError):
                autopilot.create(replace(plan, output_directory="../../outside"))
        self.assertFalse((self.outside / "state.json").exists())

    def test_output_directory_rejects_existing_symlink_escape(self) -> None:
        link = self.root / "linked-results"
        try:
            link.symlink_to(self.outside, target_is_directory=True)
        except (OSError, NotImplementedError) as exc:
            self.skipTest(f"directory symlinks unavailable: {exc}")
        with self.assertRaisesRegex(ValueError, "symlink"):
            state.validate_output_directory(self.root, "linked-results/run")

    def test_mocked_output_symlink_is_rejected_on_hosts_without_symlink_privileges(self) -> None:
        with patch.object(Path, "is_symlink", return_value=True):
            with self.assertRaisesRegex(ValueError, "symlink"):
                state.validate_output_directory(self.root, "results/final")

    def test_run_directory_symlink_escape_is_rejected_without_touching_target(self) -> None:
        run_id = "12345678-1234-4234-8234-123456789abc"
        parent = self.root / "artifacts" / "autopilot"
        parent.mkdir(parents=True)
        link = parent / run_id
        try:
            link.symlink_to(self.outside, target_is_directory=True)
        except (OSError, NotImplementedError) as exc:
            self.skipTest(f"directory symlinks unavailable: {exc}")
        with self.assertRaisesRegex(ValueError, "symlink"):
            state.run_dir(self.root, run_id)
        self.assertEqual("preserve me\n", self.sentinel.read_text(encoding="utf-8"))

    def test_mocked_run_directory_symlink_is_rejected_on_hosts_without_symlink_privileges(self) -> None:
        run_id = "12345678-1234-4234-8234-123456789abc"
        with patch.object(Path, "is_symlink", return_value=True):
            with self.assertRaisesRegex(ValueError, "symlink"):
                state.run_dir(self.root, run_id)

    def test_state_and_log_symlink_files_are_rejected(self) -> None:
        run_id = "12345678-1234-4234-8234-123456789abc"
        run_dir = state.run_dir(self.root, run_id)
        run_dir.mkdir(parents=True)
        for name, get_path in (
            ("state.json", state.state_path),
            ("contingencies.jsonl", state.contingency_path),
        ):
            link = run_dir / name
            try:
                link.symlink_to(self.sentinel)
            except (OSError, NotImplementedError) as exc:
                self.skipTest(f"file symlinks unavailable: {exc}")
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "symlink"):
                get_path(self.root, run_id)
            link.unlink()
        self.assertEqual("preserve me\n", self.sentinel.read_text(encoding="utf-8"))

    def test_mocked_state_and_log_symlinks_are_rejected(self) -> None:
        run_id = "12345678-1234-4234-8234-123456789abc"
        with patch.object(Path, "is_symlink", return_value=False):
            run_dir = state.run_dir(self.root, run_id)
            run_dir.mkdir(parents=True)
        with patch.object(Path, "is_symlink", return_value=True):
            with self.assertRaisesRegex(ValueError, "symlink"):
                state.state_path(self.root, run_id)
            with self.assertRaisesRegex(ValueError, "symlink"):
                state.contingency_path(self.root, run_id)

    def test_duplicate_initialize_does_not_overwrite_existing_run(self) -> None:
        with patch("rq1.autopilot.executor.git_state", return_value=("commit", True, None)):
            autopilot = Autopilot(self.root)
            plan = autopilot.plan(RunMode.BOOTSTRAP)
            autopilot.create(plan)
            with self.assertRaises(FileExistsError):
                autopilot.create(plan)

    def test_concurrent_state_creation_is_create_only(self) -> None:
        with patch("rq1.autopilot.executor.git_state", return_value=("commit", True, None)):
            autopilot = Autopilot(self.root)
            plan = autopilot.plan(RunMode.BOOTSTRAP)
        target = state.state_path(self.root, plan.run_plan_id)

        def concurrent_writer(_source: Path, destination: Path) -> None:
            Path(destination).write_text("concurrent state wins\n", encoding="utf-8")
            raise FileExistsError(destination)

        with patch("rq1.autopilot.state.os.link", side_effect=concurrent_writer):
            with self.assertRaises(FileExistsError):
                autopilot.create(plan)
        self.assertEqual("concurrent state wins\n", target.read_text(encoding="utf-8"))

    def test_consumption_rejects_tampered_stored_plan(self) -> None:
        with patch("rq1.autopilot.executor.git_state", return_value=("commit", True, None)):
            autopilot = Autopilot(self.root)
            plan = autopilot.plan(RunMode.BOOTSTRAP)
            autopilot.create(plan)
        path = state.state_path(self.root, plan.run_plan_id)
        original = json.loads(path.read_text(encoding="utf-8"))
        for key, value in (
            ("output_directory", "../../outside"),
            ("run_plan_id", "12345678-1234-4234-8234-123456789abc"),
            ("content_hash", "0" * 64),
        ):
            with self.subTest(key=key):
                mutated = json.loads(json.dumps(original))
                mutated["plan"][key] = value
                path.write_text(json.dumps(mutated), encoding="utf-8")
                with self.assertRaises(ValueError):
                    autopilot.status(plan.run_plan_id)
        path.write_text(json.dumps(original), encoding="utf-8")

    def test_consumption_preserves_legacy_plan_hash_only_with_default_fields(self) -> None:
        with patch("rq1.autopilot.executor.git_state", return_value=("commit", True, None)):
            autopilot = Autopilot(self.root)
            plan = autopilot.plan(RunMode.BOOTSTRAP)
            autopilot.create(plan)
        path = state.state_path(self.root, plan.run_plan_id)
        original = json.loads(path.read_text(encoding="utf-8"))
        legacy_keys = ("schema_version", "run_plan_id", "mode", "repository_commit", "output_directory")
        legacy_hash = state.canonical_hash({key: original["plan"][key] for key in legacy_keys})
        original["plan"]["content_hash"] = legacy_hash
        original["plan_hash"] = legacy_hash
        path.write_text(json.dumps(original), encoding="utf-8")
        self.assertEqual(plan.run_plan_id, autopilot.status(plan.run_plan_id)["run_id"])

        tampered = json.loads(json.dumps(original))
        tampered["plan"]["timeout_values"] = {"unreviewed": 1}
        path.write_text(json.dumps(tampered), encoding="utf-8")
        with self.assertRaises(ValueError):
            autopilot.status(plan.run_plan_id)


if __name__ == "__main__":
    unittest.main()
