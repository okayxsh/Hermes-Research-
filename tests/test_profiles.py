from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rq1.profiles.lifecycle import (
    FakeProfileBackend,
    ProfileLifecycle,
    ProfileLifecycleError,
    base_profile_plans,
    profile_plan,
    recovery_profile_template,
    validate_profile_name,
    verify_fake_profile_lifecycle,
)


class ProfileLifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "repository"
        self.root.mkdir()
        self.backend = FakeProfileBackend(Path(self.temp.name) / "profiles")
        self.lifecycle = ProfileLifecycle(self.root, self.backend)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_approved_profile_names_and_rejections(self) -> None:
        self.assertEqual("rq1-pilot", validate_profile_name("rq1-pilot"))
        self.assertEqual("rq1-recovery-L25", validate_profile_name("rq1-recovery-L25"))
        self.assertEqual("rq1-test-isolation", validate_profile_name("rq1-test-isolation"))
        for value in ("", "default", "../rq1-pilot", "rq1 arbitrary", "rq1-eval-l0"):
            with self.assertRaises(ProfileLifecycleError):
                validate_profile_name(value)

    def test_base_plans_and_future_recovery_template(self) -> None:
        pilot, acquisition = base_profile_plans(self.root)
        self.assertTrue(pilot.allow_skill_writes)
        self.assertTrue(acquisition.allow_skill_writes)
        recovery = recovery_profile_template(self.root)
        self.assertFalse(recovery.instantiate)
        self.assertTrue(recovery.read_only_snapshot)
        self.assertFalse(recovery.allow_skill_writes)

    def test_fake_creation_is_idempotent_by_refusal_and_manifest_is_clean(self) -> None:
        plan = profile_plan("rq1-pilot", self.root)
        manifest = self.lifecycle.create(plan)
        self.assertTrue(manifest.validation_result["valid"])
        self.assertEqual("validated", manifest.lifecycle_state)
        with self.assertRaises(ProfileLifecycleError):
            self.lifecycle.create(plan)
        stored = self.lifecycle.write_manifest(manifest)
        self.assertTrue(stored.is_file())

    def test_profiles_share_repository_but_not_state(self) -> None:
        pilot, acquisition = base_profile_plans(self.root)
        self.lifecycle.create(pilot)
        self.lifecycle.create(acquisition)
        pilot_path = Path(self.lifecycle.inspect(pilot.name).profile_path or "")
        acquisition_path = Path(self.lifecycle.inspect(acquisition.name).profile_path or "")
        (pilot_path / "skills" / "pilot-temp.md").write_text("only pilot\n", encoding="utf-8")
        self.assertFalse((acquisition_path / "skills" / "pilot-temp.md").exists())
        self.assertTrue(self.lifecycle.validate(pilot).validation_result["valid"])
        self.assertTrue(self.lifecycle.validate(acquisition).validation_result["valid"])

    def test_contamination_and_configuration_drift_fail_validation(self) -> None:
        plan = profile_plan("rq1-acquisition", self.root)
        baseline = self.lifecycle.create(plan)
        path = Path(self.lifecycle.inspect(plan.name).profile_path or "")
        (path / "sessions" / "old.json").write_text("{}\n", encoding="utf-8")
        (path / "skills" / "unrelated.md").write_text("bad\n", encoding="utf-8")
        result = self.lifecycle.validate(plan, baseline=baseline)
        self.assertFalse(result.validation_result["valid"])
        self.assertIn("previous sessions are present", result.contamination_result["findings"])
        self.assertIn("unexpected skills: unrelated.md", result.contamination_result["findings"])

    def test_archive_and_fake_report_are_machine_readable(self) -> None:
        plan = profile_plan("rq1-pilot", self.root)
        manifest = self.lifecycle.create(plan)
        archive = self.lifecycle.archive_manifest(manifest)
        self.assertEqual("rq1-pilot", json.loads(archive.read_text(encoding="utf-8"))["profile_name"])
        report = verify_fake_profile_lifecycle(self.root)
        self.assertTrue(report["mock_profile_lifecycle_passed"])
        self.assertTrue((self.root / "artifacts" / "stage_reports" / "phase4-hermes-profiles.json").is_file())

    def test_uninstantiated_recovery_profile_refuses_creation(self) -> None:
        with self.assertRaises(ProfileLifecycleError):
            self.lifecycle.create(recovery_profile_template(self.root))
