"""Offline containment checks; temporary paths and synthetic manifests only."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rq1.cli import build_parser
from rq1.university_gpu import feasibility
from rq1.university_gpu.cli import run
from rq1.university_gpu.scheduler import RunStore, SchedulerError
from tests.test_university_gpu_feasibility import candidate_manifest, make_fake_frozen_manifest
from tests.test_university_gpu_package import make_manifest, freeze


class FeasibilityPathGuardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.output = self.base / "approved"
        self.output.mkdir()
        self.manifest = make_fake_frozen_manifest(candidate_manifest())
        self.run_root = self.output / self.manifest["run_id"]
        self.run_root.mkdir()
        self.manifest_path = self.run_root / "manifest.json"
        self.manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")

    def test_valid_run_and_reject_outside_or_traversal(self):
        self.assertEqual(feasibility.checked_run_path(self.output, self.run_root, self.manifest_path), self.run_root.resolve())
        with self.assertRaises(feasibility.FeasibilityError):
            feasibility.checked_run_path(self.base / "another-root", self.run_root, self.manifest_path)
        with self.assertRaises(feasibility.FeasibilityError):
            feasibility.checked_run_path(self.output, self.output / ".." / "approved" / self.run_root.name, self.manifest_path)
        with self.assertRaises(feasibility.FeasibilityError):
            feasibility.checked_run_path(self.output, self.output / "../unsafe", self.manifest_path)

    def test_mismatched_or_malformed_external_manifest_is_rejected(self):
        other = self.base / "other.json"
        altered = dict(self.manifest, run_id="../../escape")
        other.write_text(json.dumps(altered), encoding="utf-8")
        with self.assertRaises(feasibility.FeasibilityError):
            feasibility.checked_run_path(self.output, self.run_root, other)
        other.write_text("[]", encoding="utf-8")
        with self.assertRaises(feasibility.FeasibilityError):
            feasibility.checked_run_path(self.output, self.run_root, other)

    def test_stop_refuses_outside_run_before_writing(self):
        outside = self.base / "outside" / self.run_root.name
        outside.mkdir(parents=True)
        (outside / "manifest.json").write_text(json.dumps(self.manifest), encoding="utf-8")
        args = build_parser().parse_args(["university-gpu", "feasibility", "stop", "--output-root", str(self.output),
                                          "--run-root", str(outside)])
        self.assertNotEqual(run(self.base, args), 0)
        self.assertFalse((outside / "STOP").exists())

    def test_symlinked_run_and_manifest_are_rejected(self):
        link = self.output / (self.manifest["run_id"] + "-linked")
        try:
            link.symlink_to(self.run_root, target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("host does not allow symlink creation")
        with self.assertRaises(feasibility.FeasibilityError):
            feasibility.checked_run_path(self.output, link, self.manifest_path)
        manifest_link = self.base / "manifest-link.json"
        manifest_link.symlink_to(self.manifest_path)
        with self.assertRaises(feasibility.FeasibilityError):
            feasibility.checked_run_path(self.output, self.run_root, manifest_link)

    def test_mocked_symlink_is_rejected_without_touching_target(self):
        with patch.object(type(self.run_root), "is_symlink", lambda path: path == self.run_root):
            with self.assertRaises(feasibility.FeasibilityError):
                feasibility.checked_run_path(self.output, self.run_root, self.manifest_path)
            with self.assertRaises(SchedulerError):
                RunStore(self.run_root, self.manifest)

    def test_managed_store_rejects_posix_and_windows_path_injection(self):
        store = RunStore(self.run_root, self.manifest)
        for relative in ("../outside", r"..\outside", "/outside", r"C:\outside"):
            with self.subTest(relative=relative), self.assertRaises(SchedulerError):
                store.path(relative)

    def test_generic_stop_requires_existing_matching_frozen_run(self):
        missing = self.base / "unrelated"
        args = build_parser().parse_args(["university-gpu", "stop", "--run-root", str(missing)])
        self.assertNotEqual(run(self.base, args), 0)
        self.assertFalse(missing.exists())
        manifest = freeze(make_manifest())
        valid = self.output / manifest["run_id"]
        valid.mkdir()
        (valid / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        args = build_parser().parse_args(["university-gpu", "stop", "--run-root", str(valid)])
        self.assertEqual(run(self.base, args), 0)
        self.assertTrue((valid / "STOP").is_file())


if __name__ == "__main__":
    unittest.main()
