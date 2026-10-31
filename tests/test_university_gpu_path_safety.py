from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from rq1.university_gpu.manifest import (
    ManifestError,
    build_manifest,
    resolve_run_root,
    validate_manifest,
    validate_run_id,
)
from rq1.university_gpu.ops import OperatorError, run_design_launch
from tests.test_university_gpu_package import freeze, make_manifest


class UniversityGpuPathSafetyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "repo"
        self.output = Path(self.temp.name) / "runs"
        self.root.mkdir()
        self.output.mkdir()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_safe_historical_style_run_ids_remain_valid(self) -> None:
        for run_id in ("test-pilot", "rq1-feasibility-200", "2026.10.31", "a_b-C.2"):
            with self.subTest(run_id=run_id):
                self.assertEqual(run_id, validate_run_id(run_id))
                self.assertEqual(self.output / run_id, resolve_run_root(self.output, run_id))

    def test_run_id_rejects_dot_segments_separators_and_absolute_paths(self) -> None:
        for run_id in (".", "..", "../escape", "safe/child", r"safe\child", "/tmp/escape", r"C:\escape", r"\\server\share", "-leading", "CON", "con.txt", "safe."):
            with self.subTest(run_id=run_id), self.assertRaises(ManifestError):
                validate_run_id(run_id)

    def test_builder_and_manifest_validator_reject_traversal_run_id(self) -> None:
        with self.assertRaisesRegex(ManifestError, "run_id"):
            build_manifest(
                stage="pilot", run_id="../escape", tasks=[], relevance={}, library_skill_ids=[],
                randomization_seed=1, source_commit="a" * 40, task_manifest_sha256="1" * 64,
                controlled_failure_sha256="2" * 64, library_sha256="3" * 64,
                alfworld_data_sha256="4" * 64, relevance_sha256="5" * 64,
                tokenizer_sha256="6" * 64, tokenizer_revision="rev", filler_sha256="7" * 64,
                gate_scores_sha256="8" * 64, gate_labels_sha256="9" * 64,
                gate_threshold=0.1, gate_calibration_sha256="a" * 64,
            )
        manifest = make_manifest()
        manifest["run_id"] = "../../escape"
        self.assertTrue(any("run_id" in issue for issue in validate_manifest(manifest)))

    def test_run_root_rejects_symlink_component_in_mocked_host(self) -> None:
        with patch.object(Path, "is_symlink", return_value=True):
            with self.assertRaisesRegex(ManifestError, "symlink"):
                resolve_run_root(self.output, "test-pilot")

    def test_run_root_rejects_symlinked_output_root(self) -> None:
        with patch.object(Path, "is_symlink", autospec=True, side_effect=lambda path: path == self.output):
            with self.assertRaisesRegex(ManifestError, "output root"):
                resolve_run_root(self.output, "test-pilot")

    def test_run_root_rejects_lexical_parent_component_but_allows_absolute_root(self) -> None:
        parent_path = self.output.parent / ".." / self.output.name
        with self.assertRaisesRegex(ManifestError, "lexical"):
            resolve_run_root(parent_path, "test-pilot")
        self.assertEqual(self.output / "test-pilot", resolve_run_root(self.output, "test-pilot"))

    def test_launch_rejects_traversal_run_id_before_preflight_or_creation(self) -> None:
        manifest = freeze(make_manifest())
        traversal_manifest = dict(manifest, run_id="../../escape")
        approval = {
            "gpu_inference": True,
            "scientific_run": True,
            "manifest_sha256": manifest["manifest_sha256"],
            "approved_by": "test-only",
            "approval_reference": "test-only",
        }
        with patch("rq1.university_gpu.ops.require_posix_process_management"), self.assertRaisesRegex(OperatorError, "unsafe or invalid run output path"):
            run_design_launch(
                self.root,
                {"run_output_dir": str(self.output)},
                traversal_manifest,
                approval=approval,
                yes=True,
                cutoff_epoch=time.time() + 7200,
            )
        self.assertEqual([], list(self.output.iterdir()))


if __name__ == "__main__":
    unittest.main()
