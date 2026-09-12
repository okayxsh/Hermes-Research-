from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class RunPodPreparationTests(unittest.TestCase):
    def test_deployment_spec_is_non_billable_reviewable_json(self) -> None:
        payload = json.loads((ROOT / "configs" / "runpod" / "deployment.example.json").read_text(encoding="utf-8"))
        self.assertEqual("SECURE", payload["cloud"])
        self.assertEqual("ON_DEMAND", payload["billing"])
        self.assertFalse(payload["interruptible"])
        self.assertEqual("NVIDIA RTX A6000", payload["gpu"]["preferred"])
        self.assertEqual(120, payload["persistent_volume_gb"])
        self.assertEqual("/workspace/persistent", payload["volume_mount_path"])

    def test_bootstrap_dry_run_does_not_need_linux_or_a_gpu(self) -> None:
        env = os.environ.copy()
        result = subprocess.run(
            ["bash", "scripts/runpod/bootstrap.sh", "--dry-run", "--repo-root", str(ROOT)],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn("Nothing was installed", result.stdout)
        self.assertNotIn("runpodctl pod create", result.stdout)

    def test_validator_dry_run_is_safe_and_marks_real_checks_pending(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "runpod" / "validate_real_stack.py"),
                "--repo-root",
                str(ROOT),
                "--persistent-root",
                str(ROOT / "results"),
                "--dry-run",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn("Hermes CLI and real integration capability", result.stdout)

    def test_launch_gate_requires_manual_approval_before_validation_lookup(self) -> None:
        result = subprocess.run(
            ["bash", "scripts/runpod/start_rq1.sh", "start", "--run-id", "test"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(0, result.returncode)
        self.assertIn("--approve-launch", result.stderr)

    def test_docs_preserve_no_provisioning_and_persistent_copy_boundary(self) -> None:
        text = (ROOT / "docs" / "RUNPOD_DEPLOYMENT.md").read_text(encoding="utf-8")
        self.assertIn("documented, deliberately not run", text)
        self.assertIn("results/final/<EXPERIMENT_ID>/", text)
        self.assertIn("valid_unseen", text)
        self.assertIn("manual approval boundary", text)


if __name__ == "__main__":
    unittest.main()
