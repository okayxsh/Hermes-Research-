from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rq1.hermes.capabilities import probe_hermes_capabilities
from rq1.profiles.lifecycle import ProfileLifecycleError, profile_plan, real_profile_lifecycle


@unittest.skipUnless(os.environ.get("RQ1_RUN_REAL_HERMES_PROFILE_TESTS") == "1", "set RQ1_RUN_REAL_HERMES_PROFILE_TESTS=1 to opt into real Hermes profile tests")
class OptionalRealHermesProfileTests(unittest.TestCase):
    def test_capability_gated_temporary_profile_creation(self) -> None:
        root = Path(__file__).resolve().parents[1]
        report = probe_hermes_capabilities(project_root=root)
        required = (
            report.installed,
            report.profile_supported,
            report.no_skills_supported,
            report.profile_inspection_supported,
            report.profile_location_supported,
            report.project_plugin_activation_supported,
        )
        if not all(required):
            self.skipTest("installed Hermes does not expose every required safe profile capability")
        lifecycle = real_profile_lifecycle(root)
        try:
            manifest = lifecycle.create(profile_plan("rq1-test-real-profile-lifecycle", root))
        except ProfileLifecycleError as exc:
            self.fail(f"capability-gated real temporary profile creation failed: {exc}")
        self.assertTrue(manifest.validation_result["valid"])
