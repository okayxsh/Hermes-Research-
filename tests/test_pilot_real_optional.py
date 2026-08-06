from __future__ import annotations

import os
import unittest
from pathlib import Path

from rq1.pilot.models import PilotMode
from rq1.pilot.runner import PilotRunner


class OptionalRealPilotTests(unittest.TestCase):
    def test_real_pilot_is_explicit_and_capability_gated(self) -> None:
        if os.environ.get("RQ1_RUN_REAL_PILOT_TESTS") != "1":
            self.skipTest("set RQ1_RUN_REAL_PILOT_TESTS=1 on the approved university machine")
        report = PilotRunner(Path.cwd()).create_and_run(PilotMode.REAL, start="pilot_00", end="pilot_02")
        self.assertFalse(report["mock_orchestration_ready"])
        self.assertIn(report["go_no_go"]["decision"], {"go", "no_go"})


if __name__ == "__main__":
    unittest.main()
