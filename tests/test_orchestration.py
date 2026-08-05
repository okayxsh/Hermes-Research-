from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rq1.orchestration.state_registry import StageRegistry, StageTransitionError


class StageRegistryTests(unittest.TestCase):
    def test_prerequisite_and_transition(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            registry = StageRegistry(Path(directory) / "state.json")
            self.assertFalse(registry.can_start("install")[0])
            registry.mark_running("preflight", "attempt-1")
            next_name = registry.finish("preflight", "passed", "report.json")
            self.assertEqual("install", next_name)
            self.assertTrue(registry.can_start("install")[0])

    def test_passed_stage_cannot_restart(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            registry = StageRegistry(Path(directory) / "state.json")
            registry.mark_running("preflight", "attempt-1")
            registry.finish("preflight", "passed", "report.json")
            with self.assertRaises(StageTransitionError):
                registry.mark_running("preflight", "attempt-2")
