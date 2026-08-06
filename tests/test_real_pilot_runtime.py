from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rq1.pilot.catalog import PILOT_TEST_MAP
from rq1.pilot.models import EvidenceLevel, PilotMode, PilotStatus, RuntimeExecution
from rq1.pilot.real import RealPilotRuntime
from rq1.pilot.real_runtime.router import build_handlers
from rq1.pilot.runner import PilotRunner


class RealRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        (self.root / "configs").mkdir(); (self.root / "data" / "schemas").mkdir(parents=True)
        (self.root / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
        (self.root / "uv.lock").write_text("fixture\n", encoding="utf-8")
        (self.root / "AGENTS.md").write_text("controlled recovery\n", encoding="utf-8")

    def tearDown(self) -> None: self.temp.cleanup()

    def test_router_covers_every_catalog_test(self) -> None:
        self.assertEqual(set(range(37)), set(build_handlers()))

    def test_model_and_recovery_have_precise_capability_blocks(self) -> None:
        runtime = RealPilotRuntime(self.root)
        model = runtime.execute(PILOT_TEST_MAP["pilot_03"], run_id="r", attempt_id="a", output_dir=self.root / "a")
        self.assertEqual(PilotStatus.BLOCKED, model.status)
        self.assertEqual("ollama_unavailable", model.details["block_code"])
        perturbation = runtime.execute(PILOT_TEST_MAP["pilot_18"], run_id="r", attempt_id="b", output_dir=self.root / "b")
        self.assertEqual(PilotStatus.BLOCKED, perturbation.status)
        self.assertIn(perturbation.details["block_code"], {"real_replay_unverified", "canonical_perturbation_unsupported"})

    def test_real_like_all_capable_fixture_can_satisfy_readiness_without_mock_promotion(self) -> None:
        class AllCapableRuntime:
            simulated = False
            def __init__(self, _root: Path) -> None: pass
            def execute(self, _spec, **_kwargs):
                return RuntimeExecution(PilotStatus.PASSED, EvidenceLevel.REAL_INTEGRATED, {"operation_executed": True, "fixture": "installed-like", "simulated": False})
        with patch("rq1.pilot.runner.RealPilotRuntime", AllCapableRuntime):
            report = PilotRunner(self.root).create_and_run(PilotMode.REAL)
        self.assertTrue(report["experimental_ready"])
        self.assertFalse(report["mock_orchestration_ready"])
        self.assertEqual("go", report["go_no_go"]["decision"])

    def test_real_mock_evidence_cannot_promote(self) -> None:
        class MockEvidenceRuntime:
            simulated = False
            def __init__(self, _root: Path) -> None: pass
            def execute(self, _spec, **_kwargs): return RuntimeExecution(PilotStatus.PASSED, EvidenceLevel.MOCK, {"operation_executed": True, "simulated": True})
        with patch("rq1.pilot.runner.RealPilotRuntime", MockEvidenceRuntime):
            report = PilotRunner(self.root).create_and_run(PilotMode.REAL, test_id="pilot_12", include_prerequisites=True)
        self.assertEqual("blocked", report["tests"]["pilot_12"]["status"])


if __name__ == "__main__": unittest.main()
