from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rq1.logging.run_registry import RunRegistry
from rq1.pilot.catalog import PILOT_TESTS, PILOT_TEST_MAP, select_tests, validate_catalog
from rq1.pilot.gates import evidence_satisfies, validate_task_manifest
from rq1.pilot.models import EvidenceLevel, PilotMode, PilotStatus
from rq1.pilot.real import RealPilotRuntime
from rq1.pilot.registry import PilotRegistry, PilotRegistryError
from rq1.pilot.runner import PilotRunner, add_manual_evidence, input_fingerprint
from rq1.cli import build_parser


class PilotTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "configs").mkdir()
        (self.root / "configs" / "base.yaml").write_text('{"schema_version":1}\n', encoding="utf-8")
        (self.root / "data" / "schemas").mkdir(parents=True)
        (self.root / "data" / "schemas" / "fixture.schema.json").write_text('{"type":"object"}\n', encoding="utf-8")
        (self.root / "pyproject.toml").write_text('[project]\nname="fixture"\n', encoding="utf-8")
        (self.root / "uv.lock").write_text("fixture\n", encoding="utf-8")
        (self.root / "AGENTS.md").write_text("controlled recovery\n", encoding="utf-8")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_catalog_is_complete_acyclic_and_typed(self) -> None:
        self.assertEqual([], validate_catalog())
        self.assertEqual(37, len(PILOT_TESTS))
        self.assertEqual("pilot_00", PILOT_TESTS[0].test_id)
        self.assertEqual("pilot_36", PILOT_TESTS[-1].test_id)
        self.assertTrue(all(item.timeout_seconds > 0 for item in PILOT_TESTS))

    def test_selection_and_prerequisite_expansion(self) -> None:
        self.assertEqual(["pilot_17"], [item.test_id for item in select_tests(test_id="pilot_17")])
        expanded = select_tests(test_id="pilot_17", include_prerequisites=True)
        self.assertEqual("pilot_00", expanded[0].test_id)
        self.assertEqual("pilot_17", expanded[-1].test_id)
        self.assertEqual(9, len(select_tests(group="recovery")))
        with self.assertRaises(ValueError): select_tests(start="pilot_05", end="pilot_03")

    def test_task_manifest_rejects_unseen_and_malformed_tasks(self) -> None:
        errors = validate_task_manifest({"split": "valid_unseen", "tasks": []}, allowed_splits={"train", "valid_seen"})
        self.assertTrue(any("valid_unseen" in item for item in errors))
        self.assertTrue(validate_task_manifest({"split": "valid_seen", "tasks": ["bad"]}, allowed_splits={"valid_seen"}))
        self.assertEqual([], validate_task_manifest({"split": "valid_seen", "tasks": [{"task_id": "seen-1"}]}, allowed_splits={"valid_seen"}))

    def test_evidence_levels_never_promote_mock_to_real(self) -> None:
        self.assertTrue(evidence_satisfies(PilotMode.FAKE, PILOT_TEST_MAP["pilot_12"], EvidenceLevel.MOCK))
        self.assertFalse(evidence_satisfies(PilotMode.REAL, PILOT_TEST_MAP["pilot_12"], EvidenceLevel.MOCK))

    def test_registry_transitions_and_stale_running(self) -> None:
        registry = PilotRegistry(self.root)
        run_id = "phase6-fake-registry"
        registry.create(run_id, "fake", "fingerprint", ["pilot_00"], "hermes3:8b")
        registry.transition(run_id, "pilot_00", PilotStatus.RUNNING, attempt_id="a")
        self.assertEqual(["pilot_00"], registry.mark_stale_running_interrupted(run_id))
        registry.transition(run_id, "pilot_00", PilotStatus.RUNNING, attempt_id="b")
        registry.transition(run_id, "pilot_00", PilotStatus.PASSED, attempt_id="b", attempt_path="report.json", evidence_level="static")
        with self.assertRaises(PilotRegistryError): registry.transition(run_id, "pilot_00", PilotStatus.RUNNING)

    def test_complete_fake_pilot_passes_all_tests_but_returns_no_go(self) -> None:
        report = PilotRunner(self.root).create_and_run(PilotMode.FAKE)
        self.assertTrue(report["mock_orchestration_ready"])
        self.assertFalse(report["experimental_ready"])
        self.assertEqual("no_go", report["go_no_go"]["decision"])
        self.assertEqual({"passed": 37}, report["status_counts"])
        output = self.root / "artifacts" / "pilot_reports" / report["pilot_run_id"]
        for name in ("pilot-report.json", "pilot-report.md", "capability-matrix.json", "runtime-benchmark.json", "proposed-recovery-protocol.json", "go-no-go.json"):
            self.assertTrue((output / name).is_file(), name)

    def test_fake_six_families_and_failure_matrix_are_recorded(self) -> None:
        report = PilotRunner(self.root).create_and_run(PilotMode.FAKE)
        state = PilotRegistry(self.root).load(report["pilot_run_id"])
        def attempt(test_id: str) -> dict:
            return json.loads((self.root / state["tests"][test_id]["attempts"][-1]).read_text(encoding="utf-8"))
        self.assertEqual(6, len(attempt("pilot_12")["details"]["task_families"]))
        self.assertEqual(11, attempt("pilot_26")["details"]["classified"])
        self.assertFalse(attempt("pilot_26")["details"]["live_services_stopped"])
        paired = [attempt(test_id)["details"] for test_id in ("pilot_21", "pilot_22", "pilot_23")]
        self.assertEqual(1, len({item["checkpoint_digest"] for item in paired}))
        self.assertEqual(1, len({item["context_digest"] for item in paired}))

    def test_resume_skips_passed_attempts(self) -> None:
        runner = PilotRunner(self.root)
        report = runner.create_and_run(PilotMode.FAKE)
        run_id = report["pilot_run_id"]
        before = len(RunRegistry(self.root / "state" / "run_registry.sqlite").pilot_attempt_bindings(run_id))
        resumed = runner.resume(run_id)
        after = len(RunRegistry(self.root / "state" / "run_registry.sqlite").pilot_attempt_bindings(run_id))
        self.assertTrue(resumed["mock_orchestration_ready"])
        self.assertEqual(before, after)

    def test_resume_restarts_interrupted_test_with_new_attempt(self) -> None:
        registry = PilotRegistry(self.root)
        run_id = "phase6-fake-interrupted"
        fingerprint = input_fingerprint(self.root, PilotMode.FAKE, "hermes3:8b")
        registry.create(run_id, "fake", fingerprint, ["pilot_00"], "hermes3:8b")
        registry.transition(run_id, "pilot_00", PilotStatus.RUNNING, attempt_id="old-attempt")
        PilotRunner(self.root).resume(run_id)
        state = registry.load(run_id)
        self.assertEqual("passed", state["tests"]["pilot_00"]["status"])
        self.assertNotEqual("old-attempt", state["tests"]["pilot_00"]["latest_attempt"])

    def test_unmet_individual_prerequisite_blocks_without_execution(self) -> None:
        report = PilotRunner(self.root).create_and_run(PilotMode.FAKE, test_id="pilot_28")
        self.assertFalse(report["mock_orchestration_ready"])
        self.assertEqual("blocked", report["tests"]["pilot_28"]["status"])

    def test_real_runtime_fails_closed(self) -> None:
        runtime = RealPilotRuntime(self.root)
        execution = runtime.execute(PILOT_TEST_MAP["pilot_03"], run_id="r", attempt_id="a", output_dir=self.root / "out")
        self.assertEqual(PilotStatus.BLOCKED, execution.status)
        self.assertFalse(execution.details["real_operation_executed"])
        final = runtime.execute(PILOT_TEST_MAP["pilot_36"], run_id="r", attempt_id="b", output_dir=self.root / "out2")
        self.assertEqual(PilotStatus.PASSED, final.status)

    def test_manual_evidence_is_hashed_and_does_not_change_status(self) -> None:
        registry = PilotRegistry(self.root)
        run_id = "phase6-fake-manual"
        registry.create(run_id, "fake", "fingerprint", ["pilot_00"], "hermes3:8b")
        source = self.root / "manual.json"; source.write_text('{"observed":true,"api_token":"secret"}\n', encoding="utf-8")
        evidence = add_manual_evidence(self.root, run_id, "pilot_00", source, EvidenceLevel.STATIC)
        self.assertEqual(64, len(evidence["sha256"]))
        stored = json.loads((self.root / evidence["path"]).read_text(encoding="utf-8"))
        self.assertEqual("[REDACTED]", stored["api_token"])
        state = registry.load(run_id)
        self.assertEqual("not_started", state["tests"]["pilot_00"]["status"])

    def test_cli_contract_parses_all_primary_pilot_commands(self) -> None:
        parser = build_parser()
        self.assertEqual("run", parser.parse_args(["pilot", "run", "--mode", "fake"]).pilot_command)
        self.assertEqual("pilot_17", parser.parse_args(["pilot", "prerequisites", "--test", "pilot_17"]).test_id)
        evidence = parser.parse_args(["pilot", "evidence", "add", "--run-id", "r", "--test", "pilot_00", "--path", "e.json", "--level", "static"])
        self.assertEqual("add", evidence.action)

    def test_run_registry_additive_bindings_are_populated(self) -> None:
        report = PilotRunner(self.root).create_and_run(PilotMode.FAKE)
        registry = RunRegistry(self.root / "state" / "run_registry.sqlite")
        self.assertEqual(37, len(registry.pilot_attempt_bindings(report["pilot_run_id"])))
        self.assertEqual(9, len(registry.recovery_evidence_bindings(report["pilot_run_id"])))


if __name__ == "__main__":
    unittest.main()
