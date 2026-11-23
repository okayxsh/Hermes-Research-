from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from rq1.scientific_prep import cli
from rq1.scientific_prep.safety import (EXIT_BLOCKED, EXIT_OK, EXIT_REFUSED, OutputRefused, assert_preparation_payload,
                                        prepare_output_dir, stamp)
from test_scientific_prep_exposure import POLICIES, eval_rules
from test_scientific_prep_inventory import T1, T3, fixture

REPO = cli.REPO_ROOT


def run(argv):
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = cli.main(argv)
    text = buffer.getvalue()
    try:
        return code, json.loads(text)
    except ValueError:
        return code, text


class SafetyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_historical_and_engineering_areas_cannot_be_written(self):
        for area in ("artifacts", "results", "runs", "state", "data/task_lists", "university-gpu/prospective"):
            with self.assertRaises(OutputRefused):
                prepare_output_dir(REPO / area, "inventory", "x", repo_root=REPO)
        protected = self.root / "E-backup"
        protected.mkdir()
        with self.assertRaises(OutputRefused):
            prepare_output_dir(protected / "sub", "inventory", "x", repo_root=REPO, extra_protected=[protected])
        with mock.patch.dict(os.environ, {"RQ1_SCI_PREP_PROTECTED_PATHS": str(protected)}):
            with self.assertRaises(OutputRefused):
                prepare_output_dir(protected, "inventory", "y", repo_root=REPO)
        self.assertFalse(any(protected.iterdir()))

    def test_versioned_directories_are_never_reused(self):
        first = prepare_output_dir(self.root / "out", "inventory", "run-1", repo_root=REPO)
        self.assertTrue(first.is_dir())
        with self.assertRaises(OutputRefused):
            prepare_output_dir(self.root / "out", "inventory", "run-1", repo_root=REPO)
        with self.assertRaises(OutputRefused):
            prepare_output_dir(self.root / "out", "inventory", "../escape", repo_root=REPO)

    def test_no_approval_or_manifest_markers(self):
        for bad in ({"manifest_type": "x"}, {"status": "approved"}, {"nested": [{"approved": True}]}, {"x": {"status": "FROZEN"}}):
            with self.assertRaises(OutputRefused):
                assert_preparation_payload(bad)
        with self.assertRaises(OutputRefused):
            stamp({"approved": True}, "s")
        value = stamp({"rows": [1]}, "s")
        self.assertEqual((value["preparation_status"], value["approved"], value["scientific_manifest"]), ("preparation_only", False, False))


class CliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        fixture(self.root / "data")
        self.out = self.root / "outputs"
        self.env = mock.patch.dict(os.environ, {"RQ1_VALID_UNSEEN_ACCESS": "evaluation-task-preparation"})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def test_valid_unseen_requires_explicit_authorization(self):
        with mock.patch.dict(os.environ, {"RQ1_VALID_UNSEEN_ACCESS": ""}):
            code, payload = run(["inventory", "--data-root", str(self.root / "data"), "--output-root", str(self.out), "--label", "a"])
        self.assertEqual(code, EXIT_BLOCKED)
        self.assertFalse(self.out.exists())

    def test_dry_run_writes_nothing_and_missing_input_blocks(self):
        code, payload = run(["inventory", "--data-root", str(self.root / "data" / "json_2.1.1"), "--dry-run",
                             "--output-root", str(self.out), "--label", "a"])
        self.assertEqual((code, payload["status"], payload["written"]), (EXIT_OK, "dry_run", False))
        self.assertFalse(self.out.exists())
        code, payload = run(["inventory", "--data-root", str(self.root / "missing"), "--output-root", str(self.out), "--label", "b"])
        self.assertEqual(code, EXIT_BLOCKED)

    def test_end_to_end_inventory_and_exposure_are_stable(self):
        args = ["inventory", "--data-root", str(self.root / "data"), "--output-root", str(self.out)]
        self.assertEqual(run(args + ["--label", "a"])[0], EXIT_OK)
        self.assertEqual(run(args + ["--label", "b"])[0], EXIT_OK)
        a = (self.out / "inventory-a" / "task-inventory.json").read_bytes()
        self.assertEqual(a, (self.out / "inventory-b" / "task-inventory.json").read_bytes())     # repeated runs are byte-stable
        self.assertEqual(run(args + ["--label", "a"])[0], EXIT_REFUSED)                          # never overwritten
        self.assertTrue((self.out / "inventory-a" / "valid-unseen-access.jsonl").is_file())
        evidence = self.root / "evidence"
        evidence.mkdir()
        (evidence / "evaluation.json").write_text(json.dumps({"tasks": [{"task_id": "valid_unseen:" + T3}], "exclusions": [
            {"task_id": "valid_unseen:" + T1, "reason": "replaced_before_freeze_no_oracle_validated_controlled_failure"}]}), encoding="utf-8")
        (evidence / "sources.json").write_text(json.dumps({"schema": "rq1-sci-prep-exposure-sources-v1", "screening_policies": POLICIES,
            "sources": [{"id": "evaluation", "kind": "task_manifest", "path": "evaluation.json", "role": "evaluation", "required": True,
                         "scope_splits": ["valid_unseen"], "selected": {"class": "EXPOSED"}, "exclusions": eval_rules()}]}), encoding="utf-8")
        code, payload = run(["exposure", "--inventory", str(self.out / "inventory-a" / "task-inventory.json"),
                             "--sources", str(evidence / "sources.json"), "--output-root", str(self.out), "--label", "e1"])
        self.assertIn(code, (EXIT_OK, 1), payload)
        produced = sorted(p.name for p in (self.out / "exposure-e1").iterdir())
        self.assertEqual(produced, sorted(["candidate-screening-summary.json", "exposure-ledger.csv", "exposure-ledger.json",
                                           "run.json", "template-overlap-summary.json", "uncertainty-ledger.json"]))
        for name in produced:
            if name.endswith(".json") and name != "run.json":
                value = json.loads((self.out / "exposure-e1" / name).read_text(encoding="utf-8"))
                self.assertEqual(value["preparation_status"], "preparation_only")
                assert_preparation_payload(value)
        self.assertEqual(payload["pool_counts"]["confirmatory_valid_unseen"]["outcome_screened"], 1)

    def test_oracle_validation_requires_explicit_consent(self):
        code, payload = run(["oracle-validate", "--inventory", "x", "--ledger", "y", "--screening", "z", "--pool",
                             "pilot_valid_seen", "--policy", "outcome_screened", "--data-root", str(self.root / "data"),
                             "--output-root", str(self.out), "--label", "o"])
        self.assertEqual(code, EXIT_REFUSED)
        self.assertFalse(self.out.exists())

    def _prepared_pilot(self):
        from scientific_prep_fakes import write_task
        task = "pick_and_place_simple-Apple-None-Cabinet-21/trial_T20190906_000009_000009"
        write_task(self.root / "data", "valid_seen", task, "pick_and_place_simple")
        args = ["inventory", "--data-root", str(self.root / "data"), "--splits", "valid_seen", "--output-root", str(self.out)]
        self.assertEqual(run(args + ["--label", "inv"])[0], EXIT_OK)
        evidence = self.root / "ev"
        evidence.mkdir()
        (evidence / "sources.json").write_text(json.dumps({"schema": "rq1-sci-prep-exposure-sources-v1", "screening_policies": POLICIES,
            "sources": [{"id": "demo", "kind": "inline_task_ids", "role": "demonstration", "scope_splits": ["valid_seen"],
                         "class": "EXPOSED", "provenance": "fixture", "task_ids": []}]}), encoding="utf-8")
        run(["exposure", "--inventory", str(self.out / "inventory-inv" / "task-inventory.json"), "--sources",
             str(evidence / "sources.json"), "--output-root", str(self.out), "--label", "exp"])
        return "valid_seen:" + task

    def _oracle_args(self, label):
        return ["oracle-validate", "--inventory", str(self.out / "inventory-inv" / "task-inventory.json"),
                "--ledger", str(self.out / "exposure-exp" / "exposure-ledger.json"),
                "--screening", str(self.out / "exposure-exp" / "candidate-screening-summary.json"),
                "--pool", "pilot_valid_seen", "--policy", "outcome_screened", "--data-root", str(self.root / "data"),
                "--output-root", str(self.out), "--label", label, "--yes"]

    def _patched(self, driver_cls):
        from types import SimpleNamespace
        from scientific_prep_fakes import ROUTE
        return [mock.patch("rq1.freeze.validation.git_state", return_value=("f" * 40, True, None)),
                mock.patch("rq1.bridge.adapters.capabilities.probe_alfworld_capabilities",
                           return_value=SimpleNamespace(real_adapter_ready=True, details="", version="0.4.2")),
                mock.patch("rq1.hermes.episode_driver.RealEpisodeDriver", driver_cls),
                mock.patch("rq1.recovery.reference_route.derive_handcoded_reference",
                           return_value=SimpleNamespace(actions=ROUTE))]

    def test_oracle_validate_cli_wiring_with_fake_environment(self):
        from scientific_prep_fakes import FakeDriver
        task_id = self._prepared_pilot()

        class Driver(FakeDriver):
            def __init__(self, root, *, data_dir=None, bridge_log_root=None):
                super().__init__()
            def __enter__(self):
                return self
            def __exit__(self, *_):
                return None
        with contextlib.ExitStack() as stack:
            for patcher in self._patched(Driver):
                stack.enter_context(patcher)
            code, payload = run(self._oracle_args("oracle-1") + ["--task-id", task_id])
        self.assertEqual((code, payload["validated"], payload["attempted"]), (EXIT_OK, 1, 1), payload)
        summary = json.loads((self.out / "oracle-oracle-1" / "oracle-validation-summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["rows"][0]["task_id"], task_id)
        self.assertEqual(summary["rows"][0]["status"], "ENVIRONMENT_VALIDATED")
        self.assertEqual((summary["model_calls"], summary["preparation_status"]), (0, "preparation_only"))
        with contextlib.ExitStack() as stack:                           # a task outside the screened pool is refused
            for patcher in self._patched(Driver):
                stack.enter_context(patcher)
            code, payload = run(self._oracle_args("oracle-2") + ["--task-id", "valid_seen:not-in-pool"])
        self.assertEqual(code, EXIT_REFUSED)

    def test_oracle_validate_infrastructure_fault_aborts_without_task_outcome(self):
        self._prepared_pilot()

        class BrokenDriver:
            def __init__(self, root, *, data_dir=None, bridge_log_root=None):
                pass
            def __enter__(self):
                return self
            def __exit__(self, *_):
                return None
            def session(self, **_):
                raise RuntimeError("bridge worker could not start")
        with contextlib.ExitStack() as stack:
            for patcher in self._patched(BrokenDriver):
                stack.enter_context(patcher)
            code, payload = run(self._oracle_args("oracle-3"))
        self.assertEqual(code, EXIT_BLOCKED)
        summary = json.loads((self.out / "oracle-oracle-3" / "oracle-validation-summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["rows"][0]["code"], "aborted_infrastructure_error")
        self.assertNotIn("FAILED", {r["status"] for r in summary["rows"]})        # never converted into a task failure

    def test_horizons_command_is_arithmetic_only(self):
        code, payload = run(["horizons", "--prefix", "48", "--continuation", "30"])
        self.assertEqual(code, EXIT_OK)
        self.assertNotIn("ENVIRONMENT_VALIDATED", {r["status"] for r in payload["rows"]})


if __name__ == "__main__":
    unittest.main()
