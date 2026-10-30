from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from rq1.university_gpu import feasibility
from rq1.university_gpu.feasibility import (
    MODEL_DIGEST,
    MODEL_QUANTIZATION,
    MODEL_SETTINGS,
    MODEL_TAG,
    PROFILE_ID,
    RESULT_SCHEMA,
    TOTAL_ACTION_BUDGET,
    FeasibilityError,
    validate_feasibility_manifest,
)
from rq1.university_gpu.manifest import MODEL_DIGEST as HISTORICAL_DIGEST, validate_manifest
from rq1.university_gpu.models import load_json, sha256_file
from rq1.university_gpu.prospective_identity import verify_candidate
from rq1.university_gpu.scheduler import Coordinator, RunStore, SchedulerError
from rq1.university_gpu.snapshot import backup_checkpoint, export_progress, verify_progress


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_PATH = ROOT / "university-gpu" / "prospective" / "feasibility-200-candidate.json"


def candidate_manifest() -> dict:
    return load_json(CANDIDATE_PATH)


def make_fake_frozen_manifest(candidate: dict) -> dict:
    value = copy.deepcopy(candidate)
    candidate_sha = value["manifest_sha256"]
    value.update({"status": "engineering_frozen", "run_id": f"{PROFILE_ID}-offline-test",
                  "source_commit": "a" * 40, "candidate_manifest_sha256": candidate_sha})
    value["model"]["identity_status"] = "REVIEWED_PROSPECTIVE_IDENTITY"
    value["retrieval"]["status"] = "VERIFIED_LOCAL_SNAPSHOT"
    value["accum18_library"] = {"condition": "D18", "skill_count": 18, "sha256": "b" * 64,
                                "status": "VERIFIED_AGAINST_FROZEN_REFERENCE",
                                "reference_manifest_sha256": "c" * 64,
                                "skill_ids": [f"skill-{index:02d}" for index in range(18)]}
    value["tokenizer"] = {"sha256": "d" * 64, "revision": "e" * 40,
                          "status": "REVIEWED_PROSPECTIVE_TOKENIZER"}
    input_names = ("data_reconciliation_report_sha256", "data_identity_review_sha256",
                   "oracle_replay_report_sha256", "accum18_library_sha256",
                   "accum18_reference_manifest_sha256", "sbert_snapshot_sha256", "tokenizer_sha256",
                   "tokenizer_identity_report_sha256", "tokenizer_identity_review_sha256",
                   "model_identity_report_sha256")
    value["verified_inputs"] = {key: f"{index + 1:064x}" for index, key in enumerate(input_names)}
    value["engineering_approval"] = {
        "schema": "rq1-engineering-feasibility-approval-v1",
        "approval_scope": "engineering_feasibility_only",
        "candidate_manifest_sha256": candidate_sha,
        "gpu_inference": True,
        "scientific_run": False,
        "not_for_scientific_claims": True,
        "prior_pick_place_exposure_acknowledged": True,
        "approved_by": "offline-test-fixture-only",
        "approval_reference": "not-an-actual-approval",
    }
    value["units"] = feasibility._unit_rows(value["tasks"], value["accum18_library"]["sha256"])
    value["manifest_sha256"] = feasibility._manifest_digest(value)
    return value


class FakeWorker:
    def __init__(self, worker_id: str) -> None:
        self.worker_id = worker_id
        self.assignments: list[str] = []

    def execute(self, unit, attempt_dir: Path, attempt_id: str):
        self.assignments.append(str(unit["unit_key"]))
        attempt_dir.mkdir(parents=True, exist_ok=True)
        (attempt_dir / "raw-attempt.jsonl").write_text(
            json.dumps({"attempt_id": attempt_id, "unit_key": unit["unit_key"], "provider": "fake"}) + "\n",
            encoding="utf-8")
        condition = str(unit["condition"])
        ids = [] if condition == "D0" else ["skill-00", "skill-01", "skill-02"]
        measurements = {
            "result_schema": RESULT_SCHEMA,
            "scientific_evidence": False,
            "evaluation_profile": PROFILE_ID,
            "model": MODEL_TAG,
            "model_digest": MODEL_DIGEST,
            "condition": condition,
            "seed": unit["seed"],
            "inference_seed": unit["seed"],
            "inference_settings": dict(MODEL_SETTINGS),
            "total_action_budget": TOTAL_ACTION_BUDGET,
            "recovery_action_budget": unit["recovery_action_budget"],
            "checkpoint_digest": unit["checkpoint_digest"],
            "perturbation_digest": unit["post_detour_digest"],
        }
        return {
            "unit_key": unit["unit_key"], "attempt_id": attempt_id, "status": "completed",
            "profile_id": PROFILE_ID, "scientific_evidence": False, "model_tag": MODEL_TAG,
            "model_digest": MODEL_DIGEST,
            "model_quantization": MODEL_QUANTIZATION, "total_action_budget": TOTAL_ACTION_BUDGET,
            "task_id": unit["task_id"], "task_family": unit["task_family"], "condition": condition,
            "seed": unit["seed"], "success": False, "recovery_success": False,
            "retrieved_skill_ids": ids, "retrieval_similarity_scores": [0.5] * len(ids),
            "actions": 0, "steps": 0, "invalid_actions": 0, "latency_ms": 1,
            "runtime_seconds": 0.01, "log_paths": [], "measurements": measurements,
        }


class FakeOracleSession:
    def __init__(self, driver, output_dir: Path, task_id: str) -> None:
        self.driver = driver
        self.output_dir = output_dir
        self.task = driver.tasks[task_id]
        self.records = []
        self.events = []
        self.state = None
        self.done = False
        self.success = False
        self.limit = 0

    def __enter__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return self

    def __exit__(self, *_args):
        log = self.output_dir / "episode-events.jsonl"
        log.write_text("".join(json.dumps(row) + "\n" for row in self.events), encoding="utf-8")

    def start(self, task_id: str, split: str, seed: int, action_limit: int):
        assert task_id == self.task["task_id"] and split == "valid_seen" and seed == 11
        self.limit = action_limit
        self.events.append({"event": "alfworld_start", "request": {"task_id": task_id, "split": split,
                                                                       "seed": seed, "action_limit": action_limit}})
        self.state = {"marker": "start", "task_id": task_id, "step_number": 0,
                      "done": False, "success": False}
        return self.state

    def apply_scripted(self, actions, *, phase: str):
        assert list(actions) == self.task["prefix_actions"]
        for action in actions:
            self.step(action, phase=phase)

    def step(self, action: str, *, phase: str):
        self.records.append(SimpleNamespace(action=action, phase=phase))
        step_number = len(self.records)
        prefix_length = len(self.task["prefix_actions"])
        final_count = len(self.task["reference_actions"]) + 1
        if phase == "oracle_checkpoint_replay" and step_number == prefix_length:
            marker = "checkpoint"
        elif phase == "oracle_controlled_detour":
            marker = "post_detour"
        elif phase == "oracle_reference_completion" and step_number == final_count:
            marker = "terminal"
            self.done = True
            self.success = True
        else:
            marker = "running"
        response = {"action_valid": True, "done": self.done, "success": self.success}
        self.state = {"marker": marker, "task_id": self.task["task_id"], "step_number": step_number,
                      "done": self.done, "success": self.success}
        self.events.append({"event": "tool_result", "payload": {"tool": "alfworld_step",
                                                                    "request": {"action": action},
                                                                    "response": response}})
        return response


class FakeOracleDriver:
    def __init__(self, tasks):
        self.tasks = {task["task_id"]: task for task in tasks}

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

class FeasibilityProfileTests(unittest.TestCase):
    def setUp(self) -> None:
        self.candidate = candidate_manifest()

    def test_six_task_budget_formula_and_expected_allowances(self) -> None:
        self.assertEqual(validate_feasibility_manifest(self.candidate), [])
        expected = {"clean_and_place": 151, "cool_and_place": 193, "heat_and_place": 193,
                    "look_at_object": 194, "pick_and_place": 195, "pick_two_and_place": 125}
        self.assertEqual({task["task_family"]: task["recovery_action_budget"] for task in self.candidate["tasks"]}, expected)
        for task in self.candidate["tasks"]:
            self.assertEqual(task["recovery_action_budget"], 200 - len(task["prefix_actions"]) - 1)
            self.assertEqual(len(task["prefix_actions"]) + 1 + task["recovery_action_budget"], 200)
            self.assertLessEqual(len(task["reference_actions"]) + 1, 200)

    def test_two_arm_matrix_is_paired_and_not_the_six_arm_pilot(self) -> None:
        units = self.candidate["units"]
        self.assertEqual(len(units), 12)
        self.assertEqual({condition: sum(row["condition"] == condition for row in units)
                          for condition in ("D0", "D18")}, {"D0": 6, "D18": 6})
        self.assertEqual({row["seed"] for row in units}, {11})
        self.assertEqual({row["scientific_evidence"] for row in units}, {False})
        self.assertEqual({row["task_id"] for row in units}, {task["task_id"] for task in self.candidate["tasks"]})
        self.assertEqual(self.candidate["episode_count"], 12)
        self.assertEqual(self.candidate["planned_matrix_action_allowance"], 2400)
        self.assertIn("every retry adds up to 200 actions", self.candidate["planned_matrix_allowance_scope"])
        self.assertEqual(self.candidate["option_b_author_decision_status"], "pending")

    def test_historical_protocol_and_model_identity_remain_separate(self) -> None:
        self.assertEqual(HISTORICAL_DIGEST, "4eb23ef187e2c5462566d6a1d3bbbc2f1346d0b4327cbb66d58fffbcc9b2b05c")
        self.assertNotEqual(MODEL_DIGEST, HISTORICAL_DIGEST)
        self.assertEqual(self.candidate["model"]["digest"], MODEL_DIGEST)
        self.assertEqual(self.candidate["model"]["historical_digest"], HISTORICAL_DIGEST)
        self.assertTrue(all(task["original_protocol_budget"]["total_action_budget"] == 50
                            for task in self.candidate["tasks"]))
        self.assertTrue(validate_manifest(self.candidate, require_frozen=True))

    def test_budget_or_matrix_mutation_fails_even_if_hash_is_refreshed(self) -> None:
        changed = copy.deepcopy(self.candidate)
        changed["tasks"][0]["recovery_action_budget"] -= 1
        changed["manifest_sha256"] = feasibility._manifest_digest(changed)
        self.assertTrue(any("recovery budget must equal" in item for item in validate_feasibility_manifest(changed)))

        changed = copy.deepcopy(self.candidate)
        changed["units"].pop()
        changed["manifest_sha256"] = feasibility._manifest_digest(changed)
        self.assertTrue(validate_feasibility_manifest(changed))

    def test_data_identity_reconciliation_is_deterministic_and_lists_scope_differences(self) -> None:
        candidate_rows = []
        for task in self.candidate["tasks"]:
            source = task["source_task"]
            candidate_rows.append(SimpleNamespace(
                task_id=task["task_id"], family=task["task_family"], task_family=task["task_family"],
                source_sha256=source["source_sha256"], game_sha256=source["game_sha256"],
                source_identifier=source["source_identifier"], split="valid_seen"))
        discovery_records = list(candidate_rows) + [SimpleNamespace(
            task_id="valid_seen:excluded:unmapped", family="unmapped", task_family="unmapped",
            source_sha256="1" * 64, game_sha256="2" * 64, source_identifier="excluded", split="valid_seen")]
        default_records = [SimpleNamespace(
            task_id=f"train:extra-{i}", task_family="pick_and_place", source_sha256="3" * 64,
            game_sha256="4" * 64, split="train") for i in range(2)] + candidate_rows
        seen_records = list(candidate_rows)
        fake_discovery = SimpleNamespace(data_root_identity=feasibility.EXPECTED_DISCOVERY_IDENTITY,
                                         records=discovery_records, exclusions=[])

        def fake_index(_root, splits=None):
            entries = default_records if splits is None else seen_records
            identity = feasibility.EXPECTED_ADAPTER_IDENTITY if splits is None else "5" * 64
            return SimpleNamespace(identity=identity, entries=entries)

        with patch.object(feasibility, "discover_tasks", return_value=fake_discovery), \
                patch.object(feasibility, "build_task_index", side_effect=fake_index), \
                patch.object(feasibility, "check_selected_task_sources",
                             return_value={"status": "PASS", "selected_task_count": 6, "rows": []}):
            first = feasibility.reconcile_data_identity(Path("unused"), self.candidate)
            second = feasibility.reconcile_data_identity(Path("unused"), self.candidate)
        self.assertEqual(first["report_sha256"], second["report_sha256"])
        self.assertEqual(first["status"], "REVIEW_REQUIRED")
        self.assertFalse(first["identity_digest_equal"])
        self.assertEqual(first["comparisons"]["discovery_vs_adapter_valid_seen_scope"]["only_left"],
                         ["valid_seen:excluded:unmapped"])
        self.assertEqual(first["comparisons"]["discovery_valid_seen_vs_adapter_default_scope"]["only_left"],
                         ["train:extra-0", "train:extra-1"])
        self.assertEqual(first["comparisons"]["discovery_valid_seen_vs_adapter_default_scope"]["only_right"],
                         ["valid_seen:excluded:unmapped"])

    def test_model_manifest_digest_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.json"
            manifest.write_text("{}", encoding="utf-8")
            provenance = {"model_manifest_sha256": sha256_file(manifest)}
            with self.assertRaisesRegex(ValueError, "saved model manifest SHA-256"):
                verify_candidate(manifest, root, root / "tokenizer.json", provenance, lambda _text: [], root / "link.bin",
                                 expected_model_digest="0" * 64)

    def test_tokenizer_missing_or_reference_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.json"
            config_bytes = b"config"
            blob_digest = hashlib.sha256(config_bytes).hexdigest()
            model_manifest = {"config": {"digest": "sha256:" + blob_digest, "size": len(config_bytes)}, "layers": []}
            manifest.write_text(json.dumps(model_manifest), encoding="utf-8")
            (root / ("sha256-" + blob_digest)).write_bytes(config_bytes)
            provenance = {"model_manifest_sha256": sha256_file(manifest)}
            with self.assertRaises(OSError):
                verify_candidate(manifest, root, root / "missing-tokenizer.json", provenance, lambda _text: [],
                                 root / "link.bin", expected_model_digest=sha256_file(manifest))

            manifest.write_text(json.dumps(model_manifest), encoding="utf-8")
            tokenizer = root / "tokenizer.json"
            tokenizer.write_text("{}", encoding="utf-8")
            link = root / "link.bin"
            link.write_bytes(b"reviewed linkage evidence")
            tokenizer_sha = sha256_file(tokenizer)
            provenance = {"model_manifest_sha256": sha256_file(manifest), "tokenizer_sha256": tokenizer_sha,
                          "revision": "a" * 40, "source_repository": "https://example.invalid/model",
                          "model_tokenizer_link_evidence_sha256": sha256_file(link),
                          "samples": [{"text": f"sample {i}", "token_ids": [i], "reference_source": "fixture"}
                                      for i in range(5)]}
            with self.assertRaisesRegex(ValueError, "reference token comparison"):
                verify_candidate(manifest, root, tokenizer, provenance, lambda _text: [999], link,
                                 expected_model_digest=sha256_file(manifest))

    def test_candidate_cannot_enter_scientific_launch_path(self) -> None:
        calls = []
        with self.assertRaisesRegex(FeasibilityError, "unapproved candidate manifests"):
            feasibility.run_feasibility_launch(
                ROOT, {}, self.candidate, approval={}, yes=True, cutoff_epoch=10**12,
                machine_report_fn=lambda **_kwargs: calls.append("machine"),
            )
        self.assertEqual(calls, [])
        self.assertTrue(validate_manifest(self.candidate, require_frozen=True))

    def test_resume_immutability_checkpoint_export_and_recovery(self) -> None:
        frozen = make_fake_frozen_manifest(self.candidate)
        self.assertEqual(validate_feasibility_manifest(frozen, require_frozen=True), [])
        workers = [FakeWorker(f"worker-{index}") for index in range(4)]
        with tempfile.TemporaryDirectory() as temporary:
            tmp = Path(temporary)
            run_root = tmp / "durable-run"
            store = RunStore(run_root, frozen)
            first = Coordinator(store, workers).run(
                cutoff_epoch=10**12, max_attempts=1, minimum_start_seconds=0, storage_minimum_bytes=0)
            self.assertEqual(first["completed_units"], 1)
            first_key = next(iter(store.completed()))
            first_result_path = store.results_dir / f"{first_key}.json"
            first_digest = sha256_file(first_result_path)

            resumed = Coordinator(store, workers).run(
                resume=True, retry_failed=True, cutoff_epoch=10**12,
                minimum_start_seconds=0, storage_minimum_bytes=0)
            self.assertEqual(resumed["status"], "completed")
            self.assertEqual(resumed["completed_units"], 12)
            self.assertEqual(sha256_file(first_result_path), first_digest)
            with self.assertRaisesRegex(SchedulerError, "cannot be overwritten"):
                unit = next(row for row in frozen["units"] if row["unit_key"] == first_key)
                result = store.completed()[first_key]["result"]
                store.record_result(unit, result, attempt_id="duplicate", worker_id="test")

            checkpoint = backup_checkpoint(store)
            checkpoint_manifest = load_json(checkpoint / "CHECKPOINT-MANIFEST.json")
            self.assertTrue(any(row["path"].startswith("attempts/") for row in checkpoint_manifest["files"]))
            self.assertEqual(Coordinator(store, workers).validate()["passed"], True)
            Coordinator(store, workers).merge()
            destination = tmp / "external-export"
            exported = export_progress(ROOT, run_root, store.manifest_path, destination)
            self.assertEqual(exported["status"], "EXPORTED")
            self.assertEqual(verify_progress(destination)["status"], "PASS")

    def test_oracle_raw_logs_must_live_outside_checkout(self) -> None:
        with self.assertRaisesRegex(FeasibilityError, "outside the Git checkout"):
            feasibility.replay_candidate_oracles(
                ROOT, ROOT / "missing-data", self.candidate,
                evidence_dir=ROOT / "artifacts" / "oracle-raw",
                output=ROOT / "artifacts" / "oracle-report.json",
                driver_factory=lambda *_args, **_kwargs: self.fail("must not start driver"),
            )

    def test_oracle_replay_records_full_successful_witness_without_model_calls(self) -> None:
        from rq1.pilot.real_runtime import harnesses

        class Driver(FakeOracleDriver):
            def session(self, *, output_dir: Path, run_id: str, attempt_id: str, profile: str):
                return FakeOracleSession(self, output_dir, self._active_task_id)

        class Harness:
            def __init__(self, _driver, *, frozen_perturbation, **_kwargs):
                self.frozen = frozen_perturbation
                self.checkpoint_digest = None

            def start_and_replay(self, _task_id, _split, _seed, _prefix):
                self.checkpoint_digest = self.frozen["checkpoint_digest"]

            def apply_controlled_action_perturbation(self, checkpoint_id):
                return SimpleNamespace(action=self.frozen["detour_action"],
                                       post_state_digest=self.frozen["post_detour_digest"])

            def close(self):
                return None

        driver = Driver(self.candidate["tasks"])

        def driver_factory(*_args, **_kwargs):
            return driver

        def harness_factory(_driver, *, run_id, **kwargs):
            index = int(run_id.rsplit("-", 1)[-1]) - 1
            task = self.candidate["tasks"][index]
            driver._active_task_id = task["task_id"]
            return Harness(_driver, **kwargs)

        def digest(state):
            task = driver.tasks[state["task_id"]]
            return task["checkpoint_digest"] if state["marker"] == "checkpoint" else task["post_detour_digest"]

        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(feasibility, "check_selected_task_sources",
                             return_value={"status": "PASS", "selected_task_count": 6, "rows": []}), \
                patch.object(harnesses, "bridge_state_digest", side_effect=digest):
            base = Path(temporary)
            report = feasibility.replay_candidate_oracles(
                ROOT, base / "alfworld", self.candidate, evidence_dir=base / "raw",
                output=base / "oracle-report.json", driver_factory=driver_factory,
                harness_factory=harness_factory)
            raw_logs_exist = all((base / "raw" / Path(row["evidence_dir"]).name / "solvability-oracle"
                                  / "episode-events.jsonl").is_file() for row in report["tasks"])
        self.assertEqual(report["status"], "PASS", json.dumps(report, indent=2))
        self.assertEqual(report["model_calls"], 0)
        self.assertEqual(len(report["tasks"]), 6)
        self.assertEqual(feasibility.validate_oracle_report(report, self.candidate), [])
        for row, task in zip(report["tasks"], self.candidate["tasks"]):
            self.assertTrue(row["oracle_success"])
            self.assertEqual(row["total_actions"], len(task["reference_actions"]) + 1)
        self.assertTrue(raw_logs_exist)


if __name__ == "__main__":
    unittest.main()
