from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from pathlib import Path
from typing import Any, Mapping

from rq1.skills.library import TASK_FAMILIES
from rq1.university_gpu.analysis import AnalysisError, analyze
from rq1.university_gpu.calibration import CalibrationError, propose_gate_calibration, validate_gate_calibration
from rq1.university_gpu.interventions import intervene
from rq1.university_gpu.manifest import CONDITIONS, build_manifest, freeze_manifest, validate_manifest
from rq1.university_gpu.models import GPUDevice, content_hash, sha256_file
from rq1.university_gpu.ollama import server_environment, worker_specs
from rq1.university_gpu.ops import OperatorError, _input_checks, directory_sha256, export_run, run_design_launch
from rq1.university_gpu.relevance import RelevanceError, parse_relevance_csv
from rq1.university_gpu.scheduler import Coordinator, RunStore, SchedulerError
from rq1.university_gpu.snapshot import SnapshotError, backup_checkpoint, export_progress, verify_progress
from rq1.university_gpu.benchmark import report as benchmark_report
from rq1.university_gpu.worker_process import WorkerProcessClient
from rq1.university_gpu.progress import report as progress_report
from rq1.university_gpu.figures import write_artifacts
from rq1.university_gpu.hardware import machine_report, parse_compute_processes, parse_nvidia_smi_csv, verify_worker_placements


def _task(stage: str, family: str, index: int) -> dict[str, Any]:
    split = "valid_seen" if stage == "pilot" else "valid_unseen"
    return {
        "task_id": f"{split}:{family}:{index}", "task_family": family, "order_index": index,
        "checkpoint_id": f"cp-{family}-{index}", "checkpoint_digest": f"{index:064x}",
        "post_detour_digest": f"{index + 1:064x}", "detour_action": "go to kitchen",
        "expected_next_action": "go to bedroom", "prefix_actions": ["go to hallway"],
        "reference_actions": ["go to hallway", "go to bedroom"], "recovery_action_budget": 12,
    }


def make_manifest(stage: str = "pilot") -> dict[str, Any]:
    per_family = 1 if stage == "pilot" else 5
    tasks = [_task(stage, family, index) for family in TASK_FAMILIES for index in range(per_family)]
    skill_ids = [f"skill-{index:02d}" for index in range(18)]
    relevance = {task["task_id"]: {"relevant": [skill_ids[0], skill_ids[1]], "irrelevant": skill_ids[2:]}
                 for task in tasks}
    return build_manifest(
        stage=stage, run_id=f"test-{stage}", tasks=tasks, relevance=relevance,
        library_skill_ids=skill_ids, randomization_seed=20261026, source_commit="a" * 40,
        task_manifest_sha256="1" * 64, controlled_failure_sha256="2" * 64,
        library_sha256="3" * 64, alfworld_data_sha256="8" * 64, relevance_sha256="4" * 64,
        tokenizer_sha256="5" * 64, tokenizer_revision="gemma4-tokenizer-rev-2026-10-20",
        filler_sha256="6" * 64, gate_scores_sha256="9" * 64, gate_labels_sha256="a" * 64,
        gate_threshold=0.25, gate_calibration_sha256="7" * 64,
    )


def freeze(manifest: Mapping[str, Any], *, acknowledge: bool = False) -> dict[str, Any]:
    approval = {"approved": True, "approved_by": "unit-test", "approval_reference": "test-record",
                "manifest_sha256": manifest["manifest_sha256"]}
    if acknowledge:
        approval["previous_task_outcomes_acknowledged"] = True
        approval.update({"eligibility_verified": True, "blinded_relevance_verified": True,
                         "leakage_review_complete": True, "pilot_go_approved": True,
                         "precision_power_review_complete": True, "power_report_sha256": "b" * 64,
                         "pilot_validation_sha256": "c" * 64})
    with tempfile.TemporaryDirectory() as temporary:
        return freeze_manifest(manifest, approval, Path(temporary) / "frozen.json")


class CharacterTokenizer:
    def encode(self, text: str) -> list[int]:
        return [ord(character) for character in text]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(chr(token_id) for token_id in token_ids)


class FakeWorker:
    def __init__(self, worker_id: str, *, delay: float = 0, fail_first: bool = False) -> None:
        self.worker_id = worker_id
        self.delay = delay
        self.fail_first = fail_first
        self.attempts: dict[str, int] = {}

    def execute(self, unit: Mapping[str, Any], attempt_dir: Path, attempt_id: str) -> Mapping[str, Any]:
        key = str(unit["unit_key"])
        self.attempts[key] = self.attempts.get(key, 0) + 1
        if self.delay:
            time.sleep(self.delay)
        if self.fail_first and self.attempts[key] == 1:
            raise RuntimeError("mock infrastructure interruption")
        return {"unit_key": key, "attempt_id": attempt_id, "status": "completed", "task_id": unit["task_id"],
                "task_family": unit["task_family"], "condition": unit["condition"], "seed": unit["seed"],
                "recovery_success": unit["condition"] in {"D18", "Rel", "Gate"},
                "retrieved_skill_ids": [] if unit["condition"] == "D0" else ["skill-00", "skill-02", "skill-03"]}


class UniversityGpuManifestTests(unittest.TestCase):
    def test_balanced_pilot_and_confirmatory_manifests(self) -> None:
        for stage, expected in (("pilot", 36), ("confirmatory", 540)):
            manifest = make_manifest(stage)
            self.assertEqual(manifest["episode_count"], expected)
            self.assertEqual(manifest["condition_counts"], {condition: expected // 6 for condition in CONDITIONS})
            self.assertEqual(len({unit["unit_key"] for unit in manifest["units"]}), expected)
            self.assertEqual(validate_manifest(manifest), [])

    def test_confirmatory_freeze_requires_exposure_acknowledgment(self) -> None:
        manifest = make_manifest("confirmatory")
        with self.assertRaisesRegex(ValueError, "acknowledge"):
            freeze(manifest)
        frozen = freeze(manifest, acknowledge=True)
        self.assertEqual(validate_manifest(frozen, require_frozen=True), [])

    def test_confirmatory_freeze_requires_pilot_leakage_and_power_review(self) -> None:
        manifest = make_manifest("confirmatory")
        approval = {"approved": True, "approved_by": "test", "approval_reference": "test",
                    "manifest_sha256": manifest["manifest_sha256"], "previous_task_outcomes_acknowledged": True}
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "eligibility_verified"):
                freeze_manifest(manifest, approval, Path(temporary) / "manifest.json")

    def test_manifest_rejects_identity_tampering_even_with_recomputed_hash(self) -> None:
        manifest = make_manifest()
        changed = dict(manifest)
        changed["model"] = {"tag": "stale-llama", "digest": "0" * 64, "quantization": "Q4_K_M"}
        changed["manifest_sha256"] = content_hash({key: value for key, value in changed.items()
                                                   if key not in {"generated_at", "manifest_sha256", "status", "approval"}})
        self.assertTrue(any("model identity" in item for item in validate_manifest(changed)))

    def test_malformed_manifest_fields_fail_with_diagnostics(self) -> None:
        for field, value, expected in (("cell_index", [], "invalid integer field: cell_index"),
                                       ("task_seed_cells", "bad", "task_seed_cells must be an integer"),
                                       ("tasks", {}, "task list must be an array")):
            with self.subTest(field=field):
                changed = json.loads(json.dumps(make_manifest()))
                if field == "cell_index":
                    changed["units"][0][field] = value
                else:
                    changed[field] = value
                changed["manifest_sha256"] = content_hash({key: value for key, value in changed.items()
                                                           if key not in {"generated_at", "manifest_sha256", "status", "approval"}})
                self.assertTrue(any(expected in item for item in validate_manifest(changed)))

    def test_relevance_coverage_and_leakage_are_fail_closed(self) -> None:
        task = "valid_seen:family:0"
        ids = {f"skill-{index:02d}" for index in range(18)}
        rows = [f"{task},{skill},RELEVANT,RELEVANT,RELEVANT" if index < 2
                else f"{task},{skill},IRRELEVANT,IRRELEVANT,IRRELEVANT"
                for index, skill in enumerate(sorted(ids))]
        valid = ("task_id,skill_id,rater_a,rater_b,adjudicated\n" + "\n".join(rows) + "\n").encode()
        self.assertEqual(set(parse_relevance_csv(valid, allowed_tasks={task}, library_skill_ids=ids)), {task})
        incomplete = ("task_id,skill_id,rater_a,rater_b,adjudicated\n" + "\n".join(rows[:-1]) + "\n").encode()
        with self.assertRaises(RelevanceError):
            parse_relevance_csv(incomplete, allowed_tasks={task}, library_skill_ids=ids)
        with self.assertRaises(RelevanceError):
            parse_relevance_csv(valid, allowed_tasks={"valid_unseen:leaked"}, library_skill_ids=ids)

    def test_gate_calibration_is_outcome_blind_and_hashable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scores = root / "scores.csv"
            labels = root / "labels.csv"
            score_rows = ["task_id,skill_id,cosine_similarity"]
            label_rows = ["task_id,skill_id,adjudicated"]
            for task_index in range(6):
                for skill_index in range(18):
                    task = f"valid_seen:family-{task_index}:{task_index}"
                    skill = f"skill-{skill_index:02d}"
                    score = 0.8 if skill_index < 9 else 0.1
                    label = "RELEVANT" if skill_index < 9 else "IRRELEVANT"
                    score_rows.append(f"{task},{skill},{score}")
                    label_rows.append(f"{task},{skill},{label}")
            scores.write_text("\n".join(score_rows) + "\n", encoding="utf-8")
            labels.write_text("\n".join(label_rows) + "\n", encoding="utf-8")
            proposal = propose_gate_calibration(scores, labels)
            self.assertEqual(proposal["status"], "proposed")
            self.assertFalse(proposal["recovery_outcomes_used"])
            self.assertEqual(validate_gate_calibration(proposal), [])
            with self.assertRaises(CalibrationError):
                propose_gate_calibration(scores, labels.with_name("missing.csv"))

    def test_memory_controls_preserve_baseline_and_match_budget(self) -> None:
        tokenizer = CharacterTokenizer()
        raw = {"recovery_memory": {"retrieved_skills": [
            {"rank": 1, "skill_id": "skill-00", "text": "known useful text"},
            {"rank": 2, "skill_id": "skill-02", "text": "other text"},
            {"rank": 3, "skill_id": "skill-03", "text": "more text"}],
            "no_retrieved_skills_available": False}}
        top = [{"skill_id": f"skill-{index:02d}", "score": 0.9 - index / 10} for index in range(3)]
        texts = {f"skill-{index:02d}": "word" for index in range(18)}
        selected = {"valid_seen:family:0": {"relevant": "skill-04", "irrelevant": "skill-05"}}
        filler = "x"
        d18, d18_meta = intervene(arm="D18", task_id="valid_seen:family:0", raw_memory=raw,
                                  retrieval_top=top, skill_texts=texts, selected_skill_ids=selected,
                                  tokenizer=tokenizer, gate_threshold=0.5, filler_unit=filler)
        self.assertEqual(d18, raw)
        gate_all, gate_meta = intervene(arm="Gate", task_id="valid_seen:family:0", raw_memory=raw,
                                        retrieval_top=top, skill_texts=texts, selected_skill_ids=selected,
                                        tokenizer=tokenizer, gate_threshold=-1, filler_unit=filler)
        self.assertEqual(gate_all, raw)
        self.assertEqual(gate_meta["token_count"], d18_meta["token_count"])
        self.assertEqual(d18_meta["token_count"], len(tokenizer.encode(json.dumps(raw["recovery_memory"],
                                                                                  ensure_ascii=False, sort_keys=True))))
        d0, _ = intervene(arm="D0", task_id="valid_seen:family:0", raw_memory={"recovery_memory": {
            "retrieved_skills": [], "no_retrieved_skills_available": True}}, retrieval_top=[], skill_texts=texts,
            selected_skill_ids=selected, tokenizer=tokenizer, gate_threshold=0.5, filler_unit=filler)
        self.assertEqual(d0["recovery_memory"]["retrieved_skills"], [])
        for arm in ("Rel", "Irr", "Fill", "Gate"):
            memory, metadata = intervene(arm=arm, task_id="valid_seen:family:0", raw_memory=raw,
                                         retrieval_top=top, skill_texts=texts, selected_skill_ids=selected,
                                         tokenizer=tokenizer, gate_threshold=0.5, filler_unit=filler)
            count = len(tokenizer.encode(json.dumps(memory["recovery_memory"], ensure_ascii=False,
                                                     sort_keys=True)))
            self.assertEqual(count, d18_meta["token_count"], arm)
            self.assertEqual(metadata["token_count"], d18_meta["token_count"], arm)


class UniversityGpuExecutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = freeze(make_manifest())
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.run_root = self.root / "run"
        self.store = RunStore(self.run_root, self.manifest)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_gpu_inventory_and_uuid_process_placement(self) -> None:
        inventory = "GPU-A, 0, A100, 40960, 30000\nGPU-B, 1, A100, 40960, 30000\n"
        devices = parse_nvidia_smi_csv(inventory)
        specs = worker_specs(devices, count=2, base_port=11534, root=self.root / "workers")
        self.assertEqual({item.gpu_uuid for item in specs}, {"GPU-A", "GPU-B"})
        with tempfile.TemporaryDirectory() as model_dir:
            environment = server_environment(specs[0], model_dir=Path(model_dir))
            self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], specs[0].gpu_uuid)
            self.assertEqual(environment["OLLAMA_HOST"], f"127.0.0.1:{specs[0].port}")
        rows = parse_compute_processes("101, GPU-A, 500\n102, GPU-B, 600\n")
        placement = verify_worker_placements({"w1": 101, "w2": 102}, rows,
                                             descendant_provider=lambda pid: {pid})
        self.assertTrue(placement["passed"])
        self.assertEqual(placement["assignments"], {"w1": "GPU-A", "w2": "GPU-B"})
        wrong_rows = parse_compute_processes("101, GPU-A, 500\n102, GPU-A, 600\n")
        wrong = verify_worker_placements({"w1": 101, "w2": 102}, wrong_rows,
                                         descendant_provider=lambda pid: {pid})
        self.assertFalse(wrong["passed"])

    def test_existing_run_must_preserve_the_entire_frozen_approval_record(self) -> None:
        self.store.initialize()
        changed = json.loads(json.dumps(self.manifest))
        changed["approval"]["approved_by"] = "different-operator"
        with self.assertRaisesRegex(SchedulerError, "approval record"):
            RunStore(self.run_root, changed).initialize()

    def test_launch_rejects_blank_approver_without_touching_gpu(self) -> None:
        approval = {"gpu_inference": True, "scientific_run": True, "manifest_sha256": self.manifest["manifest_sha256"],
                    "approved_by": "", "approval_reference": ""}
        with self.assertRaisesRegex(OperatorError, "approver and reference"):
            run_design_launch(Path(__file__).parents[1], {}, self.manifest, approval=approval, yes=True)
        approval.update({"approved_by": "operator", "approval_reference": "record"})
        with self.assertRaisesRegex(OperatorError, "access deadline is required"):
            run_design_launch(Path(__file__).parents[1], {}, self.manifest, approval=approval, yes=True)
        with self.assertRaisesRegex(OperatorError, "access deadline is required"):
            run_design_launch(Path(__file__).parents[1], {}, self.manifest, approval=approval, yes=True,
                              cutoff_epoch=float("nan"))
        with self.assertRaisesRegex(OperatorError, "30-minute"):
            run_design_launch(Path(__file__).parents[1], {}, self.manifest, approval=approval, yes=True,
                              cutoff_epoch=time.time() + 60)

    def test_mocked_four_worker_run_records_heartbeats_and_merges(self) -> None:
        workers = [FakeWorker(f"w{index}", delay=0.62 if index == 0 else 0) for index in range(4)]
        outcome = Coordinator(self.store, workers).run(heartbeat_seconds=0.05)
        self.assertEqual(outcome["status"], "completed")
        self.assertEqual(outcome["completed_units"], 36)
        self.assertTrue(any(event["event"] == "attempt_heartbeat" for event in self.store.event_rows()))
        validation = Coordinator(self.store, workers).validate()
        self.assertTrue(validation["passed"])
        merged = Coordinator(self.store, workers).merge()
        self.assertEqual(merged["units"], 36)

    def test_retry_uses_new_physical_attempt_and_preserves_failure_details(self) -> None:
        worker = FakeWorker("w1", fail_first=True)
        first = Coordinator(self.store, [worker]).run()
        self.assertEqual(first["status"], "incomplete")
        self.assertEqual(first["completed_units"], 0)
        second = Coordinator(self.store, [worker]).run(resume=True, retry_failed=True)
        self.assertEqual(second["status"], "completed")
        events = self.store.event_rows()
        attempts = [event for event in events if event["event"] == "attempt_started"]
        self.assertEqual(len(attempts), 72)
        self.assertEqual(len({event["attempt_id"] for event in attempts}), 72)
        self.assertTrue(any(event["event"] == "attempt_failed" and event["task_id"] for event in events))

    def test_safe_stop_waits_for_active_unit_and_cutoff_starts_none(self) -> None:
        coordinator = Coordinator(self.store, [FakeWorker("slow", delay=0.25)])
        timer = threading.Timer(0.06, coordinator.request_stop, kwargs={"reason": "test stop"})
        timer.start()
        stopped = coordinator.run()
        timer.join()
        self.assertEqual(stopped["status"], "paused")
        self.assertEqual(stopped["completed_units"], 1)
        self.assertTrue((self.run_root / "STOP").is_file())

        cutoff_root = self.root / "cutoff"
        cutoff_store = RunStore(cutoff_root, self.manifest)
        cutoff = Coordinator(cutoff_store, [FakeWorker("never")]).run(cutoff_epoch=time.time() - 1)
        self.assertEqual(cutoff["status"], "paused")
        self.assertEqual(cutoff["attempts_started"], 0)
        self.assertTrue(cutoff["cutoff_reached"])

    def test_corrupt_and_duplicate_authoritative_records_block_validation(self) -> None:
        workers = [FakeWorker("w1")]
        Coordinator(self.store, workers).run()
        result_path = next((self.run_root / "results" / "units").glob("*.json"))
        original = json.loads(result_path.read_text(encoding="utf-8"))
        result_path.write_text(json.dumps({**original, "result_sha256": "0" * 64}), encoding="utf-8")
        with self.assertRaises(SchedulerError):
            self.store.completed()
        result_path.write_text(json.dumps(original), encoding="utf-8")
        duplicate_path = result_path.with_name(result_path.stem + ".duplicate.json")
        duplicate_path.write_text(json.dumps(original), encoding="utf-8")
        with self.assertRaises(SchedulerError):
            self.store.completed()

    def test_missing_binary_outcome_cannot_become_an_authoritative_result(self) -> None:
        self.store.initialize()
        unit = self.manifest["units"][0]
        result = FakeWorker("worker").execute(unit, self.root, "attempt")
        result["recovery_success"] = None
        with self.assertRaises(SchedulerError):
            self.store.record_result(unit, result, attempt_id="attempt", worker_id="worker")

    def test_analysis_is_paired_clustered_and_rater_sensitive(self) -> None:
        rows = []
        labels = {}
        for task in self.manifest["tasks"]:
            for skill_index in range(18):
                skill = f"skill-{skill_index:02d}"
                relevant = skill_index < 2
                labels[(task["task_id"], skill)] = {
                    "adjudicated": "RELEVANT" if relevant else "IRRELEVANT",
                    "rater_a": "RELEVANT" if relevant else "IRRELEVANT",
                    "rater_b": "IRRELEVANT" if skill_index == 0 else ("RELEVANT" if relevant else "IRRELEVANT"),
                }
        for unit in self.manifest["units"]:
            rows.append({"unit_key": unit["unit_key"], "status": "completed", "task_id": unit["task_id"],
                         "task_family": unit["task_family"], "condition": unit["condition"], "seed": unit["seed"],
                         "recovery_success": unit["condition"] in {"D18", "Rel", "Gate"},
                         "retrieved_skill_ids": [] if unit["condition"] == "D0" else ["skill-00", "skill-02", "skill-03"]})
        report = analyze(rows, self.manifest, relevance_labels=labels, bootstrap_replicates=1000)
        self.assertEqual(report["conditions"]["Rel"]["recovery_rate"], 1.0)
        self.assertEqual(report["paired_task_cluster_contrasts"]["Rel-D18"]["task_clusters"], 6)
        self.assertFalse(report["analysis"]["power_or_sample_sufficiency_established"])
        self.assertNotEqual(report["retrieval_noise_sensitivity"]["adjudicated"],
                            report["retrieval_noise_sensitivity"]["rater_b"])
        analysis_path = self.root / "analysis.json"
        analysis_path.write_text(json.dumps(report), encoding="utf-8")
        artifacts = write_artifacts(analysis_path, self.root / "analysis-artifacts")
        self.assertEqual(artifacts["status"], "PASS")
        self.assertIn("Rel", (self.root / "analysis-artifacts" / "conditions.svg").read_text(encoding="utf-8"))
        with self.assertRaises(AnalysisError):
            analyze(rows[:-1], self.manifest, bootstrap_replicates=1000)

    def test_archive_source_hashes_are_read_only_and_exports_are_validation_gated(self) -> None:
        frozen_config = Path(__file__).parents[1] / "configs" / "models" / "gemma4-12b.yaml"
        before = sha256_file(frozen_config)
        with self.assertRaises(SchedulerError):
            Coordinator(self.store, [FakeWorker("offline")]).merge()
        self.assertEqual(sha256_file(frozen_config), before)
        with self.assertRaises(FileNotFoundError):
            export_run(self.run_root, self.root / "missing-manifest.json", self.root / "export")
        Coordinator(self.store, [FakeWorker("complete")]).run()
        Coordinator(self.store, [FakeWorker("complete")]).merge()
        manifest_path = self.root / "frozen-manifest.json"
        manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")
        result = export_run(self.run_root, manifest_path, self.root / "export")
        self.assertEqual(result["status"], "PASS")
        self.assertTrue((self.root / "export" / "EXPORT-MANIFEST.json").is_file())
        portable = self.root / "portable"
        self.assertTrue(export_progress(Path(__file__).parents[1], self.run_root, manifest_path, portable)["complete"])
        self.assertTrue(verify_progress(portable)["complete"])
        env = dict(os.environ, PYTHONPATH=str(portable / "analysis-src"))
        analyzed = subprocess.run([sys.executable, "-B", "-m", "rq1.cli", "university-gpu", "analyze",
                                   "--results", str(portable / "results.jsonl"), "--manifest", str(portable / "run/manifest.json"),
                                   "--output", str(self.root / "analysis.json"), "--bootstrap-replicates", "1000"],
                                  cwd=portable / "analysis-src", env=env, capture_output=True, text=True)
        self.assertEqual(analyzed.returncode, 0, analyzed.stdout + analyzed.stderr)
        rendered = subprocess.run([sys.executable, "-B", "-m", "rq1.cli", "university-gpu", "analysis-artifacts",
                                   "--analysis", str(self.root / "analysis.json"), "--output-dir", str(self.root / "figures")],
                                  cwd=portable / "analysis-src", env=env, capture_output=True, text=True)
        self.assertEqual(rendered.returncode, 0, rendered.stdout + rendered.stderr)
        self.assertTrue((self.root / "figures" / "conditions.svg").is_file())
        self.assertEqual(sha256_file(frozen_config), before)

    def test_dataset_tree_digest_changes_with_file_content(self) -> None:
        dataset = self.root / "dataset"
        dataset.mkdir()
        data_file = dataset / "json_2.1.1" / "tasks.json"
        data_file.parent.mkdir()
        data_file.write_text("frozen task bytes", encoding="utf-8")
        first = directory_sha256(dataset)
        data_file.write_text("different bytes", encoding="utf-8")
        self.assertNotEqual(first, directory_sha256(dataset))

    def test_partial_export_is_verified_and_corruption_is_detected(self) -> None:
        coordinator = Coordinator(self.store, [FakeWorker("slow", delay=0.2)])
        timer = threading.Timer(0.05, coordinator.request_stop)
        timer.start()
        outcome = coordinator.run()
        timer.join()
        self.assertEqual(outcome["completed_units"], 1)
        dest = self.root / "partial-export"
        exported = export_progress(Path(__file__).parents[1], self.run_root, self.store.manifest_path, dest)
        self.assertFalse(exported["complete"])
        self.assertEqual(verify_progress(dest)["missing"], 35)
        self.assertEqual(progress_report(dest / "run", dest / "run" / "manifest.json")["status"], "INCOMPLETE")
        env = dict(os.environ, PYTHONPATH=str(dest / "analysis-src"))
        checked = subprocess.run([sys.executable, "-B", "-m", "rq1.cli", "university-gpu", "verify-export",
                                  "--destination", str(dest)], cwd=dest / "analysis-src", env=env, capture_output=True, text=True)
        self.assertEqual(checked.returncode, 0, checked.stderr + checked.stdout)
        (dest / "results.jsonl").write_text("corrupt\n", encoding="utf-8")
        with self.assertRaises(SnapshotError):
            verify_progress(dest)

    def test_interrupted_export_does_not_publish_a_partial_destination(self) -> None:
        self.store.initialize()
        dest = self.root / "export"
        with patch("rq1.university_gpu.snapshot.shutil.copy2", side_effect=OSError("simulated interruption")):
            with self.assertRaises(OSError):
                export_progress(Path(__file__).parents[1], self.run_root, self.store.manifest_path, dest)
        self.assertFalse(dest.exists())
        self.assertTrue(list(self.root.glob(".export.incomplete-*")))
        export_progress(Path(__file__).parents[1], self.run_root, self.store.manifest_path, dest)
        self.assertFalse(verify_progress(dest)["complete"])

    def test_failed_worker_is_quarantined_while_other_workers_continue(self) -> None:
        class DeadWorker(FakeWorker):
            quarantine_on_failure = True

            def execute(self, unit, attempt_dir, attempt_id):
                raise RuntimeError("simulated model service failure")

        outcome = Coordinator(self.store, [DeadWorker("dead"), FakeWorker("healthy")]).run()
        self.assertEqual(outcome["completed_units"], 35)
        self.assertEqual(len(outcome["failures"]), 1)
        self.assertTrue(list((self.run_root / "backups").glob("checkpoint-*")))

    def test_backup_failure_stops_new_work_without_losing_completed_unit(self) -> None:
        with patch("rq1.university_gpu.snapshot.backup_checkpoint", side_effect=OSError("disk full")):
            outcome = Coordinator(self.store, [FakeWorker("worker")]).run()
        self.assertEqual(outcome["completed_units"], 10)
        self.assertEqual(outcome["failures"][0]["failure_class"], "backup")
        self.assertTrue((self.run_root / "STOP").exists())

    def test_interrupted_checkpoint_is_never_published_as_complete(self) -> None:
        self.store.initialize()
        self.store.append_event({"event": "attempt_started", "unit_key": self.manifest["units"][0]["unit_key"], "attempt_id": "test"})
        real_copy = shutil.copy2
        copied = []
        def interrupt_second(source, target):
            if copied:
                raise OSError("interrupted after first file")
            copied.append(str(source))
            return real_copy(source, target)
        with patch("rq1.university_gpu.snapshot.shutil.copy2", side_effect=interrupt_second):
            with self.assertRaises(OSError):
                backup_checkpoint(self.store)
        self.assertFalse(list((self.run_root / "backups").glob("checkpoint-*")))
        self.assertTrue(list((self.run_root / "backups").glob(".partial-*")))

    def test_low_storage_and_corrupt_input_are_blocked(self) -> None:
        from types import SimpleNamespace
        runner = lambda args, **kwargs: subprocess.CompletedProcess(args, 0, "GPU-A, 0, Mock, 32768, 32768\n", "")
        with patch("rq1.university_gpu.hardware.shutil.disk_usage", return_value=SimpleNamespace(free=1)):
            checks = machine_report(output_dir=self.root, runner=runner)["checks"]
        self.assertEqual(next(row["status"] for row in checks if row["name"] == "output_storage"), "FAIL")
        with patch("rq1.university_gpu.scheduler.shutil.disk_usage", return_value=SimpleNamespace(free=1)):
            result = Coordinator(self.store, [FakeWorker("worker")]).run()
        self.assertEqual(result["attempts_started"], 0)
        self.assertEqual(result["failures"][0]["failure_class"], "storage")
        bad = self.root / "tokenizer.json"
        bad.write_text("corrupted", encoding="utf-8")
        checks = _input_checks({"tokenizer_path": str(bad)}, self.manifest)
        self.assertEqual(next(row["status"] for row in checks if row["name"] == "tokenizer_sha256"), "FAIL")

    def test_four_worker_benchmark_stops_and_resumes(self) -> None:
        workers = [FakeWorker(f"worker-{i}", delay=0.03) for i in range(4)]
        coordinator = Coordinator(self.store, workers)
        result = coordinator.run(max_attempts=4)
        self.assertEqual(result["completed_units"], 4)
        self.assertEqual(benchmark_report(self.run_root, self.store.manifest_path)["status"], "PASS")
        result = coordinator.run(resume=True, max_attempts=4)
        self.assertEqual(result["completed_units"], 8)

    def test_frozen_worker_stream_times_out(self) -> None:
        class FrozenStream:
            def readline(self):
                time.sleep(0.2)
                return ""
        with self.assertRaisesRegex(SchedulerError, "timed out"):
            WorkerProcessClient._readline(FrozenStream(), timeout=0.01, label="mock")


if __name__ == "__main__":
    unittest.main()
