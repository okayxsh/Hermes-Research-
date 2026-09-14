"""Hermetic tests for the balanced 181-240 acquisition extension (Decision 011).

Every unit here is a scripted, non-scientific stand-in; no frozen extension task
is executed.
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from rq1.acquisition import extension, extension_launch, extension_preflight
from rq1.acquisition import launch
from rq1.acquisition.executor import RealAcquisitionExecutor
from rq1.acquisition.extension_protocol import (
    EXTENSION_DECISION_RECORD,
    EXTENSION_FROZEN_DIR,
    EXTENSION_PROTOCOL_CONFIG,
    EXTENSION_RUN_ID,
    FROZEN_MODEL_DIGEST,
    PARENT,
    ParentReference,
    extension_protocol_definition,
    extension_protocol_sha256,
)
from rq1.acquisition.gates import FROZEN_TASK_DIR, queue_identity_sha256
from rq1.acquisition.models import AcquisitionPlan
from rq1.acquisition.protocol import protocol_definition, protocol_sha256
from rq1.acquisition.runner import AcquisitionError, AcquisitionRunner, acquisition_units
from rq1.acquisition.skill_creation import prompt_hashes
from rq1.acquisition.skill_pool import EMPTY_POOL_HASH, SkillPoolError, pool_hash, rebuild_pool
from rq1.cli import build_parser
from rq1.experiment.models import canonical_hash
from rq1.experiment.persistence import CompatibilityError, ExperimentStore
from rq1.experiment.runner import RunnerOptions
from rq1.freeze.validation import (
    ACQUISITION_EXTENSION_ENVIRONMENT_REQUIRED,
    ACQUISITION_EXTENSION_EVIDENCE_MODE,
    build_freeze,
    write_freeze,
)
from rq1.hermes.episode_driver import INFERENCE_SEED, provider_settings
from rq1.skills.library import TASK_FAMILIES
from rq1.tasks.discovery import discover_tasks
from rq1.tasks.freeze import freeze_manifest
from rq1.tasks.models import SelectionPolicy
from rq1.tasks.selection import propose_manifest, select_tasks
from rq1.tasks.validation import validate_manifest
from rq1.utils.config import load_json_yaml
from rq1.utils.hashing import sha256_file

from test_acquisition_execution import APPROVED, COMMIT, GOOD_SKILL, NEAR_SKILL, FakeDriver, write_train_tasks

REPO = Path(__file__).resolve().parents[1]
QUEUE = "e" * 64
HEAT_SKILL = "TITLE: Heat before placing\nBODY: Carry the target object to a microwave, heat it, then put it in the requested receptacle."
COOL_SKILL = "TITLE: Cool before placing\nBODY: Carry the target object to a fridge, cool it, then put it in the requested receptacle."
EXTENSION_TASKS = [(f"train:pick_heat_then_place_in_recep-Cup-None-Cabinet-60{index}/trial_T{index}", "heat_and_place") for index in range(1, 4)]


def by_order(task):
    return task.order_index


def copy_prompts(root: Path) -> None:
    prompts = root / "hermes" / "prompts"
    prompts.mkdir(parents=True, exist_ok=True)
    for name in ("post_success_learning.md", "skill_validation.md"):
        shutil.copy(REPO / "hermes" / "prompts" / name, prompts / name)


def result_rows(store: ExperimentStore) -> list[dict]:
    return sorted(store.terminal_results(phase="acquisition", repair_tail=False).values(), key=lambda row: row["task_index"])


def run_units(root, driver, tasks, *, run_id, base, parent_pool=(), parent_run_id=None, offset=0, parent=None,
              initial_pool=None, resume=False, max_runs=None):
    store = ExperimentStore(root, run_id, base=base)
    plan = AcquisitionPlan(
        run_id, tuple(task for task, _ in tasks), task_families=tuple(family for _, family in tasks),
        queue_sha256=QUEUE, parent_run_id=parent_run_id, logical_index_offset=offset,
    )
    executor = RealAcquisitionExecutor(
        root, store, driver, scientific=False, queue_sha256=QUEUE, action_budget=3,
        parent_pool=parent_pool, parent_run_id=parent_run_id,
    )
    configuration = (
        extension.extension_run_configuration(root, queue_sha256=QUEUE, scientific=False, parent=parent)
        if parent is not None else launch.run_configuration(root, queue_sha256=QUEUE, scientific=False)
    )
    start = parent_pool if initial_pool is None else initial_pool
    result = AcquisitionRunner(root).run_resumable(
        plan, executor, configuration=configuration,
        options=RunnerOptions(resume=resume, max_runs=max_runs, fail_fast=True),
        output_base=base, initial_library_hash=pool_hash(start), initial_library_size=len(start),
        scientific=False, store=store, preflight=executor.preflight,
        checkpoint_extension=executor.checkpoint_state, progress=None,
    )
    return store, result


class SyntheticParent:
    """A completed six-unit parent acquisition with a frozen queue and a closeout manifest."""

    def __init__(self, root: Path) -> None:
        copy_prompts(root)
        self.data = root / "data"
        write_train_tasks(self.data, per_family=12)
        self.discovery = discover_tasks(self.data, "train")
        proposal = propose_manifest(
            "acquisition", self.discovery, SelectionPolicy("task-selection-v1", 1, 6),
            alfworld_version="0.4.2", repository_commit=COMMIT,
        )
        manifest_path = root / FROZEN_TASK_DIR / "acquisition-parent.json"
        with patch("rq1.tasks.freeze.git_state", return_value=(COMMIT, True, None)):
            self.manifest = freeze_manifest(root, proposal, {**APPROVED, "subject": {"manifest_sha256": proposal.manifest_sha256}}, manifest_path)
        run_id = "prelaunch-acquisition-check-parent"
        base = root / "artifacts" / "prelaunch" / "acquisition-check"
        plan = AcquisitionRunner(root).plan_from_manifest(self.manifest, run_id)
        world = {plan.task_ids[0]: {"success": True}, plan.task_ids[1]: {"success": True}}
        self.store = ExperimentStore(root, run_id, base=base)
        executor = RealAcquisitionExecutor(root, self.store, FakeDriver(world, [GOOD_SKILL, HEAT_SKILL]), scientific=False, queue_sha256=plan.queue_sha256, action_budget=3)
        AcquisitionRunner(root).run_resumable(
            plan, executor, configuration=launch.run_configuration(root, queue_sha256=str(plan.queue_sha256), scientific=False),
            options=RunnerOptions(fail_fast=True), output_base=base, initial_library_hash=EMPTY_POOL_HASH,
            scientific=False, store=self.store, preflight=executor.preflight, checkpoint_extension=executor.checkpoint_state, progress=None,
        )
        self.pool = rebuild_pool(result_rows(self.store))
        closeout_path = root / "artifacts" / "acquisition-closeout" / run_id / "closeout-manifest.json"
        closeout_path.parent.mkdir(parents=True)
        closeout_path.write_text(json.dumps({
            "closeout_passed": True,
            "run_id": run_id,
            "frozen_git_sha": COMMIT,
            "queue_hash": queue_identity_sha256(self.manifest),
            "final_pool_hash": pool_hash(self.pool),
            "counts": {"total": 6, "successful": 2, "final_skill_count": len(self.pool)},
            "authoritative_artifacts": {
                name: {"sha256": sha256_file(self.store.directory / name)} for name in extension.CLOSEOUT_AUTHORITATIVE_FILES
            },
        }), encoding="utf-8")
        self.reference = ParentReference(
            run_id=run_id,
            results_directory=(base / run_id).relative_to(root).as_posix(),
            completed_units=6,
            successful_units=2,
            repository_commit=COMMIT,
            queue_sha256=queue_identity_sha256(self.manifest),
            task_manifest=manifest_path.relative_to(root).as_posix(),
            task_manifest_sha256=self.manifest.manifest_sha256,
            pool_size=len(self.pool),
            pool_hash=pool_hash(self.pool),
            closeout_manifest=closeout_path.relative_to(root).as_posix(),
            closeout_manifest_sha256=sha256_file(closeout_path),
        )
        self.task_ids = {task.task_id for task in self.manifest.tasks}


class ExtensionProtocolTests(unittest.TestCase):
    def test_extension_inherits_identical_settings_and_the_certified_parent(self) -> None:
        definition = extension_protocol_definition()
        # The inherited acquisition protocol is byte-for-byte the one the parent ran.
        self.assertEqual("7e76ea460584b423684a1d406749ec3cebefa398f1ad4fcc27490450e5abc84d", protocol_sha256())
        self.assertEqual((protocol_definition(), protocol_sha256()), (definition["inherited_acquisition_protocol"], definition["inherited_acquisition_protocol_sha256"]))
        self.assertFalse(definition["scientific_settings_changed"])
        self.assertEqual(definition, load_json_yaml(REPO / EXTENSION_PROTOCOL_CONFIG))
        self.assertEqual(
            (60, 10, {"first": 181, "last": 240}, 240, 40),
            (definition["extension_task_count"], definition["extension_tasks_per_family"], definition["logical_acquisition_positions"],
             definition["combined_task_count_after_completion"], definition["combined_tasks_per_family_after_completion"]),
        )
        activation = definition["activation"]
        self.assertEqual(
            ({"total": 240, "per_family": 40}, False, False, False, False),
            (activation["hard_cap"], activation["automatic_extension"], activation["final_evaluation_started"],
             activation["evaluation_outcomes_exist"], activation["parent_results_discarded_or_rerun"]),
        )
        inherited = definition["inherited_acquisition_protocol"]
        self.assertEqual(({"total": 240, "per_family": 40}, False, 180), (inherited["later_hard_cap"], inherited["automatic_extension"], inherited["initial_task_count"]))
        self.assertEqual({"version": "task-selection-v1", "seed": 1, "requested_count": 240, "balancing": "round_robin_families", "selected_positions": {"first": 181, "last": 240}},
                         {key: definition["task_selection"][key] for key in ("version", "seed", "requested_count", "balancing", "selected_positions")})
        inference = definition["inference"]
        self.assertEqual(
            ("gemma4:12b", "4eb23ef187e2c5462566d6a1d3bbbc2f1346d0b4327cbb66d58fffbcc9b2b05c", "Q4_K_M", 50, 0, 42, False, 2048, 32768, "action-index-history-v3", 3),
            (definition["model"], definition["model_digest"], definition["model_quantization"], definition["acquisition_action_budget"],
             inference["temperature"], inference["seed"], inference["think"], inference["output_token_cap"], inference["model_context_length"],
             inference["action_selection_protocol"], inference["max_selection_attempts"]),
        )
        self.assertEqual((inherited["inference"]["initial_observation"], inherited["inference"]["inventory"]), (inference["initial_observation"], inference["inventory"]))
        self.assertEqual({"temperature": 0, "seed": 42, "num_predict": 2048, "num_ctx": 32768}, provider_settings()["options"])
        self.assertFalse(definition["scientific_retrieval_during_acquisition"])
        continuation = definition["continuation"]
        self.assertEqual((35, "create_only", 1), (continuation["first_new_pool_index"], continuation["skill_mode"], continuation["max_candidates_per_successful_episode"]))
        self.assertFalse(any(continuation[key] for key in (
            "parent_skills_modified", "parent_skills_deleted", "semantic_deduplication", "retrospective_deduplication", "provenance_rewritten", "patching",
        )))
        self.assertEqual(
            ("rq1-acquisition-gemma4-12b", 180, 86, 34, "11579cfe7ae0b232f233a9527fe96bae5985b50416b7408f2be89af2a4031341",
             "1f401869ea877969072f4d73e314db5841ff3f9aae875057ec3e4b5fe468ee09", "8bd452e76120da21d721c6e894d1ce5af4912ca9",
             "98c54bb3d33816da4771d7185a23a7cf2b739e550d87f59998214bf454d629de"),
            (PARENT.run_id, PARENT.completed_units, PARENT.successful_units, PARENT.pool_size, PARENT.pool_hash,
             PARENT.queue_sha256, PARENT.repository_commit, PARENT.closeout_manifest_sha256),
        )
        self.assertEqual(("rq1-acquisition-gemma4-12b-ext-181-240", FROZEN_MODEL_DIGEST), (EXTENSION_RUN_ID, definition["model_digest"]))
        record = " ".join((REPO / EXTENSION_DECISION_RECORD).read_text(encoding="utf-8").lower().split())
        for phrase in ("2026-09-14", "later_hard_cap", "automatic_extension: false", "no evaluation outcomes exist", "nothing is discarded or rerun",
                       "181–240", "append-only", "34 accepted skills", "checkpoint.backup.json", "non-authoritative",
                       "no quota, quality rubric, or skill rule is relaxed"):
            self.assertIn(phrase, record)
        self.assertEqual((180, 240, 40), tuple(load_json_yaml(REPO / "configs" / "tasks" / "acquisition.yaml")[key] for key in ("requested_count", "hard_cap", "hard_cap_tasks_per_family")))

    def test_cli_exposes_extension_commands_and_blocks_unsafe_runs(self) -> None:
        parser = build_parser()
        durable = ["--run-id", EXTENSION_RUN_ID, "--yes", "--backup-dir", "/backup", "--require-backup"]
        for argv in (
            ["acquisition-extension", "propose"],
            ["acquisition-extension", "plan"],
            ["acquisition-extension", "check", "--run-id", "prelaunch-acquisition-extension-check-x", "--task-id", "train:a", "--max-runs", "1"],
            ["acquisition-extension", "check", "--run-id", "prelaunch-acquisition-extension-check-x", "--resume"],
            ["acquisition-extension", "check-report", "--run-id", "prelaunch-acquisition-extension-check-x"],
            ["acquisition-extension", "prepare-approvals", "--proposal", "proposal.json", "--evidence-report", "report.json"],
            ["acquisition-extension", "freeze-tasks", "--proposal", "proposal.json", "--approval-file", "approval.json", "--yes"],
            ["acquisition-extension", "preflight", "--backup-dir", "/backup"],
            ["acquisition-extension", "run", *durable],
            ["acquisition-extension", "resume", *durable],
            ["acquisition-extension", "retry-failed", *durable],
            ["acquisition-extension", "validate", "--run-id", EXTENSION_RUN_ID],
            ["freeze", "acquisition-extension-environment", "--approval-file", "approval.json", "--pilot-report", "report.json", "--yes"],
            ["freeze", "acquisition-extension-protocol", "--approval-file", "approval.json", "--pilot-report", "report.json", "--yes"],
        ):
            parser.parse_args(argv)
        self.assertEqual(
            "python -m rq1.cli acquisition-extension run --run-id rq1-acquisition-gemma4-12b-ext-181-240 --yes --backup-dir /workspace/persistent/backups --require-backup",
            extension_preflight.extension_commands()["run"],
        )
        self.assertEqual(50, RealAcquisitionExecutor(Path("."), None, None, scientific=False, queue_sha256=None).action_budget)
        with tempfile.TemporaryDirectory() as temp:
            for run_id in (PARENT.run_id, "prelaunch-acquisition-check-x", EXTENSION_RUN_ID):
                args = argparse.Namespace(yes=True, run_id=run_id, task_manifest=None, max_runs=None, backup_dir=None, require_backup=False)
                blocked = extension_launch.extension_run(Path(temp), args, resume=False, retry_failed=False)
                self.assertEqual((False, "blocked"), (blocked["ok"], blocked["status"]))
            self.assertFalse((Path(temp) / "results").exists())


class ExtensionQueueTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.parent = SyntheticParent(self.root)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def propose(self):
        return extension.propose_extension_manifest(
            self.parent.discovery, self.parent.manifest, self.parent.reference, alfworld_version="0.4.2", repository_commit=COMMIT,
        )

    def test_queue_is_the_balanced_continuation_of_the_frozen_selection(self) -> None:
        reference, parent = self.parent.reference, self.parent.manifest
        manifest = self.propose()
        self.assertEqual([], extension.validate_extension_queue_manifest(manifest, parent, reference, require_frozen=False))
        self.assertEqual([], extension.continuation_problems(self.parent.discovery, parent, manifest))
        self.assertEqual((60, {family: 10 for family in TASK_FAMILIES}), (manifest.actual_count, dict(manifest.family_counts)))
        self.assertEqual(list(range(1, 61)), [task.order_index for task in manifest.tasks])
        identifiers = [task.task_id for task in manifest.tasks]
        self.assertEqual((60, set()), (len(set(identifiers)), set(identifiers) & self.parent.task_ids))
        self.assertTrue(all(task.split == "train" and task.task_id.startswith("train:") for task in manifest.tasks))
        full = select_tasks(self.parent.discovery, SelectionPolicy("task-selection-v1", 1, 66))[0]
        self.assertEqual([task.task_id for task in sorted(parent.tasks, key=by_order)], [task.task_id for task in full[:6]])
        self.assertEqual([task.task_id for task in full[6:]], identifiers)
        lineage = manifest.lineage
        self.assertEqual(
            (reference.run_id, reference.queue_sha256, reference.pool_hash, 6, (7, 66), {"overlap_with_parent_queue": [], "internal_duplicate_task_ids": []}),
            (lineage["parent_run_id"], lineage["parent_queue_sha256"], lineage["starting_pool_hash"], lineage["logical_index_offset"],
             (lineage["units"][0]["logical_acquisition_index"], lineage["units"][-1]["logical_acquisition_index"]), lineage["proof"]),
        )
        self.assertEqual(queue_identity_sha256(manifest), queue_identity_sha256(self.propose()))

        # Manifests without lineage keep their exact content and queue hashes.
        self.assertNotIn("lineage", parent.to_dict())
        self.assertEqual([], validate_manifest(parent, require_frozen=True))
        self.assertEqual(
            canonical_hash({
                "manifest_type": parent.manifest_type, "split": parent.split, "data_root_identity": parent.data_root_identity,
                "selection_policy": dict(parent.selection_policy), "tasks": [task.to_dict() for task in sorted(parent.tasks, key=by_order)],
            }),
            queue_identity_sha256(parent),
        )

        # Tampering fails closed.
        first_parent = sorted(parent.tasks, key=by_order)[0]
        overlapping = replace(manifest, tasks=(replace(manifest.tasks[0], task_id=first_parent.task_id), *manifest.tasks[1:]))
        self.assertIn("extension queue overlaps the parent acquisition queue", extension.validate_extension_queue_manifest(overlapping, parent, reference, require_frozen=False))
        self.assertIn("extension queue lineage differs from the parent reference", extension.validate_extension_queue_manifest(manifest, parent, replace(reference, pool_hash="0" * 64), require_frozen=False))
        reordered = replace(manifest, tasks=(replace(manifest.tasks[1], order_index=1), replace(manifest.tasks[0], order_index=2), *manifest.tasks[2:]))
        self.assertIn("extension queue is not the continuation of the frozen selection", extension.continuation_problems(self.parent.discovery, parent, reordered))
        extra = self.parent.data / "json_2.1.1" / "train" / "pick_and_place_simple-Extra-None-Target-99" / "trial_T0"
        extra.mkdir(parents=True)
        (extra / "traj_data.json").write_text(json.dumps({"task_type": "pick_and_place_simple"}), encoding="utf-8")
        (extra / "game.tw-pddl").write_text(json.dumps({"game": "extra"}), encoding="utf-8")
        self.assertIn("ALFWorld TRAIN data identity differs from the parent frozen queue", extension.continuation_problems(discover_tasks(self.parent.data, "train"), parent, manifest))

        # Freezing keeps the queue identity and yields a continuation plan.
        with patch("rq1.tasks.freeze.git_state", return_value=(COMMIT, True, None)):
            frozen = freeze_manifest(self.root, manifest, {**APPROVED, "subject": {"manifest_sha256": manifest.manifest_sha256}}, self.root / EXTENSION_FROZEN_DIR / "acquisition-extension-test.json")
        self.assertEqual((queue_identity_sha256(manifest), manifest.lineage), (queue_identity_sha256(frozen), frozen.lineage))
        self.assertEqual([], extension.validate_extension_queue_manifest(frozen, parent, reference, require_frozen=True))
        plan = AcquisitionRunner(self.root).plan_from_manifest(frozen, EXTENSION_RUN_ID)
        self.assertEqual((reference.run_id, 6, 60), (plan.parent_run_id, plan.logical_index_offset, len(plan.task_ids)))
        units = acquisition_units(plan, initial_library_hash=reference.pool_hash, initial_library_size=reference.pool_size)
        self.assertEqual(
            (7, 66, reference.run_id, reference.pool_size, reference.pool_hash),
            (units[0].payload["logical_acquisition_index"], units[-1].identity["logical_acquisition_index"], units[0].identity["parent_run_id"], units[0].library_size, units[0].library_hash),
        )
        with self.assertRaises(AcquisitionError):
            AcquisitionRunner(self.root).plan_from_manifest(replace(frozen, lineage=None), EXTENSION_RUN_ID)
        with self.assertRaises(AcquisitionError):
            AcquisitionRunner(self.root).plan_from_manifest(replace(parent, lineage={"parent_run_id": "x", "logical_index_offset": 1}), "run")


class ExtensionExecutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.parent = SyntheticParent(self.root)
        self.reference = self.parent.reference
        self.base = self.root / "artifacts" / "prelaunch" / "acquisition-extension-check"

    def tearDown(self) -> None:
        self.temp.cleanup()

    def run_extension(self, driver, *, parent_pool=None, **kwargs):
        return run_units(
            self.root, driver, EXTENSION_TASKS, run_id="prelaunch-acquisition-extension-check-test", base=self.base,
            parent_pool=self.parent.pool if parent_pool is None else parent_pool, parent_run_id=self.reference.run_id,
            offset=self.reference.completed_units, parent=self.reference, **kwargs,
        )

    def test_extension_starts_from_parent_pool_appends_only_and_resumes_without_replay(self) -> None:
        parent_files = {name: (self.parent.store.directory / name).read_bytes() for name in ("results.jsonl", "checkpoint.json", "checkpoint.backup.json", "skill_pool.json")}
        world = {task: {"success": True} for task, _ in EXTENSION_TASKS}
        store, first = self.run_extension(FakeDriver(world, [GOOD_SKILL]), max_runs=1)
        self.assertEqual("paused", first["status"])
        row = result_rows(store)[0]
        self.assertEqual((2, self.reference.pool_hash), (row["skill_pool_size_before"], row["skill_pool_hash_before"]))
        self.assertEqual((1, 7, self.reference.run_id), (row["task_index"], row["logical_acquisition_index"], row["parent_run_id"]))
        # A candidate identical to a parent skill is an exact duplicate.
        self.assertEqual(("rejected", ["exact_normalized_duplicate"]), (row["skill_candidate"]["status"], row["skill_candidate"]["rejection_reasons"]))

        second_driver = FakeDriver(world, [COOL_SKILL, NEAR_SKILL])
        _, second = self.run_extension(second_driver, resume=True)
        self.assertEqual(("completed", 2), (second["status"], len(second_driver.sessions)))
        rows = result_rows(store)
        self.assertEqual(([1, 2, 3], [7, 8, 9]), ([item["task_index"] for item in rows], [item["logical_acquisition_index"] for item in rows]))
        pool = rebuild_pool(rows, base=self.parent.pool)
        self.assertEqual([skill.identity() for skill in self.parent.pool], [skill.identity() for skill in pool[:2]])
        appended = pool[2:]
        self.assertEqual([3, 4], [skill.pool_index for skill in appended])
        self.assertEqual(
            [(self.reference.run_id, 8, 2), (self.reference.run_id, 9, 3)],
            [(skill.provenance["parent_run_id"], skill.provenance["logical_acquisition_index"], skill.source_task_index) for skill in appended],
        )
        with self.assertRaises(SkillPoolError):
            rebuild_pool(rows)

        third_driver = FakeDriver(world, [])
        _, third = self.run_extension(third_driver, resume=True)
        self.assertEqual(("completed", 0, []), (third["status"], third["attempted_this_invocation"], third_driver.sessions))
        self.assertEqual(3, len(store.read_results()))

        state = store.load_checkpoint()[0]["phase_state"]
        self.assertEqual(
            {"parent_run_id": self.reference.run_id, "starting_pool_size": 2, "starting_pool_hash": self.reference.pool_hash, "appended_skills": 2,
             "pool_per_family": {family: sum(skill.task_family == family for skill in pool) for family in TASK_FAMILIES}},
            state["continuation"],
        )
        self.assertEqual((4, pool_hash(pool), 2), (state["skill_pool"]["size"], state["skill_pool"]["hash"], state["per_family"]["heat_and_place"]["accepted_skills"]))
        self.assertTrue(all(item["scientific_retrieval_count"] == 0 and item["retrieved_skill_ids"] == [] for item in rows))
        configuration = json.loads((store.manifests / "acquisition.json").read_text(encoding="utf-8"))["configuration"]
        runtime = configuration["runtime_settings"]
        self.assertEqual(
            ("gemma4:12b", 2048, 50, 32768, 0, 42, "action-index-history-v3", 3),
            (configuration["model_name"], runtime["output_token_cap"], runtime["acquisition_action_budget"], runtime["model_context_length"],
             runtime["temperature"], runtime["seed"], runtime["action_selection_protocol"], runtime["max_selection_attempts"]),
        )
        self.assertEqual((self.reference.run_id, self.reference.pool_hash, 6), tuple(configuration["extension"][key] for key in ("parent_run_id", "starting_pool_hash", "logical_index_offset")))
        self.assertEqual(parent_files, {name: (self.parent.store.directory / name).read_bytes() for name in parent_files})
        self.assertFalse(store.directory.resolve().is_relative_to(self.parent.store.directory.resolve()))
        self.assertEqual([], extension.load_parent(self.root, self.reference).problems)
        plain = AcquisitionPlan("plain", (EXTENSION_TASKS[0][0],), task_families=("heat_and_place",))
        self.assertNotEqual(acquisition_units(plain)[0].run_key, row["run_key"])

    def test_extension_fails_closed_on_a_different_starting_pool_or_missing_lineage(self) -> None:
        world = {task: {"success": True} for task, _ in EXTENSION_TASKS}
        store, _ = self.run_extension(FakeDriver(world, [COOL_SKILL]), max_runs=1)
        driver = FakeDriver(world, [NEAR_SKILL])
        shorter = self.parent.pool[:1]
        with self.assertRaises(CompatibilityError):
            self.run_extension(driver, parent_pool=shorter, resume=True)
        with self.assertRaises(SkillPoolError):
            self.run_extension(driver, parent_pool=shorter, initial_pool=self.parent.pool, resume=True)
        self.assertEqual([], driver.sessions)
        with self.assertRaises(ValueError):
            RealAcquisitionExecutor(self.root, store, driver, scientific=False, queue_sha256=QUEUE, parent_pool=self.parent.pool)
        other = ExperimentStore(self.root, "prelaunch-acquisition-extension-check-nolineage", base=self.base)
        executor = RealAcquisitionExecutor(self.root, other, driver, scientific=False, queue_sha256=QUEUE, action_budget=3, parent_pool=self.parent.pool, parent_run_id=self.reference.run_id)
        plan = AcquisitionPlan(other.experiment_id, (EXTENSION_TASKS[0][0],), task_families=("heat_and_place",), queue_sha256=QUEUE)
        with self.assertRaises(SkillPoolError):
            AcquisitionRunner(self.root).run_resumable(
                plan, executor, configuration=launch.run_configuration(self.root, queue_sha256=QUEUE, scientific=False),
                options=RunnerOptions(fail_fast=True), output_base=self.base, initial_library_hash=self.reference.pool_hash,
                initial_library_size=self.reference.pool_size, scientific=False, store=other,
                preflight=executor.preflight, checkpoint_extension=executor.checkpoint_state, progress=None,
            )
        self.assertEqual([], driver.sessions)

    def test_stale_checkpoint_backup_is_non_authoritative_and_never_replays(self) -> None:
        tasks = EXTENSION_TASKS[:2]
        world = {task: {"success": False} for task, _ in tasks}
        base = self.root / "artifacts" / "prelaunch" / "acquisition-check"
        store, result = run_units(self.root, FakeDriver(world), tasks, run_id="prelaunch-acquisition-check-backup", base=base)
        self.assertEqual("completed", result["status"])
        primary = json.loads(store.checkpoint_path.read_text(encoding="utf-8"))
        backup = json.loads(store.backup_checkpoint_path.read_text(encoding="utf-8"))
        # The rotating backup is the previous generation: every unit completed, status still "running".
        self.assertEqual((("completed", 2), ("running", 2)), ((primary["status"], primary["completed_run_count"]), (backup["status"], backup["completed_run_count"])))
        differing = {key for key in set(primary) | set(backup) if primary.get(key) != backup.get(key)}
        self.assertIn("status", differing)
        self.assertTrue(differing <= {"status", "timestamp"})
        results_bytes, backup_bytes = store.results_path.read_bytes(), store.backup_checkpoint_path.read_bytes()
        store.checkpoint_path.write_text("{", encoding="utf-8")
        checkpoint, source = store.load_checkpoint()
        self.assertEqual(("backup", "running"), (source, checkpoint["status"]))
        driver = FakeDriver(world)
        _, resumed = run_units(self.root, driver, tasks, run_id="prelaunch-acquisition-check-backup", base=base, resume=True)
        self.assertEqual(("completed", 0, []), (resumed["status"], resumed["attempted_this_invocation"], driver.sessions))
        self.assertEqual((results_bytes, backup_bytes), (store.results_path.read_bytes(), store.backup_checkpoint_path.read_bytes()))
        checkpoint, source = store.load_checkpoint()
        self.assertEqual(("primary", "completed", 2), (source, checkpoint["status"], checkpoint["completed_run_count"]))


class ExtensionGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.parent = SyntheticParent(self.root)
        self.reference = self.parent.reference

    def tearDown(self) -> None:
        self.temp.cleanup()

    def gate(self, commit=COMMIT, clean=True, parent=None):
        with patch("rq1.acquisition.extension.git_state", return_value=(commit, clean, None)):
            return extension.validate_extension_gates(self.root, parent=parent or self.reference)

    def test_parent_verification_is_read_only_and_fails_closed(self) -> None:
        state = extension.load_parent(self.root, self.reference)
        self.assertEqual(([], 2, 6), (state.problems, len(state.pool), len(state.records)))
        self.assertIn("parent final skill pool hash differs from the recorded hash", extension.load_parent(self.root, replace(self.reference, pool_hash="0" * 64)).problems)
        self.assertIn("parent run does not have exactly the recorded completed and successful units", extension.load_parent(self.root, replace(self.reference, completed_units=5)).problems)
        self.assertIn(
            "parent closeout manifest is missing, differs from its recorded SHA-256, or does not certify the parent",
            extension.load_parent(self.root, replace(self.reference, closeout_manifest_sha256="0" * 64)).problems,
        )
        snapshot = self.parent.store.directory / "skill_pool.json"
        original = snapshot.read_bytes()
        snapshot.write_bytes(original + b" ")
        self.assertIn("parent authoritative files changed since closeout", extension.load_parent(self.root, self.reference).problems)
        snapshot.write_bytes(original)
        path = extension.ensure_starting_pool(self.root, state, self.reference)
        self.assertEqual(path, extension.ensure_starting_pool(self.root, state, self.reference))
        recorded = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(
            (self.reference.run_id, 2, self.reference.pool_hash, self.reference.closeout_manifest_sha256, [skill.to_dict() for skill in state.pool]),
            (recorded["source_run_id"], recorded["pool_size"], recorded["pool_hash"], recorded["source_closeout_manifest"]["sha256"], recorded["skills"]),
        )
        path.write_text(json.dumps({**recorded, "pool_size": 3}), encoding="utf-8")
        self.assertEqual(["extension starting-pool snapshot differs from the verified parent final pool"], extension.starting_pool_problems(self.root, state, self.reference))

    def test_extension_gate_blocks_until_approved_freezes_and_permits_the_scientific_continuation(self) -> None:
        reference, root = self.reference, self.root
        state = extension.load_parent(root, reference)
        pool_path = extension.ensure_starting_pool(root, state, reference)
        manifest = extension.propose_extension_manifest(self.parent.discovery, self.parent.manifest, reference, alfworld_version="0.4.2", repository_commit=COMMIT)
        self.assertEqual(set(extension.EXTENSION_APPROVAL_PENDING_REASONS), set(self.gate().reasons))

        with patch("rq1.tasks.freeze.git_state", return_value=(COMMIT, True, None)):
            frozen = freeze_manifest(root, manifest, {**APPROVED, "subject": {"manifest_sha256": manifest.manifest_sha256}}, root / EXTENSION_FROZEN_DIR / f"acquisition-extension-{manifest.manifest_sha256[:16]}.json")
        queue = queue_identity_sha256(frozen)
        prompts = prompt_hashes(root)
        environment_inputs = {key: "recorded" for key in ACQUISITION_EXTENSION_ENVIRONMENT_REQUIRED}
        environment_inputs.update({
            "repository_commit": COMMIT, "model_tag": "gemma4:12b", "model_quantization": "Q4_K_M", "model_digest": FROZEN_MODEL_DIGEST,
            "provider_settings": provider_settings(), "inference_seed": INFERENCE_SEED, "task_queue_sha256": queue, "prompt_hashes": prompts,
            "alfworld_data_identity": frozen.data_root_identity, "parent_run_id": reference.run_id,
        })
        protocol_inputs = {
            "repository_commit": COMMIT, "protocol": extension_protocol_definition(reference), "protocol_sha256": extension_protocol_sha256(reference),
            "inherited_protocol_sha256": protocol_sha256(), "task_queue_sha256": queue, "acquisition_action_budget": 50, "inference_seed": INFERENCE_SEED,
            "prompt_hashes": prompts, "decision_record_sha256": "d", "parent_run_id": reference.run_id, "parent_pool_size": reference.pool_size,
            "parent_pool_hash": reference.pool_hash, "parent_queue_sha256": reference.queue_sha256,
            "parent_closeout_manifest_sha256": reference.closeout_manifest_sha256, "starting_pool_sha256": sha256_file(pool_path),
        }
        evidence = {"mode": ACQUISITION_EXTENSION_EVIDENCE_MODE, "passed": True, "scientific_evidence": False, "repository_commit": COMMIT, "run_id": "prelaunch-acquisition-extension-check-test"}
        with patch("rq1.freeze.validation.git_state", return_value=(COMMIT, True, None)):
            for kind, inputs in (("acquisition-extension-environment", environment_inputs), ("acquisition-extension-protocol", protocol_inputs)):
                unapproved = {"approval_kind": kind, "approval": {"status": "UNAPPROVED", "approved_by": None, "approved_at": None}, "inputs": inputs}
                with self.assertRaises(ValueError):
                    build_freeze(root, kind, unapproved, evidence)
                with self.assertRaises(ValueError):
                    build_freeze(root, kind, {**unapproved, "approval": APPROVED}, {**evidence, "mode": "non_scientific_acquisition_check"})
                path = write_freeze(root, build_freeze(root, kind, {**unapproved, "approval": APPROVED}, evidence))
                self.assertEqual(f"{kind}-freeze.json", path.name)
        self.assertFalse((root / "artifacts" / "freezes" / "acquisition-environment-freeze.json").exists())
        self.assertEqual((True, ()), (self.gate().valid, self.gate().reasons))
        self.assertEqual(1, len(list((root / FROZEN_TASK_DIR).glob("acquisition-*.json"))))
        self.assertFalse(self.gate(commit="d" * 40).valid)
        self.assertFalse(self.gate(clean=False).valid)
        self.assertFalse(self.gate(parent=replace(reference, pool_hash="0" * 64)).valid)

        run_id = "rq1-acquisition-extension-test"
        store = ExperimentStore(root, run_id)
        plan = AcquisitionRunner(root).plan_from_manifest(frozen, run_id)
        world = {task: {"success": True} for task in plan.task_ids[:2]}
        driver = FakeDriver(world, [COOL_SKILL, NEAR_SKILL])
        executor = RealAcquisitionExecutor(root, store, driver, scientific=True, queue_sha256=plan.queue_sha256, action_budget=3, parent_pool=state.pool, parent_run_id=reference.run_id)
        options = dict(
            configuration=extension.extension_run_configuration(root, queue_sha256=str(plan.queue_sha256), scientific=True, parent=reference),
            options=RunnerOptions(max_runs=2, fail_fast=True), initial_library_hash=reference.pool_hash, initial_library_size=reference.pool_size,
            scientific=True, store=store, preflight=executor.preflight, checkpoint_extension=executor.checkpoint_state, progress=None,
        )
        with patch("rq1.acquisition.gates.git_state", return_value=(COMMIT, True, None)), self.assertRaises(AcquisitionError):
            AcquisitionRunner(root).run_resumable(plan, executor, **options)
        self.assertEqual([], driver.sessions)

        def extension_gate(gate_root, task_manifest_path=None):
            return extension.validate_extension_gates(gate_root, task_manifest_path=task_manifest_path, parent=reference)

        with patch("rq1.acquisition.extension.git_state", return_value=(COMMIT, True, None)):
            result = AcquisitionRunner(root).run_resumable(plan, executor, gate=extension_gate, **options)
        self.assertEqual(("paused", root / "results" / "final" / run_id), (result["status"], store.directory))
        rows = result_rows(store)
        self.assertEqual([(True, reference.run_id, 7), (True, reference.run_id, 8)], [(item["scientific_evidence"], item["parent_run_id"], item["logical_acquisition_index"]) for item in rows])
        self.assertEqual((2, 4), (rows[0]["skill_pool_size_before"], len(rebuild_pool(rows, base=state.pool))))

        # The hard cap counts every scientific acquisition run: a second run of the same
        # queue is refused before any episode, while the existing run may still resume.
        second = argparse.Namespace(yes=True, run_id="rq1-acquisition-extension-second", task_manifest=None, max_runs=None, backup_dir=None, require_backup=False)
        with patch("rq1.acquisition.extension.git_state", return_value=(COMMIT, True, None)), patch.object(launch, "ACQUISITION_HARD_CAP", 60):
            blocked = extension_launch.extension_run(root, second, resume=False, retry_failed=False, parent=reference)
            new_run_plan = extension_launch.extension_plan(root, argparse.Namespace(task_manifest=None), parent=reference)
            resume_plan = extension_launch.extension_plan(root, argparse.Namespace(task_manifest=None, run_id=run_id), parent=reference)
        self.assertEqual(("blocked", False, 60), (blocked["status"], blocked["hard_cap"]["permitted"], blocked["hard_cap"]["allocated_units_other_runs"]))
        self.assertFalse((root / "results" / "final" / "rq1-acquisition-extension-second").exists())
        self.assertEqual((False, True), (new_run_plan["launch_permitted"], resume_plan["launch_permitted"]))


def write_phase_manifest(root: Path, run_id: str, families: dict, *, scientific: bool = True) -> None:
    directory = root / "results" / "final" / run_id / "manifests"
    directory.mkdir(parents=True)
    planned = [{"payload": {"task_family": family}} for family, count in families.items() for _ in range(count)]
    payload = {"configuration": {"scientific_evidence": scientific}, "planned_runs": planned, "planned_run_count": len(planned)}
    (directory / "acquisition.json").write_text(json.dumps(payload), encoding="utf-8")


class HardCapTests(unittest.TestCase):
    def test_no_scientific_acquisition_beyond_240_units_or_40_per_family(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            queue_180 = [family for family in TASK_FAMILIES for _ in range(30)]
            queue_60 = [family for family in TASK_FAMILIES for _ in range(10)]
            self.assertTrue(launch.hard_cap_status(root, PARENT.run_id, queue_180)["permitted"])
            write_phase_manifest(root, PARENT.run_id, {family: 30 for family in TASK_FAMILIES})
            self.assertTrue(launch.hard_cap_status(root, EXTENSION_RUN_ID, queue_60)["permitted"])
            write_phase_manifest(root, EXTENSION_RUN_ID, {family: 10 for family in TASK_FAMILIES})
            write_phase_manifest(root, "non-scientific-check", {family: 10 for family in TASK_FAMILIES}, scientific=False)
            # Resuming either completed run adds no unit.
            for run_id, queue in ((PARENT.run_id, queue_180), (EXTENSION_RUN_ID, queue_60)):
                status = launch.hard_cap_status(root, run_id, queue)
                self.assertEqual((True, 240 - len(queue)), (status["permitted"], status["allocated_units_other_runs"]))
            for run_id, queue in (("rq1-acquisition-gemma4-12b-ext-second", queue_60), ("rq1-acquisition-new", queue_180), ("rq1-acquisition-episode-241", ["cool_and_place"])):
                status = launch.hard_cap_status(root, run_id, queue)
                self.assertFalse(status["permitted"])
                self.assertTrue(any("hard cap of 240" in problem for problem in status["problems"]))
                self.assertTrue(any("40 tasks per family" in problem for problem in status["problems"]))
            broken = root / "results" / "final" / "unreadable" / "manifests"
            broken.mkdir(parents=True)
            (broken / "acquisition.json").write_text("{", encoding="utf-8")
            self.assertFalse(launch.hard_cap_status(root, EXTENSION_RUN_ID, queue_60)["permitted"])


if __name__ == "__main__":
    unittest.main()
