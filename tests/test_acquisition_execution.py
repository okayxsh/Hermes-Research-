"""Hermetic tests for the RQ1 acquisition execution path (Decision 007)."""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from rq1.acquisition import environment, launch
from rq1.acquisition.executor import RealAcquisitionExecutor
from rq1.acquisition.gates import queue_identity_sha256, validate_acquisition_gates, validate_queue_manifest
from rq1.acquisition.models import AcquisitionPlan
from rq1.acquisition.protocol import ACQUISITION_ACTION_BUDGET, DECISION_RECORD, PROTOCOL_CONFIG, protocol_definition, protocol_sha256
from rq1.acquisition.runner import AcquisitionError, AcquisitionRunner
from rq1.acquisition.skill_creation import parse_skill_response, validate_skill
from rq1.acquisition.skill_pool import EMPTY_POOL_HASH, SNAPSHOT_NAME, SkillPoolError, pool_hash, rebuild_pool
from rq1.cli import build_parser
from rq1.experiment import persistence
from rq1.experiment.persistence import CompatibilityError, ExperimentStore
from rq1.experiment.runner import RunnerOptions
from rq1.freeze.validation import ACQUISITION_ENVIRONMENT_REQUIRED, build_freeze, write_freeze
from rq1.hermes.episode_driver import INFERENCE_SEED, EpisodeDriverError, RealEpisodeSession
from rq1.pilot import prelaunch
from rq1.retrieval.text import build_skill_text
from rq1.skills.library import TASK_FAMILIES
from rq1.tasks.discovery import discover_tasks
from rq1.tasks.freeze import TaskFreezeError, freeze_manifest
from rq1.tasks.models import SelectionPolicy
from rq1.tasks.selection import propose_manifest
from rq1.utils.config import load_json_yaml

REPO = Path(__file__).resolve().parents[1]
COMMIT = "c" * 40
QUEUE = "q" * 64
NATIVE = (
    "pick_and_place_simple", "pick_two_obj_and_place", "look_at_obj_in_light",
    "pick_clean_then_place_in_recep", "pick_heat_then_place_in_recep", "pick_cool_then_place_in_recep",
)
GOOD_SKILL = "TITLE: Clean before placing\nBODY: Find the target object, rinse it at a sink basin, then put it in the requested receptacle."
NEAR_SKILL = "TITLE: Clean before placing\nBODY: Find the target object, rinse it at the sink basin, then put it in the requested receptacle."
LEAKY_SKILL = "TITLE: Cart routine\nBODY: go to cart 1 and put the cloth there."
TASKS = [(f"train:pick_clean_then_place_in_recep-Cloth-None-Cart-40{index}/trial_T{index}", "clean_and_place") for index in range(1, 6)]
APPROVED = {"status": "APPROVED", "approved_by": "test reviewer", "approved_at": "2026-09-13T00:00:00Z"}


class ScriptedWorker:
    """Registry-worker stand-in; ``finish task`` wins only for successful tasks."""

    ACTIONS = ["go to cart 1", "finish task", "look"]

    def __init__(self, world: dict) -> None:
        self.world = world
        self.task_id: str | None = None
        self.steps = 0

    def start(self) -> None:
        return None

    def close(self) -> None:
        return None

    def dispatch(self, tool, args, hook_kwargs):
        if tool == "alfworld_start":
            self.task_id = args["task_id"]
        spec = self.world.get(self.task_id, {})
        base = {
            "episode_id": "python-owned-episode", "task_id": self.task_id, "split": "train", "task_family": "clean_and_place",
            "instruction": self.task_id, "inventory": [], "admissible_actions": list(self.ACTIONS), "done": False, "success": False,
            "action_valid": True,
        }
        if tool == "alfworld_start":
            observation = "-= Welcome to TextWorld, ALFRED! =-\n\nYou are in the middle of a room.\n\nYour task is to: put a clean cloth in cart."
            return {"result": {**base, "observation": observation, "step_number": 0, "action_valid": None}}
        if tool == "alfworld_step":
            if spec.get("infrastructure_failure"):
                raise EpisodeDriverError("simulated bridge outage")
            self.steps += 1
            won = bool(spec.get("success")) and args["action"] == "finish task"
            return {"result": {**base, "observation": "You won." if won else "Nothing happens.", "step_number": self.steps, "done": won, "success": won}}
        if tool == "alfworld_abort":
            return {"result": {**base, "observation": "aborted", "admissible_actions": [], "done": True}}
        raise AssertionError(tool)


class FakeDriver:
    root = Path(".")
    bridge_url = "http://127.0.0.1:9"
    bridge_timeout_seconds = 5.0
    model_name = "hermes3:8b"
    ollama_url = "http://127.0.0.1:11434"
    model_timeout_seconds = 5.0
    inference_seed = INFERENCE_SEED

    def __init__(self, world: dict, skill_responses=()) -> None:
        self.world = world
        self.skill_responses = list(skill_responses)
        self.prompts: list[str] = []
        self.sessions: list[dict] = []

    def session(self, *, output_dir, run_id, attempt_id=None, profile="test"):
        session = RealEpisodeSession(self, output_dir=output_dir, run_id=run_id, attempt_id=attempt_id or "attempt", profile=profile)
        session.worker = ScriptedWorker(self.world)

        def model(prompt: str) -> str:
            self.prompts.append(prompt)
            if prompt.startswith("POST-SUCCESS LEARNING:"):
                return self.skill_responses.pop(0)
            return "ACTION_INDEX: 1"

        session._model_response = model
        self.sessions.append({"profile": profile, "output_dir": str(output_dir)})
        return session

    def skill_prompts(self) -> list[str]:
        return [prompt for prompt in self.prompts if prompt.startswith("POST-SUCCESS LEARNING:")]


def write_train_tasks(root: Path, per_family: int = 31) -> None:
    for native in NATIVE:
        for index in range(per_family):
            path = root / "json_2.1.1" / "train" / f"{native}-Object-None-Target-{index}" / "trial_T0"
            path.mkdir(parents=True)
            (path / "traj_data.json").write_text(json.dumps({"task_type": native}), encoding="utf-8")
            (path / "game.tw-pddl").write_text(json.dumps({"game": f"{native}-{index}"}), encoding="utf-8")


class ProtocolFreezeTests(unittest.TestCase):
    def test_pre_run_decision_values_are_frozen_in_code_config_and_record(self) -> None:
        definition = protocol_definition()
        self.assertEqual(50, ACQUISITION_ACTION_BUDGET)
        self.assertEqual(50, definition["acquisition_action_budget"])
        self.assertFalse(definition["action_budget_varies_by_task_or_family"])
        creation = definition["skill_creation"]
        self.assertEqual(
            ("same_experimental_agent", False, "create_only", False, 1, 0, 0),
            (creation["author"], creation["separate_summariser"], creation["mode"], creation["patching"],
             creation["max_candidates_per_successful_episode"], creation["failed_episode_candidates"],
             creation["infrastructure_failure_candidates"]),
        )
        duplicates = definition["duplicate_policy"]
        self.assertEqual("exact_normalized_duplicate_only", duplicates["acquisition_rejection"])
        self.assertFalse(any(duplicates[key] for key in ("semantic_deduplication", "embedding_deduplication", "llm_duplicate_judge", "retrospective_library_deduplication")))
        self.assertFalse(definition["scientific_retrieval_during_acquisition"])
        self.assertEqual(INFERENCE_SEED, definition["inference"]["seed"])
        self.assertEqual((180, 30, False), (definition["initial_task_count"], definition["tasks_per_family"], definition["automatic_extension"]))
        self.assertEqual(definition, load_json_yaml(REPO / PROTOCOL_CONFIG))
        self.assertTrue(load_json_yaml(REPO / "configs" / "libraries.yaml")["do_not_deduplicate"])
        record = (REPO / DECISION_RECORD).read_text(encoding="utf-8").lower()
        for phrase in ("2026-09-13", "50 alfworld", "before", "create-only", "exact normalized duplicate", "near-duplicates"):
            self.assertIn(phrase, record)
        self.assertNotIn("patch", (REPO / "hermes" / "prompts" / "post_success_learning.md").read_text(encoding="utf-8").lower().replace("never modify or patch", ""))
        self.assertEqual((prelaunch.SBERT_MODEL, prelaunch.SBERT_REVISION), (environment.SBERT_MODEL, environment.SBERT_REVISION))


class QueueAndGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.data = self.root / "data"
        write_train_tasks(self.data)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def propose(self, seed: int = 1, count: int = 180):
        return propose_manifest("acquisition", discover_tasks(self.data, "train"), SelectionPolicy("task-selection-v1", seed, count), alfworld_version="0.4.2", repository_commit=COMMIT)

    def test_queue_is_180_train_tasks_balanced_and_deterministic(self) -> None:
        first, second = self.propose(), self.propose()
        self.assertEqual([], validate_queue_manifest(first, require_frozen=False))
        self.assertEqual(queue_identity_sha256(first), queue_identity_sha256(second))
        self.assertEqual({family: 30 for family in TASK_FAMILIES}, dict(first.family_counts))
        self.assertTrue(all(task.split == "train" and task.task_id.startswith("train:") for task in first.tasks))
        self.assertEqual(list(range(1, 181)), [task.order_index for task in first.tasks])
        self.assertEqual(6, len({task.family for task in first.tasks[:6]}))
        other = self.propose(seed=2)
        self.assertNotEqual(queue_identity_sha256(first), queue_identity_sha256(other))
        self.assertIn("acquisition queue selection policy differs from the frozen protocol", validate_queue_manifest(other, require_frozen=False))
        self.assertIn("acquisition queue must contain TRAIN tasks only", validate_queue_manifest(replace(first, split="valid_seen"), require_frozen=False))
        self.assertIn("acquisition queue must contain exactly 180 tasks", validate_queue_manifest(self.propose(count=179), require_frozen=False))

    def test_queue_hash_survives_freezing_which_requires_bound_human_approval(self) -> None:
        manifest = self.propose()
        destination = self.root / "frozen.json"
        with patch("rq1.tasks.freeze.git_state", return_value=(COMMIT, True, None)):
            with self.assertRaises(TaskFreezeError):
                freeze_manifest(self.root, manifest, {"status": "UNAPPROVED", "approved_by": None, "approved_at": None}, destination)
            with self.assertRaises(TaskFreezeError):
                freeze_manifest(self.root, manifest, {**APPROVED, "subject": {"manifest_sha256": "other"}}, destination)
            frozen = freeze_manifest(self.root, manifest, {**APPROVED, "subject": {"manifest_sha256": manifest.manifest_sha256}}, destination)
        self.assertEqual([], validate_queue_manifest(frozen, require_frozen=True))
        self.assertEqual(queue_identity_sha256(manifest), queue_identity_sha256(frozen))
        self.assertNotEqual(manifest.manifest_sha256, frozen.manifest_sha256)

    def _approved_state(self) -> None:
        manifest = self.propose()
        with patch("rq1.tasks.freeze.git_state", return_value=(COMMIT, True, None)):
            frozen = freeze_manifest(self.root, manifest, APPROVED, self.root / "artifacts" / "task_manifests" / "frozen" / "acquisition-test.json")
        queue = queue_identity_sha256(frozen)
        prompts = {"hermes/prompts/post_success_learning.md": "a", "hermes/prompts/skill_validation.md": "b"}
        environment_inputs = {key: "recorded" for key in ACQUISITION_ENVIRONMENT_REQUIRED}
        environment_inputs.update({
            "repository_commit": COMMIT, "model_tag": "hermes3:8b", "inference_seed": INFERENCE_SEED,
            "task_queue_sha256": queue, "prompt_hashes": prompts, "alfworld_data_identity": frozen.data_root_identity,
        })
        protocol_inputs = {
            "repository_commit": COMMIT, "protocol": protocol_definition(), "protocol_sha256": protocol_sha256(),
            "task_queue_sha256": queue, "acquisition_action_budget": 50, "inference_seed": INFERENCE_SEED,
            "prompt_hashes": prompts, "decision_record_sha256": "d",
        }
        evidence = {"mode": "non_scientific_acquisition_check", "passed": True, "scientific_evidence": False, "repository_commit": COMMIT, "run_id": "prelaunch-acquisition-check-test"}
        with patch("rq1.freeze.validation.git_state", return_value=(COMMIT, True, None)):
            for kind, inputs in (("acquisition-environment", environment_inputs), ("acquisition-protocol", protocol_inputs)):
                with self.assertRaises(ValueError):
                    build_freeze(self.root, kind, {"approval_kind": kind, "approval": {"status": "UNAPPROVED", "approved_by": None, "approved_at": None}, "inputs": inputs}, evidence)
                with self.assertRaises(ValueError):
                    build_freeze(self.root, kind, {"approval_kind": kind, "approval": APPROVED, "inputs": inputs}, {**evidence, "passed": False})
                write_freeze(self.root, build_freeze(self.root, kind, {"approval_kind": kind, "approval": APPROVED, "inputs": inputs}, evidence))

    def test_unapproved_state_blocks_and_approved_state_permits_launch(self) -> None:
        with patch("rq1.acquisition.gates.git_state", return_value=(COMMIT, True, None)):
            blocked = validate_acquisition_gates(self.root)
        self.assertFalse(blocked.valid)
        self.assertTrue(any("frozen acquisition task manifest" in reason for reason in blocked.reasons))
        self._approved_state()
        with patch("rq1.acquisition.gates.git_state", return_value=(COMMIT, True, None)):
            self.assertEqual((True, ()), (validate_acquisition_gates(self.root).valid, validate_acquisition_gates(self.root).reasons))
        with patch("rq1.acquisition.gates.git_state", return_value=("d" * 40, True, None)):
            self.assertFalse(validate_acquisition_gates(self.root).valid)
        with patch("rq1.acquisition.gates.git_state", return_value=(COMMIT, False, None)):
            self.assertFalse(validate_acquisition_gates(self.root).valid)


class AcquisitionExecutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        prompts = self.root / "hermes" / "prompts"
        prompts.mkdir(parents=True)
        for name in ("post_success_learning.md", "skill_validation.md"):
            shutil.copy(REPO / "hermes" / "prompts" / name, prompts / name)
        self.base = self.root / "artifacts" / "prelaunch" / "acquisition-check"

    def tearDown(self) -> None:
        self.temp.cleanup()

    def run_units(self, driver, tasks, *, run_id="prelaunch-acquisition-check-test", resume=False, retry_failed=False, max_runs=None, queue=QUEUE):
        store = ExperimentStore(self.root, run_id, base=self.base)
        plan = AcquisitionPlan(run_id, tuple(task for task, _ in tasks), task_families=tuple(family for _, family in tasks), queue_sha256=queue)
        executor = RealAcquisitionExecutor(self.root, store, driver, scientific=False, queue_sha256=queue, action_budget=3)
        result = AcquisitionRunner(self.root).run_resumable(
            plan, executor,
            configuration=launch.run_configuration(self.root, queue_sha256=queue, scientific=False),
            options=RunnerOptions(resume=resume, retry_failed=retry_failed, max_runs=max_runs, fail_fast=True),
            output_base=self.base, initial_library_hash=EMPTY_POOL_HASH, scientific=False, store=store,
            preflight=executor.preflight, checkpoint_extension=executor.checkpoint_state, progress=None,
        )
        return store, result

    @staticmethod
    def rows(store: ExperimentStore) -> list[dict]:
        return sorted(store.terminal_results(phase="acquisition", repair_tail=False).values(), key=lambda row: row["task_index"])

    def test_successful_episode_creates_one_validated_skill_with_provenance(self) -> None:
        world = {TASKS[0][0]: {"success": True}, TASKS[1][0]: {"success": True}, TASKS[2][0]: {"success": False}, TASKS[3][0]: {"success": True}}
        driver = FakeDriver(world, [GOOD_SKILL, GOOD_SKILL, LEAKY_SKILL])
        store, result = self.run_units(driver, TASKS[:4])
        self.assertEqual("completed", result["status"])
        rows = self.rows(store)
        self.assertEqual([True, True, False, True], [row["success"] for row in rows])
        self.assertEqual(["accepted", "rejected", "not_generated_episode_unsuccessful", "rejected"], [row["skill_candidate"]["status"] for row in rows])
        self.assertEqual(["exact_normalized_duplicate"], rows[1]["skill_candidate"]["rejection_reasons"])
        self.assertIn("leakage_object_instance", rows[3]["skill_candidate"]["rejection_reasons"])
        self.assertEqual("action_budget_exhausted", rows[2]["termination_reason"])
        self.assertEqual(3, len(driver.skill_prompts()))
        pool = rebuild_pool(rows)
        self.assertEqual(1, len(pool))
        skill = pool[0]
        self.assertNotEqual(EMPTY_POOL_HASH, pool_hash(pool))
        self.assertEqual([0, 1, 1, 1], [row["skill_pool_size_before"] for row in rows])
        self.assertEqual([1, 1, 1, 1], [row["library_size_after"] for row in rows])
        self.assertEqual({pool_hash(pool)}, {row["skill_library_hash_after"] for row in rows})
        self.assertEqual(
            (TASKS[0][0], "clean_and_place", 1, 1, rows[0]["run_key"], rows[0]["attempt_id"]),
            (skill.source_task_id, skill.task_family, skill.pool_index, skill.source_task_index, skill.source_run_key, skill.source_attempt_id),
        )
        self.assertEqual(("hermes3:8b", INFERENCE_SEED, 0, "create"), (skill.provenance["model"], skill.provenance["inference_seed"], skill.provenance["temperature"], skill.provenance["operation"]))
        self.assertEqual(build_skill_text(title="Clean before placing", body="Find the target object, rinse it at a sink basin, then put it in the requested receptacle."), skill.text)
        self.assertFalse(any(character.isdigit() for character in skill.text))
        self.assertTrue(all(row["scientific_retrieval_count"] == 0 and row["retrieved_skill_ids"] == [] for row in rows))
        self.assertEqual(4, len({session["output_dir"] for session in driver.sessions}))
        self.assertEqual({"rq1-acquisition"}, {session["profile"] for session in driver.sessions})
        state = store.load_checkpoint()[0]["phase_state"]
        self.assertEqual((1, pool_hash(pool)), (state["skill_pool"]["size"], state["skill_pool"]["hash"]))
        self.assertEqual({"planned": 4, "completed": 4, "successful": 3, "scientific_failures": 1, "infrastructure_failures": 0, "accepted_skills": 1}, state["per_family"]["clean_and_place"])
        self.assertEqual((None, QUEUE), (state["next_queue_index"], state["queue_sha256"]))
        self.assertEqual(pool_hash(pool), json.loads((store.directory / SNAPSHOT_NAME).read_text(encoding="utf-8"))["pool_hash"])

    def test_near_duplicates_are_preserved(self) -> None:
        world = {TASKS[0][0]: {"success": True}, TASKS[1][0]: {"success": True}}
        store, _ = self.run_units(FakeDriver(world, [GOOD_SKILL, NEAR_SKILL]), TASKS[:2])
        self.assertEqual(2, len(rebuild_pool(self.rows(store))))

    def test_infrastructure_failure_creates_no_skill_and_halts_for_chronological_retry(self) -> None:
        world = {TASKS[0][0]: {"success": True, "infrastructure_failure": True}, TASKS[1][0]: {"success": True}}
        driver = FakeDriver(world, [GOOD_SKILL])
        store, result = self.run_units(driver, TASKS[:2])
        self.assertEqual("failed", result["status"])
        self.assertEqual(["failed"], [row["status"] for row in self.rows(store)])
        error = store.read_errors()[-1]
        self.assertEqual(("episode", True, True), (error["details"]["stage"], error["safe_to_continue"], error["mutation_state_known"]))
        self.assertEqual([], driver.skill_prompts())
        self.assertEqual(1, len(driver.sessions))
        world[TASKS[0][0]]["infrastructure_failure"] = False
        retry_driver = FakeDriver(world, [GOOD_SKILL, NEAR_SKILL])
        self.run_units(retry_driver, TASKS[:2], resume=True, retry_failed=True)
        rows = self.rows(store)
        self.assertEqual(("completed", True), (rows[0]["status"], "supersedes_attempt_id" in rows[0]))
        _, resumed = self.run_units(retry_driver, TASKS[:2], resume=True)
        self.assertEqual("completed", resumed["status"])
        self.assertEqual(2, len(rebuild_pool(self.rows(store))))

    def test_resume_restores_pool_and_never_reruns_completed_units(self) -> None:
        world = {task: {"success": True} for task, _ in TASKS[:2]}
        store, first = self.run_units(FakeDriver(world, [GOOD_SKILL]), TASKS[:2], max_runs=1)
        self.assertEqual("paused", first["status"])
        second_driver = FakeDriver(world, [NEAR_SKILL])
        _, second = self.run_units(second_driver, TASKS[:2], resume=True)
        self.assertEqual(("completed", 1), (second["status"], len(second_driver.sessions)))
        rows = self.rows(store)
        self.assertEqual([0, 1], [row["skill_pool_size_before"] for row in rows])
        self.assertEqual(2, len(store.read_results()))
        third_driver = FakeDriver(world, [])
        _, third = self.run_units(third_driver, TASKS[:2], resume=True)
        self.assertEqual((0, []), (third["attempted_this_invocation"], third_driver.sessions))

    def test_resume_fails_closed_on_queue_configuration_or_commit_drift(self) -> None:
        world = {task: {"success": True} for task, _ in TASKS[:2]}
        self.run_units(FakeDriver(world, [GOOD_SKILL]), TASKS[:2], max_runs=1)
        with self.assertRaises(CompatibilityError):
            self.run_units(FakeDriver(world, [NEAR_SKILL]), TASKS[:2], resume=True, queue="z" * 64)
        with self.assertRaises(CompatibilityError):
            self.run_units(FakeDriver(world, [NEAR_SKILL]), list(reversed(TASKS[:2])), resume=True)
        original = persistence.runtime_manifest

        def with_commit(commit: str):
            def fake(root):
                value = original(root)
                value["git_commit"] = commit
                return value
            return fake

        with patch("rq1.experiment.persistence.runtime_manifest", with_commit("a" * 40)):
            self.run_units(FakeDriver(world, [GOOD_SKILL]), TASKS[:2], run_id="prelaunch-acquisition-check-commit", max_runs=1)
        with patch("rq1.experiment.persistence.runtime_manifest", with_commit("b" * 40)), self.assertRaises(CompatibilityError):
            self.run_units(FakeDriver(world, [NEAR_SKILL]), TASKS[:2], run_id="prelaunch-acquisition-check-commit", resume=True)

    def test_inconsistent_skill_pool_state_fails_closed(self) -> None:
        world = {task: {"success": True} for task, _ in TASKS[:2]}
        store, _ = self.run_units(FakeDriver(world, [GOOD_SKILL]), TASKS[:2], max_runs=1)
        snapshot = store.directory / SNAPSHOT_NAME
        value = json.loads(snapshot.read_text(encoding="utf-8"))
        value["skills"].append({**value["skills"][0], "pool_index": 2, "skill_id": "skill_extra"})
        value["pool_size"] = 2
        snapshot.write_text(json.dumps(value), encoding="utf-8")
        driver = FakeDriver(world, [NEAR_SKILL])
        with self.assertRaises(SkillPoolError):
            self.run_units(driver, TASKS[:2], resume=True)
        self.assertEqual([], driver.sessions)
        snapshot.unlink()
        record = json.loads(store.results_path.read_text(encoding="utf-8").splitlines()[0])
        record["skill_candidate"]["skill"]["body"] = "Something else entirely."
        store.results_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
        with self.assertRaises(SkillPoolError):
            self.run_units(FakeDriver(world, [NEAR_SKILL]), TASKS[:2], resume=True)

    def test_decision_003_validation_is_deterministic_and_preserves_near_duplicates(self) -> None:
        task = "train:pick_clean_then_place_in_recep-Cloth-None-Cart-401/trial_T20190909_010101_000001"

        def reasons(title, body, actions=(), existing=()):
            return validate_skill(title=title, body=body, source_task_id=task, executed_actions=actions, existing_texts=set(existing))

        body = "Rinse the object at a sink basin before placing it."
        accepted = build_skill_text(title="Clean before placing", body=body)
        self.assertEqual([], reasons("Clean before placing", body))
        self.assertIn("leakage_task_id", reasons("Routine", "Repeat what worked in train_12."))
        self.assertIn("leakage_room_number", reasons("Routine", "Search kitchen 2 first."))
        self.assertIn("leakage_object_instance", reasons("Routine", "Take cloth 1 to the cart."))
        self.assertIn("source_task_identifier", reasons("Routine", "Like pick_clean_then_place_in_recep-Cloth-None-Cart-401."))
        self.assertIn("executed_instance_action_verbatim", reasons("Routine", "take cloth 1 from cart 1", actions=("take cloth 1 from cart 1",)))
        self.assertEqual(["exact_normalized_duplicate"], reasons("  Clean   before placing ", "Rinse the object at a sink basin\n before placing it.", existing=[accepted]))
        self.assertEqual([], reasons("Clean Before Placing", body, existing=[accepted]))
        self.assertEqual([], reasons("Clean before placing", "Rinse the object at the sink basin before placing it.", existing=[accepted]))
        self.assertEqual([], reasons("Look first", "look around the room before acting", actions=("look",)))

    def test_skill_response_contract(self) -> None:
        self.assertTrue(parse_skill_response(" NO_SKILL \n").declined)
        parsed = parse_skill_response("TITLE: Clean first\nBODY: Rinse the object\nthen place it.")
        self.assertEqual(("Clean first", "Rinse the object then place it."), (parsed.title, parsed.body))
        for bad in ("BODY: only", "Sure!\nTITLE: A\nBODY: B", "TITLE: A", "TITLE:\nBODY: B", ""):
            self.assertIsNone(parse_skill_response(bad))

    def test_post_success_learning_requires_a_successful_episode(self) -> None:
        driver = FakeDriver({TASKS[0][0]: {"success": False}})
        with driver.session(output_dir=self.root / "episode", run_id="run") as session:
            session.start(TASKS[0][0], "train", 0, 3)
            with self.assertRaises(EpisodeDriverError):
                session.complete_post_success_learning("prompt")

    def test_scientific_path_requires_gates_and_separated_outputs(self) -> None:
        driver = FakeDriver({TASKS[0][0]: {"success": True}}, [GOOD_SKILL])
        plan = AcquisitionPlan("rq1-acquisition-test", (TASKS[0][0],), task_families=("clean_and_place",), queue_sha256=QUEUE)
        runner = AcquisitionRunner(self.root)
        store = ExperimentStore(self.root, plan.run_id)
        with self.assertRaises(AcquisitionError):
            runner.run_resumable(
                plan, RealAcquisitionExecutor(self.root, store, driver, scientific=True, queue_sha256=QUEUE, action_budget=3),
                configuration=launch.run_configuration(self.root, queue_sha256=QUEUE, scientific=True),
                initial_library_hash=EMPTY_POOL_HASH, scientific=True, store=store, progress=None,
            )
        outside = ExperimentStore(self.root, "prelaunch-acquisition-check-outside")
        with self.assertRaises(AcquisitionError):
            runner.run_resumable(
                replace(plan, run_id=outside.experiment_id),
                RealAcquisitionExecutor(self.root, outside, driver, scientific=False, queue_sha256=QUEUE, action_budget=3),
                configuration=launch.run_configuration(self.root, queue_sha256=QUEUE, scientific=False),
                initial_library_hash=EMPTY_POOL_HASH, scientific=False, store=outside, progress=None,
            )
        self.assertEqual([], driver.sessions)
        blocked = launch.scientific_run(
            self.root,
            argparse.Namespace(yes=True, run_id="rq1-acquisition-test", task_manifest=None, max_runs=None, backup_dir=None, require_backup=False),
            resume=False, retry_failed=False,
        )
        self.assertEqual((False, "blocked"), (blocked["ok"], blocked["status"]))

    def test_cli_exposes_operational_acquisition_commands(self) -> None:
        parser = build_parser()
        for argv in (
            ["acquisition", "run", "--run-id", "rq1-acquisition", "--yes", "--backup-dir", "/backup", "--require-backup"],
            ["acquisition", "check", "--run-id", "prelaunch-acquisition-check-x", "--task-id", "train:a", "--max-runs", "2"],
            ["acquisition", "check-report", "--run-id", "prelaunch-acquisition-check-x"],
            ["acquisition", "prepare-approvals", "--proposal", "proposal.json", "--evidence-report", "report.json"],
            ["freeze", "acquisition-protocol", "--approval-file", "approval.json", "--pilot-report", "report.json", "--yes"],
        ):
            parser.parse_args(argv)
        self.assertNotIn("no final run was started", (REPO / "src" / "rq1" / "cli.py").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
