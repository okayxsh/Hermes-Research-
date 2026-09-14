"""Hermetic tests for the amended controlled-recovery evaluation (Decision 012).

No valid_unseen data, model, or real ALFWorld is used; every episode is scripted.
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from rq1.bridge.adapters.task_index import TaskIndexError, build_task_index
from rq1.bridge.adapters.unseen_access import SCIENTIFIC_EVALUATION, TASK_PREPARATION, UNSEEN_ACCESS_ENV, UnseenAccessError, allowed_real_splits, require_unseen_access
from rq1.evaluation.amended_executor import AmendedEvaluationExecutor, recovery_timing
from rq1.evaluation.amended_libraries import (
    AmendedLibrary,
    LibraryAmendmentError,
    PoolEntry,
    build_amended_libraries,
    nesting_problems,
    parse_review_csv,
    raw_feasibility,
    render_review_csv,
    review_rows,
    select_core,
)
from rq1.evaluation.amended_matrix import (
    EvaluationTask,
    build_matrix,
    coverage_problems,
    experiment_units,
    matrix_sha256,
    merge_shards,
    shard_manifest,
    shard_run_id,
)
from rq1.evaluation.amended_protocol import (
    CONDITIONS,
    EVALUATION_SEEDS,
    LIBRARY_SIZES,
    PROTOCOL_CONFIG,
    evaluation_protocol_definition,
)
from rq1.experiment.models import RunExecutionContext, canonical_hash
from rq1.freeze.validation import EVALUATION_EVIDENCE_MODE, REQUIRED_INPUTS, build_freeze
from rq1.pilot.real_runtime.harnesses import bridge_state_digest
from rq1.recovery.controlled_failure import ControlledFailureError
from rq1.recovery.reference_route import ReferenceRouteError, derive_handcoded_reference
from rq1.retrieval.text import build_skill_text, skill_text_hash
from rq1.skills.library import TASK_FAMILIES
from rq1.utils.config import load_json_yaml

from test_real_episode_driver import FakeDriver, LengthEmbedder, NoOracleHarness, TASK_ID

REPO = Path(__file__).resolve().parents[1]
RAW_COUNTS = {"pick_and_place": 11, "pick_two_and_place": 8, "look_at_object": 4, "clean_and_place": 9, "heat_and_place": 15, "cool_and_place": 3}
REFERENCE = ("go to bed 1", "go to desk 1", "go to sidetable 1")


def synthetic_pool(counts=RAW_COUNTS) -> tuple[PoolEntry, ...]:
    entries, index = [], 0
    ranks = {family: 0 for family in TASK_FAMILIES}
    for round_ in range(max(counts.values())):
        for family in TASK_FAMILIES:
            if round_ < counts[family]:
                index += 1
                ranks[family] += 1
                text = build_skill_text(title=f"{family} skill {ranks[family]}", body=f"General {family} guidance number {ranks[family]}.")
                entries.append(PoolEntry(index, f"skill_{family}_{ranks[family]}", family, ranks[family], index * 4, "parent_units_1_180",
                                         "rq1-acquisition-gemma4-12b", f"train:{family}-{ranks[family]}", None, text, skill_text_hash(text)))
    return tuple(entries)


def reviewed(pool, decisions: dict[str, list[str]]) -> list[dict[str, str]]:
    rows = parse_review_csv(render_review_csv(review_rows(pool)))
    by_family_rank = {(row["task_family"], int(row["family_chronological_rank"])): row for row in rows}
    for family, verdicts in decisions.items():
        for rank, verdict in enumerate(verdicts, 1):
            row = by_family_rank[(family, rank)]
            row.update({"reviewer_quality_pass": verdict, "reviewer_quality_reason": "PASS_Q1_Q7" if verdict == "PASS" else "Q1_NOT_GENERALIZED",
                        "reviewed_at": "2026-09-14T12:00:00Z"})
    return rows


def synthetic_tasks() -> tuple[EvaluationTask, ...]:
    tasks = []
    for family in TASK_FAMILIES:
        for number in range(1, 6):
            task_id = f"valid_unseen:{family}-task-{number}/trial_T{number}"
            tasks.append(EvaluationTask(task_id, family, len(tasks) + 1, f"cp:{task_id}", "c" * 64, "p" * 64, "go to bed 1", "go to desk 1",
                                        ("go to bed 1",), REFERENCE, 20))
    return tuple(tasks)


class ProtocolAndRubricTests(unittest.TestCase):
    def test_amended_protocol_is_frozen_in_code_config_and_records(self) -> None:
        definition = evaluation_protocol_definition()
        self.assertEqual(definition, load_json_yaml(REPO / PROTOCOL_CONFIG))
        self.assertEqual({"NoLib": 0, "Core-6": 6, "Accum-12": 12, "Accum-18": 18}, LIBRARY_SIZES)
        self.assertEqual(("NoLib", "Core-6", "Accum-12", "Accum-18"), CONDITIONS)
        self.assertEqual((360, [11, 29, 47], 30), (definition["matrix"]["units"], definition["matrix"]["seeds"], definition["tasks"]["task_count"]))
        self.assertEqual("A", definition["libraries"]["rule"])
        self.assertFalse(any(definition["amendment"][key] for key in ("evaluation_outcomes_observed", "skills_fabricated", "acquisition_episodes_discarded",
                                                                       "quality_rule_loosened", "semantic_deduplication")))
        retrieval = definition["retrieval"]
        self.assertEqual(("sentence-transformers/all-mpnet-base-v2", "e8c3b32edf5434bc2275fc9bab85f82640a19130", 3, 768, False, False),
                         (retrieval["model"], retrieval["revision"], retrieval["top_k"], retrieval["embedding_dimension"],
                          retrieval["action_history_in_query"], retrieval["scores_visible_to_model"]))
        self.assertEqual("bbfecb04a834241144af31e9200a3dd0df83ff0fbb2453d56a4bb383a6bfbe11", retrieval["snapshot_sha256"])
        self.assertFalse(definition["controlled_failure"]["object_relocation"])
        self.assertEqual(("gemma4:12b", "4eb23ef187e2c5462566d6a1d3bbbc2f1346d0b4327cbb66d58fffbcc9b2b05c", 2048, 32768, 0),
                         tuple(definition["model"][key] for key in ("tag", "digest", "num_predict", "num_ctx", "temperature")))
        self.assertEqual("conditional_recovery_rate", definition["metrics"]["primary"])
        rubric = " ".join((REPO / "docs" / "SKILL_QUALITY_RUBRIC.md").read_text(encoding="utf-8").split())
        for phrase in ("Q1 Generalized", "Q7 Structurally valid", "PASSES only if every question Q1–Q7", "Near-duplicates are allowed", "first PASS"):
            self.assertIn(phrase, rubric)
        record = " ".join((REPO / "docs" / "decisions" / "012-pre-evaluation-library-amendment.md").read_text(encoding="utf-8").split())
        for phrase in ("Rule A", "0 / 6 / 12 / 18", "no evaluation outcome", "UNAPPROVED", "valid_unseen"):
            self.assertIn(phrase, record)

    def test_evaluation_freeze_kinds_require_evaluation_evidence(self) -> None:
        for kind in ("evaluation-amendment", "evaluation-environment", "evaluation-protocol"):
            self.assertIn(kind, REQUIRED_INPUTS)
        inputs = {key: "x" for key in REQUIRED_INPUTS["evaluation-protocol"]} | {"repository_commit": "c" * 40}
        approval = {"approval_kind": "evaluation-protocol", "approval": {"status": "APPROVED", "approved_by": "r", "approved_at": "t"}, "inputs": inputs}
        evidence = {"mode": EVALUATION_EVIDENCE_MODE, "passed": True, "scientific_evidence": False, "repository_commit": "c" * 40, "run_id": "prelaunch-evaluation-check-x"}
        with patch("rq1.freeze.validation.git_state", return_value=("c" * 40, True, None)):
            self.assertEqual("evaluation-protocol", build_freeze(Path("."), "evaluation-protocol", approval, evidence).kind)
            with self.assertRaises(ValueError):
                build_freeze(Path("."), "evaluation-protocol", approval, {**evidence, "mode": "non_scientific_acquisition_check"})


class UnseenAccessTests(unittest.TestCase):
    def test_valid_unseen_requires_explicit_named_authorization(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(UNSEEN_ACCESS_ENV, None)
            self.assertNotIn("valid_unseen", allowed_real_splits())
            with self.assertRaises(UnseenAccessError):
                require_unseen_access()
            with self.assertRaises(ReferenceRouteError):
                derive_handcoded_reference(Path("/nonexistent"), "valid_unseen:task/trial", "valid_unseen")
        with patch.dict(os.environ, {UNSEEN_ACCESS_ENV: "yes"}):
            self.assertNotIn("valid_unseen", allowed_real_splits())
        with patch.dict(os.environ, {UNSEEN_ACCESS_ENV: TASK_PREPARATION}):
            self.assertIn("valid_unseen", allowed_real_splits())
            with self.assertRaises(UnseenAccessError):
                require_unseen_access(SCIENTIFIC_EVALUATION)

    def test_bridge_index_includes_valid_unseen_only_when_authorized(self) -> None:
        from rq1.bridge.adapters.base import IndexedTask
        from rq1.bridge.adapters.task_index import TaskIndex

        task_id = "valid_unseen:look_at_obj_in_light-Book-None-DeskLamp-1/trial_T1"
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for split in ("train", "valid_seen", "valid_unseen"):
                (root / "json_2.1.1" / split).mkdir(parents=True)
            entry = IndexedTask(task_id, "valid_unseen", "look_at_obj_in_light", Path("json_2.1.1/valid_unseen/x/traj_data.json"),
                                Path("json_2.1.1/valid_unseen/x/game.tw-pddl"), "s", "g", "d")
            index = TaskIndex(root, (entry,), "identity")
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop(UNSEEN_ACCESS_ENV, None)
                with self.assertRaises(TaskIndexError):
                    index.resolve(task_id, "valid_unseen")
                with self.assertRaises(TaskIndexError):
                    build_task_index(root, splits=("valid_unseen",))
            with patch.dict(os.environ, {UNSEEN_ACCESS_ENV: SCIENTIFIC_EVALUATION}):
                self.assertEqual("valid_unseen", index.resolve(task_id, "valid_unseen").split)
                try:
                    build_task_index(root, splits=("valid_unseen",))
                except TaskIndexError as exc:
                    self.assertNotIn("intentionally excluded", str(exc))


class LibraryConstructorTests(unittest.TestCase):
    def test_raw_counts_support_18_and_core_is_earliest_pass(self) -> None:
        pool = synthetic_pool()
        self.assertEqual({"feasible": True, "families_below": {}}, {key: raw_feasibility(pool)[key] for key in ("feasible", "families_below")})
        self.assertFalse(raw_feasibility(synthetic_pool({**RAW_COUNTS, "cool_and_place": 2}))["feasible"])
        decisions = {family: ["PASS"] for family in TASK_FAMILIES}
        decisions["look_at_object"] = ["FAIL", "PASS"]
        decisions["cool_and_place"] = ["FAIL", "FAIL", "PASS"]
        selection = select_core(pool, reviewed(pool, decisions))
        self.assertTrue(selection.complete, selection.problems)
        self.assertEqual(("skill_look_at_object_2", "skill_cool_and_place_3", "skill_heat_and_place_1"),
                         (selection.core["look_at_object"], selection.core["cool_and_place"], selection.core["heat_and_place"]))
        libraries = build_amended_libraries(pool, selection.core)
        self.assertEqual({condition: LIBRARY_SIZES[condition] for condition in CONDITIONS}, {condition: libraries[condition].size for condition in CONDITIONS})
        self.assertEqual([], nesting_problems(libraries))
        ids = {condition: [skill["skill_id"] for skill in libraries[condition].skills] for condition in CONDITIONS}
        self.assertTrue(set(ids["Core-6"]) < set(ids["Accum-12"]) < set(ids["Accum-18"]))
        self.assertEqual(len({libraries[condition].core_sha256 for condition in CONDITIONS[1:]}), 1)
        # Rule A: extras are the earliest remaining skills, including ones that FAILED review.
        look = [skill["skill_id"] for skill in libraries["Accum-12"].skills if skill["task_family"] == "look_at_object"]
        self.assertEqual(["skill_look_at_object_2", "skill_look_at_object_1"], look)
        cool = [skill["skill_id"] for skill in libraries["Accum-18"].skills if skill["task_family"] == "cool_and_place"]
        self.assertEqual(["skill_cool_and_place_3", "skill_cool_and_place_1", "skill_cool_and_place_2"], cool)
        self.assertEqual((), libraries["NoLib"].skills)
        self.assertEqual(libraries["Accum-18"].content_sha256, build_amended_libraries(pool, selection.core)["Accum-18"].content_sha256)

    def test_core_review_fails_closed(self) -> None:
        pool = synthetic_pool()
        pending = select_core(pool, reviewed(pool, {family: ["PASS"] for family in TASK_FAMILIES[:5]}))
        self.assertEqual((False, ("cool_and_place",)), (pending.complete, pending.pending_families))
        failed = select_core(pool, reviewed(pool, {**{family: ["PASS"] for family in TASK_FAMILIES}, "cool_and_place": ["FAIL", "FAIL", "FAIL"]}))
        self.assertEqual((False, ("cool_and_place",)), (failed.complete, failed.failed_families))
        self.assertTrue(any("no PASS core candidate" in problem for problem in failed.problems))
        rows = reviewed(pool, {family: ["PASS"] for family in TASK_FAMILIES})
        tampered = [dict(row) for row in rows]
        tampered[0]["skill_text"] = "TITLE: edited\nBODY: edited"
        self.assertFalse(select_core(pool, tampered).complete)
        bad_time = [dict(row) for row in rows]
        next(row for row in bad_time if row["reviewer_quality_pass"])["reviewed_at"] = "today"
        self.assertFalse(select_core(pool, bad_time).complete)
        bad_value = [dict(row) for row in rows]
        next(row for row in bad_value if row["reviewer_quality_pass"])["reviewer_quality_pass"] = "yes"
        self.assertFalse(select_core(pool, bad_value).complete)
        self.assertFalse(select_core(pool, rows[:-1]).complete)
        with self.assertRaises(LibraryAmendmentError):
            build_amended_libraries(pool, {family: f"skill_{family}_1" for family in TASK_FAMILIES[:5]})


class MatrixAndShardTests(unittest.TestCase):
    def test_360_units_balanced_shards_and_deterministic_merge(self) -> None:
        tasks = synthetic_tasks()
        matrix = build_matrix(tasks)
        self.assertEqual((360, []), (len(matrix), coverage_problems(matrix)))
        self.assertEqual(matrix_sha256(matrix), matrix_sha256(build_matrix(tuple(reversed(tasks)))))
        for shard in range(1, 7):
            manifest = shard_manifest(matrix, shard)
            self.assertEqual((60, {condition: 15 for condition in CONDITIONS}), (manifest["unit_count"], manifest["condition_counts"]))
            self.assertEqual(f"rq1-evaluation-gemma4-12b-shard-{shard}-of-6", manifest["run_id"])
        self.assertEqual({(task.task_id, seed, condition) for task in tasks for seed in EVALUATION_SEEDS for condition in CONDITIONS},
                         {(unit["task_id"], unit["seed"], unit["condition"]) for unit in matrix})
        for cell in range(90):
            members = [unit for unit in matrix if unit["cell_index"] == cell]
            self.assertEqual((set(CONDITIONS), 1), ({unit["condition"] for unit in members}, len({unit["shard"] for unit in members})))
        rotations = {tuple(unit["condition"] for unit in matrix if unit["cell_index"] == cell) for cell in range(0, 90, 6)}
        self.assertEqual(4, len(rotations))
        sizes = dict(LIBRARY_SIZES)
        hashes = {condition: canonical_hash(condition) for condition in CONDITIONS}
        units = experiment_units(matrix, 3, sizes, hashes)
        self.assertEqual((list(range(1, 61)), [unit["unit_key"] for unit in matrix if unit["shard"] == 3]),
                         ([unit.task_index for unit in units], [unit.run_key for unit in units]))
        records = {shard: [{"run_key": unit.run_key, "status": "completed"} for unit in experiment_units(matrix, shard, sizes, hashes)] for shard in range(1, 7)}
        merged, problems = merge_shards(matrix, records)
        self.assertEqual((360, []), (len(merged), problems))
        broken = {**records, 1: records[1][1:], 2: records[2] + [records[3][0]]}
        _, problems = merge_shards(matrix, broken)
        self.assertTrue(any("no terminal record" in problem for problem in problems))
        self.assertTrue(any("not its frozen shard" in problem for problem in problems))
        with self.assertRaises(Exception):
            build_matrix(tasks[:-1])
        with self.assertRaises(Exception):
            shard_run_id(7)


class ExecutorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def frozen_digests(self) -> dict[str, str]:
        harness = NoOracleHarness(FakeDriver(), output_dir=self.root / "freeze-probe", run_id="probe", attempt_id="probe", reference_actions=REFERENCE, total_action_limit=50)
        harness.start_and_replay(TASK_ID, "valid_seen", 11, REFERENCE[:1])
        perturbation = harness.apply_controlled_action_perturbation("cp")
        harness.close()
        return {"checkpoint_digest": harness.checkpoint_digest, "detour_action": perturbation.action, "post_detour_digest": perturbation.post_state_digest}

    def task(self, digests: dict[str, str], budget: int = 3) -> EvaluationTask:
        return EvaluationTask(TASK_ID, "look_at_object", 1, "cp", digests["checkpoint_digest"], digests["post_detour_digest"], digests["detour_action"],
                              "go to desk 1", REFERENCE[:1], REFERENCE, budget)

    def library(self, condition: str) -> AmendedLibrary:
        skills = tuple({"skill_id": f"s{index}", "task_family": family, "text": build_skill_text(title=f"T{index}", body=f"B{index}"),
                        "text_sha256": "x", "role": "core" if index < 6 else "extra", "family_chronological_rank": 1, "pool_index": index,
                        "logical_acquisition_index": index} for index, family in enumerate(TASK_FAMILIES * 3) if index < LIBRARY_SIZES[condition])
        return AmendedLibrary(condition, skills)

    def run_unit(self, driver, condition: str, digests: dict[str, str], seed: int = 29):
        task = self.task(digests)
        matrix_unit = {"global_order": 1, "unit_key": "k", "identity": {"phase": "evaluation", "task_id": TASK_ID, "seed": seed, "condition": condition},
                       "cell_index": 0, "position_in_cell": 0, "shard": 1, "task_id": TASK_ID, "task_family": "look_at_object", "seed": seed,
                       "condition": condition, "checkpoint_id": "cp", "recovery_action_budget": 3}
        unit = experiment_units([matrix_unit], 1, dict(LIBRARY_SIZES), {c: "h" for c in CONDITIONS})[0]
        executor = AmendedEvaluationExecutor(driver, tasks={TASK_ID: task}, libraries={c: self.library(c) for c in CONDITIONS}, embedder=LengthEmbedder(),
                                             scientific=False, split="valid_seen", provenance={"evaluation_matrix_sha256": "m"})
        output = self.root / condition / str(seed)
        context = RunExecutionContext("prelaunch-evaluation-check-test", f"attempt-{condition}", 1, output, lambda: False)
        return executor(unit, context)

    def test_library_retrieves_once_nolib_never_and_results_carry_provenance(self) -> None:
        digests = self.frozen_digests()
        memory_driver = FakeDriver(respond=lambda prompt: "ACTION_INDEX: 3")
        outcome = self.run_unit(memory_driver, "Accum-18", digests)
        values = outcome.measurements
        self.assertEqual((1, False, 3, 29, 29), (values["retrieval_events"], values["no_retrieval"], len(values["retrieval"]["top"]), values["seed"], memory_driver.inference_seed))
        self.assertEqual(("recovery_budget_exhausted", False, 3, None), (values["termination_reason"], values["recovery_success"], values["post_failure_actions"], values["recovery_latency_actions"]))
        self.assertEqual((digests["checkpoint_digest"], digests["post_detour_digest"]), (values["checkpoint_digest"], values["perturbation_digest"]))
        for key in ("task_id", "task_family", "seed", "condition", "library_hash", "model_digest", "evaluation_matrix_sha256", "checkpoint_id",
                    "perturbation", "failure_context", "retrieval", "recovery_actions", "termination_reason", "recovery_success"):
            self.assertIn(key, values)
        recovery_prompts = [prompt for prompt in memory_driver.prompts if "RECOVERY MEMORY:" in prompt]
        self.assertEqual(3, len(recovery_prompts))
        for prompt in recovery_prompts:
            self.assertNotIn("score", prompt.split("RECOVERY MEMORY:")[1].split("ADMISSIBLE ACTIONS:")[0])
            self.assertNotIn("Accum-18", prompt)
        nolib_driver = FakeDriver(respond=lambda prompt: "ACTION_INDEX: 3")
        nolib = self.run_unit(nolib_driver, "NoLib", digests, seed=11).measurements
        self.assertEqual((0, True, [], 11), (nolib["retrieval_events"], nolib["no_retrieval"], nolib["retrieval"]["top"], nolib_driver.inference_seed))
        self.assertTrue(all('"no_retrieved_skills_available": true' in prompt for prompt in nolib_driver.prompts if "RECOVERY MEMORY:" in prompt))

    def test_frozen_checkpoint_drift_fails_closed(self) -> None:
        digests = self.frozen_digests()
        driver = FakeDriver(respond=lambda prompt: "ACTION_INDEX: 3")
        from rq1.experiment.models import RunFailure

        with self.assertRaises(RunFailure) as caught:
            self.run_unit(driver, "Core-6", {**digests, "checkpoint_digest": "0" * 64})
        self.assertEqual("checkpoint_digest_mismatch", caught.exception.details["code"])
        with self.assertRaises(RunFailure):
            self.run_unit(driver, "Core-6", {**digests, "post_detour_digest": "0" * 64})
        self.assertEqual([], [prompt for prompt in driver.prompts if "RECOVERY MEMORY:" in prompt])

    def test_recovery_timing_uses_injection_and_successful_step(self) -> None:
        events = [{"event": "tool_result", "timestamp": 1.0, "payload": {"tool": "alfworld_step", "response": {"done": True, "success": True}}},
                  {"event": "recovery_memory_injected", "timestamp": 10.0, "payload": {}},
                  {"event": "tool_result", "timestamp": 12.5, "payload": {"tool": "alfworld_step", "response": {"done": False}}},
                  {"event": "tool_result", "timestamp": 14.25, "payload": {"tool": "alfworld_step", "response": {"done": True, "success": True}}}]
        self.assertEqual(4.25, recovery_timing(events)["recovery_latency_seconds"])
        self.assertIsNone(recovery_timing(events[:3])["recovery_latency_seconds"])

    def test_state_digest_is_order_stable(self) -> None:
        self.assertEqual(bridge_state_digest({"observation": "o", "admissible_actions": ["a"], "step_number": 1, "extra": 1}),
                         bridge_state_digest({"step_number": 1, "admissible_actions": ["a"], "observation": "o"}))


if __name__ == "__main__":
    unittest.main()
