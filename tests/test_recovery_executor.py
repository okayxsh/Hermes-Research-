"""Hermetic tests for controlled failure, evaluation executor, acquisition executor."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rq1.acquisition.real_executor import run_acquisition_episode
from rq1.evaluation.recovery_executor import RecoveryEpisodeSpec, run_recovery_episode
from rq1.recovery.controlled_failure import (
    CANONICAL_FAILURE_MESSAGE,
    ControlledFailureError,
    FailureEnvironment,
    FailureTrajectory,
    apply_controlled_failure,
    select_relocation_destination,
)
from rq1.recovery.models import RecoveryState
from rq1.retrieval import RetrievalQuery, build_retrieval_boundary


class FakeFailureEnvironment:
    def __init__(self, reachable=("countertop 1", "fridge 1", "pantry 1"), solvable: bool = True) -> None:
        self.object_id = "mug"
        self.locations = {"mug": "countertop 1"}
        self.reachable = tuple(reachable)
        self._solvable = solvable

    def required_object_id(self) -> str:
        return self.object_id

    def object_location(self, object_id: str) -> str:
        return self.locations[object_id]

    def reachable_locations(self):
        return self.reachable

    def relocate_object(self, object_id: str, destination: str) -> str:
        self.locations[object_id] = destination
        return "digest-" + destination

    def is_solvable(self) -> bool:
        return self._solvable


class DictEmbedder:
    def __init__(self, table, default=None) -> None:
        self.table = table
        self.default = default or [0.0, 0.0, 0.0]

    def encode(self, texts):
        return [self.table.get(text, list(self.default)) for text in texts]


def _state() -> RecoveryState:
    return RecoveryState(
        "valid_seen:task", "valid_seen", "heat_and_place", "heat the mug",
        "You are in the kitchen.", ("mug",), ("go to countertop 1",), 1, False, False, True,
    )


class FakeRecoveryHarness:
    def __init__(self, env: FakeFailureEnvironment | None = None) -> None:
        self.env = env or FakeFailureEnvironment()
        self.injected: dict | None = None
        self.replayed = None

    def start_and_replay(self, task_id, split, seed, prefix_actions):
        self.replayed = (task_id, split, seed, tuple(prefix_actions))
        return _state()

    def current_state(self):
        return _state()

    def failure_environment(self):
        return self.env

    def inject_recovery_memory(self, message):
        self.injected = message

    def run_recovery(self, action_budget):
        return (
            {"step": 1, "action": "go to fridge 1", "action_valid": True, "done": False},
            {"step": 2, "action": "take mug", "action_valid": True, "done": True},
        )

    def recovery_succeeded(self):
        return True

    def failure_trajectory(self):
        return FailureTrajectory(
            pre_failure_actions=("go to countertop 1",),
            post_failure_actions=("go to fridge 1", "take mug"),
        )


def _query_text() -> str:
    return RetrievalQuery(
        task_instruction="heat the mug",
        observation="You are in the kitchen.",
        inventory=("mug",),
        failure_message=CANONICAL_FAILURE_MESSAGE,
    ).text()


def _memory_boundary():
    skills = [
        ("s-heat", "TITLE: heat object\nBODY: use the microwave"),
        ("s-clean", "TITLE: clean object\nBODY: use a cloth"),
        ("s-cool", "TITLE: cool object\nBODY: use the fridge"),
        ("s-look", "TITLE: examine object\nBODY: pick it up"),
    ]
    table = {
        "TITLE: heat object\nBODY: use the microwave": [1.0, 0.0],
        "TITLE: clean object\nBODY: use a cloth": [0.0, 1.0],
        "TITLE: cool object\nBODY: use the fridge": [0.9, 0.1],
        "TITLE: examine object\nBODY: pick it up": [0.8, 0.2],
        _query_text(): [1.0, 0.0],
    }
    return build_retrieval_boundary(skills, DictEmbedder(table), embedding_model="all-mpnet-base-v2")


def _spec(condition="Accum-60", size=4) -> RecoveryEpisodeSpec:
    return RecoveryEpisodeSpec(
        run_id="run-1", attempt_id="att-1", task_id="valid_seen:task", task_family="heat_and_place",
        split="valid_seen", seed=11, condition=condition, library_name=condition, library_size=size,
        library_hash="hash-1", checkpoint_id="cp-1",
        prefix_actions=("go to countertop 1",), action_budget=12,
    )


class ControlledFailureTests(unittest.TestCase):
    def test_destination_is_deterministic_and_not_original(self) -> None:
        destination = select_relocation_destination("countertop 1", ("pantry 1", "fridge 1", "countertop 1"))
        self.assertEqual(destination, "fridge 1")

    def test_no_destination_fails_closed(self) -> None:
        with self.assertRaises(ControlledFailureError) as caught:
            select_relocation_destination("countertop 1", ("countertop 1",))
        self.assertEqual(caught.exception.code, "no_reachable_destination")

    def test_unsolvable_perturbation_fails_closed(self) -> None:
        with self.assertRaises(ControlledFailureError) as caught:
            apply_controlled_failure(FakeFailureEnvironment(solvable=False), checkpoint_id="cp-1")
        self.assertEqual(caught.exception.code, "perturbation_unsolvable")

    def test_failure_message_never_reveals_new_location(self) -> None:
        failure = apply_controlled_failure(FakeFailureEnvironment(), checkpoint_id="cp-1")
        self.assertEqual(failure.failure_message, CANONICAL_FAILURE_MESSAGE)
        self.assertNotIn(failure.new_location, failure.failure_message)
        self.assertNotEqual(failure.new_location, failure.original_location)


class EvaluationExecutorTests(unittest.TestCase):
    def test_memory_condition_retrieves_exactly_once(self) -> None:
        harness = FakeRecoveryHarness()
        with tempfile.TemporaryDirectory() as tmp:
            result = run_recovery_episode(harness, _spec(), _memory_boundary(), log_dir=Path(tmp))
            self.assertEqual(result.retrieval_count, 1)
            self.assertFalse(result.no_retrieval)
            self.assertEqual(len(result.retrieval_event["top"]), 3)
            self.assertEqual(result.outcome.retrieved_skill_ids[0], "s-heat")
            # Scores logged for audit, never injected into the agent message.
            self.assertIn("score", result.retrieval_event["top"][0])
            for item in result.recovery_memory["recovery_memory"]["retrieved_skills"]:
                self.assertNotIn("score", item)
            self._assert_logs(Path(tmp))

    def test_nolib_uses_same_path_and_reports_no_retrieval(self) -> None:
        boundary = build_retrieval_boundary([], DictEmbedder({}), embedding_model="all-mpnet-base-v2")
        harness = FakeRecoveryHarness()
        with tempfile.TemporaryDirectory() as tmp:
            result = run_recovery_episode(harness, _spec("NoLib", 0), boundary, log_dir=Path(tmp))
            self.assertEqual(result.retrieval_count, 1)
            self.assertTrue(result.no_retrieval)
            self.assertEqual(result.outcome.retrieved_skill_ids, ())
            self.assertEqual(result.recovery_memory["recovery_memory"]["retrieved_skills"], [])
            self.assertTrue(result.recovery_memory["recovery_memory"]["no_retrieved_skills_available"])
            self._assert_logs(Path(tmp))

    def test_failure_context_and_recovery_actions_are_logged(self) -> None:
        harness = FakeRecoveryHarness()
        with tempfile.TemporaryDirectory() as tmp:
            result = run_recovery_episode(harness, _spec(), _memory_boundary(), log_dir=Path(tmp))
            self.assertEqual(result.failure_context["task_instruction"], "heat the mug")
            self.assertEqual(result.failure_context["inventory"], ["mug"])
            self.assertEqual(len(result.recovery_steps), 2)
            failure = json.loads((Path(tmp) / "failure.json").read_text(encoding="utf-8"))
            self.assertIn("failure_context", failure)
            self.assertIn("trajectory", failure)
            steps = (Path(tmp) / "recovery_steps.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(steps), 2)

    def test_injected_message_is_not_the_query_or_scores(self) -> None:
        harness = FakeRecoveryHarness()
        with tempfile.TemporaryDirectory() as tmp:
            run_recovery_episode(harness, _spec(), _memory_boundary(), log_dir=Path(tmp))
            self.assertIsNotNone(harness.injected)
            serialized = json.dumps(harness.injected)
            self.assertNotIn(CANONICAL_FAILURE_MESSAGE, serialized)

    def _assert_logs(self, tmp: Path) -> None:
        retrieval = (tmp / "retrieval.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(retrieval), 1)
        record = json.loads(retrieval[0])
        self.assertEqual(record["condition"] in {"Accum-60", "NoLib"}, True)
        self.assertIn("query_text_hash", record)
        self.assertTrue((tmp / "failure.json").is_file())


class AcquisitionExecutorTests(unittest.TestCase):
    def _harness(self, success: bool, with_candidate: bool = True):
        class Harness:
            def run_episode(self, task_id, split, seed, action_limit):
                candidate = {"title": "heat", "body": "use microwave"} if with_candidate else None
                return {"success": success, "steps": 3, "actions": 3, "invalid_actions": 0, "skill_candidate": candidate}

            def post_run_library_hash(self):
                return "library-hash"

            def post_run_library_size(self):
                return 5

        return Harness()

    def test_successful_episode_creates_one_candidate_with_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = run_acquisition_episode(
                self._harness(True), task_id="train:a", task_family="heat_and_place",
                attempt_id="att", log_dir=Path(tmp),
            )
            self.assertEqual(result.retrieval_count, 0)
            self.assertEqual(result.outcome.retrieved_skill_ids, ())
            self.assertIsNotNone(result.skill_candidate)
            self.assertEqual(result.skill_candidate["task_family"], "heat_and_place")
            self.assertEqual(result.skill_candidate["source_task_id"], "train:a")
            self.assertEqual(result.skill_candidate["source_attempt_id"], "att")
            self.assertEqual(result.outcome.skill_library_hash_after, "library-hash")
            self.assertEqual(result.outcome.library_size_after, 5)
            # Acquisition never writes a retrieval log.
            self.assertFalse((Path(tmp) / "retrieval.jsonl").exists())

    def test_failed_episode_creates_no_positive_skill(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = run_acquisition_episode(
                self._harness(False), task_id="train:a", task_family="heat_and_place",
                attempt_id="att", log_dir=Path(tmp),
            )
            self.assertIsNone(result.skill_candidate)

    def test_rejects_non_train_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                run_acquisition_episode(
                    self._harness(True), task_id="valid_seen:a", task_family="heat_and_place",
                    attempt_id="att", log_dir=Path(tmp), split="valid_seen",
                )


if __name__ == "__main__":
    unittest.main()
