"""Tests for frozen task/seed/count protocol and longest-trajectory selection."""
from __future__ import annotations

import unittest
from pathlib import Path

from rq1.freeze.validation import ENVIRONMENT_REQUIRED, PROTOCOL_REQUIRED
from rq1.tasks.models import TaskRecord
from rq1.tasks.selection import (
    ACQUISITION_HARD_CAP,
    ACQUISITION_INITIAL_TASKS,
    FROZEN_EVALUATION_TASKS,
    FROZEN_REPETITIONS,
    FROZEN_SEEDS,
    FROZEN_TASKS_PER_FAMILY,
    select_longest_per_family,
)
from rq1.utils.config import load_json_yaml


def _record(task_id: str, family: str) -> TaskRecord:
    return TaskRecord(task_id, "valid_unseen", family, task_id, "s", "g", 0)


class FrozenCountTests(unittest.TestCase):
    def test_seeds_and_counts_are_frozen(self) -> None:
        self.assertEqual(FROZEN_SEEDS, (11, 29, 47))
        self.assertEqual(FROZEN_REPETITIONS, 3)
        self.assertEqual(FROZEN_EVALUATION_TASKS, 30)
        self.assertEqual(FROZEN_TASKS_PER_FAMILY, 5)
        self.assertEqual(ACQUISITION_INITIAL_TASKS, 180)
        self.assertEqual(ACQUISITION_HARD_CAP, 240)

    def test_evaluation_config_matches_constants(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = load_json_yaml(root / "configs" / "tasks" / "evaluation.yaml")
        self.assertEqual(config["requested_count"], 30)
        self.assertEqual(config["tasks_per_family"], 5)
        self.assertEqual(config["seeds"], list(FROZEN_SEEDS))
        self.assertEqual(config["repetitions"], 3)
        self.assertEqual(config["core_episode_count"], 360)
        self.assertEqual(
            config["selection_rule"], "longest_expert_trajectories_tie_break_task_id"
        )

    def test_acquisition_config_matches_constants(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = load_json_yaml(root / "configs" / "tasks" / "acquisition.yaml")
        self.assertEqual(config["requested_count"], 180)
        self.assertEqual(config["hard_cap"], 240)

    def test_freeze_requires_seeds_library_hashes_retriever(self) -> None:
        for field in ("seeds", "library_hashes", "retriever_model"):
            self.assertIn(field, ENVIRONMENT_REQUIRED)
        for field in ("seeds", "retriever_model"):
            self.assertIn(field, PROTOCOL_REQUIRED)


class LongestPerFamilyTests(unittest.TestCase):
    def test_selects_longest_per_family_with_tie_break(self) -> None:
        records = [
            _record("a-short", "pick_and_place"),
            _record("a-long", "pick_and_place"),
            _record("b-short", "heat_and_place"),
            _record("b-long", "heat_and_place"),
        ]
        lengths = {"a-short": 3, "a-long": 9, "b-short": 4, "b-long": 8}
        selected, exclusions = select_longest_per_family(records, lengths, tasks_per_family=1)
        self.assertEqual([item.task_id for item in selected], ["a-long", "b-long"])
        self.assertEqual({item["task_id"] for item in exclusions}, {"a-short", "b-short"})

    def test_tie_break_is_deterministic_by_task_id(self) -> None:
        records = [_record("zz", "pick_and_place"), _record("aa", "pick_and_place")]
        lengths = {"zz": 5, "aa": 5}
        selected, _ = select_longest_per_family(records, lengths, tasks_per_family=1)
        self.assertEqual([item.task_id for item in selected], ["aa"])

    def test_returns_fewer_than_quota_when_family_short(self) -> None:
        records = [_record("only", "pick_and_place")]
        lengths = {"only": 5}
        selected, _ = select_longest_per_family(records, lengths, tasks_per_family=5)
        # Selection is not padded silently; the caller must fail closed.
        self.assertEqual(len(selected), 1)


if __name__ == "__main__":
    unittest.main()
