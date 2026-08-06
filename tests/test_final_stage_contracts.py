from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rq1.acquisition.models import AcquisitionAttempt, SkillOperation
from rq1.acquisition.runner import AcquisitionRunner, validate_history
from rq1.evaluation.queue import build_paired_queue
from rq1.evaluation.validation import validate_queue
from rq1.orchestration.state_registry import StageRegistry
from rq1.snapshots.builder import build_snapshots
from rq1.snapshots.validation import validate_snapshot_chain


class FinalStageContractsTests(unittest.TestCase):
    def test_acquisition_rejects_leakage_and_failed_source(self) -> None:
        attempts = [AcquisitionAttempt("r", "train:one", "a1", "failed", None, None)]
        operations = [SkillOperation(1, "create", "skill", "x", "train:one", "a1", "log")]
        self.assertTrue(validate_history(attempts, operations))

    def test_acquisition_plan_is_train_only_and_deterministic(self) -> None:
        runner = AcquisitionRunner(Path.cwd())
        plan = runner.plan([{"task_id": "train:b", "split": "train"}, {"task_id": "train:a", "split": "train"}], "r")
        self.assertEqual(("train:a", "train:b"), plan.task_ids)
        with self.assertRaises(Exception): runner.plan([{"task_id": "valid_seen:a", "split": "valid_seen"}])

    def test_snapshots_are_nested_and_l0_empty(self) -> None:
        operations = [SkillOperation(1, "create", "one", "a", "train:a", "a", "l"), SkillOperation(2, "create", "two", "b", "train:b", "b", "l")]
        with tempfile.TemporaryDirectory() as directory:
            manifests = build_snapshots(acquisition_run="r", operations=operations, cutoffs=[("L0", 0), ("L2", 2)], commit="c", destination=Path(directory))
        self.assertEqual([], validate_snapshot_chain(manifests))

    def test_paired_queue_keeps_context_constant(self) -> None:
        items = build_paired_queue(run_id="r", tasks=[{"task_id": "valid_unseen:task", "task_family": "heat", "split": "valid_unseen"}], snapshots=[{"snapshot_id": "L0", "directory_sha256": "0"}, {"snapshot_id": "L1", "directory_sha256": "1"}], checkpoint={"checkpoint_id": "cp", "observable_state_digest": "c"}, perturbation={"perturbation_id": "p", "observable_post_state_digest": "p"}, context_digest="ctx", repetitions=1, seeds=[7])
        self.assertEqual([], validate_queue(items))
        self.assertEqual(2, len(items))

    def test_old_placeholder_final_stage_is_invalidated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            registry = StageRegistry(Path(directory) / "state.json")
            registry.initialize(); states = registry.status(); states["acquisition"].status = "passed"; registry._save(states)
            self.assertEqual("invalidated", registry.status()["acquisition"].status)

