from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rq1.acquisition.models import AcquisitionAttempt, SkillOperation
from rq1.acquisition.runner import AcquisitionRunner, acquisition_units, validate_history
from rq1.evaluation.activation import ActivationError
from rq1.evaluation.runner import evaluation_units, run_resumable_evaluation
from rq1.freeze.models import FreezeValidation
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

    def test_durable_unit_identity_preserves_acquisition_and_evaluation_order(self) -> None:
        runner = AcquisitionRunner(Path.cwd())
        plan = runner.plan([
            {"task_id": "train:b", "split": "train"},
            {"task_id": "train:a", "split": "train"},
        ], "experiment")
        acquisition = acquisition_units(
            plan, initial_library_hash="empty", initial_library_size=0
        )
        self.assertEqual(["train:a", "train:b"], [item.task_id for item in acquisition])
        self.assertEqual([1, 2], [item.task_index for item in acquisition])

        queue = build_paired_queue(
            run_id="experiment",
            tasks=[{"task_id": "valid_unseen:task", "task_family": "heat", "split": "valid_unseen"}],
            snapshots=[{"snapshot_id": "L0", "directory_sha256": "zero"}, {"snapshot_id": "L1", "directory_sha256": "one"}],
            checkpoint={"checkpoint_id": "cp", "observable_state_digest": "checkpoint"},
            perturbation={"perturbation_id": "pert", "observable_post_state_digest": "post"},
            context_digest="context",
            repetitions=1,
            seeds=[7],
        )
        evaluation = evaluation_units(queue, {"L0": 0, "L1": 1})
        self.assertEqual(["L0", "L1"], [item.condition for item in evaluation])
        self.assertEqual([0, 1], [item.library_size for item in evaluation])
        self.assertEqual(2, len({item.run_key for item in evaluation}))

    def test_old_placeholder_final_stage_is_invalidated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            registry = StageRegistry(Path(directory) / "state.json")
            registry.initialize(); states = registry.status(); states["acquisition"].status = "passed"; registry._save(states)
            self.assertEqual("invalidated", registry.status()["acquisition"].status)

    def test_durable_acquisition_still_requires_approved_gates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner = AcquisitionRunner(root)
            plan = runner.plan([{"task_id": "train:one", "split": "train"}], "run")
            called = False

            def executor(unit, context):
                nonlocal called
                called = True

            with patch(
                "rq1.acquisition.runner.validate_acquisition_gates",
                return_value=FreezeValidation(False, ("blocked",), None, None),
            ), self.assertRaises(Exception):
                runner.run_resumable(
                    plan, executor,
                    configuration={"version": "v1", "model_name": "m", "runtime_settings": {}},
                    initial_library_hash="empty",
                )
            self.assertFalse(called)

    def test_durable_evaluation_checks_activation_before_executor(self) -> None:
        called = False

        def executor(unit, context):
            nonlocal called
            called = True

        with tempfile.TemporaryDirectory() as directory, patch(
            "rq1.evaluation.runner.require_runtime_opt_in",
            side_effect=ActivationError("not active"),
        ), self.assertRaises(ActivationError):
            run_resumable_evaluation(
                Path(directory), Path("activation.json"), "run", [], executor,
                configuration={"version": "v1", "model_name": "m", "runtime_settings": {}},
                library_sizes={},
            )
        self.assertFalse(called)
