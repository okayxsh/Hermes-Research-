from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rq1.experiment.models import ExperimentUnit, RunFailure, RunOutcome
from rq1.experiment.persistence import (
    CompatibilityError,
    DuplicateResultError,
    ExperimentStateError,
    ExperimentStore,
    durable_append_jsonl,
)
from rq1.experiment.runner import DurableExperimentRunner, RunnerOptions


class SimulatedCrash(RuntimeError):
    pass


class ExperimentResumeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "pyproject.toml").write_text("[project]\nname='test'\n", encoding="utf-8")
        self.base = self.root / "results" / "final"
        self.config = {
            "version": "test-v1",
            "model_name": "fake-model",
            "runtime_settings": {"temperature": 0},
            "task_manifest_hash": "tasks-v1",
        }

    def tearDown(self) -> None:
        self.temp.cleanup()

    @staticmethod
    def units(phase: str = "evaluation", count: int = 3) -> list[ExperimentUnit]:
        return [
            ExperimentUnit(
                phase=phase,
                task_id=f"task:{index}",
                task_index=index,
                condition="L0",
                seed=index,
                identity={
                    "phase": phase,
                    "task_id": f"task:{index}",
                    "condition": "L0",
                    "seed": index,
                },
                payload={"frozen": True},
                library_name="L0",
                library_size=0,
                library_hash="a" * 64,
            )
            for index in range(1, count + 1)
        ]

    @staticmethod
    def success(unit: ExperimentUnit, context: object) -> RunOutcome:
        return RunOutcome(
            success=True,
            recovery_success=True,
            measurements={"existing_metric": unit.task_index},
            retrieved_skill_ids=("skill-1",),
            retrieval_similarity_scores=(0.75,),
            actions=2,
            steps=2,
            invalid_actions=0,
            latency_ms=10,
        )

    def store(self, run_id: str = "experiment") -> ExperimentStore:
        return ExperimentStore(self.root, run_id, base=self.base)

    def test_fresh_run_checkpoints_after_each_result(self) -> None:
        store = self.store()
        result = DurableExperimentRunner(store, progress=None).run(
            "evaluation", self.units(), self.config, self.success,
            RunnerOptions(max_runs=2),
        )
        self.assertEqual("paused", result["status"])
        self.assertEqual(2, len(store.read_results()))
        checkpoint = json.loads(store.checkpoint_path.read_text(encoding="utf-8"))
        self.assertEqual(2, checkpoint["completed_run_count"])
        self.assertEqual("task:3", checkpoint["next_run"]["task_id"])
        self.assertTrue(store.backup_checkpoint_path.is_file())
        self.assertTrue(store.manifest_path.is_file())

    def test_resume_skips_completed_runs(self) -> None:
        store = self.store()
        DurableExperimentRunner(store, progress=None).run(
            "evaluation", self.units(), self.config, self.success,
            RunnerOptions(max_runs=2),
        )
        called: list[str] = []

        def execute(unit: ExperimentUnit, context: object) -> RunOutcome:
            called.append(unit.task_id)
            return self.success(unit, context)

        result = DurableExperimentRunner(store, progress=None).run(
            "evaluation", self.units(), self.config, execute,
            RunnerOptions(resume=True),
        )
        self.assertEqual(["task:3"], called)
        self.assertEqual("completed", result["status"])
        self.assertEqual(3, len(store.read_results()))

    def test_crash_after_result_write_before_checkpoint_does_not_duplicate(self) -> None:
        store = self.store()
        crashed = False

        def fault(point: str, unit: ExperimentUnit, record: object) -> None:
            nonlocal crashed
            if point == "after_result_fsync" and not crashed:
                crashed = True
                raise SimulatedCrash("power loss")

        with self.assertRaises(SimulatedCrash):
            DurableExperimentRunner(store, fault_hook=fault, progress=None).run(
                "evaluation", self.units(), self.config, self.success
            )
        self.assertEqual(1, len(store.read_results()))
        called: list[str] = []

        def execute(unit: ExperimentUnit, context: object) -> RunOutcome:
            called.append(unit.task_id)
            return self.success(unit, context)

        DurableExperimentRunner(store, progress=None).run(
            "evaluation", self.units(), self.config, execute,
            RunnerOptions(resume=True),
        )
        self.assertEqual(["task:2", "task:3"], called)
        self.assertEqual(3, len(store.read_results()))

    def test_crash_before_result_is_recorded_and_run_restarts(self) -> None:
        store = self.store("pre-result-crash")
        crashed = False

        def fault(point: str, unit: ExperimentUnit, record: object) -> None:
            nonlocal crashed
            if point == "before_result_append" and not crashed:
                crashed = True
                raise SimulatedCrash("power loss before commit")

        with self.assertRaises(SimulatedCrash):
            DurableExperimentRunner(store, fault_hook=fault, progress=None).run(
                "evaluation", self.units(count=1), self.config, self.success
            )
        self.assertEqual([], store.read_results())
        DurableExperimentRunner(store, progress=None).run(
            "evaluation", self.units(count=1), self.config, self.success,
            RunnerOptions(resume=True),
        )
        self.assertEqual(1, len(store.read_results()))
        self.assertEqual("crashed_or_interrupted", store.read_errors()[0]["status"])

    def test_result_is_mirrored_before_post_result_crash(self) -> None:
        store = self.store("mirrored-crash")
        backup = self.root / "persistent"

        def fault(point: str, unit: ExperimentUnit, record: object) -> None:
            if point == "after_result_fsync":
                raise SimulatedCrash("lost process")

        with self.assertRaises(SimulatedCrash):
            DurableExperimentRunner(store, fault_hook=fault, progress=None).run(
                "evaluation", self.units(count=1), self.config, self.success,
                RunnerOptions(backup_dir=backup, require_backup=True),
            )
        mirrored = ExperimentStore(
            self.root, "mirrored-crash", base=backup
        )
        self.assertEqual(1, len(mirrored.read_results()))

    def test_corrupt_primary_checkpoint_uses_backup_and_results(self) -> None:
        store = self.store()
        DurableExperimentRunner(store, progress=None).run(
            "evaluation", self.units(), self.config, self.success,
            RunnerOptions(max_runs=1),
        )
        store.checkpoint_path.write_text("{broken", encoding="utf-8")
        result = DurableExperimentRunner(store, progress=None).run(
            "evaluation", self.units(), self.config, self.success,
            RunnerOptions(resume=True, max_runs=1),
        )
        self.assertEqual(2, result["completed"])
        self.assertEqual(2, len(store.read_results()))

    def test_unexplained_duplicate_is_rejected(self) -> None:
        store = self.store()
        DurableExperimentRunner(store, progress=None).run(
            "evaluation", self.units(phase="evaluation", count=1), self.config,
            self.success,
        )
        first = store.read_results()[0]
        duplicate = dict(first)
        duplicate["attempt_id"] = "unexpected-duplicate"
        durable_append_jsonl(store.results_path, duplicate)
        with self.assertRaises(DuplicateResultError):
            DurableExperimentRunner(store, progress=None).run(
                "evaluation", self.units(count=1), self.config, self.success,
                RunnerOptions(resume=True),
            )

    def test_incompatible_configuration_is_rejected(self) -> None:
        store = self.store()
        DurableExperimentRunner(store, progress=None).run(
            "evaluation", self.units(), self.config, self.success,
            RunnerOptions(max_runs=1),
        )
        changed = {**self.config, "model_name": "different-model"}
        with self.assertRaises(CompatibilityError):
            DurableExperimentRunner(store, progress=None).run(
                "evaluation", self.units(), changed, self.success,
                RunnerOptions(resume=True),
            )

    def test_incompatible_task_condition_seed_or_library_is_rejected(self) -> None:
        store = self.store()
        DurableExperimentRunner(store, progress=None).run(
            "evaluation", self.units(), self.config, self.success,
            RunnerOptions(max_runs=1),
        )
        changed = self.units()
        original = changed[0]
        changed[0] = ExperimentUnit(
            phase=original.phase,
            task_id=original.task_id,
            task_index=original.task_index,
            condition="L1",
            seed=99,
            identity={**original.identity, "condition": "L1", "seed": 99},
            payload=original.payload,
            library_name="L1",
            library_size=1,
            library_hash="b" * 64,
        )
        with self.assertRaises(CompatibilityError):
            DurableExperimentRunner(store, progress=None).run(
                "evaluation", changed, self.config, self.success,
                RunnerOptions(resume=True),
            )

    def test_runtime_version_drift_is_rejected(self) -> None:
        store = self.store()
        DurableExperimentRunner(store, progress=None).run(
            "evaluation", self.units(count=1), self.config, self.success
        )
        manifest = json.loads(store.manifest_path.read_text(encoding="utf-8"))
        manifest["runtime"]["python_version"] = "0.0-incompatible"
        store.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaises(CompatibilityError):
            DurableExperimentRunner(store, progress=None).run(
                "evaluation", self.units(count=1), self.config, self.success,
                RunnerOptions(resume=True),
            )

    def test_resume_unknown_run_does_not_create_output_directory(self) -> None:
        store = self.store("missing")
        with self.assertRaises(ExperimentStateError):
            DurableExperimentRunner(store, progress=None).run(
                "evaluation", self.units(count=1), self.config, self.success,
                RunnerOptions(resume=True),
            )
        self.assertFalse(store.directory.exists())

    def test_interrupted_run_has_no_result_and_restarts(self) -> None:
        store = self.store()

        def interrupt(unit: ExperimentUnit, context: object) -> RunOutcome:
            raise KeyboardInterrupt

        first = DurableExperimentRunner(store, progress=None).run(
            "evaluation", self.units(count=1), self.config, interrupt
        )
        self.assertEqual("interrupted", first["status"])
        self.assertEqual([], store.read_results())
        self.assertEqual("interrupted", store.read_errors()[0]["status"])
        second = DurableExperimentRunner(store, progress=None).run(
            "evaluation", self.units(count=1), self.config, self.success,
            RunnerOptions(resume=True),
        )
        self.assertEqual("completed", second["status"])
        self.assertEqual(1, len(store.read_results()))

    def test_failed_attempt_is_terminal_until_explicit_retry(self) -> None:
        store = self.store()

        def fail_first(unit: ExperimentUnit, context: object) -> RunOutcome:
            if unit.task_index == 1:
                raise RunFailure(
                    "provider error", safe_to_continue=True,
                    mutation_state_known=True,
                )
            return self.success(unit, context)

        first = DurableExperimentRunner(store, progress=None).run(
            "evaluation", self.units(), self.config, fail_first
        )
        self.assertEqual(1, first["failed"])
        normal = DurableExperimentRunner(store, progress=None).run(
            "evaluation", self.units(), self.config, self.success,
            RunnerOptions(resume=True),
        )
        self.assertEqual(0, normal["attempted_this_invocation"])
        retried = DurableExperimentRunner(store, progress=None).run(
            "evaluation", self.units(), self.config, self.success,
            RunnerOptions(resume=True, retry_failed=True),
        )
        self.assertEqual(0, retried["failed"])
        rows = store.read_results()
        self.assertEqual(4, len(rows))
        self.assertEqual(rows[0]["attempt_id"], rows[-1]["supersedes_attempt_id"])

    def test_bounded_failed_retry_reports_next_failed_attempt(self) -> None:
        store = self.store("retry-bounded")

        def fail(unit: ExperimentUnit, context: object) -> RunOutcome:
            raise RunFailure(
                "isolated provider error", safe_to_continue=True,
                mutation_state_known=True,
            )

        DurableExperimentRunner(store, progress=None).run(
            "evaluation", self.units(count=2), self.config, fail
        )
        retried = DurableExperimentRunner(store, progress=None).run(
            "evaluation", self.units(count=2), self.config, self.success,
            RunnerOptions(resume=True, retry_failed=True, max_runs=1),
        )
        self.assertEqual("paused", retried["status"])
        self.assertEqual("task:2", retried["next_run"]["task_id"])

    def test_partial_final_jsonl_tail_is_preserved_and_repaired(self) -> None:
        store = self.store()
        DurableExperimentRunner(store, progress=None).run(
            "evaluation", self.units(count=1), self.config, self.success
        )
        with store.results_path.open("ab") as handle:
            handle.write(b'{"partial":')
        values = store.read_results()
        self.assertEqual(1, len(values))
        self.assertTrue(list((store.logs / "recovery").glob("results-partial-*.bin")))

    def test_interior_jsonl_corruption_is_fatal(self) -> None:
        store = self.store()
        store.initialize()
        store.results_path.write_bytes(b'{"run_key":"one"}\n{broken\n{"run_key":"two"}\n')
        with self.assertRaises(ExperimentStateError):
            store.read_results()

    def test_cooperative_stop_commits_finished_run_only(self) -> None:
        store = self.store()
        runner = DurableExperimentRunner(store, progress=None)

        def stop_after_current(unit: ExperimentUnit, context: object) -> RunOutcome:
            runner.request_stop(15)
            return self.success(unit, context)

        first = runner.run(
            "evaluation", self.units(), self.config, stop_after_current
        )
        self.assertEqual("interrupted", first["status"])
        self.assertEqual(1, len(store.read_results()))
        resumed = DurableExperimentRunner(store, progress=None).run(
            "evaluation", self.units(), self.config, self.success,
            RunnerOptions(resume=True),
        )
        self.assertEqual("completed", resumed["status"])

    def test_uncertain_acquisition_failure_blocks_resume(self) -> None:
        store = self.store()

        def fail(unit: ExperimentUnit, context: object) -> RunOutcome:
            raise RuntimeError("unknown skill write outcome")

        first = DurableExperimentRunner(store, progress=None).run(
            "acquisition", self.units(phase="acquisition", count=2),
            self.config, fail,
        )
        self.assertEqual("blocked", first["status"])
        with self.assertRaises(ExperimentStateError):
            DurableExperimentRunner(store, progress=None).run(
                "acquisition", self.units(phase="acquisition", count=2),
                self.config, self.success, RunnerOptions(resume=True),
            )

    def test_acquisition_interrupt_blocks_unknown_skill_write_state(self) -> None:
        store = self.store("acquisition-interrupt")

        def interrupt(unit: ExperimentUnit, context: object) -> RunOutcome:
            raise KeyboardInterrupt

        result = DurableExperimentRunner(store, progress=None).run(
            "acquisition", self.units(phase="acquisition", count=1),
            self.config, interrupt,
        )
        self.assertEqual("blocked", result["status"])
        checkpoint = store.load_checkpoint()[0]
        self.assertFalse(checkpoint["blocking_error"]["mutation_state_known"])

    def test_acquisition_checkpoint_tracks_verified_library_state(self) -> None:
        store = self.store("acquisition-library")

        def acquire(unit: ExperimentUnit, context: object) -> RunOutcome:
            return RunOutcome(
                success=unit.task_index == 1,
                skill_library_hash_after=str(unit.task_index) * 64,
                library_size_after=unit.task_index,
            )

        result = DurableExperimentRunner(store, progress=None).run(
            "acquisition", self.units(phase="acquisition", count=2),
            self.config, acquire,
        )
        self.assertEqual("completed", result["status"])
        checkpoint = store.load_checkpoint()[0]
        self.assertEqual(2, checkpoint["library_size"])
        self.assertEqual("2" * 64, checkpoint["skill_library_hash"])

    def test_backup_mirrors_critical_state(self) -> None:
        store = self.store()
        backup = self.root / "persistent"
        DurableExperimentRunner(store, progress=None).run(
            "evaluation", self.units(count=1), self.config, self.success,
            RunnerOptions(backup_dir=backup, require_backup=True),
        )
        mirrored = backup / store.experiment_id
        for name in ("checkpoint.json", "results.jsonl", "errors.jsonl", "run_manifest.json"):
            self.assertTrue((mirrored / name).is_file(), name)

    def test_checkpoint_smoke_cli_stops_at_three_then_resumes_at_four(self) -> None:
        from rq1.cli import main

        with patch("rq1.cli._root", return_value=self.root):
            first = main([
                "experiment", "checkpoint-test", "--run-id", "smoke",
                "--max-runs", "3", "--total-runs", "6",
            ])
            second = main([
                "experiment", "checkpoint-test", "--run-id", "smoke",
                "--resume", "--max-runs", "3", "--total-runs", "6",
            ])
        self.assertEqual(0, first)
        self.assertEqual(0, second)
        store = ExperimentStore(
            self.root, "smoke", base=self.root / "results" / "checkpoint-tests"
        )
        rows = store.read_results()
        self.assertEqual(6, len(rows))
        self.assertEqual(6, len({row["run_key"] for row in rows}))
        self.assertEqual("completed", store.load_checkpoint()[0]["status"])

    def test_copied_output_resumes_on_a_new_root(self) -> None:
        import shutil

        store = self.store("portable")
        DurableExperimentRunner(store, progress=None).run(
            "evaluation", self.units(), self.config, self.success,
            RunnerOptions(max_runs=1),
        )
        destination_root = self.root / "replacement"
        destination = destination_root / "results" / "final" / "portable"
        destination.parent.mkdir(parents=True)
        shutil.copytree(store.directory, destination)
        moved = ExperimentStore(destination_root, "portable")
        result = DurableExperimentRunner(moved, progress=None).run(
            "evaluation", self.units(), self.config, self.success,
            RunnerOptions(resume=True),
        )
        self.assertEqual("completed", result["status"])
        self.assertEqual(3, len(moved.read_results()))


if __name__ == "__main__":
    unittest.main()
