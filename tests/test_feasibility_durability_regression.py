"""Offline authority/snapshot regressions; providers are explicit test fixtures."""
from __future__ import annotations

import shutil
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.test_university_gpu_package import FakeWorker, freeze, make_manifest
from rq1.university_gpu.models import load_json
from rq1.university_gpu.scheduler import Coordinator, RunStore
from rq1.university_gpu.snapshot import SnapshotError, backup_checkpoint, export_progress, verify_progress

ROOT = Path(__file__).resolve().parents[1]


class DurabilityRegressionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest = freeze(make_manifest())
        self.store = RunStore(self.root / "run", self.manifest)
        self.store.initialize()

    def _attempt(self, name, closed=False):
        unit = self.manifest["units"][0]
        attempt = self.store.attempts_dir / unit["unit_key"] / name
        attempt.mkdir(parents=True)
        self.store.append_event({"event": "attempt_started", "unit_key": unit["unit_key"],
                                 "attempt_id": name, "attempt_dir": str(attempt)})
        log = attempt / "trace.jsonl"
        log.write_text("closed evidence\n")
        if closed:
            self.store.append_event({"event": "unit_completed", "unit_key": unit["unit_key"],
                                     "attempt_id": name, "error_type": "fixture interruption"})
        return log

    def test_live_writer_is_not_copied_but_closed_completion_trace_is(self):
        closed = self._attempt("closed", closed=True)
        active = self._attempt("active")
        stop = threading.Event()
        started = threading.Event()
        def writer():
            with active.open("a") as stream:
                started.set()
                while not stop.wait(0.001):
                    stream.write("ongoing\n")
                    stream.flush()
        thread = threading.Thread(target=writer)
        thread.start()
        self.addCleanup(thread.join)
        self.addCleanup(stop.set)
        started.wait(2)
        real_copy = shutil.copy2
        def checked_copy(source, target):
            self.assertNotEqual(Path(source), active)
            return real_copy(source, target)
        with patch("rq1.university_gpu.snapshot.shutil.copy2", side_effect=checked_copy):
            checkpoint = backup_checkpoint(self.store)
        report = load_json(checkpoint / "CHECKPOINT-MANIFEST.json")
        self.assertEqual(report["omitted_open_attempt_ids"], ["active"])
        self.assertEqual(report["closed_attempt_ids"], ["closed"])
        self.assertEqual((checkpoint / closed.relative_to(self.store.root)).read_bytes(), closed.read_bytes())
        self.assertFalse((checkpoint / active.relative_to(self.store.root)).exists())

    def test_closed_trace_mutation_aborts_checkpoint_publication(self):
        closed = self._attempt("closed", closed=True)
        real_copy = shutil.copy2
        def mutate(source, target):
            value = real_copy(source, target)
            if Path(source) == closed:
                closed.write_text("mutated after copy")
            return value
        with patch("rq1.university_gpu.snapshot.shutil.copy2", side_effect=mutate):
            with self.assertRaisesRegex(SnapshotError, "changed"):
                backup_checkpoint(self.store)
        self.assertFalse(list((self.store.root / "backups").glob("checkpoint-*")))
        self.assertTrue(list((self.store.root / "backups").glob(".partial-*")))

    def test_failed_live_writer_is_omitted_and_relocated_completed_trace_copies(self):
        closed = self._attempt("closed", closed=True)
        failed = self._attempt("timedout")
        self.store.append_event({"event": "attempt_failed", "attempt_id": "timedout",
                                 "unit_key": self.manifest["units"][0]["unit_key"]})
        relocated = self.root / "relocated"
        shutil.copytree(self.store.root, relocated)
        moved = RunStore(relocated, self.manifest)
        checkpoint = backup_checkpoint(moved)
        report = load_json(checkpoint / "CHECKPOINT-MANIFEST.json")
        self.assertIn("timedout", report["omitted_open_attempt_ids"])
        self.assertTrue((checkpoint / closed.relative_to(self.store.root)).is_file())
        self.assertFalse((checkpoint / failed.relative_to(self.store.root)).exists())

    def test_final_backup_failure_is_distinct_from_completed_execution(self):
        real_backup = backup_checkpoint
        def fail_final(store):
            if len(store.completed()) == len(self.manifest["units"]):
                raise OSError("fixture final backup failure")
            return real_backup(store)
        with patch("rq1.university_gpu.snapshot.backup_checkpoint", side_effect=fail_final):
            outcome = Coordinator(self.store, [FakeWorker("fixture-worker")]).run()
        self.assertEqual(outcome["execution_status"], "completed")
        self.assertEqual(outcome["status"], "durability_failed")
        self.assertEqual(outcome["backup_status"], "failed")
        self.assertEqual(outcome["export_status"], "not_performed")
        self.assertEqual(outcome["independent_copy_status"], "not_verified")
        self.assertEqual(len(self.store.completed()), 36)
        self.assertTrue(Coordinator(self.store, [FakeWorker("check")]).validate()["passed"])

    def test_incomplete_progress_exports_without_complete_merge(self):
        outcome = Coordinator(self.store, [FakeWorker("fixture-worker")]).run(max_attempts=1)
        self.assertEqual(outcome["execution_status"], "paused")
        destination = self.root / "partial-export"
        exported = export_progress(ROOT, self.store.root, self.store.manifest_path, destination)
        self.assertFalse(exported["complete"])
        self.assertEqual(exported["export_status"], "verified_local_copy")
        self.assertEqual(exported["independent_copy_status"], "not_verified")
        verified = verify_progress(destination)
        self.assertFalse(verified["complete"])
        self.assertEqual(verified["completed"], 1)


if __name__ == "__main__":
    unittest.main()
