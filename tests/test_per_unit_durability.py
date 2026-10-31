from __future__ import annotations

import json
import os
import errno
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from tests.test_university_gpu_package import FakeWorker, freeze, make_manifest
import tests.test_feasibility_evidence_regression as evidence_fixtures
from rq1.university_gpu.models import load_json, sha256_file
from rq1.university_gpu.scheduler import Coordinator, RunLock, RunStore, SchedulerError
from rq1.university_gpu.snapshot import (
    SnapshotError, backup_completed_unit, export_progress, verify_progress,
    verify_unit_checkpoint, _fsync_path,
)

ROOT = Path(__file__).resolve().parents[1]


class PerUnitDurabilityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.manifest = freeze(make_manifest())
        self.store = RunStore(self.base / "run", self.manifest)
        self.store.initialize()

    def run_limited(self, limit: int, workers=None):
        workers = workers or [FakeWorker("worker", delay=0.002)]
        return Coordinator(self.store, workers).run(
            max_attempts=limit, storage_minimum_bytes=0, minimum_start_seconds=0,
            heartbeat_seconds=0.01,
        )

    def test_interruption_after_units_1_5_and_11_checkpoints_each_unit(self):
        for completed_count in (1, 5, 11):
            with self.subTest(completed=completed_count):
                if completed_count > 1:
                    self.store = RunStore(self.base / f"run-{completed_count}", self.manifest)
                    self.store.initialize()
                outcome = self.run_limited(completed_count)
                self.assertEqual(outcome["completed_units"], completed_count)
                self.assertEqual(outcome["checkpointed_units"], completed_count)
                self.assertEqual(outcome["uncheckpointed_units"], [])
                for key in self.store.completed():
                    self.assertEqual(verify_unit_checkpoint(self.store, key)["status"], "PASS")

    def test_feasibility_twelve_unit_interruption_and_resume_with_raw_traces(self):
        from tests.test_university_gpu_feasibility import candidate_manifest, make_fake_frozen_manifest, FakeWorker as FeasibilityFakeWorker
        manifest = make_fake_frozen_manifest(candidate_manifest())
        for count in (1, 5, 11):
            with self.subTest(completed=count):
                store = RunStore(self.base / f"feasibility-{count}", manifest)
                coordinator = Coordinator(store, [FeasibilityFakeWorker("synthetic-worker")])
                first = coordinator.run(max_attempts=count, storage_minimum_bytes=0, minimum_start_seconds=0)
                self.assertEqual(first["completed_units"], count)
                self.assertEqual(first["checkpointed_units"], count)
                hashes = {key: value["result_sha256"] for key, value in store.completed().items()}

                destination = self.base / f"feasibility-{count}-export"
                exported = export_progress(ROOT, store.root, store.manifest_path, destination)
                self.assertFalse(exported["complete"])
                self.assertEqual(exported["completed"], count)
                expected_results = [store.completed()[str(unit["unit_key"])]["result"]
                                    for unit in manifest["units"]
                                    if str(unit["unit_key"]) in store.completed()]
                self.assertEqual([json.loads(line) for line in (destination / "results.jsonl")
                                  .read_text(encoding="utf-8").splitlines()], expected_results)
                copied_destination = self.base / f"feasibility-{count}-export-copy"
                shutil.copytree(destination, copied_destination)
                environment = dict(os.environ, PYTHONPATH=str(copied_destination / "analysis-src"))
                checked = subprocess.run(
                    [sys.executable, "-B", "-m", "rq1.cli", "university-gpu", "verify-export",
                     "--destination", str(copied_destination)],
                    cwd=copied_destination / "analysis-src", env=environment,
                    capture_output=True, text=True)
                self.assertEqual(checked.returncode, 0, checked.stdout + checked.stderr)
                verified_copy = verify_progress(copied_destination)
                self.assertFalse(verified_copy["complete"])
                self.assertEqual(verified_copy["completed"], count)

                resumed = coordinator.run(resume=True, max_attempts=1, storage_minimum_bytes=0, minimum_start_seconds=0)
                self.assertEqual(resumed["completed_units"], count + 1)
                self.assertEqual(resumed["checkpointed_units"], count + 1)
                for key, digest in hashes.items():
                    self.assertEqual(store.completed()[key]["result_sha256"], digest)
                    self.assertEqual(verify_unit_checkpoint(store, key)["status"], "PASS")

    def test_two_workers_completing_concurrently_get_distinct_checkpoints(self):
        outcome = self.run_limited(2, [FakeWorker("worker-a", delay=0.02), FakeWorker("worker-b", delay=0.02)])
        completed = self.store.completed()
        self.assertEqual(outcome["completed_units"], 2)
        self.assertEqual(len(completed), 2)
        self.assertEqual(outcome["checkpointed_units"], 2)
        self.assertEqual(len({item["attempt_id"] for item in completed.values()}), 2)
        self.assertTrue(all(verify_unit_checkpoint(self.store, key)["status"] == "PASS" for key in completed))

    def test_feasibility_two_workers_checkpoint_distinct_raw_attempts(self):
        from tests.test_university_gpu_feasibility import candidate_manifest, make_fake_frozen_manifest, FakeWorker as FeasibilityFakeWorker
        store = RunStore(self.base / "feasibility-concurrent", make_fake_frozen_manifest(candidate_manifest()))
        outcome = Coordinator(store, [FeasibilityFakeWorker("worker-a"), FeasibilityFakeWorker("worker-b")]).run(
            max_attempts=2, storage_minimum_bytes=0, minimum_start_seconds=0)
        self.assertEqual(outcome["completed_units"], 2)
        self.assertEqual(outcome["checkpointed_units"], 2)
        self.assertEqual(len({item["attempt_id"] for item in store.completed().values()}), 2)

    def test_interruption_during_checkpoint_publish_preserves_result_and_resume_repairs(self):
        real_rename = Path.rename
        def interrupt_checkpoint(path, target):
            if path.parent.name == "units" and path.name.startswith(".partial-"):
                raise OSError("simulated crash before checkpoint rename")
            return real_rename(path, target)
        with patch("rq1.university_gpu.snapshot.Path.rename", new=interrupt_checkpoint):
            failed = self.run_limited(1)
        self.assertEqual(failed["status"], "durability_failed")
        self.assertEqual(failed["completed_units"], 1)
        self.assertEqual(len(self.store.completed()), 1)
        self.assertTrue(list((self.store.root / "backups" / "units").glob(".partial-*")))
        key = next(iter(self.store.completed()))
        result_before = sha256_file(self.store.results_dir / f"{key}.json")
        resumed = Coordinator(self.store, [FakeWorker("resume")]).run(
            resume=True, max_attempts=1, storage_minimum_bytes=0, minimum_start_seconds=0,
        )
        self.assertEqual(resumed["status"], "paused")
        self.assertEqual(sha256_file(self.store.results_dir / f"{key}.json"), result_before)
        self.assertEqual(verify_unit_checkpoint(self.store, key)["status"], "PASS")

    def test_unpublished_partial_directory_is_ignored_and_not_counted(self):
        outcome = self.run_limited(1)
        key = next(iter(self.store.completed()))
        units_dir = self.store.root / "backups" / "units"
        partial = units_dir / f".partial-{key}-truncated"
        partial.mkdir(parents=True)
        (partial / "CHECKPOINT-MANIFEST.json").write_text("{truncated", encoding="utf-8")
        self.assertEqual(outcome["checkpointed_units"], 1)
        self.assertEqual(verify_unit_checkpoint(self.store, key)["status"], "PASS")
        self.assertTrue(partial.exists())

    def test_existing_checkpoint_with_missing_or_corrupt_metadata_fails_closed(self):
        self.run_limited(1)
        key = next(iter(self.store.completed()))
        checkpoint = self.store.root / "backups" / "units" / key
        metadata = checkpoint / "CHECKPOINT-MANIFEST.json"
        original = metadata.read_bytes()
        metadata.write_text("{}", encoding="utf-8")
        before = sha256_file(self.store.results_dir / f"{key}.json")
        outcome = Coordinator(self.store, [FakeWorker("must-not-replace")]).run(
            resume=True, max_attempts=1, storage_minimum_bytes=0, minimum_start_seconds=0,
        )
        self.assertEqual(outcome["status"], "durability_failed")
        self.assertEqual(outcome["backup_status"], "failed")
        self.assertEqual(sha256_file(self.store.results_dir / f"{key}.json"), before)
        self.assertEqual(metadata.read_text(encoding="utf-8"), "{}")
        with self.assertRaises(SnapshotError):
            verify_unit_checkpoint(self.store, key)
        metadata.write_bytes(original)

    def test_orphan_checkpoint_without_authoritative_result_is_not_rerun(self):
        self.run_limited(1)
        key = next(iter(self.store.completed()))
        result = self.store.results_dir / f"{key}.json"
        result.unlink()
        for event in self.store.events_dir.glob("*.json"):
            if load_json(event).get("event") == "unit_completed":
                event.unlink()
        outcome = Coordinator(self.store, [FakeWorker("must-not-rerun")]).run(
            resume=True, max_attempts=1, storage_minimum_bytes=0, minimum_start_seconds=0,
        )
        self.assertEqual(outcome["status"], "durability_failed")
        self.assertEqual(outcome["attempts_started"], 0)
        self.assertFalse(result.exists())
        self.assertTrue(any("without matching authoritative" in row["error"] for row in outcome["failures"]))

    def test_completed_event_without_result_is_never_rerun_even_before_checkpoint(self):
        with patch("rq1.university_gpu.snapshot.backup_completed_unit", side_effect=OSError("checkpoint interrupted")):
            outcome = self.run_limited(1)
        self.assertEqual(outcome["status"], "durability_failed")
        key = next(iter(self.store.completed()))
        (self.store.results_dir / f"{key}.json").unlink()
        with self.assertRaisesRegex(SchedulerError, "completion event exists without"):
            Coordinator(self.store, [FakeWorker("must-not-rerun")]).run(
                resume=True, max_attempts=1, storage_minimum_bytes=0, minimum_start_seconds=0)
        self.assertFalse((self.store.results_dir / f"{key}.json").exists())
        self.assertTrue((self.store.root / "STOP").exists())
        from rq1.university_gpu.ops import status
        observed = status(self.store.root, self.store.manifest_path)
        self.assertEqual(observed["durability_status"], "durability_failed")
        self.assertEqual(observed["completion_events_without_results"], [key])
        validation = Coordinator(self.store, [FakeWorker("offline")]).validate()
        self.assertTrue(any("completion event exists without" in issue for issue in validation["problems"]))

    def test_resume_and_repeated_resume_preserve_single_authority_and_checkpoint(self):
        self.run_limited(1)
        first = self.store.completed()
        first_key = next(iter(first))
        result_path = self.store.results_dir / f"{first_key}.json"
        original = sha256_file(result_path)
        resumed = Coordinator(self.store, [FakeWorker("resume")]).run(
            resume=True, storage_minimum_bytes=0, minimum_start_seconds=0,
        )
        self.assertEqual(resumed["execution_status"], "completed")
        complete_hashes = {p.name: sha256_file(p) for p in self.store.results_dir.glob("*.json")}
        again = Coordinator(self.store, [FakeWorker("resume-again")]).run(
            resume=True, storage_minimum_bytes=0, minimum_start_seconds=0,
        )
        self.assertEqual(again["execution_status"], "completed")
        self.assertEqual({p.name: sha256_file(p) for p in self.store.results_dir.glob("*.json")}, complete_hashes)
        self.assertEqual(sha256_file(result_path), original)
        self.assertEqual(len(self.store.completed()), len(complete_hashes))
        self.assertEqual(again["checkpointed_units"], len(complete_hashes))

    def test_resume_retries_missing_final_full_snapshot_without_rerunning_units(self):
        from tests.test_university_gpu_feasibility import (
            candidate_manifest, make_fake_frozen_manifest, FakeWorker as FeasibilityFakeWorker,
        )
        from rq1.university_gpu import snapshot
        manifest = make_fake_frozen_manifest(candidate_manifest())
        store = RunStore(self.base / "feasibility-final-snapshot", manifest)
        coordinator = Coordinator(store, [FeasibilityFakeWorker("synthetic-worker")])
        real_backup_checkpoint = snapshot.backup_checkpoint
        calls = []
        def fail_second(store_arg):
            calls.append(1)
            if len(calls) == 2:
                raise OSError(errno.EIO, "simulated final full snapshot failure")
            return real_backup_checkpoint(store_arg)
        with patch("rq1.university_gpu.snapshot.backup_checkpoint", new=fail_second):
            first = coordinator.run(
                max_attempts=12, storage_minimum_bytes=0, minimum_start_seconds=0,
                heartbeat_seconds=0.01,
            )
        self.assertEqual(first["completed_units"], 12)
        self.assertEqual(first["execution_status"], "completed")
        self.assertEqual(first["status"], "durability_failed")
        self.assertEqual(first["full_snapshot_status"], "failed")
        self.assertEqual(len(calls), 2)
        self.assertTrue((store.root / "STOP").exists())
        before = {key: value["result_sha256"] for key, value in store.completed().items()}
        self.assertEqual(len(before), 12)
        # The accepted 10-unit boundary bundle survives; the failed final bundle does not.
        self.assertEqual(len(list((store.root / "backups").glob("checkpoint-0010-*"))), 1)
        self.assertEqual(len(list((store.root / "backups").glob("checkpoint-0012-*"))), 0)

        resumed = coordinator.run(
            resume=True, max_attempts=1, storage_minimum_bytes=0, minimum_start_seconds=0,
            heartbeat_seconds=0.01,
        )
        self.assertEqual(resumed["attempts_started"], 0)
        self.assertEqual(resumed["completed_units"], 12)
        self.assertEqual(resumed["full_snapshot_status"], "succeeded")
        self.assertNotEqual(resumed["status"], "durability_failed")
        self.assertEqual(resumed["execution_status"], "completed")
        self.assertEqual({key: value["result_sha256"] for key, value in store.completed().items()}, before)
        final_snapshots = list((store.root / "backups").glob("checkpoint-0012-*"))
        self.assertEqual(len(final_snapshots), 1)
        final_snapshot = final_snapshots[0]
        final_manifest = load_json(final_snapshot / "CHECKPOINT-MANIFEST.json")

        again = coordinator.run(
            resume=True, max_attempts=1, storage_minimum_bytes=0, minimum_start_seconds=0,
            heartbeat_seconds=0.01,
        )
        self.assertEqual(again["attempts_started"], 0)
        self.assertEqual(again["completed_units"], 12)
        self.assertEqual(again["full_snapshot_status"], "succeeded")
        self.assertNotEqual(again["status"], "durability_failed")
        self.assertEqual({key: value["result_sha256"] for key, value in store.completed().items()}, before)
        self.assertEqual(list((store.root / "backups").glob("checkpoint-0012-*")), [final_snapshot])
        self.assertEqual(load_json(final_snapshot / "CHECKPOINT-MANIFEST.json"), final_manifest)

    def test_corrupt_same_count_full_snapshot_fails_closed(self):
        from rq1.university_gpu.snapshot import backup_checkpoint
        for index, mutation in enumerate(("metadata", "file")):
            with self.subTest(mutation=mutation):
                store = RunStore(self.base / f"corrupt-full-snapshot-{index}", self.manifest)
                Coordinator(store, [FakeWorker("worker")]).run(
                    max_attempts=10, storage_minimum_bytes=0, minimum_start_seconds=0)
                snapshots = list((store.root / "backups").glob("checkpoint-0010-*"))
                self.assertEqual(len(snapshots), 1)
                snapshot = snapshots[0]
                metadata = snapshot / "CHECKPOINT-MANIFEST.json"
                if mutation == "metadata":
                    before = metadata.read_bytes()
                    metadata.write_text("{malformed", encoding="utf-8")
                else:
                    report = load_json(metadata)
                    victim = snapshot / report["files"][0]["path"]
                    before = victim.read_bytes()
                    victim.write_bytes(before + b"tampered")
                with self.assertRaises(SnapshotError):
                    backup_checkpoint(store)
                self.assertEqual(list((store.root / "backups").glob("checkpoint-0010-*")), [snapshot])
                if mutation == "metadata":
                    self.assertEqual(metadata.read_bytes(), b"{malformed")
                else:
                    self.assertEqual((snapshot / report["files"][0]["path"]).read_bytes(),
                                     before + b"tampered")

    def test_valid_stale_same_count_full_snapshot_is_preserved_beside_new_bundle(self):
        from rq1.university_gpu.snapshot import backup_checkpoint
        self.run_limited(10)
        snapshots = list((self.store.root / "backups").glob("checkpoint-0010-*"))
        self.assertEqual(len(snapshots), 1)
        stale = snapshots[0]
        stale_manifest = load_json(stale / "CHECKPOINT-MANIFEST.json")
        key, envelope = next(iter(self.store.completed().items()))
        event_path = self.store.append_event({
            "event": "attempt_heartbeat", "unit_key": key,
            "attempt_id": envelope["attempt_id"], "worker_id": envelope["worker_id"]})
        current = backup_checkpoint(self.store)
        self.assertNotEqual(current, stale)
        self.assertEqual(sorted((self.store.root / "backups").glob("checkpoint-0010-*")),
                         sorted([stale, current]))
        self.assertEqual(load_json(stale / "CHECKPOINT-MANIFEST.json"), stale_manifest)
        event_rel = event_path.relative_to(self.store.root).as_posix()
        self.assertNotIn(event_rel, {row["path"] for row in stale_manifest["files"]})
        current_manifest = load_json(current / "CHECKPOINT-MANIFEST.json")
        self.assertIn(event_rel, {row["path"] for row in current_manifest["files"]})

    def test_checkpoint_metadata_cannot_omit_a_raw_trace_even_with_rehashed_metadata(self):
        fixture = evidence_fixtures.EvidenceRegressionTests("test_valid_partial_export_is_portable_and_resume_keeps_authority")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.publish()
        key = fixture.unit["unit_key"]
        checkpoint = backup_completed_unit(fixture.store, key)
        metadata = checkpoint / "CHECKPOINT-MANIFEST.json"
        value = load_json(metadata)
        value["files"] = [row for row in value["files"] if not row["path"].endswith("result/retrieval.jsonl")]
        from rq1.university_gpu.models import content_hash
        value["files_sha256"] = content_hash(value["files"])
        metadata.write_text(json.dumps(value), encoding="utf-8")
        with self.assertRaisesRegex(SnapshotError, "inventory is incomplete"):
            verify_unit_checkpoint(fixture.store, key)

    def test_file_fsync_errors_are_not_suppressed(self):
        path = self.base / "fsync-test"
        path.write_bytes(b"checkpoint data")
        with patch("rq1.university_gpu.snapshot.os.fsync", side_effect=OSError(errno.EIO, "I/O error")):
            with self.assertRaises(OSError):
                _fsync_path(path)

    def test_final_checkpoint_verification_failure_is_not_reported_as_durable(self):
        from rq1.university_gpu import snapshot
        verify = snapshot.verify_unit_checkpoint
        calls = []
        def fail_final(store, key):
            calls.append(key)
            if len(calls) == 2:
                raise SnapshotError("simulated final verification failure")
            return verify(store, key)
        with patch.object(snapshot, "verify_unit_checkpoint", side_effect=fail_final):
            outcome = self.run_limited(1)
        self.assertEqual(outcome["status"], "durability_failed")
        self.assertEqual(outcome["unit_checkpoint_status"], "failed")
        self.assertEqual(outcome["backup_status"], "failed")
        self.assertTrue(any(row.get("failure_class") == "checkpoint_verification" for row in outcome["failures"]))

    def test_non_object_checkpoint_metadata_is_visible_in_status(self):
        self.run_limited(1)
        key = next(iter(self.store.completed()))
        metadata = self.store.root / "backups" / "units" / key / "CHECKPOINT-MANIFEST.json"
        metadata.write_text("[]", encoding="utf-8")
        from rq1.university_gpu.ops import status
        observed = status(self.store.root, self.store.manifest_path)
        self.assertEqual(observed["durability_status"], "durability_failed")
        self.assertIn(key, observed["local_unit_checkpoints"]["invalid"])

    def test_concurrent_stale_lock_takeovers_have_only_one_owner(self):
        import socket
        lock_path = self.base / "stale.lock"
        lock_path.write_text(json.dumps({"pid": 424242, "host": socket.gethostname()}), encoding="utf-8")
        gate = threading.Barrier(2)
        release = threading.Event()
        def attempt():
            gate.wait(timeout=5)
            try:
                with RunLock(lock_path, resume=True):
                    release.wait(timeout=5)
                    return "acquired"
            except SchedulerError:
                release.set()
                return "refused"
        with patch("rq1.university_gpu.scheduler.require_posix_process_management"), \
                patch("rq1.university_gpu.scheduler._pid_alive", side_effect=lambda pid: pid != 424242):
            with ThreadPoolExecutor(max_workers=2) as pool:
                outcomes = list(pool.map(lambda _: attempt(), range(2)))
        self.assertCountEqual(outcomes, ["acquired", "refused"])

    def test_full_snapshot_flush_failure_is_visible(self):
        from rq1.university_gpu.snapshot import backup_checkpoint
        with patch("rq1.university_gpu.snapshot._fsync_checkpoint_tree", side_effect=OSError(errno.EIO, "flush failed")):
            with self.assertRaises(OSError):
                backup_checkpoint(self.store)
        self.assertTrue(list((self.store.root / "backups").glob(".partial-*")))

    def test_failure_after_result_publication_is_not_a_worker_failure(self):
        original = self.store.append_event
        def interrupt(value):
            if value.get("event") == "unit_completed":
                raise OSError("simulated event persistence failure")
            return original(value)
        with patch.object(self.store, "append_event", side_effect=interrupt):
            outcome = self.run_limited(1)
        self.assertEqual(outcome["status"], "durability_failed")
        self.assertEqual(len(self.store.completed()), 1)
        self.assertTrue(any(row.get("failure_class") == "durability" for row in outcome["failures"]))
        self.assertFalse(any(row.get("event") == "attempt_failed" for row in self.store.event_rows()))
        resumed = Coordinator(self.store, [FakeWorker("resume")]).run(
            resume=True, max_attempts=1, storage_minimum_bytes=0, minimum_start_seconds=0)
        self.assertEqual(resumed["checkpointed_units"], 2)

    def test_external_anchor_mismatch_blocks_checkpoint_publication(self):
        fixture = evidence_fixtures.EvidenceRegressionTests("test_valid_partial_export_is_portable_and_resume_keeps_authority")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.publish()
        from rq1.university_gpu.evidence_anchor import create_anchors
        anchor_dir = fixture.base / "anchors"
        create_anchors(fixture.store.root, fixture.store.manifest_path, anchor_dir)
        anchor = next(anchor_dir.glob("*.json"))
        anchor.write_text("{}", encoding="utf-8")
        with patch.dict(os.environ, {"RQ1_EVIDENCE_ANCHOR_DIR": str(anchor_dir)}):
            with self.assertRaisesRegex(SchedulerError, "external evidence anchor"):
                backup_completed_unit(fixture.store, fixture.unit["unit_key"])

    def test_backup_destination_failure_is_visible_and_partial_export_still_works(self):
        with patch("rq1.university_gpu.snapshot.shutil.copy2", side_effect=OSError("destination unavailable")):
            outcome = self.run_limited(1)
        self.assertEqual(outcome["status"], "durability_failed")
        self.assertEqual(outcome["backup_status"], "failed")
        self.assertEqual(outcome["completed_units"], 1)
        self.assertEqual(outcome["checkpointed_units"], 0)
        destination = self.base / "partial-export"
        exported = export_progress(ROOT, self.store.root, self.store.manifest_path, destination)
        self.assertFalse(exported["complete"])
        self.assertEqual(len(exported["source_checkpoint_status_at_export"]["missing"]), 1)
        self.assertEqual(verify_progress(destination)["status"], "PASS")
        self.assertEqual(verify_progress(destination)["completed"], 1)

    def test_symlinked_checkpoint_path_is_rejected_before_mutation(self):
        outside = self.base / "outside"
        outside.mkdir()
        backups = self.store.root / "backups"
        try:
            backups.symlink_to(outside, target_is_directory=True)
        except OSError as exc:
            self.skipTest(f"symlinks unavailable on this host: {exc}")
        with self.assertRaises(SchedulerError):
            self.store.path("backups/units")
        self.assertEqual(list(outside.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
