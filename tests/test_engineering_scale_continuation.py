from __future__ import annotations

import copy
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from rq1.university_gpu import engineering_scale as scale
from rq1.university_gpu import scale_continuation as continuation
from rq1.university_gpu.models import atomic_write_json, content_hash, load_json, sha256_file
from rq1.university_gpu.scheduler import Coordinator, RunStore
from rq1.university_gpu.scale_protection import protect_segment
from tests.test_engineering_scale import frozen
from tests.test_engineering_scale_recovery import Worker

ROOT = Path(__file__).resolve().parents[1]


def fixture(backup: Path):
    manifest = frozen()
    manifest.update(source_commit=continuation.ORIGINAL_COMMIT, backup_destination=str(backup))
    approval = manifest["engineering_approval"]
    approval.update(test_only=False, authorization_type="engineering_gpu", profile_id=scale.PROFILE_ID,
        model_tag=scale.MODEL_TAG, model_digest=scale.MODEL_DIGEST, model_quantization=scale.MODEL_QUANTIZATION,
        total_action_budget=200, run_id=manifest["run_id"], conditions=["D0", "D18"],
        approver_id="Yash", approved_by="Yash", owner_reviewer_id="Yash", review_mode="owner_reviewed",
        independent_review=False, approval_provenance="owner-reviewed synthetic fixture",
        source_commit=continuation.ORIGINAL_COMMIT, backup_destination=str(backup), access_end_epoch=1)
    manifest["manifest_sha256"] = scale._manifest_digest(manifest)
    return manifest, approval


def authorization(manifest, identity):
    return {"schema": continuation.SCHEMA, "review_mode": "owner_reviewed", "independent_review": False,
        "reviewer_id": "Yash", "approval_reference": "synthetic-test-only", "approval_provenance": "owner-reviewed fixture",
        "engineering_only": True, "scientific_run": False, "gpu_inference": True, "test_only": False,
        "run_id": manifest["run_id"], "manifest_sha256": manifest["manifest_sha256"],
        "original_source_commit": manifest["source_commit"], "continuation_source_commit": "b" * 40,
        "backup_destination": manifest["backup_destination"], "checkpoint": identity, "access_end_epoch": time.time() + 10000}


def pins(manifest, approval, auth):
    return {"RQ1_ENGINEERING_AUTHORITY_MODE": "owner_reviewed",
        "RQ1_OWNER_REVIEWED_ENGINEERING_APPROVAL_SHA256": content_hash(approval),
        "RQ1_OWNER_REVIEWED_ENGINEERING_MANIFEST_SHA256": manifest["manifest_sha256"],
        continuation.PIN_ENV: content_hash(auth)}


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name); self.backup = self.base / "backup"; self.backup.mkdir()
        self.m, self.approval = fixture(self.backup)
        self.identity = {"completed_count": 218, "completed_unit_keys_sha256": "a" * 64,
            "checkpoint_manifests_sha256": "b" * 64, "receipt_sha256": "c" * 64, "receipt_archive_sha256": "d" * 64}
        self.auth = authorization(self.m, self.identity)

    def verify(self, auth=None, manifest=None, identity=None, cutoff=None):
        return continuation.verify_authorization(ROOT, manifest or self.m, self.approval, auth or self.auth,
            identity or self.identity, cutoff_epoch=cutoff or time.time() + 6000)

    def test_valid_owner_review_and_original_expired_approval(self):
        with patch.dict(os.environ, pins(self.m, self.approval, self.auth)), patch.object(continuation, "check_source"):
            self.assertEqual(self.verify(), content_hash(self.auth))

    def test_rejects_changed_scope_checkpoint_deadline_and_pin(self):
        for changes in ({"independent_review": True}, {"reviewer_id": "other"}, {"scientific_run": True},
                        {"manifest_sha256": "f" * 64}, {"original_source_commit": "e" * 40},
                        {"backup_destination": "/wrong"}, {"access_end_epoch": 1}, {"test_only": True},
                        {"checkpoint": dict(self.identity, completed_count=219)}):
            changed = dict(self.auth, **changes)
            with patch.dict(os.environ, pins(self.m, self.approval, changed)), patch.object(continuation, "check_source"):
                with self.assertRaises(continuation.ContinuationError): self.verify(auth=changed)
        with patch.dict(os.environ, dict(pins(self.m, self.approval, self.auth), **{continuation.PIN_ENV:"0"*64})):
            with self.assertRaises(continuation.ContinuationError): self.verify()

    def test_rejects_wrong_original_pin_and_non_owner_mode(self):
        for changes in ({"RQ1_ENGINEERING_AUTHORITY_MODE":"independent"},
                        {"RQ1_OWNER_REVIEWED_ENGINEERING_MANIFEST_SHA256":"0"*64},
                        {"RQ1_OWNER_REVIEWED_ENGINEERING_APPROVAL_SHA256":"0"*64}):
            with patch.dict(os.environ, dict(pins(self.m, self.approval, self.auth), **changes)):
                with self.assertRaises(continuation.ContinuationError): self.verify()

    def test_source_commit_mismatch_and_scientific_profile_block(self):
        with patch.dict(os.environ, pins(self.m, self.approval, self.auth)):
            with self.assertRaises(continuation.ContinuationError): self.verify()
            altered = copy.deepcopy(self.m); altered["purpose"] = "scientific"
            with self.assertRaises(continuation.ContinuationError): self.verify(manifest=altered)

    def test_source_ancestry_diff_allowlist_and_dirty_checkout(self):
        import subprocess
        root = self.base / "synthetic-repo"; root.mkdir()
        def git(*args):
            return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True).stdout.strip()
        git("init"); git("config", "user.email", "fixture@example.invalid"); git("config", "user.name", "fixture")
        (root / "base.txt").write_text("base", encoding="utf-8"); git("add", "base.txt"); git("commit", "-m", "base")
        base = git("rev-parse", "HEAD")
        allowed = root / "src/rq1/university_gpu/scale_continuation.py"; allowed.parent.mkdir(parents=True)
        allowed.write_text("fixture", encoding="utf-8"); git("add", "."); git("commit", "-m", "allowed")
        current = git("rev-parse", "HEAD")
        with patch.object(continuation, "ORIGINAL_COMMIT", base):
            continuation.check_source(root, base, current)
            (root / "base.txt").write_text("dirty", encoding="utf-8")
            with self.assertRaises(continuation.ContinuationError): continuation.check_source(root, base, current)
            git("add", "base.txt"); git("commit", "-m", "unrelated")
            with self.assertRaises(continuation.ContinuationError): continuation.check_source(root, base, git("rev-parse", "HEAD"))

    def test_launch_or_failed_retry_is_rejected(self):
        with self.assertRaises(scale.FeasibilityError):
            scale.run_supervised(ROOT, {"run_output_dir":str(self.base)}, self.m,
                approval=self.approval, continuation=self.auth, yes=True, resume=False)
        with self.assertRaises(scale.FeasibilityError):
            scale.run_supervised(ROOT, {"run_output_dir":str(self.base)}, self.m,
                approval=self.approval, continuation=self.auth, yes=True, resume=True, retry_failed=True)

    def test_missing_protected_run_blocks_before_server_start(self):
        devices = [{"uuid": f"GPU-{i}", "index": i, "name": "NVIDIA RTX 5000 Ada Generation",
                    "memory_total_mib": 32760, "memory_free_mib": 30000} for i in range(4)]
        server = Mock()
        with patch.dict(os.environ, pins(self.m, self.approval, self.auth)), \
             patch.object(continuation, "check_source"), patch.object(scale, "require_posix_process_management"), \
             patch.object(scale, "verify_frozen_feasibility_assets", return_value={"status": "PASS"}):
            with self.assertRaises(continuation.ContinuationError):
                scale.run_feasibility_launch(ROOT, {"run_output_dir":str(self.base), "worker_count":4}, self.m,
                    approval=self.approval, continuation=self.auth, resume=True, yes=True,
                    cutoff_epoch=time.time()+6000, machine_report_fn=lambda **kw: {"status":"PASS", "devices":devices},
                    server_factory=server)
        server.assert_not_called()
        self.assertFalse((self.base / self.m["run_id"]).exists())

    def test_original_218_checkpoint_set_remains_bound_after_progress(self):
        root = self.base / "fixture-run"; root.mkdir()
        copy = self.backup / "protected-copy"; copy.mkdir()
        archive = copy / "export.tar.gz"; archive.write_bytes(b"fixture archive")
        keys = [f"{i:064x}" for i in range(218)]
        checkpoint_hashes = {}
        for key in keys:
            path = root / "backups" / "units" / key / "CHECKPOINT-MANIFEST.json"
            path.parent.mkdir(parents=True); path.write_text(key, encoding="utf-8")
            checkpoint_hashes[key] = sha256_file(path)
        receipt = root / "protection" / "0218.json"
        atomic_write_json(receipt, {"unit_keys": keys, "destination": str(copy)}, overwrite=False)
        identity = {"completed_count":218, "completed_unit_keys_sha256":content_hash(keys),
            "checkpoint_manifests_sha256":content_hash(checkpoint_hashes), "receipt_sha256":sha256_file(receipt),
            "receipt_archive_sha256":sha256_file(archive)}
        with patch.object(RunStore, "completed", return_value={key:{} for key in keys}):
            continuation.verify_authorized_checkpoint_preserved(root, self.m, identity)
            archive.write_bytes(b"changed")
            with self.assertRaises(continuation.ContinuationError):
                continuation.verify_authorized_checkpoint_preserved(root, self.m, identity)

    def test_deadline_is_rechecked_after_checkpoint_before_gpu_start(self):
        root = self.base / self.m["run_id"]; RunStore(root, self.m).initialize()
        devices = [{"uuid": f"GPU-{i}", "index": i, "name": "NVIDIA RTX 5000 Ada Generation",
                    "memory_total_mib":32760, "memory_free_mib":30000} for i in range(4)]
        server = Mock()
        with patch.object(continuation, "verify_authorization", side_effect=["ok", continuation.ContinuationError("expired")]), \
             patch.object(continuation, "checkpoint_identity", return_value=self.identity), \
             patch.object(continuation, "verify_authorized_checkpoint_preserved"), \
             patch.object(scale, "require_posix_process_management"), \
             patch.object(scale, "verify_frozen_feasibility_assets", return_value={"status":"PASS"}):
            with self.assertRaisesRegex(continuation.ContinuationError, "expired"):
                scale.run_feasibility_launch(ROOT, {"run_output_dir":str(self.base), "worker_count":4}, self.m,
                    approval=self.approval, continuation=self.auth, resume=True, continuation_start=False, yes=True,
                    cutoff_epoch=time.time()+6000, machine_report_fn=lambda **kw: {"status":"PASS", "devices":devices},
                    server_factory=server)
        server.assert_not_called()

    def test_positive_launch_records_both_sources_without_touching_old_worker_files(self):
        from types import SimpleNamespace
        root = self.base / self.m["run_id"]; RunStore(root, self.m).initialize()
        old = root / "workers" / "worker-1" / "ollama-server.log"
        old.parent.mkdir(parents=True); old.write_text("original log", encoding="utf-8")
        devices = [{"uuid": f"GPU-{i}", "index": i, "name": "NVIDIA RTX 5000 Ada Generation",
                    "memory_total_mib":32760, "memory_free_mib":30000} for i in range(4)]
        class Server:
            def __init__(self, spec, *_args, **_kwargs):
                self.spec = spec; self.process = SimpleNamespace(pid=100)
            def start(self): pass
            def stop(self): pass
        def worker(worker_id, *_args, **_kwargs):
            return SimpleNamespace(worker_id=worker_id, start=lambda:None, stop=lambda:None)
        config = {"run_output_dir":str(self.base), "worker_count":4, "ollama_models_dir":str(self.base),
            "alfworld_data_dir":"fixture", "sbert_cache":"fixture", "tokenizer_path":"fixture", "accum18_library_path":"fixture"}
        outcome = {"status":"paused", "cutoff_reached":False, "completed_units":0, "attempts_started":0, "failures":[]}
        placement = {"passed":True, "assignments":{f"gpu-worker-{i+1}":f"GPU-{i}" for i in range(4)}}
        with patch.dict(os.environ, pins(self.m, self.approval, self.auth)), patch.object(continuation, "check_source"), \
             patch.object(continuation, "checkpoint_identity", return_value=self.identity), \
             patch.object(continuation, "verify_authorized_checkpoint_preserved"), \
             patch.object(scale, "require_posix_process_management"), \
             patch.object(scale, "verify_frozen_feasibility_assets", return_value={"status":"PASS"}), \
             patch("rq1.university_gpu.ops._library_payload", return_value=([], {}, {"Accum-18":{"skills":[]}})), \
             patch("rq1.university_gpu.scheduler.Coordinator.run", return_value=outcome):
            result = scale.run_feasibility_launch(ROOT, config, self.m, approval=self.approval, continuation=self.auth,
                resume=True, yes=True, cutoff_epoch=time.time()+6000,
                machine_report_fn=lambda **kw:{"status":"PASS", "devices":devices}, server_factory=Server,
                worker_factory=worker, probe_fn=lambda *a,**kw:{"status":"PASS"},
                smoke_fn=lambda *a,**kw:{"status":"PASS", "scientific_evidence":False}, placement_fn=lambda *a:placement)
        self.assertEqual(result["run"]["status"], "paused")
        self.assertEqual(old.read_text(encoding="utf-8"), "original log")
        launch = load_json(next((root / "launches").glob("*.json")))
        self.assertEqual(launch["original_source_commit"], continuation.ORIGINAL_COMMIT)
        self.assertEqual(launch["continuation_source_commit"], "b"*40)
        config_file = next((root / "continuation-runtime").glob("*/worker-configs/*.json"))
        provenance = load_json(config_file)["provenance"]
        self.assertEqual(provenance["source_commit"], "b"*40)
        self.assertEqual(provenance["original_source_commit"], continuation.ORIGINAL_COMMIT)

    def test_storage_failure_blocks_before_authorization_record_or_launch(self):
        root = self.base / self.m["run_id"]; RunStore(root, self.m).initialize()
        launch = Mock()
        with patch.object(continuation, "checkpoint_identity", return_value=self.identity), \
             patch.object(continuation, "verify_authorization", return_value="f"*64), \
             patch.object(scale, "verify_frozen_feasibility_assets", side_effect=scale.FeasibilityError("storage below reserve")), \
             patch.object(scale, "run_feasibility_launch", launch):
            with self.assertRaisesRegex(scale.FeasibilityError, "storage"):
                scale.run_supervised(ROOT, {"run_output_dir":str(self.base)}, self.m, approval=self.approval,
                    continuation=self.auth, yes=True, resume=True, cutoff_epoch=time.time()+6000)
        launch.assert_not_called()
        self.assertFalse((root / "continuation-authorizations").exists())

    def test_competing_supervisor_is_not_displaced(self):
        from rq1.university_gpu.scheduler import RunLock, SchedulerError
        root = self.base / self.m["run_id"]; RunStore(root, self.m).initialize()
        lock_path = self.base / (self.m["run_id"] + ".supervisor.lock")
        with RunLock(lock_path, resume=False), patch("rq1.university_gpu.scheduler.require_posix_process_management"), \
             patch("rq1.university_gpu.scheduler._pid_alive", return_value=True):
            with self.assertRaises(SchedulerError):
                scale.run_supervised(ROOT, {"run_output_dir":str(self.base)}, self.m, approval=self.approval,
                    continuation=self.auth, yes=True, resume=True, cutoff_epoch=time.time()+6000)
            self.assertTrue(lock_path.exists())

    def test_missing_or_corrupt_backup_and_unknown_attempt_fail_closed(self):
        m = frozen(acceptance=True); m["backup_destination"] = str(self.backup)
        m["engineering_approval"]["backup_destination"] = str(self.backup); m["manifest_sha256"] = scale._manifest_digest(m)
        store = RunStore(self.base / "run", m)
        Coordinator(store, [Worker("w")]).run(max_attempts=1, storage_minimum_bytes=0, minimum_start_seconds=0)
        with self.assertRaises((FileNotFoundError, continuation.ContinuationError)):
            continuation.checkpoint_identity(store.root, m)
        receipt = protect_segment(ROOT, store.root, store.manifest_path)
        identity = continuation.checkpoint_identity(store.root, m)
        self.assertEqual(identity["completed_count"], 1)
        archive = Path(receipt["destination"]) / "export.tar.gz"; original = archive.read_bytes(); archive.write_bytes(b"test corruption")
        with self.assertRaises(ValueError): continuation.checkpoint_identity(store.root, m)
        archive.write_bytes(original)
        store.append_event({"event": "attempt_started", "unit_key": m["units"][1]["unit_key"], "attempt_id": "unknown-fixture"})
        with self.assertRaisesRegex(continuation.ContinuationError, "unknown"):
            continuation.checkpoint_identity(store.root, m)

    def test_export_records_both_source_identities(self):
        from rq1.university_gpu.snapshot import export_progress, verify_progress
        m = frozen(acceptance=True); m["backup_destination"] = str(self.backup)
        m["engineering_approval"]["backup_destination"] = str(self.backup); m["manifest_sha256"] = scale._manifest_digest(m)
        store = RunStore(self.base / "run", m)
        Coordinator(store, [Worker("w")]).run(max_attempts=1, storage_minimum_bytes=0, minimum_start_seconds=0)
        auth = {"continuation_source_commit": "b" * 40}
        atomic_write_json(store.root / "launches" / "fixture.json", {"launch_id":"fixture", "manifest_sha256":m["manifest_sha256"],
            "original_source_commit":m["source_commit"], "continuation_source_commit":"b"*40,
            "continuation_authorization":auth}, overwrite=False)
        destination = self.base / "export"
        export_progress(ROOT, store.root, store.manifest_path, destination)
        verify_progress(destination)
        row = load_json(destination / "EXPORT-MANIFEST.json")["continuation_launches"][0]
        self.assertEqual(row["original_source_commit"], m["source_commit"])
        self.assertEqual(row["continuation_source_commit"], "b"*40)
        self.assertEqual(row["authorization_sha256"], content_hash(auth))

    @patch("os.fsync", return_value=None)  # Synthetic speed-up; durability is covered by separate failure-injection tests.
    def test_synthetic_218_to_500_supervised_continuation(self, _mock_fsync):
        run_root = self.base / self.m["run_id"]
        store = RunStore(run_root, self.m); workers = [Worker(str(i)) for i in range(4)]
        first = Coordinator(store, workers).run(max_attempts=218, storage_minimum_bytes=0, minimum_start_seconds=0)
        self.assertEqual(first["completed_units"], 218)
        protect_segment(ROOT, run_root, store.manifest_path)
        identity = continuation.checkpoint_identity(run_root, self.m)
        before = {str(p.relative_to(run_root)): sha256_file(p) for p in run_root.rglob("*") if p.is_file()}
        auth = authorization(self.m, identity)
        # A full Windows filesystem integration may take longer than a real access
        # window. This synthetic-only deadline must not trigger the cutoff test.
        auth["access_end_epoch"] = time.time() + 100000
        def fake_launch(root, config, manifest, **kwargs):
            completed = len(store.completed()); boundary = min(500, (completed // 50 + 1) * 50)
            outcome = Coordinator(store, workers).run(resume=True, retry_failed=False, max_attempts=boundary-completed,
                storage_minimum_bytes=0, minimum_start_seconds=0)
            return {"run": outcome}
        with patch.dict(os.environ, pins(self.m, self.approval, auth)), patch.object(continuation, "check_source"), \
             patch.object(scale, "verify_frozen_feasibility_assets", return_value={"status":"PASS"}), \
             patch.object(scale, "run_feasibility_launch", side_effect=fake_launch):
            result = scale.run_supervised(ROOT, {"run_output_dir":str(self.base)}, self.m, approval=self.approval,
                continuation=auth, yes=True, resume=True, retry_failed=False, cutoff_epoch=auth["access_end_epoch"])
        self.assertEqual(result["run"]["completed_units"], 500)
        self.assertEqual(sum(len(worker.assignments) for worker in workers), 500)
        for name, digest in before.items():
            self.assertEqual(sha256_file(run_root / name), digest, name)
