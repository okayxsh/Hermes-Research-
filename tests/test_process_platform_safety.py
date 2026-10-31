from __future__ import annotations

import json
import socket
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from rq1.experiment.persistence import ExperimentLock
from rq1.orchestration.locks import StageLock
from rq1.university_gpu import feasibility
from rq1.university_gpu.models import WorkerSpec
from rq1.university_gpu.ollama import OllamaServer
from rq1.university_gpu.ops import run_design_launch
from rq1.university_gpu.scheduler import RunLock, SchedulerError, _pid_alive
from rq1.university_gpu.worker_process import WorkerProcessClient
from rq1.utils.process_safety import UnsupportedProcessManagementError


class ProcessPlatformSafetyTests(unittest.TestCase):
    def _lock_fixtures(self, root: Path) -> tuple[RunLock, ExperimentLock, StageLock]:
        run_path = root / "RUN.lock"
        run_path.write_text(json.dumps({"pid": 42, "host": socket.gethostname()}), encoding="utf-8")
        experiment_path = root / "experiment.lock"
        experiment_path.write_text(json.dumps({"pid": 42, "hostname": socket.gethostname()}), encoding="utf-8")
        stage_path = root / "stage.lock"
        stage_path.write_text("42", encoding="utf-8")
        return RunLock(run_path, resume=True), ExperimentLock(experiment_path), StageLock(stage_path)

    def test_native_windows_pid_probes_fail_before_os_kill(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_lock, experiment_lock, stage_lock = self._lock_fixtures(root)
            kill = Mock()
            with patch("rq1.utils.process_safety.os.name", "nt"), \
                    patch("rq1.university_gpu.scheduler.os.kill", kill):
                with self.assertRaises(UnsupportedProcessManagementError):
                    _pid_alive(42)
                with self.assertRaises(UnsupportedProcessManagementError):
                    run_lock.__enter__()
                with self.assertRaises(UnsupportedProcessManagementError):
                    experiment_lock._stale()
                with self.assertRaises(UnsupportedProcessManagementError):
                    stage_lock._stale()
            kill.assert_not_called()

    def test_posix_pid_probes_preserve_live_and_stale_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_lock, experiment_lock, stage_lock = self._lock_fixtures(root)
            with patch("rq1.utils.process_safety.os.name", "posix"), \
                    patch("rq1.university_gpu.scheduler.os.kill") as kill:
                self.assertTrue(_pid_alive(42))
                with self.assertRaises(SchedulerError):
                    run_lock.__enter__()
                self.assertFalse(experiment_lock._stale())
                self.assertFalse(stage_lock._stale())
                self.assertEqual(kill.call_count, 4)
                kill.side_effect = ProcessLookupError
                self.assertFalse(_pid_alive(42))
                with run_lock:
                    self.assertTrue(run_lock.owned)
                self.assertTrue(experiment_lock._stale())
                self.assertTrue(stage_lock._stale())
                self.assertEqual(kill.call_count, 8)
                kill.side_effect = OSError("unknown liveness error")
                self.assertTrue(_pid_alive(42))

    def test_native_windows_launches_reject_before_preflight_or_asset_access(self) -> None:
        root = Path.cwd()
        with patch("rq1.utils.process_safety.os.name", "nt"):
            with patch.object(feasibility, "validate_feasibility_manifest", return_value=[]), \
                    patch.object(feasibility, "_validate_engineering_approval"), \
                    patch.object(feasibility, "verify_frozen_feasibility_assets") as verify_assets:
                with self.assertRaises(UnsupportedProcessManagementError):
                    feasibility.run_feasibility_launch(
                        root, {"worker_count": 4}, {"status": "engineering_frozen"},
                        approval={}, yes=True, cutoff_epoch=10**12,
                    )
                verify_assets.assert_not_called()
            approval = {
                "gpu_inference": True, "scientific_run": True, "manifest_sha256": "a" * 64,
                "approved_by": "operator", "approval_reference": "approval-record",
            }
            with self.assertRaises(UnsupportedProcessManagementError):
                run_design_launch(
                    root, {"run_output_dir": str(root)}, {"manifest_sha256": "a" * 64, "stage": "pilot", "run_id": "test-run"}, approval=approval,
                    yes=True, cutoff_epoch=10**12,
                )

    def test_native_windows_worker_and_ollama_start_reject_before_process_creation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            worker = WorkerProcessClient("worker-1", root / "worker.json", cwd=root, gpu_uuid="GPU-1")
            spec = WorkerSpec("worker-1", "GPU-1", "http://127.0.0.1:11434", 11434, root / "ollama")
            server = OllamaServer(spec, root / "models")
            popen = Mock()
            with patch("rq1.utils.process_safety.os.name", "nt"), \
                    patch("rq1.university_gpu.worker_process.subprocess.Popen", popen), \
                    patch("rq1.university_gpu.ollama.subprocess.Popen", popen):
                with self.assertRaises(UnsupportedProcessManagementError):
                    worker.start()
                with self.assertRaises(UnsupportedProcessManagementError):
                    server.start(runner=popen)
            popen.assert_not_called()

    def test_native_windows_worker_and_ollama_stop_reject_before_signaling(self) -> None:
        root = Path.cwd()
        worker = WorkerProcessClient("worker-1", root / "worker.json", cwd=root, gpu_uuid="GPU-1")
        worker_process = Mock()
        worker_process.poll.return_value = None
        worker.process = worker_process
        server = OllamaServer(
            WorkerSpec("worker-1", "GPU-1", "http://127.0.0.1:11434", 11434, root), root
        )
        server_process = Mock()
        server_process.poll.return_value = None
        server.process = server_process
        with patch("rq1.utils.process_safety.os.name", "nt"):
            with self.assertRaises(UnsupportedProcessManagementError):
                worker.stop()
            with self.assertRaises(UnsupportedProcessManagementError):
                server.stop()
        worker_process.terminate.assert_not_called()
        worker_process.kill.assert_not_called()
        worker_process.stdin.write.assert_not_called()
        server_process.terminate.assert_not_called()
        server_process.kill.assert_not_called()


if __name__ == "__main__":
    unittest.main()
