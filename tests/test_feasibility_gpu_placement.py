from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import rq1.university_gpu.feasibility_placement as placement_module
from rq1.university_gpu.feasibility_placement import capture_feasibility_placement


GPUS = [
    ("w1", "GPU-1", "0000:34:00.0"),
    ("w2", "GPU-2", "0000:ac:00.0"),
    ("w3", "GPU-3", "0000:ca:00.0"),
    ("w4", "GPU-4", "0000:16:00.0"),
]


class FeasibilityPlacementTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.servers = []
        for worker_id, gpu_uuid, pci in GPUS:
            worker_dir = self.root / worker_id
            worker_dir.mkdir()
            log = worker_dir / "ollama-server.log"
            log.write_text(self._log_lines(pci), encoding="utf-8")
            stat = log.stat()
            spec = SimpleNamespace(worker_id=worker_id, gpu_uuid=gpu_uuid, worker_dir=worker_dir,
                                   port=11534 + len(self.servers))
            process = SimpleNamespace(pid=1000 + len(self.servers), poll=lambda: None)
            server = SimpleNamespace(spec=spec, process=process, log_start_offset=0,
                                     log_identity=(stat.st_dev, stat.st_ino))
            self.servers.append(server)

    def tearDown(self) -> None:
        self.temp.cleanup()

    @staticmethod
    def _log_lines(pci: str) -> str:
        return (
            f'time=2026-11-02T09:00:00Z level=INFO source=ggml.go:123 msg="inference compute" '
            f'library=CUDA name=CUDA0 pci_id={pci}\n'
            'time=2026-11-02T09:00:01Z level=INFO source=ggml.go:456 '
            'msg="model weights" device=CUDA0 size="1 GiB"\n'
        )

    def _runner(self, *, mapping: str | None = None, pid_output: str = "", pid_returncode: int = 0):
        def run(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
            if "--query-compute-apps=pid,gpu_uuid,used_memory" in args:
                return subprocess.CompletedProcess(args, pid_returncode, pid_output, "PID evidence unavailable")
            if "pci.bus_id" in " ".join(args):
                output = mapping if mapping is not None else "\n".join(
                    f"{uuid}, {pci}" for _, uuid, pci in GPUS
                )
                return subprocess.CompletedProcess(args, 0, output, "")
            self.fail(f"unexpected nvidia-smi command: {args}")

        return run

    def _capture(self, *, runner=None, pid_capture=None):
        return capture_feasibility_placement(
            self.servers,
            runner=runner or self._runner(),
            pid_capture=pid_capture or (lambda pids: {
                "passed": False, "assignments": {},
                "errors": [f"{worker_id} has no unique observed GPU placement: []" for worker_id in pids],
            }),
            wsl_check=lambda: True,
            owner_check=lambda _server: True,
        )

    def test_valid_fresh_logs_cross_checked_to_four_distinct_live_gpus(self) -> None:
        result = self._capture()
        self.assertTrue(result["passed"], result)
        self.assertEqual(result["method"], "wsl_log_pci")
        self.assertEqual(result["assignments"], {worker: gpu for worker, gpu, _pci in GPUS})
        self.assertEqual(len(set(result["assignments"].values())), 4)

    def test_correct_pid_mapping_is_preferred(self) -> None:
        expected = {worker: gpu for worker, gpu, _pci in GPUS}
        result = self._capture(pid_capture=lambda _pids: {"passed": True, "assignments": expected, "errors": []})
        self.assertTrue(result["passed"], result)
        self.assertEqual(result["method"], "pid")
        self.assertEqual(result["assignments"], expected)

    def test_stale_log_prefix_does_not_count_as_fresh_cuda_evidence(self) -> None:
        first = self.servers[0]
        log = first.spec.worker_dir / "ollama-server.log"
        old = 'msg="inference compute" library=CUDA name=CUDA0 pci_id=0000:34:00.0\n' \
              'msg="model weights" device=CUDA0 size="1 GiB"\n'
        log.write_text(old, encoding="utf-8")
        first.log_start_offset = len(old.encode("utf-8"))
        first.log_identity = (log.stat().st_dev, log.stat().st_ino)
        result = self._capture()
        self.assertFalse(result["passed"])

    def test_missing_cuda_model_loading_evidence_fails(self) -> None:
        first = self.servers[0]
        (first.spec.worker_dir / "ollama-server.log").write_text(
            'msg="inference compute" library=CUDA name=CUDA0 pci_id=0000:34:00.0\n', encoding="utf-8")
        result = self._capture()
        self.assertFalse(result["passed"])

    def test_log_pci_must_match_workers_expected_uuid(self) -> None:
        first = self.servers[0]
        (first.spec.worker_dir / "ollama-server.log").write_text(self._log_lines("0000:ac:00.0"), encoding="utf-8")
        result = self._capture()
        self.assertFalse(result["passed"])

    def test_two_distinct_pci_discovery_lines_in_one_fresh_worker_fail(self) -> None:
        first = self.servers[0]
        log = first.spec.worker_dir / "ollama-server.log"
        log.write_text(self._log_lines("0000:34:00.0") + self._log_lines("0000:ac:00.0"), encoding="utf-8")
        first.log_identity = (log.stat().st_dev, log.stat().st_ino)
        result = self._capture()
        self.assertFalse(result["passed"])

    def test_partial_conflicting_pid_evidence_is_not_ignored(self) -> None:
        result = self._capture(pid_capture=lambda _pids: {
            "passed": False,
            "assignments": {"w1": "GPU-WRONG"},
            "errors": ["conflicting PID UUID"],
        })
        self.assertFalse(result["passed"])

    def test_nonempty_ambiguous_pid_evidence_is_not_replaced_by_log_fallback(self) -> None:
        result = self._capture(pid_capture=lambda _pids: {
            "passed": False,
            "assignments": {},
            "errors": ["w1 has no unique observed GPU placement: ['GPU-1', 'GPU-2']"],
        })
        self.assertFalse(result["passed"])

    def test_fallback_is_refused_outside_wsl(self) -> None:
        result = capture_feasibility_placement(
            self.servers, runner=self._runner(),
            pid_capture=lambda pids: {
                "passed": False, "assignments": {},
                "errors": [f"{worker_id} has no unique observed GPU placement: []" for worker_id in pids],
            },
            wsl_check=lambda: False, owner_check=lambda _server: True,
        )
        self.assertFalse(result["passed"])

    def test_fallback_requires_live_ollama_process_ownership(self) -> None:
        result = capture_feasibility_placement(
            self.servers, runner=self._runner(),
            pid_capture=lambda pids: {
                "passed": False, "assignments": {},
                "errors": [f"{worker_id} has no unique observed GPU placement: []" for worker_id in pids],
            },
            wsl_check=lambda: True,
            owner_check=lambda server: server is not self.servers[0],
        )
        self.assertFalse(result["passed"])

    def test_dead_worker_process_fails(self) -> None:
        self.servers[0].process.poll = lambda: 17
        result = self._capture()
        self.assertFalse(result["passed"])

    def test_replaced_worker_log_identity_fails(self) -> None:
        first = self.servers[0]
        first.log_identity = (first.log_identity[0], first.log_identity[1] + 1)
        result = self._capture()
        self.assertFalse(result["passed"])

    def test_ambiguous_pci_mapping_fails(self) -> None:
        output = "\n".join(("GPU-1, 0000:34:00.0", "GPU-2, 0000:34:00.0",
                              "GPU-3, 0000:ca:00.0", "GPU-4, 0000:16:00.0"))
        result = self._capture(runner=self._runner(mapping=output))
        self.assertFalse(result["passed"])

    def test_missing_or_malformed_live_mapping_fails(self) -> None:
        result = self._capture(runner=self._runner(mapping="GPU-1, not-a-pci-bus-id"))
        self.assertFalse(result["passed"])

    def test_extra_live_gpu_mapping_fails(self) -> None:
        rows = [f"{uuid}, {pci}" for _, uuid, pci in GPUS]
        rows.append("GPU-EXTRA, 0000:ff:00.0")
        result = self._capture(runner=self._runner(mapping="\n".join(rows)))
        self.assertFalse(result["passed"])

    def test_nvidia_smi_timeout_fails_closed(self) -> None:
        def timeout(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
            if "--query-compute-apps=pid,gpu_uuid,used_memory" in args:
                return subprocess.CompletedProcess(args, 0, "", "")
            raise subprocess.TimeoutExpired(args, timeout=20)

        result = self._capture(runner=timeout)
        self.assertFalse(result["passed"])

    def test_load_tensors_cuda_model_buffer_line_is_accepted(self) -> None:
        first = self.servers[0]
        log = first.spec.worker_dir / "ollama-server.log"
        log.write_text(
            'msg="inference compute" library=CUDA name=CUDA0 pci_id=0000:34:00.0\n'
            'load_tensors: CUDA0 model buffer size = 1234.0 MiB\n',
            encoding="utf-8",
        )
        first.log_identity = (log.stat().st_dev, log.stat().st_ino)
        result = self._capture()
        self.assertTrue(result["passed"], result)

    def test_missing_worker_log_fails_closed(self) -> None:
        (self.servers[0].spec.worker_dir / "ollama-server.log").unlink()
        result = self._capture()
        self.assertFalse(result["passed"])

    def _call_owns_log(self, server, *, cuda_uuid: str | None = None, ollama_host: str | None = None) -> bool:
        proc_root = self.root / "mock-proc"
        proc = proc_root / str(server.process.pid)
        fd = proc / "fd"
        fd.mkdir(parents=True, exist_ok=True)
        log = server.spec.worker_dir / "ollama-server.log"
        log_stat = log.stat()
        server.log_identity = (log_stat.st_dev, log_stat.st_ino)
        device = cuda_uuid or server.spec.gpu_uuid
        host = ollama_host or f"127.0.0.1:{server.spec.port}"
        (proc / "environ").write_bytes(
            f"CUDA_VISIBLE_DEVICES={device}\0OLLAMA_HOST={host}\0".encode("ascii"))
        (fd / "1").write_text("mock file descriptor", encoding="ascii")
        original_stat = Path.stat

        def path_factory(value):
            raw = str(value).replace("\\", "/")
            if raw == "/proc":
                return proc_root
            if raw.startswith("/proc/"):
                return proc_root / raw.removeprefix("/proc/")
            return Path(value)

        def stat_with_log_identity(path, *args, **kwargs):
            if path.name == "1" and path.parent.name == "fd":
                return SimpleNamespace(st_dev=log_stat.st_dev, st_ino=log_stat.st_ino)
            return original_stat(path, *args, **kwargs)

        with patch.object(placement_module, "Path", side_effect=path_factory), \
                patch.object(Path, "stat", stat_with_log_identity):
            return placement_module._owns_log(server)

    def test_owns_log_verifies_mocked_proc_environment_and_stdout_identity(self) -> None:
        self.assertTrue(self._call_owns_log(self.servers[0]))

    def test_owns_log_rejects_wrong_cuda_environment(self) -> None:
        self.assertFalse(self._call_owns_log(self.servers[0], cuda_uuid="GPU-OTHER"))

    def test_owns_log_rejects_wrong_endpoint_environment(self) -> None:
        self.assertFalse(self._call_owns_log(self.servers[0], ollama_host="127.0.0.1:9999"))

    def test_owns_log_rejects_dead_process(self) -> None:
        self.servers[0].process.poll = lambda: 9
        self.assertFalse(self._call_owns_log(self.servers[0]))


if __name__ == "__main__":
    unittest.main()
