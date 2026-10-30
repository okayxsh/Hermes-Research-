"""Subprocess-backed episode worker with a private runtime and line protocol."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from queue import Empty, Queue
from pathlib import Path
from typing import Any, Mapping

from rq1.experiment.models import RunFailure
from rq1.university_gpu.scheduler import SchedulerError


class WorkerProcessClient:
    # A failed subprocess or model connection is quarantined for operator review.
    quarantine_on_failure = True
    def __init__(self, worker_id: str, config_path: Path, *, python: str = sys.executable, cwd: Path,
                 gpu_uuid: str, response_timeout: float = 3600,
                 module: str = "rq1.university_gpu.worker") -> None:
        self.worker_id = worker_id
        self.config_path = config_path
        self.python = python
        self.cwd = cwd
        self.gpu_uuid = gpu_uuid
        self.response_timeout = response_timeout
        self.module = module
        self.process: subprocess.Popen[str] | None = None
        self._lock = threading.Lock()
        self._stderr_handle: Any = None

    @staticmethod
    def _readline(stream: Any, *, timeout: float, label: str) -> str:
        result: Queue[str] = Queue(maxsize=1)

        def read() -> None:
            result.put(stream.readline())

        threading.Thread(target=read, name=f"rq1-worker-read-{label}", daemon=True).start()
        try:
            return result.get(timeout=timeout)
        except Empty as exc:
            raise SchedulerError(f"timed out waiting for worker {label}") from exc

    def start(self, *, timeout: float = 900) -> dict[str, Any]:
        if self.process is not None and self.process.poll() is None:
            raise SchedulerError(f"worker process already running: {self.worker_id}")
        env = os.environ.copy()
        env.update({"RQ1_WORKER_ID": self.worker_id, "CUDA_VISIBLE_DEVICES": ""})
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self._stderr_handle = (self.config_path.parent / f"{self.worker_id}.stderr.log").open("ab")
        self.process = subprocess.Popen(
            [self.python, "-m", self.module, "--config", str(self.config_path)],
            cwd=self.cwd, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._stderr_handle,
            text=True, encoding="utf-8", bufsize=1,
        )
        assert self.process.stdout is not None
        try:
            line = self._readline(self.process.stdout, timeout=timeout, label=f"{self.worker_id} startup")
            if not line:
                stderr_path = self.config_path.parent / f"{self.worker_id}.stderr.log"
                stderr = stderr_path.read_text(encoding="utf-8", errors="replace") if stderr_path.is_file() else ""
                raise SchedulerError(f"worker {self.worker_id} exited before ready: {stderr[-2000:]}")
            ready = json.loads(line)
            if ready.get("ready") is not True:
                raise SchedulerError(f"worker {self.worker_id} failed startup: {ready}")
            if ready.get("worker_id") != self.worker_id or ready.get("gpu_uuid") != self.gpu_uuid:
                raise SchedulerError(f"worker {self.worker_id} identity does not match its assigned GPU")
            return ready
        except BaseException:
            self.stop(timeout=2)
            raise

    def execute(self, unit: Mapping[str, Any], attempt_dir: Path, attempt_id: str) -> Mapping[str, Any]:
        with self._lock:
            if self.process is None or self.process.poll() is not None:
                raise SchedulerError(f"worker process is not alive: {self.worker_id}")
            assert self.process.stdin is not None and self.process.stdout is not None
            assignment = {"unit": dict(unit), "attempt_dir": str(attempt_dir), "attempt_id": attempt_id}
            self.process.stdin.write(json.dumps({"command": "execute", "assignment": assignment}, ensure_ascii=False) + "\n")
            self.process.stdin.flush()
            try:
                line = self._readline(self.process.stdout, timeout=self.response_timeout, label=f"{self.worker_id} episode")
            except SchedulerError:
                self.stop(timeout=2)
                raise
            if not line:
                stderr_path = self.config_path.parent / f"{self.worker_id}.stderr.log"
                stderr = stderr_path.read_text(encoding="utf-8", errors="replace") if stderr_path.is_file() else ""
                raise SchedulerError(f"worker {self.worker_id} closed its protocol stream: {stderr[-2000:]}")
            response = json.loads(line)
            if response.get("ok") is not True:
                raise RunFailure(
                    f"worker {self.worker_id} failed unit: {response.get('error_type')}: {response.get('error')}",
                    safe_to_continue=response.get("safe_to_continue") is True,
                    mutation_state_known=response.get("mutation_state_known") is True,
                    details={"worker_error_type": response.get("error_type"),
                             **dict(response.get("failure_details") or {})},
                )
            return response["result"]

    def stop(self, *, timeout: float = 10) -> None:
        if self.process is None:
            return
        if self.process.poll() is None:
            try:
                if self.process.stdin is not None:
                    self.process.stdin.write('{"command":"shutdown"}\n')
                    self.process.stdin.flush()
                self.process.wait(timeout=timeout)
            except (OSError, subprocess.TimeoutExpired):
                self.process.terminate()
                try:
                    self.process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=timeout)
        self.process = None
        if self._stderr_handle is not None:
            self._stderr_handle.close()
            self._stderr_handle = None
