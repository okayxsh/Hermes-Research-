from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import unittest
from unittest.mock import patch

from rq1.experiment.models import RunFailure
from rq1.university_gpu.feasibility_protocol import isolate_protocol_stdout
from rq1.university_gpu.scheduler import SchedulerError
from rq1.university_gpu.worker_process import WorkerProcessClient


class FeasibilityWorkerProtocolTests(unittest.TestCase):
    def test_valid_json_object_response_is_decoded(self) -> None:
        value = {"ok": True, "result": {"unit_key": "unit-1", "actions": 7}}
        self.assertEqual(WorkerProcessClient._decode_protocol_line(json.dumps(value) + "\n", label="worker-1"), value)

    def test_blank_protocol_line_is_rejected(self) -> None:
        with self.assertRaisesRegex(SchedulerError, "invalid worker protocol"):
            WorkerProcessClient._decode_protocol_line("\n", label="worker-1 episode")

    def test_whitespace_only_protocol_line_is_rejected(self) -> None:
        with self.assertRaisesRegex(SchedulerError, "invalid worker protocol"):
            WorkerProcessClient._decode_protocol_line(" \t\r\n", label="worker-1 episode")

    def test_malformed_json_has_bounded_diagnostic(self) -> None:
        line = "ordinary progress output " + ("x" * 100_000) + "\n"
        with self.assertRaises(SchedulerError) as caught:
            WorkerProcessClient._decode_protocol_line(line, label="worker-2 episode")
        message = str(caught.exception)
        self.assertIn("worker-2 episode", message)
        self.assertLess(len(message), 2_500)
        self.assertIn("ordinary progress output", message)

    def test_valid_json_non_object_is_rejected(self) -> None:
        for line in ('"text"\n', "[]\n", "null\n", "7\n"):
            with self.subTest(line=line), self.assertRaises(SchedulerError):
                WorkerProcessClient._decode_protocol_line(line, label="worker-3")

    def test_duplicate_protocol_keys_are_rejected(self) -> None:
        line = '{"ok":false,"ok":true,"result":{}}\n'
        with self.assertRaisesRegex(SchedulerError, "invalid JSON object"):
            WorkerProcessClient._decode_protocol_line(line, label="worker-duplicate")

    def _client_for_response(self, temporary: str, response: str) -> tuple[WorkerProcessClient, Any]:
        client = WorkerProcessClient("worker-failure", Path(temporary) / "config.json",
                                     cwd=Path(temporary), gpu_uuid="GPU-4",
                                     module="rq1.university_gpu.feasibility_worker")
        process = SimpleNamespace(pid=1234, stdin=StringIO(), stdout=StringIO(response),
                                  poll=lambda: None, wait=lambda timeout=None: 0)
        client.process = process
        return client, process

    def test_malformed_failure_with_true_retry_flags_stays_protocol_error(self) -> None:
        invalid = (
            '{"ok":false,"error":"failed","safe_to_continue":true,"mutation_state_known":true}\n',
            '{"ok":false,"error_type":"RuntimeError","safe_to_continue":true,"mutation_state_known":true}\n',
        )
        for line in invalid:
            with self.subTest(line=line), tempfile.TemporaryDirectory() as temporary:
                client, _process = self._client_for_response(temporary, line)
                with patch("rq1.university_gpu.worker_process.require_posix_process_management"):
                    with self.assertRaises(SchedulerError):
                        client.execute({"unit_key": "u"}, Path(temporary) / "attempt", "attempt-2")
                self.assertIsNone(client.process)

    def test_valid_failure_preserves_explicit_retry_safety_flags(self) -> None:
        response = json.dumps({"ok": False, "error_type": "RunFailure", "error": "known clean failure",
                              "safe_to_continue": True, "mutation_state_known": True,
                              "failure_details": {"source": "synthetic test"}}) + "\n"
        with tempfile.TemporaryDirectory() as temporary:
            client, _process = self._client_for_response(temporary, response)
            with self.assertRaises(RunFailure) as caught:
                client.execute({"unit_key": "u"}, Path(temporary) / "attempt", "attempt-3")
        self.assertIs(caught.exception.safe_to_continue, True)
        self.assertIs(caught.exception.mutation_state_known, True)
        self.assertEqual(caught.exception.details["source"], "synthetic test")

    def test_malformed_execute_response_stops_worker_without_retry_flags(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            client = WorkerProcessClient("worker-4", Path(temporary) / "config.json",
                                         cwd=Path(temporary), gpu_uuid="GPU-4",
                                         module="rq1.university_gpu.feasibility_worker")
            process = SimpleNamespace(pid=1234, stdin=StringIO(), stdout=StringIO("\n"),
                                      poll=lambda: None, wait=lambda timeout=None: 0)
            client.process = process
            with patch("rq1.university_gpu.worker_process.require_posix_process_management"):
                with self.assertRaisesRegex(SchedulerError, "invalid worker protocol"):
                    client.execute({"unit_key": "u"}, Path(temporary) / "attempt", "attempt-1")
            self.assertIsNone(client.process)
            self.assertEqual(process.stdin.getvalue().splitlines()[-1], '{"command":"shutdown"}')

    def test_stdout_and_spawned_process_output_cannot_enter_protocol_pipe(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            script = (
                "import os, subprocess, sys\n"
                "from rq1.university_gpu.feasibility_protocol import isolate_protocol_stdout\n"
                "protocol = isolate_protocol_stdout()\n"
                "print('ordinary python progress')\n"
                "os.write(1, b'ordinary fd progress\\n')\n"
                "subprocess.run([sys.executable, '-c', \"print('spawned child progress')\"], check=True)\n"
                "protocol.write('{\\\"ready\\\":true}\\n'); protocol.flush()\n"
            )
            result = subprocess.run([sys.executable, "-c", script], cwd=Path.cwd(),
                                    env={**os.environ, "PYTHONPATH": str(Path.cwd() / "src")},
                                    text=True, capture_output=True, timeout=10, check=True)
        self.assertEqual(json.loads(result.stdout), {"ready": True})
        self.assertIn("ordinary python progress", result.stderr)
        self.assertIn("ordinary fd progress", result.stderr)
        self.assertIn("spawned child progress", result.stderr)

    def test_feasibility_module_entrypoint_isolates_stdout_before_runtime_startup(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            missing_config = Path(temporary) / "deliberately-missing-config.json"
            result = subprocess.run(
                [sys.executable, "-m", "rq1.university_gpu.feasibility_worker", "--config", str(missing_config)],
                cwd=Path.cwd(), env={**os.environ, "PYTHONPATH": str(Path.cwd() / "src")},
                text=True, capture_output=True, timeout=30, check=False,
            )
        self.assertNotEqual(result.returncode, 0)
        lines = result.stdout.splitlines()
        self.assertEqual(len(lines), 1, result.stdout)
        response = WorkerProcessClient._decode_protocol_line(lines[0], label="entrypoint startup")
        self.assertIs(response.get("ready"), False)
        self.assertIn("FileNotFoundError", response.get("error_type", ""))


if __name__ == "__main__":
    unittest.main()
