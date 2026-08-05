from __future__ import annotations

import sys
import tempfile
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rq1.logging.run_registry import Run, RunRegistry
from rq1.runners.mock import run_mock_workflow


class RegistryAndMockTests(unittest.TestCase):
    def test_atomic_claiming(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            registry = RunRegistry(Path(directory) / "runs.sqlite")
            for index in range(8):
                registry.plan(Run(f"run-{index}", f"task-{index}", "valid_seen", "L0", "mock", 1, "planned"))
            claimed: list[str] = []
            lock = threading.Lock()
            def worker() -> None:
                while run := registry.claim_next("worker"):
                    with lock:
                        claimed.append(run.run_id)
            threads = [threading.Thread(target=worker) for _ in range(3)]
            for thread in threads: thread.start()
            for thread in threads: thread.join()
            self.assertEqual(8, len(claimed))
            self.assertEqual(8, len(set(claimed)))

    def test_mock_workflow_is_resumable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = run_mock_workflow(root)
            second = run_mock_workflow(root)
            self.assertEqual("completed", first["status"])
            self.assertEqual(1.0, first["metrics"]["success_rate"])
            self.assertEqual("already_complete", second["status"])
