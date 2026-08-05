from __future__ import annotations

import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rq1.bridge.app import create_bridge_server


class BridgeHttpTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.server = create_bridge_server(Path(self.temp.name) / "logs", port=0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.temp.cleanup()

    def request(self, method: str, path: str, payload: dict[str, object] | None = None) -> tuple[int, dict[str, object]]:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(self.base + path, data=data, method=method)
        if data is not None:
            request.add_header("Content-Type", "application/json")
        try:
            with urlopen(request, timeout=3) as response:
                return response.status, json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            return error.code, json.loads(error.read().decode("utf-8"))

    def test_complete_http_lifecycle(self) -> None:
        status, health = self.request("POST", "/health")
        self.assertEqual(200, status)
        self.assertEqual("fake", health["mode"])
        status, started = self.request("POST", "/episode/start", {"task_id": "http_001", "split": "valid_seen", "seed": 3, "action_limit": 8})
        self.assertEqual(200, status)
        episode_id = started["episode_id"]
        status, stepped = self.request("POST", "/episode/step", {"episode_id": episode_id, "action": "go to countertop 1"})
        self.assertEqual(200, status)
        self.assertTrue(stepped["action_valid"])
        status, current = self.request("GET", f"/episode/{episode_id}/status")
        self.assertEqual(200, status)
        self.assertEqual(1, current["step_number"])
        status, reset = self.request("POST", f"/episode/{episode_id}/reset")
        self.assertEqual(200, status)
        self.assertEqual(episode_id, reset["episode_id"])
        self.assertEqual(1, reset["reset_count"])
        status, aborted = self.request("POST", f"/episode/{episode_id}/abort", {"reason": "test"})
        self.assertEqual(200, status)
        self.assertTrue(aborted["aborted"])
        status, error = self.request("POST", "/episode/step", {"episode_id": episode_id, "action": "look"})
        self.assertEqual(409, status)
        self.assertEqual(409, error["error"]["status"])

    def test_contract_errors_and_concurrent_episodes(self) -> None:
        status, _ = self.request("POST", "/episode/start")
        self.assertEqual(400, status)
        status, _ = self.request("POST", "/episode/start", {"task_id": "x", "split": "invalid", "seed": 1, "action_limit": 1})
        self.assertEqual(422, status)
        status, _ = self.request("GET", "/episode/missing/status")
        self.assertEqual(404, status)
        identifiers: list[str] = []
        lock = threading.Lock()
        def start_fixture(index: int) -> None:
            response_status, result = self.request("POST", "/episode/start", {"task_id": f"parallel_{index}", "split": "valid_seen", "seed": index, "action_limit": 4})
            self.assertEqual(200, response_status)
            with lock:
                identifiers.append(str(result["episode_id"]))
        threads = [threading.Thread(target=start_fixture, args=(index,)) for index in range(4)]
        for thread in threads: thread.start()
        for thread in threads: thread.join()
        self.assertEqual(4, len(set(identifiers)))
        status, health = self.request("POST", "/health")
        self.assertEqual(4, health["active_episode_count"])
