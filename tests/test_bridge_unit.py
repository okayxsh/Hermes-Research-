from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rq1.bridge.environment import FakeALFWorldAdapter, real_adapter_capability
from rq1.bridge.episode_manager import BridgeError, EpisodeManager
from rq1.bridge.models import EpisodeStartRequest, RequestValidationError


class BridgeUnitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.manager = EpisodeManager(FakeALFWorldAdapter, Path(self.temp.name) / "logs")
        self.request = EpisodeStartRequest("fixture_001", "valid_seen", 7, 10)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_start_is_deterministic_and_uses_unique_ids(self) -> None:
        first = self.manager.start(self.request)
        second = self.manager.start(self.request)
        self.assertNotEqual(first.episode_id, second.episode_id)
        self.assertEqual(first.task_family, second.task_family)
        self.assertEqual(first.observation, second.observation)
        self.assertEqual(2, self.manager.active_episode_count)

    def test_invalid_action_counts_and_action_limit_ends_episode(self) -> None:
        response = self.manager.start(EpisodeStartRequest("fixture_002", "valid_seen", 1, 1))
        step = self.manager.step(response.episode_id, "not an admissible action")
        self.assertFalse(step.action_valid)
        self.assertEqual(1, step.action_count)
        self.assertTrue(step.done)
        self.assertFalse(step.success)
        with self.assertRaises(BridgeError) as raised:
            self.manager.step(response.episode_id, "look")
        self.assertEqual(409, raised.exception.status_code)

    def test_reset_keeps_id_and_raw_log_is_append_only(self) -> None:
        initial = self.manager.start(self.request)
        moved = self.manager.step(initial.episode_id, "go to countertop 1")
        self.assertEqual(1, moved.step_number)
        reset = self.manager.reset(initial.episode_id)
        self.assertEqual(initial.episode_id, reset.episode_id)
        self.assertEqual(1, reset.reset_count)
        self.assertEqual(0, reset.action_count)
        self.assertEqual(0, reset.step_number)
        aborted = self.manager.abort(initial.episode_id, "fixture complete")
        self.assertTrue(aborted.aborted)
        events = [json.loads(line)["event"] for line in (Path(self.temp.name) / "logs" / f"{initial.episode_id}.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(["start", "step", "reset", "abort", "terminal"], events)

    def test_validation_and_real_capability_fail_closed(self) -> None:
        with self.assertRaises(RequestValidationError):
            EpisodeStartRequest.from_payload({"task_id": "x", "split": "test", "seed": 1, "action_limit": 1})
        capability = real_adapter_capability()
        self.assertFalse(capability.available)
        self.assertIn("unverified", capability.details.lower())
