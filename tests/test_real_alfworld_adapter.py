from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rq1.bridge.adapters.base import IndexedTask, RealALFWorldUnavailable
from rq1.bridge.adapters.capabilities import ALFWorldCapabilityReport, probe_alfworld_capabilities
from rq1.bridge.adapters.task_index import TaskIndex, TaskIndexError, build_task_index
from rq1.bridge.app import create_bridge_server
from rq1.bridge.episode_manager import BridgeError
from rq1.bridge.environment import FakeALFWorldAdapter
from rq1.bridge.models import EpisodeStartRequest


def ready_report() -> ALFWorldCapabilityReport:
    return ALFWorldCapabilityReport(True, True, True, True, True, True, True, True, True, False, True, False, False, False, True, "0.4.2", "fixture")


def write_task(root: Path, split: str, name: str, task_type: int = 1, solvable: bool = True) -> str:
    task = root / "json_2.1.1" / split / name
    task.mkdir(parents=True)
    (task / "traj_data.json").write_text(json.dumps({"task_type": task_type}), encoding="utf-8")
    (task / "game.tw-pddl").write_text(json.dumps({"solvable": solvable}), encoding="utf-8")
    (root / "logic").mkdir(exist_ok=True)
    return f"{split}:{name}"


class FixtureEnvironment:
    def __init__(self) -> None:
        self.phase = 0
        self.actions: list[str] = []

    def reset(self):
        self.phase = 0
        return ["initial observation"], {"admissible_commands": [["look", "finish"]], "won": [False], "extra.gamefile": ["fixture"]}

    def step(self, commands):
        self.actions.extend(commands)
        self.phase += 1
        if commands[0] == "finish":
            return ["completed"], [1], [True], {"admissible_commands": [[]], "won": [True]}
        return ["after action"], [0], [False], {"admissible_commands": [["finish"]], "won": [False]}


class RealALFWorldAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.task_id = write_task(self.root, "valid_seen", "pick_and_place-001")
        write_task(self.root, "train", "bootstrap-task")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_index_is_deterministic_and_rejects_wrong_split_or_unseen(self) -> None:
        write_task(self.root, "train", "pick_two-002", 6)
        first, second = build_task_index(self.root), build_task_index(self.root)
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual("pick_and_place_simple", first.resolve(self.task_id, "valid_seen").task_family)
        with self.assertRaises(TaskIndexError):
            first.resolve(self.task_id, "train")
        with self.assertRaises(TaskIndexError):
            build_task_index(self.root, splits=("valid_unseen",))

    def test_index_rejects_malformed_data_and_duplicate_resolution(self) -> None:
        malformed = self.root / "json_2.1.1" / "valid_seen" / "bad"
        malformed.mkdir(parents=True)
        (malformed / "traj_data.json").write_text("not-json", encoding="utf-8")
        (malformed / "game.tw-pddl").write_text("{}", encoding="utf-8")
        with self.assertRaises(TaskIndexError):
            build_task_index(self.root)
        entry = IndexedTask("valid_seen:duplicate", "valid_seen", "pick_and_place_simple", Path("a"), Path("b"), "a", "b", "id")
        index = TaskIndex(self.root, (entry, entry), "id")
        with self.assertRaises(TaskIndexError): index.resolve(entry.task_id, "valid_seen")

    def test_adapter_maps_observed_state_and_keeps_status_cached(self) -> None:
        from rq1.bridge.adapters.alfworld_v042 import RealALFWorldAdapter
        environment = FixtureEnvironment()
        with patch("rq1.bridge.adapters.alfworld_v042.probe_alfworld_capabilities", return_value=ready_report()):
            adapter = RealALFWorldAdapter(self.root, environment_factory=lambda *_args: environment)
            start = adapter.start(EpisodeStartRequest(self.task_id, "valid_seen", 7, 8))
            self.assertEqual(("look", "finish"), start.admissible_actions)
            self.assertEqual("unavailable_in_observed_alfworld_v042_surface", start.field_sources["inventory"])
            step = adapter.step("look")
            self.assertTrue(step.action_valid)
            self.assertEqual("after action", step.observation)
            self.assertEqual("cached", adapter.status().freshness)
            terminal = adapter.step("finish")
            self.assertTrue(terminal.done)
            self.assertTrue(terminal.success)
            reset = adapter.reset()
            self.assertEqual(0, reset.step_number)
            self.assertEqual("initial observation", reset.observation)
            aborted = adapter.abort("fixture")
            self.assertTrue(aborted.done)
            self.assertEqual("controller_side", aborted.field_sources["abort"])

    def test_invalid_action_and_reset_mismatch_are_reported(self) -> None:
        from rq1.bridge.adapters.alfworld_v042 import RealALFWorldAdapter
        environment = FixtureEnvironment()
        with patch("rq1.bridge.adapters.alfworld_v042.probe_alfworld_capabilities", return_value=ready_report()):
            adapter = RealALFWorldAdapter(self.root, environment_factory=lambda *_args: environment)
            adapter.start(EpisodeStartRequest(self.task_id, "valid_seen", 1, 4))
            self.assertFalse(adapter.step("unknown").action_valid)
            environment.reset = lambda: (["changed initial"], {"admissible_commands": [["look"]], "won": [False]})
            with self.assertRaises(RealALFWorldUnavailable): adapter.reset()

    def test_missing_capability_never_falls_back_to_fake(self) -> None:
        from rq1.bridge.adapters.alfworld_v042 import RealALFWorldAdapter
        unavailable = ALFWorldCapabilityReport(False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, None, "missing")
        with patch("rq1.bridge.adapters.alfworld_v042.probe_alfworld_capabilities", return_value=unavailable):
            adapter = RealALFWorldAdapter(self.root)
            with self.assertRaises(RealALFWorldUnavailable): adapter.start(EpisodeStartRequest(self.task_id, "valid_seen", 1, 2))
        self.assertIsInstance(FakeALFWorldAdapter(), FakeALFWorldAdapter)

    def test_real_http_mode_and_cli_are_explicitly_gated(self) -> None:
        with self.assertRaises(BridgeError) as raised:
            create_bridge_server(self.root / "logs", port=0, mode="real")
        self.assertEqual(503, raised.exception.status_code)
        from rq1.cli import build_parser
        parsed = build_parser().parse_args(["alfworld", "smoke-test", "--split", "valid_seen", "--yes"])
        self.assertEqual("smoke-test", parsed.alfworld_command)
        self.assertTrue(parsed.yes)

    def test_capability_probe_reports_missing_and_injected_supported_surface(self) -> None:
        missing = probe_alfworld_capabilities(self.root)
        self.assertFalse(missing.real_adapter_ready)
        fake_module = types.SimpleNamespace(AlfredTWEnv=type("AlfredTWEnv", (), {"init_env": lambda self, batch_size: None}))
        original_import = __import__
        def injected_import(name, *args, **kwargs):
            if name == "alfworld.agents.environment.alfred_tw_env": return fake_module
            return original_import(name, *args, **kwargs)
        with patch("rq1.bridge.adapters.capabilities.importlib.util.find_spec", return_value=object()), \
             patch("rq1.bridge.adapters.capabilities.importlib.metadata.version", return_value="0.4.2"), \
             patch("builtins.__import__", side_effect=injected_import):
            report = probe_alfworld_capabilities(self.root)
        self.assertTrue(report.real_adapter_ready)
        self.assertFalse(report.inventory_observable)
        self.assertFalse(report.target_relocation_supported)


@unittest.skipUnless(__import__("os").environ.get("RQ1_RUN_REAL_ALFWORLD_TESTS") == "1", "set RQ1_RUN_REAL_ALFWORLD_TESTS=1 after installing ALFWorld 0.4.2 data")
class OptionalRealALFWorldTests(unittest.TestCase):
    def test_real_lifecycle_requires_valid_seen_evidence(self) -> None:
        # The explicit CLI smoke test is the supported university-machine path.
        from rq1.bridge.adapters.capabilities import default_data_dir
        index = build_task_index(default_data_dir())
        task = index.for_split("valid_seen")[0]
        from rq1.bridge.adapters.alfworld_v042 import RealALFWorldAdapter
        adapter = RealALFWorldAdapter(default_data_dir())
        started = adapter.start(EpisodeStartRequest(task.task_id, "valid_seen", 1, 12))
        self.assertTrue(started.admissible_actions)
        adapter.step(started.admissible_actions[0]); self.assertEqual("cached", adapter.status().freshness)
        adapter.reset(); self.assertTrue(adapter.abort("test cleanup").done)
