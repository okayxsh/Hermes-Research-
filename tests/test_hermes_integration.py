from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.request import Request

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rq1.bridge.app import create_bridge_server
from rq1.hermes.adapter import BridgeTransportError, FakeHermesAdapter, HermesAdapter, LocalBridgeClient
from rq1.hermes.capabilities import probe_hermes_capabilities
from rq1.hermes.models import HermesContext, HermesEventLog
from rq1.hermes.reconcile import read_jsonl, reconcile_evidence
from rq1.logging.run_registry import EpisodeBinding, RunRegistry
from rq1.hermes.verification import verify_fake_hermes_integration


class HermesIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.server = create_bridge_server(self.root / "bridge", port=0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.context = HermesContext(run_id="run-1", attempt_id="attempt-1", profile="rq1-pilot", session_id="session-1")
        self.log = self.root / "hermes.jsonl"
        self.adapter = FakeHermesAdapter(LocalBridgeClient(f"http://127.0.0.1:{self.server.server_port}"), HermesEventLog(self.log))

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.temp.cleanup()

    def test_complete_tool_lifecycle_correlation_and_reconciliation(self) -> None:
        self.assertTrue(self.adapter.health(self.context).ok)
        start = self.adapter.invoke("alfworld_start", {"task_id": "hermes_001", "split": "valid_seen", "seed": 1, "action_limit": 4}, self.context)
        self.assertTrue(start.ok)
        episode_id = start.result["episode_id"]
        self.assertTrue(self.adapter.invoke("alfworld_step", {"episode_id": episode_id, "action": "go to countertop 1"}, self.context).ok)
        self.assertTrue(self.adapter.invoke("alfworld_status", {"episode_id": episode_id}, self.context).ok)
        reset = self.adapter.invoke("alfworld_reset", {"episode_id": episode_id}, self.context)
        self.assertEqual(1, reset.result["reset_count"])
        self.assertTrue(self.adapter.invoke("alfworld_abort", {"episode_id": episode_id, "reason": "done"}, self.context).result["aborted"])
        bridge_log = self.root / "bridge" / f"{episode_id}.jsonl"
        bridge_events = read_jsonl(bridge_log)
        self.assertEqual("run-1", bridge_events[0]["correlation"]["run_id"])
        registry = RunRegistry(self.root / "registry.sqlite")
        registry.bind_episode(EpisodeBinding("run-1", "attempt-1", episode_id, "session-1", "rq1-pilot", "hermes.jsonl", str(bridge_log)))
        report = reconcile_evidence(read_jsonl(self.log), [], bridge_events, [dict(row) for row in registry.episode_bindings("run-1")])
        self.assertTrue(report["ok"], report)

    def test_validation_unknown_episode_action_limit_and_conflict_errors_are_structured(self) -> None:
        malformed = self.adapter.invoke("alfworld_start", {"task_id": "x", "split": "valid_seen", "seed": 1, "action_limit": 2, "extra": True}, self.context)
        self.assertEqual("validation_failure", malformed.error.code)
        unknown = self.adapter.invoke("alfworld_status", {"episode_id": "missing"}, self.context)
        self.assertEqual("bridge_error", unknown.error.code)
        self.assertEqual(404, unknown.error.status)
        start = self.adapter.invoke("alfworld_start", {"task_id": "limit", "split": "valid_seen", "seed": 1, "action_limit": 1}, self.context)
        limited = self.adapter.invoke("alfworld_step", {"episode_id": start.result["episode_id"], "action": "invalid"}, self.context)
        self.assertTrue(limited.result["done"])
        self.assertFalse(limited.result["action_valid"])
        conflicting = self.adapter.invoke("alfworld_status", {"episode_id": start.result["episode_id"]}, HermesContext(run_id="other-run"))
        self.assertEqual("correlation_conflict", conflicting.error.code)
        self.assertEqual(409, conflicting.error.status)

    def test_timeout_is_not_retried_for_mutations_but_health_retries_once(self) -> None:
        calls: list[str] = []

        def timeout_transport(request: Request, _timeout: float) -> tuple[int, bytes]:
            calls.append(request.full_url)
            raise BridgeTransportError("timeout", "timed out")

        adapter = HermesAdapter(LocalBridgeClient("http://127.0.0.1:8000", timeout_transport))
        result = adapter.invoke("alfworld_start", {"task_id": "x", "split": "valid_seen", "seed": 1, "action_limit": 1})
        self.assertTrue(result.error.outcome_unknown)
        self.assertEqual(1, len(calls))
        calls.clear()

        def flaky_transport(request: Request, _timeout: float) -> tuple[int, bytes]:
            calls.append(request.full_url)
            if len(calls) == 1:
                raise BridgeTransportError("bridge_unavailable", "unavailable")
            return 200, b'{"bridge_available": true}'

        health = HermesAdapter(LocalBridgeClient("http://127.0.0.1:8000", flaky_transport)).health()
        self.assertTrue(health.ok)
        self.assertEqual(2, len(calls))
        def unavailable_transport(_request: Request, _timeout: float) -> tuple[int, bytes]:
            raise BridgeTransportError("bridge_unavailable", "offline")

        unavailable = HermesAdapter(LocalBridgeClient("http://127.0.0.1:8000", unavailable_transport)).invoke(
            "alfworld_status", {"episode_id": "missing"}
        )
        self.assertEqual("bridge_unavailable", unavailable.error.code)

    def test_remote_bridge_urls_are_refused(self) -> None:
        with self.assertRaises(ValueError):
            LocalBridgeClient("http://203.0.113.10:8000")
        with self.assertRaises(ValueError):
            LocalBridgeClient("https://127.0.0.1:8000")

    def test_fake_skill_events_are_versioned_and_metrics_remain_compatible(self) -> None:
        self.adapter.emit_skill_event("skill_index_available", "alpha", "relevant", self.context)
        self.adapter.emit_skill_event("skill_loaded", "alpha", "relevant", self.context)
        self.adapter.emit_skill_event("skill_loaded", "alpha", "irrelevant", self.context)
        events = read_jsonl(self.log)
        self.assertEqual(["skill_index_available", "skill_loaded", "skill_loaded"], [event["event"] for event in events])
        self.assertEqual("irrelevant", events[-1]["relevance"])

    def test_concurrent_fake_episodes_keep_independent_correlation(self) -> None:
        episode_ids: list[str] = []
        errors: list[str] = []
        lock = threading.Lock()

        def worker(index: int) -> None:
            context = HermesContext(run_id=f"run-{index}", attempt_id=f"attempt-{index}", profile="rq1-pilot", session_id=f"session-{index}")
            result = self.adapter.invoke("alfworld_start", {"task_id": f"parallel-{index}", "split": "valid_seen", "seed": index, "action_limit": 3}, context)
            with lock:
                if result.ok:
                    episode_ids.append(result.result["episode_id"])
                else:
                    errors.append(result.error.code)

        threads = [threading.Thread(target=worker, args=(index,)) for index in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual([], errors)
        self.assertEqual(4, len(set(episode_ids)))
        for episode_id in episode_ids:
            records = read_jsonl(self.root / "bridge" / f"{episode_id}.jsonl")
            self.assertIn(records[0]["correlation"]["run_id"], {"run-0", "run-1", "run-2", "run-3"})

    def test_capability_probe_uses_only_help_and_version_commands(self) -> None:
        calls: list[tuple[str, ...]] = []
        responses = {
            ("hermes", "--version"): (0, "Hermes 1.2.3", ""),
            ("hermes", "--help"): (0, "profile plugins skills hooks config", ""),
            ("hermes", "profile", "--help"): (0, "create", ""),
            ("hermes", "profile", "create", "--help"): (0, "--no-skills", ""),
            ("hermes", "plugins", "--help"): (0, "hooks", ""),
            ("hermes", "skills", "--help"): (0, "list", ""),
        }

        def runner(command: tuple[str, ...]) -> tuple[int, str, str]:
            calls.append(command)
            return responses[command]

        report = probe_hermes_capabilities(runner=runner, project_root=self.root)
        self.assertTrue(report.plugin_supported)
        self.assertTrue(report.hook_supported)
        self.assertTrue(report.no_skills_supported)
        self.assertEqual("hermes-plugin-v1", report.selected_version_adapter)
        self.assertTrue(all(command[-1] in {"--help", "--version"} for command in calls))

    def test_project_plugin_requires_explicit_trust_and_registers_five_tools(self) -> None:
        plugin = Path(__file__).resolve().parents[1] / ".hermes" / "plugins" / "alfworld-experiment" / "__init__.py"
        spec = importlib.util.spec_from_file_location("rq1_test_plugin", plugin)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        class Context:
            def __init__(self) -> None:
                self.tools: list[dict[str, object]] = []
                self.hooks: list[str] = []

            def register_tool(self, **kwargs: object) -> None:
                self.tools.append(kwargs)

            def register_hook(self, name: str, _handler: object) -> None:
                self.hooks.append(name)

        previous = os.environ.pop("HERMES_ENABLE_PROJECT_PLUGINS", None)
        try:
            with self.assertRaises(RuntimeError):
                module.register(Context())
            os.environ["HERMES_ENABLE_PROJECT_PLUGINS"] = "1"
            registered = Context()
            module.register(registered)
            self.assertEqual(5, len(registered.tools))
            self.assertEqual("alfworld_experiment", registered.tools[0]["toolset"])
            self.assertEqual(["pre_tool_call", "post_tool_call"], registered.hooks)
        finally:
            if previous is None:
                os.environ.pop("HERMES_ENABLE_PROJECT_PLUGINS", None)
            else:
                os.environ["HERMES_ENABLE_PROJECT_PLUGINS"] = previous

    def test_fake_verification_writes_a_machine_readable_report(self) -> None:
        report = verify_fake_hermes_integration(self.root)
        self.assertTrue(report["mock_integration"])
        stored = self.root / "artifacts" / "stage_reports" / "phase3-hermes-integration.json"
        self.assertTrue(stored.is_file())
        self.assertFalse(json.loads(stored.read_text(encoding="utf-8"))["real_compatibility"])
