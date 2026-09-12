"""Machine-readable Phase 3 fake and explicitly opted-in real verification."""
from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import hashlib
from pathlib import Path
from typing import Any

from rq1.bridge.app import create_bridge_server
from rq1.bridge.adapters.capabilities import default_data_dir
from rq1.bridge.adapters.alfworld_v042 import RealALFWorldAdapter
from rq1.bridge.episode_manager import EpisodeManager
from rq1.hermes.adapter import FakeHermesAdapter, LocalBridgeClient
from rq1.hermes.capabilities import probe_hermes_capabilities
from rq1.hermes.models import HermesContext, HermesEventLog, HermesIntegrationEvent
from rq1.hermes.reconcile import read_jsonl, reconcile_evidence
from rq1.logging.run_registry import EpisodeBinding, RunRegistry
from rq1.utils.ids import new_attempt_id
from rq1.utils.time import utc_now


def _write(path: Path, value: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def verify_fake_hermes_integration(root: Path) -> dict[str, Any]:
    attempt_id = new_attempt_id()
    run_id = f"phase3-fake-{attempt_id}"
    output = root / "artifacts" / "phase3" / attempt_id
    bridge_logs = output / "bridge"
    hermes_log = output / "hermes-events.jsonl"
    plugin_log = output / "plugin-events.jsonl"
    server = create_bridge_server(bridge_logs, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    context = HermesContext(run_id=run_id, attempt_id=attempt_id, profile="rq1-pilot", session_id="fake-session")
    adapter = FakeHermesAdapter(LocalBridgeClient(f"http://127.0.0.1:{server.server_port}"), HermesEventLog(hermes_log))
    try:
        health = adapter.health(context)
        start = adapter.invoke("alfworld_start", {"task_id": "phase3_fixture", "split": "valid_seen", "seed": 13, "action_limit": 4}, context)
        episode_id = str((start.result or {}).get("episode_id", ""))
        step = adapter.invoke("alfworld_step", {"episode_id": episode_id, "action": "go to countertop 1"}, context)
        status = adapter.invoke("alfworld_status", {"episode_id": episode_id}, context)
        reset = adapter.invoke("alfworld_reset", {"episode_id": episode_id}, context)
        abort = adapter.invoke("alfworld_abort", {"episode_id": episode_id, "reason": "fake verification"}, context)
        plugin_events = HermesEventLog(plugin_log)
        for result, parameters in (
            (start, {"task_id": "phase3_fixture", "split": "valid_seen", "seed": 13, "action_limit": 4}),
            (step, {"episode_id": episode_id, "action": "go to countertop 1"}),
            (status, {"episode_id": episode_id}),
            (reset, {"episode_id": episode_id}),
            (abort, {"episode_id": episode_id, "reason": "fake verification"}),
        ):
            plugin_events.append(
                HermesIntegrationEvent(
                    "plugin_post_tool_call",
                    {"tool": result.tool, "request": parameters, "response": result.to_dict()},
                    result.metadata,
                    simulated=True,
                )
            )
        adapter.emit_skill_event("skill_index_available", "fixture_skill", "relevant", context)
        adapter.emit_skill_event("skill_selected", "fixture_skill", "relevant", context)
        adapter.emit_skill_event("skill_loaded", "fixture_skill", "relevant", context)
        bridge_log = bridge_logs / f"{episode_id}.jsonl"
        registry = RunRegistry(root / "state" / "run_registry.sqlite")
        registry.bind_episode(EpisodeBinding(run_id, attempt_id, episode_id, "fake-session", "rq1-pilot", str(hermes_log.relative_to(root)), str(bridge_log.relative_to(root))))
        reconciliation = reconcile_evidence(
            read_jsonl(hermes_log),
            read_jsonl(plugin_log),
            read_jsonl(bridge_log),
            [dict(row) for row in registry.episode_bindings(run_id)],
        )
        report = {
            "schema_version": 1,
            "generated_at": utc_now(),
            "mode": "fake",
            "mock_integration": all(item.ok for item in (health, start, step, status, reset, abort)),
            "hermes_detected": False,
            "plugin_capability": False,
            "profile_capability": False,
            "hook_capability": False,
            "real_plugin_loading": False,
            "real_tool_dispatch": False,
            "native_skill_event_capture": False,
            "real_compatibility": False,
            "results": {"health": health.to_dict(), "start": start.to_dict(), "step": step.to_dict(), "status": status.to_dict(), "reset": reset.to_dict(), "abort": abort.to_dict()},
            "reconciliation": reconciliation,
            "artifacts": {
                "hermes_log": str(hermes_log.relative_to(root)),
                "plugin_log": str(plugin_log.relative_to(root)),
                "bridge_log": str(bridge_log.relative_to(root)),
            },
            "unverified": ["No installed Hermes instance was loaded.", "No real ALFWorld adapter was selected."],
        }
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    _write(root / "artifacts" / "stage_reports" / "phase3-hermes-integration.json", report)
    return report


def verify_real_hermes_integration(root: Path) -> dict[str, Any]:
    report = probe_hermes_capabilities(project_root=root)
    base: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "mode": "real",
        "mock_integration": False,
        "hermes_detected": report.installed,
        "plugin_capability": report.plugin_supported,
        "profile_capability": report.profile_supported,
        "hook_capability": report.hook_supported,
        "real_plugin_loading": False,
        "real_tool_dispatch": False,
        "native_skill_event_capture": False,
        "real_compatibility": False,
        "capabilities": report.to_dict(),
        "status": "blocked",
        "remediation": "Set RQ1_RUN_REAL_HERMES_TESTS=1 only on a machine with an installed compatible Hermes CLI and the fake bridge available.",
    }
    if os.environ.get("RQ1_RUN_REAL_HERMES_TESTS") != "1":
        base["reason"] = "Explicit real-Hermes opt-in is absent."
    elif not (report.installed and report.plugin_supported and report.hook_supported and report.executable):
        base["reason"] = "Installed Hermes capability is missing or unsupported."
    else:
        base.update(_run_real_registry_integration(root, report.executable))
    _write(root / "artifacts" / "stage_reports" / "phase3-hermes-integration.json", base)
    return base


def _hermes_python(executable: str | None) -> str | None:
    """Resolve Hermes's own interpreter from the installed launcher, never this project's venv."""
    if not executable:
        return None
    try:
        source = Path(executable).resolve().read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    match = re.search(r'exec\s+["\']([^"\']*python[^"\']*)["\']', source)
    if not match:
        return None
    candidate = Path(match.group(1))
    return str(candidate) if candidate.is_file() else None


_REGISTRY_EXERCISE = r'''
import json
import os
import sys

sys.path.insert(0, os.environ["RQ1_PROJECT_SOURCE"])
from hermes_cli.plugins import PluginContext, _dispatch_pre_tool_call_hooks, collect_directory_manifests, get_plugin_manager
from model_tools import _emit_post_tool_call_hook
import tools.skills_tool  # Registers Hermes's genuine skill_view tool with its native lifecycle callback.

manager = get_plugin_manager()
manager.discover_and_load()
manifest = next(item for item in collect_directory_manifests() if item.name == "alfworld-experiment" and item.source == "project")
context = PluginContext(manifest, manager)
plugin = next(item for item in manager.list_plugins() if item["name"] == "alfworld-experiment")
if not plugin["enabled"]:
    raise RuntimeError("Project plugin was discovered but not enabled in the isolated Hermes home")

task_id = os.environ["RQ1_REAL_TASK_ID"]
session_id = "rq1-real-registry-session"

def invoke(name, args, number):
    hook_kwargs = {
        "task_id": task_id,
        "session_id": session_id,
        "tool_call_id": f"rq1-real-tool-{number}",
        "api_request_id": f"rq1-real-request-{number}",
    }
    blocked, modified = _dispatch_pre_tool_call_hooks(name, args, **hook_kwargs)
    if blocked:
        raise RuntimeError(f"Hermes pre_tool_call blocked {name}: {blocked}")
    effective_args = modified if modified is not None else args
    result = context.dispatch_tool(name, effective_args, **hook_kwargs)
    _emit_post_tool_call_hook(function_name=name, function_args=effective_args, result=result, **hook_kwargs)
    parsed = json.loads(result)
    if not isinstance(parsed, dict) or parsed.get("ok") is not True:
        raise RuntimeError(f"Hermes registry dispatch failed for {name}: {result}")
    return parsed

started = invoke("alfworld_start", {"task_id": task_id, "split": "valid_seen", "seed": 17, "action_limit": 2}, 1)
episode_id = started["result"]["episode_id"]
actions = started["result"].get("admissible_actions") or []
if not actions:
    raise RuntimeError("Real ALFWorld start returned no admissible action for the registry exercise")
stepped = invoke("alfworld_step", {"episode_id": episode_id, "action": actions[0]}, 2)
status = invoke("alfworld_status", {"episode_id": episode_id}, 3)
reset = invoke("alfworld_reset", {"episode_id": episode_id}, 4)
aborted = invoke("alfworld_abort", {"episode_id": episode_id, "reason": "real Hermes registry verification"}, 5)
skill = context.dispatch_tool("skill_view", {"name": "rq1-native-lifecycle"}, task_id=task_id, session_id=session_id)
skill_result = json.loads(skill)
if not isinstance(skill_result, dict) or skill_result.get("success") is not True:
    raise RuntimeError(f"Native Hermes skill_view failed: {skill}")
print("RQ1_HERMES_RESULT=" + json.dumps({
    "plugin": {"enabled": plugin["enabled"], "tools": plugin["tools"], "hooks": plugin["hooks"]},
    "episode_id": episode_id,
    "operations": {"start": started["ok"], "step": stepped["ok"], "status": status["ok"], "reset": reset["ok"], "abort": aborted["ok"]},
    "skill_view": skill_result.get("name") or "rq1-native-lifecycle",
}, sort_keys=True))
'''


def _result_line(stdout: str) -> dict[str, Any] | None:
    for line in reversed(stdout.splitlines()):
        if line.startswith("RQ1_HERMES_RESULT="):
            try:
                payload = json.loads(line.split("=", 1)[1])
            except json.JSONDecodeError:
                return None
            return payload if isinstance(payload, dict) else None
    return None


def _verified_real_task(root: Path) -> str:
    """Reuse a prior real-smoke selection; this verifier never rebuilds a task catalog just to choose one cell."""
    configured = os.environ.get("RQ1_REAL_HERMES_TASK_ID")
    if configured:
        return configured
    reports = sorted((root / "artifacts" / "alfworld_smoke").glob("*/smoke-report.json"), reverse=True)
    for report in reports:
        try:
            payload = json.loads(report.read_text(encoding="utf-8"))
            request = payload.get("requests", {}).get("start", {})
            task_id = request.get("task_id")
            if payload.get("real_operation_executed") is True and request.get("split") == "valid_seen" and isinstance(task_id, str):
                return task_id
        except (OSError, ValueError, TypeError):
            continue
    raise ValueError("No prior real valid_seen smoke task is available; set RQ1_REAL_HERMES_TASK_ID to a frozen valid_seen task ID.")


def _run_real_registry_integration(root: Path, executable: str | None) -> dict[str, Any]:
    """Exercise the installed Hermes registry against a real local ALFWorld bridge once."""
    python = _hermes_python(executable)
    if not python:
        return {
            "status": "blocked",
            "reason": "The installed Hermes launcher did not expose its own Python runtime safely.",
            "remediation": "Use a supported Hermes installation whose launcher resolves its interpreter; do not substitute the project venv.",
        }
    attempt_id = new_attempt_id()
    output = root / "artifacts" / "phase3" / attempt_id
    hermes_home = output / "hermes-home"
    plugin_log = output / "plugin-events.jsonl"
    bridge_logs = output / "bridge"
    hermes_home.mkdir(parents=True, exist_ok=True)
    # Hermes 0.21.2 requires an explicit project-plugin allow-list. This is an
    # isolated evidence home, not a profile or user-level Hermes configuration.
    (hermes_home / "config.yaml").write_text("plugins:\n  enabled:\n    - alfworld-experiment\n", encoding="utf-8")
    skill_root = hermes_home / "skills" / "rq1-native-lifecycle"
    skill_root.mkdir(parents=True, exist_ok=True)
    (skill_root / "SKILL.md").write_text(
        "---\nname: rq1-native-lifecycle\ndescription: Isolated native Hermes lifecycle evidence fixture.\n---\n"
        "This fixture is dispatched through Hermes skill_view to observe its native lifecycle hook.\n",
        encoding="utf-8",
    )
    try:
        task_id = _verified_real_task(root)
        # The real adapter performs its own capability check and opens the
        # selected indexed episode. Supplying it directly avoids an additional
        # full index scan merely to choose this already-frozen smoke cell.
        manager = EpisodeManager(lambda: RealALFWorldAdapter(data_dir=default_data_dir()), bridge_logs)
        server = create_bridge_server(bridge_logs, port=0, manager=manager)
    except Exception as exc:
        return {"status": "blocked", "reason": f"Real ALFWorld bridge preparation failed: {type(exc).__name__}: {exc}", "remediation": "Resolve the existing real ALFWorld capability gate without using valid_unseen."}
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    environment = os.environ.copy()
    environment.update({
        "HERMES_HOME": str(hermes_home), "HERMES_ENABLE_PROJECT_PLUGINS": "1",
        "RQ1_PROJECT_SOURCE": str(root / "src"), "RQ1_BRIDGE_URL": f"http://127.0.0.1:{server.server_port}",
        "RQ1_BRIDGE_TIMEOUT_SECONDS": "300",
        "RQ1_HERMES_EVENT_LOG": str(plugin_log), "RQ1_RUN_ID": f"phase3-real-{attempt_id}",
        "RQ1_ATTEMPT_ID": attempt_id, "RQ1_PROFILE": "rq1-real-registry-verification",
        "RQ1_REAL_TASK_ID": task_id,
    })
    try:
        completed = subprocess.run((python, "-c", _REGISTRY_EXERCISE), cwd=root, env=environment, capture_output=True, text=True, timeout=300, check=False)
    except subprocess.TimeoutExpired:
        return {"status": "blocked", "reason": "Installed Hermes registry exercise exceeded its 300-second safe start bound before a result was committed.", "remediation": "Resolve real ALFWorld bridge startup latency before retrying; do not treat the pre-tool hook as a completed episode."}
    except OSError as exc:
        return {"status": "blocked", "reason": f"Installed Hermes registry exercise could not start: {type(exc).__name__}: {exc}", "remediation": "Inspect the isolated evidence artifact and installed Hermes runtime."}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    registry_result = _result_line(completed.stdout)
    plugin_events = read_jsonl(plugin_log)
    episode_id = str(registry_result.get("episode_id")) if registry_result else ""
    bridge_events = read_jsonl(bridge_logs / f"{episode_id}.jsonl") if episode_id else []
    expected_tools = {"alfworld_start", "alfworld_step", "alfworld_status", "alfworld_reset", "alfworld_abort"}
    hooked_tools = {
        str(event.get("payload", {}).get("tool")) for event in plugin_events
        if event.get("event") in {"plugin_pre_tool_call", "plugin_post_tool_call"}
    }
    native_loaded = any(
        event.get("event") == "native_skill_lifecycle"
        and event.get("simulated") is False
        and event.get("payload", {}).get("action") == "loaded"
        and event.get("payload", {}).get("skill_name") == "rq1-native-lifecycle"
        for event in plugin_events
    )
    bridge_operations = {str(event.get("event")) for event in bridge_events}
    operations_ok = bool(registry_result and all(registry_result.get("operations", {}).values()))
    plugin_loaded = bool(registry_result and registry_result.get("plugin", {}).get("enabled") is True)
    dispatch_ok = operations_ok and expected_tools.issubset(hooked_tools) and {"start", "step", "status", "reset", "abort"}.issubset(bridge_operations)
    status = "passed" if plugin_loaded and dispatch_ok and native_loaded else "blocked"
    return {
        "status": status,
        "real_plugin_loading": plugin_loaded,
        "real_tool_dispatch": dispatch_ok,
        "native_skill_event_capture": native_loaded,
        "real_compatibility": status == "passed",
        "bridge_adapter": "RealALFWorldAdapter",
        "registry_result": registry_result,
        "command_result": {"returncode": completed.returncode, "stdout_sha256": hashlib.sha256(completed.stdout.encode()).hexdigest(), "stderr_sha256": hashlib.sha256(completed.stderr.encode()).hexdigest()},
        "artifacts": {"root": str(output.relative_to(root)), "plugin_log": str(plugin_log.relative_to(root)), "bridge_log": str((bridge_logs / f"{episode_id}.jsonl").relative_to(root)) if episode_id else None},
        "reason": None if status == "passed" else "Installed Hermes registry evidence was incomplete; no readiness claim was promoted.",
        "remediation": None if status == "passed" else "Inspect the recorded plugin, bridge, and subprocess evidence; do not substitute a direct bridge adapter.",
    }
