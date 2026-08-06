"""Machine-readable Phase 3 fake and explicitly opted-in real verification."""
from __future__ import annotations

import json
import os
import subprocess
import threading
import hashlib
from pathlib import Path
from typing import Any

from rq1.bridge.app import create_bridge_server
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
        # This command is read-only and uses a temporary home. A successful list
        # proves discovery only, never tool dispatch or model compatibility.
        temporary_home = root / "artifacts" / "phase3" / new_attempt_id() / "hermes-home"
        environment = os.environ.copy()
        environment.update({"HERMES_HOME": str(temporary_home), "HERMES_ENABLE_PROJECT_PLUGINS": "1", "HERMES_PLUGINS_DEBUG": "1"})
        completed = subprocess.run((report.executable, "plugins", "list"), cwd=root, env=environment, capture_output=True, text=True, timeout=30, check=False)
        base.update({
            "real_plugin_loading": completed.returncode == 0 and "alfworld-experiment" in (completed.stdout + completed.stderr),
            "status": "passed" if completed.returncode == 0 else "blocked",
            "command_result": {
                "returncode": completed.returncode,
                "stdout_sha256": hashlib.sha256(completed.stdout.encode()).hexdigest(),
                "stderr_sha256": hashlib.sha256(completed.stderr.encode()).hexdigest(),
            },
            "reason": "Discovery was attempted with the fake bridge only; dispatch and native skill capture remain unverified.",
        })
    _write(root / "artifacts" / "stage_reports" / "phase3-hermes-integration.json", base)
    return base
