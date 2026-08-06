"""Evidence reconciliation across adapter, plugin, bridge, and registry logs."""
from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Mapping


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"Expected JSON object in {path}")
        records.append(value)
    return records


def _episode_ids(events: Iterable[Mapping[str, Any]]) -> set[str]:
    ids: set[str] = set()
    for event in events:
        if isinstance(event.get("episode_id"), str):
            ids.add(str(event["episode_id"]))
        payload = event.get("payload")
        if isinstance(payload, Mapping):
            response = payload.get("response")
            if isinstance(response, Mapping) and isinstance(response.get("episode_id"), str):
                ids.add(str(response["episode_id"]))
            tool_response = payload.get("response")
            if isinstance(tool_response, Mapping):
                result = tool_response.get("result")
                if isinstance(result, Mapping) and isinstance(result.get("episode_id"), str):
                    ids.add(str(result["episode_id"]))
    return ids


def _actions(events: Iterable[Mapping[str, Any]]) -> list[str]:
    actions: list[str] = []
    for event in events:
        payload = event.get("payload")
        if not isinstance(payload, Mapping):
            continue
        request = payload.get("request")
        if isinstance(request, Mapping) and isinstance(request.get("action"), str):
            actions.append(str(request["action"]))
    return actions


def _terminal_outcome(events: Iterable[Mapping[str, Any]]) -> dict[str, bool] | None:
    for event in reversed(list(events)):
        payload = event.get("payload")
        if not isinstance(payload, Mapping):
            continue
        response = payload.get("response")
        if isinstance(response, Mapping) and isinstance(response.get("result"), Mapping):
            response = response["result"]
        if isinstance(response, Mapping) and response.get("done") is True:
            return {
                "done": True,
                "success": bool(response.get("success")),
                "aborted": bool(response.get("aborted")),
            }
    return None


def reconcile_evidence(
    adapter_events: Iterable[Mapping[str, Any]],
    plugin_events: Iterable[Mapping[str, Any]],
    bridge_events: Iterable[Mapping[str, Any]],
    bindings: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare stable IDs and outcomes without relying on log filenames."""
    adapter = list(adapter_events)
    plugin = list(plugin_events)
    bridge = list(bridge_events)
    registry = list(bindings)
    adapter_ids = _episode_ids(adapter)
    plugin_ids = _episode_ids(plugin)
    bridge_ids = _episode_ids(bridge)
    registry_ids = {str(item["episode_id"]) for item in registry if item.get("episode_id")}
    adapter_actions = _actions(adapter)
    plugin_actions = _actions(plugin)
    bridge_actions = _actions(bridge)
    adapter_terminal = _terminal_outcome(adapter)
    plugin_terminal = _terminal_outcome(plugin)
    bridge_terminal = _terminal_outcome(bridge)
    correlation_ok = all(
        isinstance(event.get("correlation"), Mapping)
        for event in bridge
        if event.get("event") in {"start", "step", "reset", "abort", "terminal"}
    )
    mismatches: list[str] = []
    if adapter_ids and bridge_ids and adapter_ids != bridge_ids:
        mismatches.append("adapter and bridge episode IDs differ")
    if plugin_ids and bridge_ids and plugin_ids != bridge_ids:
        mismatches.append("plugin and bridge episode IDs differ")
    if registry_ids and bridge_ids and not bridge_ids.issubset(registry_ids):
        mismatches.append("bridge episode IDs are missing from run-registry bindings")
    if adapter_actions and bridge_actions and adapter_actions != bridge_actions:
        mismatches.append("adapter and bridge action sequences differ")
    if plugin_actions and bridge_actions and plugin_actions != bridge_actions:
        mismatches.append("plugin and bridge action sequences differ")
    if adapter_terminal and bridge_terminal and adapter_terminal != bridge_terminal:
        mismatches.append("adapter and bridge terminal outcomes differ")
    if plugin_terminal and bridge_terminal and plugin_terminal != bridge_terminal:
        mismatches.append("plugin and bridge terminal outcomes differ")
    if not correlation_ok:
        mismatches.append("bridge lifecycle events are missing correlation objects")
    return {
        "schema_version": 1,
        "ok": not mismatches,
        "episode_ids": {
            "adapter": sorted(adapter_ids),
            "plugin": sorted(plugin_ids),
            "bridge": sorted(bridge_ids),
            "registry": sorted(registry_ids),
        },
        "actions": {"adapter": adapter_actions, "plugin": plugin_actions, "bridge": bridge_actions},
        "terminal_outcome": {"adapter": adapter_terminal, "plugin": plugin_terminal, "bridge": bridge_terminal},
        "counts": {"adapter_events": len(adapter), "plugin_events": len(plugin), "bridge_events": len(bridge), "bindings": len(registry)},
        "mismatches": mismatches,
    }
