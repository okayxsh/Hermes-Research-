"""Runtime helpers imported by the explicitly trusted project-local Hermes plugin."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

from rq1.hermes.adapter import HermesAdapter, LocalBridgeClient
from rq1.hermes.models import HermesContext, HermesEventLog, HermesIntegrationEvent, redact


TOOLSET = "alfworld_experiment"
TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "alfworld_start": {
        "name": "alfworld_start",
        "description": "Start one explicit ALFWorld bridge episode.",
        "parameters": {
            "type": "object",
            "properties": {
                "task_id": {"type": "string"}, "split": {"type": "string"},
                "seed": {"type": "integer"}, "action_limit": {"type": "integer", "minimum": 1},
            },
            "required": ["task_id", "split", "seed", "action_limit"],
            "additionalProperties": False,
        },
    },
    "alfworld_step": {
        "name": "alfworld_step",
        "description": "Send one action to an existing ALFWorld bridge episode.",
        "parameters": {"type": "object", "properties": {"episode_id": {"type": "string"}, "action": {"type": "string"}}, "required": ["episode_id", "action"], "additionalProperties": False},
    },
    "alfworld_status": {
        "name": "alfworld_status",
        "description": "Read an ALFWorld bridge episode without mutating it.",
        "parameters": {"type": "object", "properties": {"episode_id": {"type": "string"}}, "required": ["episode_id"], "additionalProperties": False},
    },
    "alfworld_abort": {
        "name": "alfworld_abort",
        "description": "Explicitly abort an active ALFWorld bridge episode.",
        "parameters": {"type": "object", "properties": {"episode_id": {"type": "string"}, "reason": {"type": "string"}}, "required": ["episode_id"], "additionalProperties": False},
    },
    "alfworld_reset": {
        "name": "alfworld_reset",
        "description": "Explicitly reset an active ALFWorld bridge episode with the same ID.",
        "parameters": {"type": "object", "properties": {"episode_id": {"type": "string"}}, "required": ["episode_id"], "additionalProperties": False},
    },
}


def _context(kwargs: Mapping[str, Any]) -> HermesContext:
    return HermesContext(
        run_id=os.environ.get("RQ1_RUN_ID"),
        attempt_id=os.environ.get("RQ1_ATTEMPT_ID"),
        profile=os.environ.get("RQ1_PROFILE"),
        session_id=str(kwargs["session_id"]) if kwargs.get("session_id") else None,
        tool_call_id=str(kwargs["tool_call_id"]) if kwargs.get("tool_call_id") else None,
        request_id=str(kwargs["api_request_id"]) if kwargs.get("api_request_id") else None,
    )


def _event_log() -> HermesEventLog:
    configured = os.environ.get("RQ1_HERMES_EVENT_LOG")
    return HermesEventLog(Path(configured) if configured else Path("artifacts/phase3/hermes-plugin-events.jsonl"))


def _adapter() -> HermesAdapter:
    return HermesAdapter(LocalBridgeClient(os.environ.get("RQ1_BRIDGE_URL", "http://127.0.0.1:8000")), _event_log())


def dispatch(tool_name: str, params: Mapping[str, Any], **kwargs: Any) -> str:
    """Return the required one-object JSON result from a bridge-backed tool."""
    return _adapter().invoke(tool_name, params, _context(kwargs)).to_json()


def pre_tool_call(tool_name: str, args: Mapping[str, Any] | None = None, **kwargs: Any) -> None:
    if tool_name.startswith("alfworld_"):
        _event_log().append(HermesIntegrationEvent("plugin_pre_tool_call", {"tool": tool_name, "args": redact(dict(args or {}))}, _context(kwargs).metadata(), False))
    return None


def post_tool_call(tool_name: str, args: Mapping[str, Any] | None = None, result: str | None = None, **kwargs: Any) -> None:
    if tool_name.startswith("alfworld_"):
        try:
            parsed: Any = json.loads(result) if result else None
        except (TypeError, json.JSONDecodeError):
            parsed = {"unparseable_result": True}
        _event_log().append(HermesIntegrationEvent("plugin_post_tool_call", {"tool": tool_name, "args": redact(dict(args or {})), "result": redact(parsed)}, _context(kwargs).metadata(), False))
    return None


def register_plugin(ctx: Any) -> None:
    """Register only on the documented current surface and explicit project trust."""
    if os.environ.get("HERMES_ENABLE_PROJECT_PLUGINS") != "1":
        raise RuntimeError("alfworld-experiment requires HERMES_ENABLE_PROJECT_PLUGINS=1")
    if not callable(getattr(ctx, "register_tool", None)) or not callable(getattr(ctx, "register_hook", None)):
        raise RuntimeError("Unsupported Hermes plugin surface: register_tool/register_hook are required")
    for name, schema in TOOL_SCHEMAS.items():
        ctx.register_tool(
            name=name,
            toolset=TOOLSET,
            schema=schema,
            handler=lambda params, _name=name, **kwargs: dispatch(_name, params, **kwargs),
            description=schema["description"],
        )
    ctx.register_hook("pre_tool_call", pre_tool_call)
    ctx.register_hook("post_tool_call", post_tool_call)
