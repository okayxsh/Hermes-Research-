"""Typed tool, event, and redaction models used by the Hermes boundary."""
from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

from rq1.bridge.models import CORRELATION_HEADERS, CorrelationMetadata
from rq1.utils.time import utc_now


class ToolValidationError(ValueError):
    """Raised before an invalid plugin call can reach the bridge."""


TOOL_NAMES = frozenset({"alfworld_start", "alfworld_step", "alfworld_status", "alfworld_abort", "alfworld_reset"})
TOOL_FIELDS = {
    "alfworld_start": frozenset({"task_id", "split", "seed", "action_limit"}),
    "alfworld_step": frozenset({"episode_id", "action"}),
    "alfworld_status": frozenset({"episode_id"}),
    "alfworld_abort": frozenset({"episode_id", "reason"}),
    "alfworld_reset": frozenset({"episode_id"}),
}
_REDACTED = "[REDACTED]"


def _string(payload: Mapping[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ToolValidationError(f"{field} must be a non-empty string")
    return value.strip()


def _integer(payload: Mapping[str, Any], field: str, minimum: int | None = None) -> int:
    value = payload.get(field)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ToolValidationError(f"{field} must be an integer")
    if minimum is not None and value < minimum:
        raise ToolValidationError(f"{field} must be at least {minimum}")
    return value


def validate_tool_payload(tool: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a normalized exact tool payload, rejecting unknown fields."""
    if tool not in TOOL_NAMES:
        raise ToolValidationError(f"Unsupported Hermes tool: {tool}")
    if not isinstance(payload, Mapping):
        raise ToolValidationError("tool parameters must be an object")
    unknown = sorted(set(payload) - TOOL_FIELDS[tool])
    if unknown:
        raise ToolValidationError(f"Unknown tool field(s): {', '.join(unknown)}")
    if tool == "alfworld_start":
        split = _string(payload, "split")
        if split not in {"train", "valid_seen", "valid_unseen"}:
            raise ToolValidationError("split must be one of: train, valid_seen, valid_unseen")
        return {
            "task_id": _string(payload, "task_id"),
            "split": split,
            "seed": _integer(payload, "seed"),
            "action_limit": _integer(payload, "action_limit", 1),
        }
    if tool == "alfworld_step":
        return {"episode_id": _string(payload, "episode_id"), "action": _string(payload, "action")}
    if tool == "alfworld_abort":
        result: dict[str, Any] = {"episode_id": _string(payload, "episode_id")}
        if "reason" in payload:
            result["reason"] = _string(payload, "reason")
        return result
    return {"episode_id": _string(payload, "episode_id")}


def redact(value: Any) -> Any:
    """Redact credentials recursively while retaining useful structured evidence."""
    if isinstance(value, Mapping):
        return {
            str(key): _REDACTED
            if any(word in str(key).lower() for word in ("token", "secret", "password", "authorization", "api_key"))
            else redact(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, tuple):
        return [redact(item) for item in value]
    return value


@dataclass(frozen=True)
class HermesContext:
    run_id: str | None = None
    attempt_id: str | None = None
    profile: str | None = None
    session_id: str | None = None
    tool_call_id: str | None = None
    request_id: str | None = None

    def with_request_id(self) -> "HermesContext":
        return self if self.request_id else HermesContext(**{**asdict(self), "request_id": str(uuid4())})

    def correlation(self) -> CorrelationMetadata:
        return CorrelationMetadata(**asdict(self))

    def metadata(self) -> dict[str, str]:
        return self.correlation().to_dict()


@dataclass(frozen=True)
class HermesToolError:
    code: str
    message: str
    status: int | None = None
    outcome_unknown: bool = False
    details: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        result = {key: value for key, value in asdict(self).items() if value is not None and value is not False}
        if self.outcome_unknown:
            result["outcome_unknown"] = True
        return result


@dataclass(frozen=True)
class HermesToolResult:
    ok: bool
    tool: str
    request_id: str
    latency_ms: int
    result: dict[str, Any] | None
    error: HermesToolError | None
    metadata: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "tool": self.tool,
            "request_id": self.request_id,
            "latency_ms": self.latency_ms,
            "result": self.result,
            "error": self.error.to_dict() if self.error else None,
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)


@dataclass(frozen=True)
class HermesIntegrationEvent:
    event: str
    payload: dict[str, Any]
    context: dict[str, str]
    simulated: bool
    relevance: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "timestamp": utc_now(),
            "event": self.event,
            "payload": redact(self.payload),
            "correlation": self.context,
            "simulated": self.simulated,
            "relevance": self.relevance,
        }


class HermesEventLog:
    """Small append-only JSONL writer shared by the adapter and project plugin."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()

    def append(self, event: HermesIntegrationEvent) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event.to_dict(), sort_keys=True) + "\n")


def correlation_headers(context: HermesContext) -> dict[str, str]:
    values = context.metadata()
    return {CORRELATION_HEADERS[field]: value for field, value in values.items()}
