"""Typed, dependency-free data models for the local bridge contract."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


ALLOWED_SPLITS = frozenset({"train", "valid_seen", "valid_unseen"})
CORRELATION_HEADERS = {
    "run_id": "X-RQ1-Run-ID",
    "attempt_id": "X-RQ1-Attempt-ID",
    "profile": "X-RQ1-Profile",
    "session_id": "X-RQ1-Session-ID",
    "tool_call_id": "X-RQ1-Tool-Call-ID",
    "request_id": "X-RQ1-Request-ID",
}


class RequestValidationError(ValueError):
    """Raised when an HTTP request does not meet the bridge contract."""


@dataclass(frozen=True)
class CorrelationMetadata:
    """Opaque experiment identifiers carried in optional localhost headers."""

    run_id: str | None = None
    attempt_id: str | None = None
    profile: str | None = None
    session_id: str | None = None
    tool_call_id: str | None = None
    request_id: str | None = None

    @classmethod
    def from_headers(cls, headers: Mapping[str, str]) -> "CorrelationMetadata":
        values: dict[str, str | None] = {}
        for field, header in CORRELATION_HEADERS.items():
            value = headers.get(header)
            if value is None:
                # BaseHTTPRequestHandler's headers are case-insensitive, but plain
                # mappings supplied by tests or callers might not be.
                value = headers.get(header.lower())
            if value is not None:
                value = value.strip()
                if not value or len(value) > 256:
                    raise RequestValidationError(f"{header} must be a non-empty value of at most 256 characters")
            values[field] = value
        return cls(**values)

    def to_dict(self) -> dict[str, str]:
        return {field: value for field, value in asdict(self).items() if value is not None}

    def is_empty(self) -> bool:
        return not self.to_dict()


def _required_string(payload: Mapping[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise RequestValidationError(f"{field} must be a non-empty string")
    return value.strip()


def _required_integer(payload: Mapping[str, Any], field: str, minimum: int | None = None) -> int:
    value = payload.get(field)
    if not isinstance(value, int) or isinstance(value, bool):
        raise RequestValidationError(f"{field} must be an integer")
    if minimum is not None and value < minimum:
        raise RequestValidationError(f"{field} must be at least {minimum}")
    return value


@dataclass(frozen=True)
class EpisodeStartRequest:
    task_id: str
    split: str
    seed: int
    action_limit: int

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "EpisodeStartRequest":
        task_id = _required_string(payload, "task_id")
        split = _required_string(payload, "split")
        if split not in ALLOWED_SPLITS:
            raise RequestValidationError(f"split must be one of: {', '.join(sorted(ALLOWED_SPLITS))}")
        return cls(task_id, split, _required_integer(payload, "seed"), _required_integer(payload, "action_limit", 1))


@dataclass(frozen=True)
class EpisodeStepRequest:
    episode_id: str
    action: str

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "EpisodeStepRequest":
        return cls(_required_string(payload, "episode_id"), _required_string(payload, "action"))


@dataclass(frozen=True)
class AdapterState:
    task_family: str
    instruction: str
    observation: str
    inventory: tuple[str, ...]
    admissible_actions: tuple[str, ...]
    reward: int | float
    step_number: int
    done: bool
    success: bool | None
    action_valid: bool | None = None
    field_sources: dict[str, str] | None = None
    freshness: str = "observed"


@dataclass(frozen=True)
class EpisodeResponse:
    episode_id: str
    task_id: str
    split: str
    task_family: str
    instruction: str
    observation: str
    inventory: tuple[str, ...]
    admissible_actions: tuple[str, ...]
    reward: int | float
    step_number: int
    action_count: int
    action_limit: int
    done: bool
    success: bool | None
    aborted: bool
    reset_count: int
    action_valid: bool | None = None
    field_sources: dict[str, str] | None = None
    freshness: str = "observed"

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["inventory"] = list(self.inventory)
        value["admissible_actions"] = list(self.admissible_actions)
        if self.field_sources is None:
            value.pop("field_sources")
        return value


@dataclass(frozen=True)
class HealthResponse:
    bridge_available: bool
    mode: str
    active_episode_count: int
    real_adapter_available: bool
    real_adapter_details: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
