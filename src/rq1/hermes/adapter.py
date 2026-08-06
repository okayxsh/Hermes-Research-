"""Local-only bridge client and fake/real Hermes adapter boundaries."""
from __future__ import annotations

import json
import socket
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from rq1.hermes.models import (
    HermesContext,
    HermesEventLog,
    HermesIntegrationEvent,
    HermesToolError,
    HermesToolResult,
    ToolValidationError,
    correlation_headers,
    validate_tool_payload,
)


class BridgeTransport(Protocol):
    def __call__(self, request: Request, timeout: float) -> tuple[int, bytes]: ...


class BridgeTransportError(RuntimeError):
    def __init__(self, code: str, message: str, *, outcome_unknown: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.outcome_unknown = outcome_unknown


@dataclass(frozen=True)
class BridgeResponse:
    status: int
    body: dict[str, Any]


def _default_transport(request: Request, timeout: float) -> tuple[int, bytes]:
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.status, response.read()
    except HTTPError as exc:
        return exc.code, exc.read()
    except (socket.timeout, TimeoutError) as exc:
        raise BridgeTransportError("timeout", "The localhost bridge request timed out") from exc
    except URLError as exc:
        raise BridgeTransportError("bridge_unavailable", "The localhost bridge is unavailable") from exc
    except OSError as exc:
        raise BridgeTransportError("bridge_unavailable", "The localhost bridge is unavailable") from exc


class LocalBridgeClient:
    """Dependency-free HTTP client that can only target the local bridge."""

    def __init__(self, base_url: str = "http://127.0.0.1:8000", transport: BridgeTransport | None = None) -> None:
        parsed = urlparse(base_url)
        if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"} or parsed.username or parsed.password:
            raise ValueError("bridge URL must be an http://127.0.0.1 or http://localhost address")
        if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
            raise ValueError("bridge URL must not contain a path, query, or fragment")
        self.base_url = base_url.rstrip("/")
        self.transport = transport or _default_transport

    def request(
        self,
        method: str,
        path: str,
        payload: Mapping[str, Any] | None,
        context: HermesContext,
        *,
        timeout: float = 5.0,
        retry_once: bool = False,
        outcome_unknown_on_timeout: bool = False,
    ) -> BridgeResponse:
        attempts = 2 if retry_once else 1
        last_error: BridgeTransportError | None = None
        for _ in range(attempts):
            data = None if payload is None else json.dumps(dict(payload), sort_keys=True).encode("utf-8")
            request = Request(self.base_url + path, data=data, method=method)
            request.add_header("Accept", "application/json")
            if data is not None:
                request.add_header("Content-Type", "application/json")
            for header, value in correlation_headers(context).items():
                request.add_header(header, value)
            try:
                status, raw = self.transport(request, timeout)
                try:
                    body = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise BridgeTransportError("invalid_bridge_response", "Bridge returned malformed JSON") from exc
                if not isinstance(body, dict):
                    raise BridgeTransportError("invalid_bridge_response", "Bridge returned a non-object JSON response")
                return BridgeResponse(status, body)
            except BridgeTransportError as exc:
                last_error = exc
        assert last_error is not None
        if outcome_unknown_on_timeout and last_error.code == "timeout":
            raise BridgeTransportError(last_error.code, str(last_error), outcome_unknown=True) from last_error
        raise last_error


_ROUTES: dict[str, tuple[str, Callable[[dict[str, Any]], str], bool, bool]] = {
    "alfworld_start": ("POST", lambda _payload: "/episode/start", True, False),
    "alfworld_step": ("POST", lambda _payload: "/episode/step", True, False),
    "alfworld_status": ("GET", lambda payload: f"/episode/{payload['episode_id']}/status", False, True),
    "alfworld_abort": ("POST", lambda payload: f"/episode/{payload['episode_id']}/abort", True, False),
    "alfworld_reset": ("POST", lambda payload: f"/episode/{payload['episode_id']}/reset", True, False),
}


class HermesAdapter:
    """Typed dispatcher for the five bridge-backed Hermes tools."""

    simulated = False

    def __init__(self, client: LocalBridgeClient, event_log: HermesEventLog | None = None) -> None:
        self.client = client
        self.event_log = event_log

    def invoke(self, tool: str, payload: Mapping[str, Any], context: HermesContext | None = None) -> HermesToolResult:
        context = (context or HermesContext()).with_request_id()
        started = time.monotonic()
        try:
            normalized = validate_tool_payload(tool, payload)
            method, path_builder, mutating, retryable = _ROUTES[tool]
            if tool == "alfworld_abort":
                outbound = {key: value for key, value in normalized.items() if key != "episode_id"}
            else:
                outbound = normalized if tool in {"alfworld_start", "alfworld_step"} else None
            response = self.client.request(
                method,
                path_builder(normalized),
                outbound,
                context,
                retry_once=retryable,
                outcome_unknown_on_timeout=mutating,
            )
            if not 200 <= response.status < 300:
                error_body = response.body.get("error")
                message = error_body.get("message") if isinstance(error_body, Mapping) else "Bridge returned an error response"
                code = "correlation_conflict" if response.status == 409 and "Correlation metadata conflict" in str(message) else "bridge_error"
                result = self._failure(tool, context, started, HermesToolError(code, str(message), response.status, details=response.body))
            elif not self._valid_success(tool, response.body):
                result = self._failure(
                    tool,
                    context,
                    started,
                    HermesToolError("invalid_bridge_response", "Bridge success response failed contract validation", response.status, details=response.body),
                )
            else:
                result = HermesToolResult(True, tool, context.request_id or "", self._latency(started), response.body, None, context.metadata())
        except ToolValidationError as exc:
            result = self._failure(tool, context, started, HermesToolError("validation_failure", str(exc)))
        except BridgeTransportError as exc:
            result = self._failure(
                tool,
                context,
                started,
                HermesToolError(exc.code, str(exc), outcome_unknown=exc.outcome_unknown),
            )
        except Exception:
            result = self._failure(tool, context, started, HermesToolError("internal_failure", "Hermes bridge adapter failed safely"))
        self._record(result, payload, context)
        return result

    def health(self, context: HermesContext | None = None) -> HermesToolResult:
        """Internal health probe; it is intentionally not a model-visible Hermes tool."""
        context = (context or HermesContext()).with_request_id()
        started = time.monotonic()
        try:
            response = self.client.request("POST", "/health", {}, context, retry_once=True)
            if response.status != 200 or response.body.get("bridge_available") is not True:
                return self._failure("health", context, started, HermesToolError("bridge_error", "Bridge health check failed", response.status, details=response.body))
            result = HermesToolResult(True, "health", context.request_id or "", self._latency(started), response.body, None, context.metadata())
        except BridgeTransportError as exc:
            result = self._failure("health", context, started, HermesToolError(exc.code, str(exc)))
        except Exception:
            result = self._failure("health", context, started, HermesToolError("internal_failure", "Hermes bridge health probe failed safely"))
        self._record(result, {}, context)
        return result

    @staticmethod
    def _valid_success(tool: str, response: Mapping[str, Any]) -> bool:
        if tool == "alfworld_start":
            return isinstance(response.get("episode_id"), str) and isinstance(response.get("observation"), str)
        return isinstance(response.get("episode_id"), str) and isinstance(response.get("done"), bool)

    @staticmethod
    def _latency(started: float) -> int:
        return max(0, round((time.monotonic() - started) * 1000))

    def _failure(self, tool: str, context: HermesContext, started: float, error: HermesToolError) -> HermesToolResult:
        return HermesToolResult(False, tool, context.request_id or "", self._latency(started), None, error, context.metadata())

    def _record(self, result: HermesToolResult, payload: Mapping[str, Any], context: HermesContext) -> None:
        if self.event_log:
            self.event_log.append(
                HermesIntegrationEvent(
                    "hermes_tool_result",
                    {"tool": result.tool, "request": dict(payload), "response": result.to_dict()},
                    context.metadata(),
                    simulated=self.simulated,
                )
            )


class FakeHermesAdapter(HermesAdapter):
    """Deterministic local adapter used by tests and fake Phase 3 verification."""

    simulated = True

    def emit_skill_event(self, operation: str, skill_id: str, relevance: str, context: HermesContext) -> None:
        if operation not in {"skill_index_available", "skill_selected", "skill_loaded", "skill_managed", "unknown_native_skill_operation"}:
            raise ToolValidationError(f"Unsupported skill event: {operation}")
        if relevance not in {"relevant", "irrelevant", "unknown"}:
            raise ToolValidationError("skill relevance must be relevant, irrelevant, or unknown")
        if self.event_log:
            self.event_log.append(
                HermesIntegrationEvent(operation, {"skill_id": skill_id, "native_operation": operation}, context.with_request_id().metadata(), True, relevance)
            )


class RealHermesAdapter(HermesAdapter):
    """Fail-closed facade until a live installed Hermes plugin test has evidence."""

    simulated = False

    def __init__(self, client: LocalBridgeClient, capability_report: Any, event_log: HermesEventLog | None = None) -> None:
        super().__init__(client, event_log)
        self.capability_report = capability_report

    def assert_supported(self) -> None:
        supported = getattr(self.capability_report, "plugin_supported", False) and getattr(self.capability_report, "hook_supported", False)
        if not supported:
            raise BridgeTransportError("unsupported_hermes_capability", "Installed Hermes plugin or hook capability is unsupported or unverified")

    def invoke(self, tool: str, payload: Mapping[str, Any], context: HermesContext | None = None) -> HermesToolResult:
        context = (context or HermesContext()).with_request_id()
        started = time.monotonic()
        try:
            self.assert_supported()
        except BridgeTransportError as exc:
            result = self._failure(tool, context, started, HermesToolError(exc.code, str(exc)))
            self._record(result, payload, context)
            return result
        return super().invoke(tool, payload, context)
