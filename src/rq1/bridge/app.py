"""Dependency-free local HTTP app for the ALFWorld bridge contract."""
from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from rq1.bridge.environment import FakeALFWorldAdapter, RealALFWorldAdapter, real_adapter_capability
from rq1.bridge.episode_manager import BridgeError, EpisodeManager
from rq1.bridge.models import (
    CorrelationMetadata,
    EpisodeStartRequest,
    EpisodeStepRequest,
    HealthResponse,
    RequestValidationError,
)


def _decode_json(handler: BaseHTTPRequestHandler, required: bool = True) -> dict[str, Any]:
    length = handler.headers.get("Content-Length")
    if length in (None, "", "0"):
        if required:
            raise BridgeError(400, "JSON request body is required")
        return {}
    try:
        raw = handler.rfile.read(int(length))
        value = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BridgeError(400, "Malformed JSON request body") from exc
    if not isinstance(value, dict):
        raise BridgeError(422, "JSON request body must be an object")
    return value


def create_bridge_server(
    log_root: Path,
    host: str = "127.0.0.1",
    port: int = 8000,
    manager: EpisodeManager | None = None,
    mode: str = "fake",
    data_dir: Path | None = None,
) -> ThreadingHTTPServer:
    """Return an unstarted localhost server; real mode is explicit and capability-gated."""
    if manager is not None and mode != "fake":
        raise ValueError("Pass either a manager or an adapter mode, not both.")
    if mode not in {"fake", "real"}:
        raise ValueError("mode must be fake or real")
    if mode == "real":
        capability = real_adapter_capability()
        if not capability.real_adapter_ready:
            raise BridgeError(503, capability.details)
        episode_manager = EpisodeManager(lambda: RealALFWorldAdapter(data_dir=data_dir), log_root)
    else:
        episode_manager = manager or EpisodeManager(FakeALFWorldAdapter, log_root)

    class BridgeRequestHandler(BaseHTTPRequestHandler):
        server_version = "RQ1ALFWorldBridge/0.2"

        def do_POST(self) -> None:  # noqa: N802
            try:
                path = urlparse(self.path).path
                correlation = CorrelationMetadata.from_headers(self.headers)
                if path == "/health":
                    payload = _decode_json(self, required=False)
                    if payload:
                        raise BridgeError(422, "health does not accept request fields")
                    capability = real_adapter_capability()
                    response = HealthResponse(True, mode, episode_manager.active_episode_count, capability.available, capability.details).to_dict()
                elif path == "/episode/start":
                    response = episode_manager.start(EpisodeStartRequest.from_payload(_decode_json(self)), correlation).to_dict()
                elif path == "/episode/step":
                    request = EpisodeStepRequest.from_payload(_decode_json(self))
                    response = episode_manager.step(request.episode_id, request.action, correlation).to_dict()
                elif path.endswith("/abort") and path.startswith("/episode/"):
                    reason = _decode_json(self, required=False).get("reason")
                    if reason is not None and (not isinstance(reason, str) or not reason.strip()):
                        raise RequestValidationError("reason must be a non-empty string when provided")
                    response = episode_manager.abort(_episode_id(path, "abort"), reason, correlation).to_dict()
                elif path.endswith("/reset") and path.startswith("/episode/"):
                    if _decode_json(self, required=False):
                        raise BridgeError(422, "reset does not accept request fields")
                    response = episode_manager.reset(_episode_id(path, "reset"), correlation).to_dict()
                else:
                    raise BridgeError(404, "Unknown route")
                self._write(HTTPStatus.OK, response)
            except RequestValidationError as exc:
                self._error(422, str(exc))
            except BridgeError as exc:
                self._error(exc.status_code, str(exc))
            except Exception:
                self._error(500, "Unexpected bridge server error")

        def do_GET(self) -> None:  # noqa: N802
            try:
                path = urlparse(self.path).path
                correlation = CorrelationMetadata.from_headers(self.headers)
                if path.startswith("/episode/") and path.endswith("/status"):
                    self._write(HTTPStatus.OK, episode_manager.status(_episode_id(path, "status"), correlation).to_dict())
                else:
                    raise BridgeError(404, "Unknown route")
            except BridgeError as exc:
                self._error(exc.status_code, str(exc))

        def _write(self, status: int | HTTPStatus, payload: dict[str, Any]) -> None:
            encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
            self.send_response(int(status))
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def _error(self, status: int, message: str) -> None:
            self._write(status, {"error": {"status": status, "message": message}})

        def log_message(self, _format: str, *_args: object) -> None:
            return

    server = ThreadingHTTPServer((host, port), BridgeRequestHandler)
    server.manager = episode_manager  # type: ignore[attr-defined]
    return server


def _episode_id(path: str, action: str) -> str:
    prefix = "/episode/"
    suffix = f"/{action}"
    value = path[len(prefix):-len(suffix)]
    if not value or "/" in value:
        raise BridgeError(404, "Unknown route")
    return value
