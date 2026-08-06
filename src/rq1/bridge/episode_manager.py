"""Thread-safe episode ownership, lifecycle control, and raw event logging."""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from rq1.bridge.models import AdapterState, CorrelationMetadata, EpisodeResponse, EpisodeStartRequest
from rq1.utils.time import utc_now


class BridgeError(RuntimeError):
    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


class RawEventLog:
    def __init__(self, root: Path, episode_id: str, correlation: CorrelationMetadata) -> None:
        self.path = root / f"{episode_id}.jsonl"
        self.correlation = correlation
        self._lock = threading.Lock()

    def append(
        self, event: str, payload: dict[str, Any], correlation: CorrelationMetadata | None = None
    ) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        event_correlation = self.correlation.to_dict()
        if correlation is not None:
            event_correlation.update(correlation.to_dict())
        record = {
            "timestamp": utc_now(),
            "event": event,
            "episode_id": self.path.stem,
            "correlation": event_correlation,
            "payload": payload,
        }
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


@dataclass
class EpisodeRecord:
    request: EpisodeStartRequest
    adapter: Any
    state: AdapterState
    logger: RawEventLog
    correlation: CorrelationMetadata
    action_count: int = 0
    reset_count: int = 0
    aborted: bool = False


class EpisodeManager:
    def __init__(self, adapter_factory: Callable[[], Any], log_root: Path) -> None:
        self._adapter_factory = adapter_factory
        self._log_root = log_root
        self._episodes: dict[str, EpisodeRecord] = {}
        self._lock = threading.RLock()

    @property
    def active_episode_count(self) -> int:
        with self._lock:
            return sum(not item.state.done for item in self._episodes.values())

    def start(self, request: EpisodeStartRequest, correlation: CorrelationMetadata | None = None) -> EpisodeResponse:
        with self._lock:
            episode_id = str(uuid4())
            correlation = correlation or CorrelationMetadata()
            logger = RawEventLog(self._log_root, episode_id, correlation)
            adapter = self._adapter_factory()
            try:
                state = adapter.start(request)
            except Exception as exc:
                logger.append("adapter_error", {"operation": "start", "error": str(exc)}, correlation)
                raise BridgeError(500, "Adapter failed to start episode") from exc
            record = EpisodeRecord(request, adapter, state, logger, correlation)
            self._episodes[episode_id] = record
            response = self._response(episode_id, record)
            logger.append("start", {"request": request.__dict__, "response": response.to_dict()}, correlation)
            return response

    def status(self, episode_id: str, correlation: CorrelationMetadata | None = None) -> EpisodeResponse:
        with self._lock:
            record = self._record(episode_id)
            self._check_correlation(record, correlation)
            return self._response(episode_id, record)

    def step(self, episode_id: str, action: str, correlation: CorrelationMetadata | None = None) -> EpisodeResponse:
        with self._lock:
            record = self._active_record(episode_id)
            self._check_correlation(record, correlation)
            try:
                state = record.adapter.step(action)
            except Exception as exc:
                record.logger.append("adapter_error", {"operation": "step", "action": action, "error": str(exc)}, correlation)
                raise BridgeError(500, "Adapter failed while stepping episode") from exc
            record.action_count += 1
            if record.action_count >= record.request.action_limit and not state.done:
                state = replace(
                    state, observation="Action limit reached.", admissible_actions=(), done=True, success=False, action_valid=state.action_valid
                )
            record.state = state
            response = self._response(episode_id, record)
            record.logger.append("step", {"request": {"action": action}, "response": response.to_dict()}, correlation)
            if not state.action_valid:
                record.logger.append("invalid_action", {"action": action, "action_count": record.action_count}, correlation)
            if state.done:
                record.logger.append("terminal", {"response": response.to_dict()}, correlation)
            return response

    def abort(
        self, episode_id: str, reason: str | None = None, correlation: CorrelationMetadata | None = None
    ) -> EpisodeResponse:
        with self._lock:
            record = self._active_record(episode_id)
            self._check_correlation(record, correlation)
            try:
                record.state = record.adapter.abort(reason)
            except Exception as exc:
                record.logger.append("adapter_error", {"operation": "abort", "error": str(exc)}, correlation)
                raise BridgeError(500, "Adapter failed while aborting episode") from exc
            record.aborted = True
            response = self._response(episode_id, record)
            record.logger.append("abort", {"reason": reason, "response": response.to_dict()}, correlation)
            record.logger.append("terminal", {"response": response.to_dict()}, correlation)
            return response

    def reset(self, episode_id: str, correlation: CorrelationMetadata | None = None) -> EpisodeResponse:
        with self._lock:
            record = self._active_record(episode_id)
            self._check_correlation(record, correlation)
            try:
                record.state = record.adapter.reset()
            except Exception as exc:
                record.logger.append("adapter_error", {"operation": "reset", "error": str(exc)}, correlation)
                raise BridgeError(500, "Adapter failed while resetting episode") from exc
            record.action_count = 0
            record.reset_count += 1
            response = self._response(episode_id, record)
            record.logger.append("reset", {"response": response.to_dict()}, correlation)
            return response

    def _record(self, episode_id: str) -> EpisodeRecord:
        try:
            return self._episodes[episode_id]
        except KeyError as exc:
            raise BridgeError(404, f"Unknown episode_id: {episode_id}") from exc

    def _active_record(self, episode_id: str) -> EpisodeRecord:
        record = self._record(episode_id)
        if record.state.done:
            raise BridgeError(409, "Episode is terminal; start a new episode instead.")
        return record

    @staticmethod
    def _check_correlation(record: EpisodeRecord, supplied: CorrelationMetadata | None) -> None:
        """Allow absent headers for Phase 2 callers; reject supplied conflicts."""
        if supplied is None or supplied.is_empty():
            return
        expected = record.correlation.to_dict()
        # Request and tool-call IDs are intentionally per-operation. Run,
        # attempt, profile, and session identifiers define episode ownership.
        for field, value in supplied.to_dict().items():
            if field in {"request_id", "tool_call_id"}:
                continue
            if expected.get(field) != value:
                raise BridgeError(409, f"Correlation metadata conflict for {field}")

    @staticmethod
    def _response(episode_id: str, record: EpisodeRecord) -> EpisodeResponse:
        state = record.state
        return EpisodeResponse(
            episode_id=episode_id, task_id=record.request.task_id, split=record.request.split,
            task_family=state.task_family, instruction=state.instruction, observation=state.observation,
            inventory=state.inventory, admissible_actions=state.admissible_actions, reward=state.reward,
            step_number=state.step_number, action_count=record.action_count, action_limit=record.request.action_limit,
            done=state.done, success=state.success, aborted=record.aborted, reset_count=record.reset_count,
            action_valid=state.action_valid,
        )
