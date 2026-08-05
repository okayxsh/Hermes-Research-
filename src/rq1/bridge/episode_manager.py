"""Thread-safe episode ownership, lifecycle control, and raw event logging."""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from rq1.bridge.models import AdapterState, EpisodeResponse, EpisodeStartRequest
from rq1.utils.time import utc_now


class BridgeError(RuntimeError):
    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


class RawEventLog:
    def __init__(self, root: Path, episode_id: str) -> None:
        self.path = root / f"{episode_id}.jsonl"
        self._lock = threading.Lock()

    def append(self, event: str, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = {"timestamp": utc_now(), "event": event, "episode_id": self.path.stem, "payload": payload}
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


@dataclass
class EpisodeRecord:
    request: EpisodeStartRequest
    adapter: Any
    state: AdapterState
    logger: RawEventLog
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

    def start(self, request: EpisodeStartRequest) -> EpisodeResponse:
        with self._lock:
            episode_id = str(uuid4())
            logger = RawEventLog(self._log_root, episode_id)
            adapter = self._adapter_factory()
            try:
                state = adapter.start(request)
            except Exception as exc:
                logger.append("adapter_error", {"operation": "start", "error": str(exc)})
                raise BridgeError(500, "Adapter failed to start episode") from exc
            record = EpisodeRecord(request, adapter, state, logger)
            self._episodes[episode_id] = record
            response = self._response(episode_id, record)
            logger.append("start", {"request": request.__dict__, "response": response.to_dict()})
            return response

    def status(self, episode_id: str) -> EpisodeResponse:
        with self._lock:
            return self._response(episode_id, self._record(episode_id))

    def step(self, episode_id: str, action: str) -> EpisodeResponse:
        with self._lock:
            record = self._active_record(episode_id)
            try:
                state = record.adapter.step(action)
            except Exception as exc:
                record.logger.append("adapter_error", {"operation": "step", "action": action, "error": str(exc)})
                raise BridgeError(500, "Adapter failed while stepping episode") from exc
            record.action_count += 1
            if record.action_count >= record.request.action_limit and not state.done:
                state = replace(
                    state, observation="Action limit reached.", admissible_actions=(), done=True, success=False, action_valid=state.action_valid
                )
            record.state = state
            response = self._response(episode_id, record)
            record.logger.append("step", {"request": {"action": action}, "response": response.to_dict()})
            if not state.action_valid:
                record.logger.append("invalid_action", {"action": action, "action_count": record.action_count})
            if state.done:
                record.logger.append("terminal", {"response": response.to_dict()})
            return response

    def abort(self, episode_id: str, reason: str | None = None) -> EpisodeResponse:
        with self._lock:
            record = self._active_record(episode_id)
            try:
                record.state = record.adapter.abort(reason)
            except Exception as exc:
                record.logger.append("adapter_error", {"operation": "abort", "error": str(exc)})
                raise BridgeError(500, "Adapter failed while aborting episode") from exc
            record.aborted = True
            response = self._response(episode_id, record)
            record.logger.append("abort", {"reason": reason, "response": response.to_dict()})
            record.logger.append("terminal", {"response": response.to_dict()})
            return response

    def reset(self, episode_id: str) -> EpisodeResponse:
        with self._lock:
            record = self._active_record(episode_id)
            try:
                record.state = record.adapter.reset()
            except Exception as exc:
                record.logger.append("adapter_error", {"operation": "reset", "error": str(exc)})
                raise BridgeError(500, "Adapter failed while resetting episode") from exc
            record.action_count = 0
            record.reset_count += 1
            response = self._response(episode_id, record)
            record.logger.append("reset", {"response": response.to_dict()})
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
