"""ALFWorld adapter implementations and the real-integration capability boundary."""
from __future__ import annotations

import hashlib
import importlib.util
from dataclasses import replace

from rq1.bridge.models import AdapterState, EpisodeStartRequest
from rq1.integrations.contracts import CapabilityResult


class FakeALFWorldAdapter:
    """A deterministic, deliberately small text-world contract fixture."""

    def __init__(self) -> None:
        self._request: EpisodeStartRequest | None = None
        self._phase = 0

    def start(self, request: EpisodeStartRequest) -> AdapterState:
        self._request = request
        self._phase = 0
        return self._state(reward=0, action_valid=None)

    def step(self, action: str) -> AdapterState:
        if self._request is None:
            raise RuntimeError("Fake adapter has not been started")
        valid_actions = self._actions()
        if action not in valid_actions:
            return self._state(reward=0, action_valid=False)
        if action == "go to countertop 1":
            self._phase = 1
        elif action == "take target":
            self._phase = 2
        elif action == "complete task":
            self._phase = 3
        return self._state(reward=1 if self._phase == 3 else 0, action_valid=True)

    def status(self) -> AdapterState:
        if self._request is None:
            raise RuntimeError("Fake adapter has not been started")
        return self._state(reward=0, action_valid=None)

    def abort(self, _reason: str | None = None) -> AdapterState:
        state = self.status()
        return replace(state, observation="Episode aborted by caller.", admissible_actions=(), done=True, success=False)

    def reset(self) -> AdapterState:
        if self._request is None:
            raise RuntimeError("Fake adapter has not been started")
        self._phase = 0
        return self._state(reward=0, action_valid=None)

    def _family(self) -> str:
        assert self._request is not None
        families = ("heat_and_place", "clean_and_place", "pick_and_place")
        digest = hashlib.sha256(f"{self._request.task_id}:{self._request.seed}".encode("utf-8")).digest()
        return families[digest[0] % len(families)]

    def _actions(self) -> tuple[str, ...]:
        return {
            0: ("look", "go to countertop 1"),
            1: ("take target",),
            2: ("complete task",),
            3: (),
        }[self._phase]

    def _state(self, reward: int, action_valid: bool | None) -> AdapterState:
        assert self._request is not None
        observations = (
            "You are in the middle of a room. A countertop is nearby.",
            "You are at countertop 1. The target object is here.",
            "You are holding the target object. Complete the task.",
            "The task is complete.",
        )
        return AdapterState(
            task_family=self._family(),
            instruction=f"Complete deterministic fixture task {self._request.task_id}.",
            observation=observations[self._phase],
            inventory=("target object",) if self._phase >= 2 else (),
            admissible_actions=self._actions(),
            reward=reward,
            step_number=self._phase,
            done=self._phase == 3,
            success=self._phase == 3,
            action_valid=action_valid,
        )


def real_adapter_capability() -> CapabilityResult:
    """Check only package discoverability; no ALFWorld import or data access occurs."""
    found = importlib.util.find_spec("alfworld") is not None
    if found:
        return CapabilityResult(False, None, "ALFWorld package detected, but its API and data are unverified by the pilot.")
    return CapabilityResult(False, None, "ALFWorld package is not installed; real bridge remains unverified and unavailable.")


class RealALFWorldAdapter:
    """Fails closed until a later capability-informed real implementation exists."""

    def __init__(self) -> None:
        details = real_adapter_capability().details
        raise RuntimeError(f"Real ALFWorld adapter is unavailable: {details}")
