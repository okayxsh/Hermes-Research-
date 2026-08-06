"""ALFWorld adapter implementations and the real-integration capability boundary."""
from __future__ import annotations

import hashlib
from dataclasses import replace

from rq1.bridge.models import AdapterState, EpisodeStartRequest
from rq1.bridge.adapters.alfworld_v042 import RealALFWorldAdapter
from rq1.bridge.adapters.capabilities import ALFWorldCapabilityReport, probe_alfworld_capabilities


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
        fixture_families = {
            "fake-pick-and-place": "pick_and_place",
            "fake-pick-two": "pick_two_and_place",
            "fake-look-at": "look_at_object",
            "fake-clean-and-place": "clean_and_place",
            "fake-heat-and-place": "heat_and_place",
            "fake-cool-and-place": "cool_and_place",
        }
        for prefix, family in fixture_families.items():
            if self._request.task_id.startswith(prefix):
                return family
        families = (
            "pick_and_place", "pick_two_and_place", "look_at_object",
            "clean_and_place", "heat_and_place", "cool_and_place",
        )
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


def real_adapter_capability() -> ALFWorldCapabilityReport:
    """Return structured local capability evidence; it never claims live compatibility."""
    return probe_alfworld_capabilities()
