"""Deterministic fake recovery environment. This is not an ALFWorld implementation."""
from __future__ import annotations

from dataclasses import replace

from rq1.recovery.models import RecoveryState, ReferenceTrajectory
from rq1.utils.time import utc_now


class FakeRecoveryEnvironment:
    def __init__(self, task_id: str = "fake-recovery-001", seed: int = 1) -> None:
        self.task_id, self.seed, self.phase, self.target_location, self.perturbed = task_id, seed, 0, "countertop 1", False

    def reset(self) -> RecoveryState:
        self.phase, self.target_location, self.perturbed = 0, "countertop 1", False
        return self.state()

    def state(self, action_valid: bool | None = None) -> RecoveryState:
        if self.phase == 0:
            observation, actions = "You are in the room.", ("go to countertop 1",)
        elif self.phase == 1 and self.perturbed:
            observation, actions = "The target is no longer at its expected location. Reassess the state and continue.", ("go to pantry 1",)
        elif self.phase == 1:
            observation, actions = "You are at countertop 1. The target is here.", ("take target",)
        elif self.phase == 2:
            observation, actions = f"You are at {self.target_location}. The target is here.", ("take target",)
        elif self.phase == 3:
            observation, actions = "You hold the target.", ("complete task",)
        else: observation, actions = "Task complete.", ()
        return RecoveryState(self.task_id, "valid_seen", "heat_and_place", "Place the target safely.", observation,
            ("target",) if self.phase == 3 else (), actions, self.phase, self.phase >= 4, self.phase >= 4,
            action_valid, {"phase": self.phase, "target_location": self.target_location, "perturbed": self.perturbed})

    def step(self, action: str) -> RecoveryState:
        allowed = self.state().admissible_actions
        if action not in allowed: return self.state(False)
        # The unperturbed reference can take the target at the countertop;
        # after relocation it must first travel to the pantry.
        self.phase = 3 if self.phase == 1 and action == "take target" else self.phase + 1
        return self.state(True)

    def relocate_target(self) -> RecoveryState:
        if self.phase != 1: raise RuntimeError("Fake target relocation is valid only at the checkpoint.")
        self.target_location, self.perturbed = "pantry 1", True
        return self.state()

    def reference_trajectory(self) -> ReferenceTrajectory:
        self.reset(); states = [self.step("go to countertop 1"), self.step("take target"), self.step("complete task")]
        self.reset()
        return ReferenceTrajectory(self.task_id, "valid_seen", "heat_and_place", "deterministic_fake",
            ("go to countertop 1", "take target", "complete task"), tuple(s.observation for s in states),
            tuple(s.inventory for s in states), tuple(s.admissible_actions for s in states), (True, True, True),
            created_at=utc_now())
