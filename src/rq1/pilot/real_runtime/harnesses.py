"""Concrete harness adapters built on the shared real episode driver.

They are deliberately small adapters: acquisition and recovery share one
Hermes-registry/real-ALFWorld loop, while their scientific responsibilities
remain separate (no acquisition retrieval; exactly one recovery boundary).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from rq1.hermes.episode_driver import EpisodeDriverError, RealEpisodeDriver, RealEpisodeSession
from rq1.recovery.controlled_failure import (
    CANONICAL_FAILURE_MESSAGE,
    ControlledActionPerturbation,
    ControlledFailureError,
    FailureTrajectory,
    select_reversible_navigation_action,
)
from rq1.recovery.models import RecoveryState


def _recovery_state(state: Mapping[str, Any], *, task_goal: str) -> RecoveryState:
    # The bridge's ``instruction`` field is the indexed task ID; the semantic
    # instruction is the natural-language goal frozen at episode start.
    return RecoveryState(
        task_id=str(state["task_id"]),
        split=str(state["split"]),
        task_family=str(state["task_family"]),
        instruction=task_goal,
        observation=str(state["observation"]),
        inventory=tuple(str(item) for item in state.get("inventory") or ()),
        admissible_actions=tuple(str(item) for item in state.get("admissible_actions") or ()),
        step_number=int(state.get("step_number", 0)),
        done=bool(state.get("done")),
        success=bool(state.get("success")),
        action_valid=state.get("action_valid") if isinstance(state.get("action_valid"), bool) else None,
    )


class RealAcquisitionHarness:
    """Train-only concrete adapter for :func:`run_acquisition_episode`."""

    def __init__(
        self,
        driver: RealEpisodeDriver,
        *,
        output_dir: Path,
        run_id: str,
        attempt_id: str,
        library_hash: str = "0" * 64,
        library_size: int = 0,
    ) -> None:
        self.driver = driver
        self.output_dir = output_dir
        self.run_id = run_id
        self.attempt_id = attempt_id
        self.library_hash = library_hash
        self.library_size = library_size

    def run_episode(self, task_id: str, split: str, seed: int, action_limit: int) -> Mapping[str, Any]:
        if split != "train":
            raise ValueError("real acquisition harness accepts train only")
        with self.driver.session(
            output_dir=self.output_dir,
            run_id=self.run_id,
            attempt_id=self.attempt_id,
            profile="rq1-prelaunch-acquisition",
        ) as session:
            started = session.start(task_id, split, seed, action_limit)
            steps = session.run_model_loop(action_limit, phase="acquisition")
            state = session.state or started
            success = state.get("success") is True and state.get("done") is True
            candidate = None
            if success:
                actions = [record.action for record in steps]
                candidate = {
                    "title": f"Observed successful {started['task_family']} procedure",
                    "body": "Successful observed action trace:\n" + "\n".join(actions),
                }
            return {
                "success": success,
                "steps": len(steps),
                "actions": len(steps),
                "invalid_actions": session.invalid_model_actions,
                "skill_candidate": candidate,
            }

    def post_run_library_hash(self) -> str:
        return self.library_hash

    def post_run_library_size(self) -> int:
        return self.library_size


class RealRecoveryHarness:
    """Real valid_seen recovery adapter with a validation-only detour oracle."""

    def __init__(
        self,
        driver: RealEpisodeDriver,
        *,
        output_dir: Path,
        run_id: str,
        attempt_id: str,
        reference_actions: Sequence[str],
        total_action_limit: int = 80,
    ) -> None:
        self.driver = driver
        self.output_dir = output_dir
        self.run_id = run_id
        self.attempt_id = attempt_id
        self.reference_actions = tuple(reference_actions)
        self.total_action_limit = total_action_limit
        self.session: RealEpisodeSession | None = None
        self._memory: Mapping[str, Any] | None = None
        self._prefix: tuple[str, ...] = ()
        self._continuation: tuple[str, ...] = ()
        self._task: tuple[str, str, int] | None = None
        self._detour_action: str | None = None

    def close(self) -> None:
        if self.session is not None:
            self.session.__exit__(None, None, None)
            self.session = None

    def start_and_replay(
        self, task_id: str, split: str, seed: int, prefix_actions: Sequence[str]
    ) -> RecoveryState:
        if split != "valid_seen":
            raise ValueError("real recovery harness permits valid_seen only")
        self._prefix = tuple(prefix_actions)
        if self.reference_actions[: len(self._prefix)] != self._prefix:
            raise ControlledFailureError("prefix is not a frozen reference-route prefix", code="prefix_mismatch")
        self._continuation = self.reference_actions[len(self._prefix) :]
        if not self._continuation:
            raise ControlledFailureError("checkpoint leaves no reference continuation", code="checkpoint_terminal")
        self._task = (task_id, split, seed)
        self.session = self.driver.session(
            output_dir=self.output_dir,
            run_id=self.run_id,
            attempt_id=self.attempt_id,
            profile="rq1-prelaunch-recovery",
        )
        self.session.__enter__()
        self.session.start(task_id, split, seed, self.total_action_limit)
        self.session.apply_scripted(self._prefix, phase="checkpoint_replay")
        return self.current_state()

    def current_state(self) -> RecoveryState:
        if self.session is None or self.session.state is None or self.session.task_goal is None:
            raise EpisodeDriverError("recovery session is not active")
        return _recovery_state(self.session.state, task_goal=self.session.task_goal)

    def failure_environment(self) -> object:
        raise ControlledFailureError(
            "real recovery uses controlled action perturbation, not target relocation",
            code="target_relocation_retired",
        )

    def _oracle(self, detour: str) -> None:
        if self._task is None:
            raise EpisodeDriverError("recovery oracle lacks task state")
        task_id, split, seed = self._task
        oracle_dir = self.output_dir / "solvability-oracle"
        with self.driver.session(
            output_dir=oracle_dir,
            run_id=self.run_id + "-oracle",
            attempt_id=self.attempt_id + "-oracle",
            profile="rq1-prelaunch-oracle-validation",
        ) as oracle:
            oracle.start(task_id, split, seed, self.total_action_limit)
            oracle.apply_scripted(self._prefix, phase="oracle_checkpoint_replay")
            detoured = oracle.step(detour, phase="oracle_controlled_detour")
            if detoured.get("done") or detoured.get("action_valid") is not True:
                raise ControlledFailureError("oracle detour was not a live non-terminal transition", code="oracle_detour_invalid")
            for index, action in enumerate(self._continuation):
                result = oracle.step(action, phase="oracle_reference_completion")
                if result.get("action_valid") is not True:
                    raise ControlledFailureError("oracle reference continuation became invalid", code="oracle_route_invalid")
                if result.get("done") and index != len(self._continuation) - 1:
                    raise ControlledFailureError("oracle route terminated before its frozen end", code="oracle_early_terminal")
            if oracle.state is None or oracle.state.get("done") is not True or oracle.state.get("success") is not True:
                raise ControlledFailureError("oracle did not complete the perturbed task", code="oracle_unsolved")

    def apply_controlled_action_perturbation(
        self, checkpoint_id: str, failure_message: str = CANONICAL_FAILURE_MESSAGE
    ) -> ControlledActionPerturbation:
        if self.session is None or self.session.state is None:
            raise EpisodeDriverError("cannot perturb an inactive recovery session")
        expected = self._continuation[0]
        admissible = self.session.state.get("admissible_actions")
        if not isinstance(admissible, list):
            raise ControlledFailureError("checkpoint has no observable admissible actions", code="admissible_actions_unavailable")
        detour = select_reversible_navigation_action(admissible, expected)
        before = str(self.session.state.get("observation", ""))
        state = self.session.step(detour, phase="controlled_detour")
        if state.get("done") or state.get("action_valid") is not True:
            raise ControlledFailureError("controlled detour was not a live non-terminal transition", code="detour_invalid")
        if str(state.get("observation", "")) == before:
            raise ControlledFailureError("controlled detour did not change the observed state", code="detour_no_state_change")
        self._oracle(detour)
        self._detour_action = detour
        digest_payload = {
            "observation": state.get("observation"),
            "admissible_actions": state.get("admissible_actions"),
            "step_number": state.get("step_number"),
        }
        digest = hashlib.sha256(json.dumps(digest_payload, sort_keys=True).encode("utf-8")).hexdigest()
        return ControlledActionPerturbation(
            checkpoint_id=checkpoint_id,
            action=detour,
            expected_action=expected,
            failure_message=failure_message,
            post_state_digest=digest,
            solvable=True,
            selection_rule="sorted_first_safe_navigation_not_expected_then_real_oracle_rejoin",
        )

    def inject_recovery_memory(self, message: Mapping[str, Any]) -> None:
        self._memory = dict(message)
        if self.session is not None:
            self.session._event("recovery_memory_injected", {"memory": dict(message)})

    def run_recovery(self, action_budget: int) -> Sequence[Mapping[str, Any]]:
        if self.session is None:
            raise EpisodeDriverError("cannot recover without an active episode")
        failures_before = len(self.session.selection_failures)
        steps: list[dict[str, Any]] = [
            record.to_dict()
            for record in self.session.run_model_loop(
                action_budget, phase="recovery", recovery_memory=self._memory
            )
        ]
        # An exhausted selection dispatches nothing; record it as a non-action
        # invalid step so the existing result schema counts it.
        steps.extend(
            {
                "step": None,
                "phase": failure["phase"],
                "action": None,
                "action_valid": False,
                "done": False,
                "success": None,
                "observation": "",
                "controller_failure": failure["reason"],
                "selection_attempts": failure["attempts"],
            }
            for failure in self.session.selection_failures[failures_before:]
        )
        return tuple(steps)

    def recovery_succeeded(self) -> bool:
        return bool(self.session and self.session.state and self.session.state.get("success") is True)

    def failure_trajectory(self) -> FailureTrajectory:
        if self.session is None:
            return FailureTrajectory()
        pre = [item for item in self.session.records if item.phase == "checkpoint_replay"]
        post = [item for item in self.session.records if item.phase in {"controlled_detour", "recovery"}]
        return FailureTrajectory(
            pre_failure_actions=tuple(item.action for item in pre),
            pre_failure_observations=tuple(item.observation for item in pre),
            post_failure_actions=tuple(item.action for item in post),
            post_failure_observations=tuple(item.observation for item in post),
        )
