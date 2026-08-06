"""Observed ALFWorld 0.4.2 text-only adapter.  No private-state mutation is used."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

from rq1.bridge.adapters.base import IndexedTask, RealALFWorldUnavailable
from rq1.bridge.adapters.capabilities import default_data_dir, probe_alfworld_capabilities
from rq1.bridge.adapters.task_index import TaskIndex, build_task_index
from rq1.bridge.models import AdapterState, EpisodeStartRequest


def _first(value: Any, default: Any = None) -> Any:
    return value[0] if isinstance(value, (list, tuple)) and value else default


def _state_digest(state: AdapterState) -> str:
    from rq1.recovery.models import RecoveryState
    from rq1.recovery.state_digest import observable_digest
    return observable_digest(RecoveryState("real", "valid_seen", state.task_family, state.instruction, state.observation,
        state.inventory, state.admissible_actions, state.step_number, state.done, bool(state.success), state.action_valid))


class RealALFWorldAdapter:
    """One explicit indexed task per adapter instance, using ALFWorld 0.4.2 only."""

    def __init__(self, data_dir: Path | None = None, environment_factory: Callable[[IndexedTask, int, int], Any] | None = None) -> None:
        self.data_dir = (data_dir or default_data_dir()).expanduser()
        self._factory = environment_factory or self._create_environment
        self._index: TaskIndex | None = None
        self._task: IndexedTask | None = None
        self._request: EpisodeStartRequest | None = None
        self._environment: Any = None
        self._state: AdapterState | None = None
        self._initial_digest: str | None = None

    def start(self, request: EpisodeStartRequest) -> AdapterState:
        report = probe_alfworld_capabilities(self.data_dir)
        if not report.real_adapter_ready:
            raise RealALFWorldUnavailable(report.details)
        self._index = build_task_index(self.data_dir)
        self._task = self._index.resolve(request.task_id, request.split)
        self._request = request
        self._environment = self._factory(self._task, request.seed, request.action_limit)
        self._state = self._observe_reset()
        self._initial_digest = _state_digest(self._state)
        return self._state

    def step(self, action: str) -> AdapterState:
        if self._state is None or self._environment is None:
            raise RealALFWorldUnavailable("Real ALFWorld adapter has not been started.")
        prior_actions = self._state.admissible_actions
        result = self._environment.step([action])
        if not isinstance(result, tuple) or len(result) != 4:
            raise RealALFWorldUnavailable("ALFWorld 0.4.2 step did not return (obs, scores, dones, infos).")
        obs, scores, dones, infos = result
        valid = action in prior_actions if self._state.field_sources and self._state.field_sources.get("admissible_actions") == "alfworld_info" else None
        self._state = self._map_state(_first(obs, ""), _first(scores, 0), _first(dones, False), infos, valid, self._state.step_number + 1)
        return self._state

    def status(self) -> AdapterState:
        if self._state is None:
            raise RealALFWorldUnavailable("Real ALFWorld adapter has not been started.")
        return replace(self._state, freshness="cached")

    def reset(self) -> AdapterState:
        if self._environment is None or self._task is None:
            raise RealALFWorldUnavailable("Real ALFWorld adapter has not been started.")
        state = self._observe_reset()
        if self._initial_digest is not None and _state_digest(state) != self._initial_digest:
            raise RealALFWorldUnavailable("ALFWorld reset did not reproduce the initial observable state; deterministic replay is not established.")
        self._state = state
        return state

    def abort(self, _reason: str | None = None) -> AdapterState:
        state = self.status()
        return replace(state, observation="Episode aborted by repository controller.", admissible_actions=(), done=True, success=False,
                       field_sources={**(state.field_sources or {}), "abort": "controller_side"})

    def _observe_reset(self) -> AdapterState:
        result = self._environment.reset()
        if not isinstance(result, tuple) or len(result) != 2:
            raise RealALFWorldUnavailable("ALFWorld 0.4.2 reset did not return (obs, info).")
        observations, infos = result
        return self._map_state(_first(observations, ""), 0, False, infos, None, 0)

    def _map_state(self, observation: Any, reward: Any, done: Any, infos: Any, action_valid: bool | None, step_number: int) -> AdapterState:
        if self._task is None:
            raise RealALFWorldUnavailable("No indexed task is associated with this adapter.")
        info = infos if isinstance(infos, dict) else {}
        commands = _first(info.get("admissible_commands"), ())
        actions = tuple(str(item) for item in commands) if isinstance(commands, (list, tuple)) else ()
        won = _first(info.get("won"), None)
        sources = {
            "task_family": "indexed_traj_data", "observation": "alfworld_reset_or_step", "reward": "alfworld_score",
            "admissible_actions": "alfworld_info" if isinstance(commands, (list, tuple)) else "unavailable",
            "success": "alfworld_info.won" if isinstance(won, bool) else "unavailable",
            "inventory": "unavailable_in_observed_alfworld_v042_surface", "instruction": "indexed_task_id",
            "data_identity": self._task.data_identity, "seed": "not_exposed_by_observed_alfworld_v042_surface",
        }
        return AdapterState(self._task.task_family, self._task.task_id, str(observation), (), actions,
            float(reward) if isinstance(reward, (int, float)) else 0, step_number, bool(done), won if isinstance(won, bool) else None,
            action_valid, sources, "observed")

    def _create_environment(self, task: IndexedTask, _seed: int, action_limit: int) -> Any:
        try:
            from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv
        except Exception as exc:
            raise RealALFWorldUnavailable("ALFWorld 0.4.2 text environment import failed.") from exc
        root = self.data_dir.resolve()
        config = {
            "dataset": {"data_path": str(root / "json_2.1.1" / "train"), "eval_id_data_path": str(root / "json_2.1.1" / "valid_seen"), "eval_ood_data_path": "", "num_train_games": -1, "num_eval_games": -1},
            "logic": {"domain": str(root / "logic" / "alfred.pddl"), "grammar": str(root / "logic" / "alfred.twl2")},
            "env": {"goal_desc_human_anns_prob": 0.0, "task_types": [1, 2, 3, 4, 5, 6], "domain_randomization": False, "expert_type": "handcoded"},
            "general": {"training_method": "dagger"}, "dagger": {"training": {"max_nb_steps_per_episode": action_limit}},
        }
        train_eval = "train" if task.split == "train" else "eval_in_distribution"
        wrapper = AlfredTWEnv(config, train_eval=train_eval)
        wrapper.game_files, wrapper.num_games = [str(root / task.game_file)], 1
        return wrapper.init_env(batch_size=1)
