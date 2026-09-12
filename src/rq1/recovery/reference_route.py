"""Validation-only expert reference routes for controlled action perturbations.

This module never drives the scientific agent.  It opens the one frozen game
file with ALFWorld's observed hand-coded expert wrapper solely to derive an
auditable reference route and to choose a checkpoint before evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


class ReferenceRouteError(RuntimeError):
    pass


@dataclass(frozen=True)
class ReferenceRoute:
    task_id: str
    split: str
    actions: tuple[str, ...]


def _game_path(data_dir: Path, task_id: str, split: str) -> Path:
    prefix = split + ":"
    if not task_id.startswith(prefix):
        raise ReferenceRouteError("task ID does not belong to the requested split")
    relative = task_id.removeprefix(prefix)
    candidate = (data_dir / "json_2.1.1" / split / relative / "game.tw-pddl").resolve()
    expected_root = (data_dir / "json_2.1.1" / split).resolve()
    if expected_root not in candidate.parents or not candidate.is_file():
        raise ReferenceRouteError("frozen task game file is unavailable")
    return candidate


def derive_handcoded_reference(data_dir: Path, task_id: str, split: str) -> ReferenceRoute:
    """Derive a route from ALFWorld's installed hand-coded expert surface."""
    if split != "valid_seen":
        raise ReferenceRouteError("recovery reference routes are valid_seen-only before final evaluation")
    try:
        from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv
    except Exception as exc:  # pragma: no cover - requires the optional runtime
        raise ReferenceRouteError("ALFWorld expert wrapper is unavailable") from exc

    root = data_dir.resolve()
    game = _game_path(root, task_id, split)
    config = {
        "dataset": {
            # This one-game root avoids rescanning the complete network volume.
            "data_path": str(game.parent),
            "eval_id_data_path": str(game.parent),
            "eval_ood_data_path": "",
            "num_train_games": -1,
            "num_eval_games": -1,
        },
        "logic": {"domain": str(root / "logic" / "alfred.pddl"), "grammar": str(root / "logic" / "alfred.twl2")},
        "env": {
            "goal_desc_human_anns_prob": 0.0,
            "task_types": [1, 2, 3, 4, 5, 6],
            "domain_randomization": False,
            "expert_type": "handcoded",
        },
        "general": {"training_method": "dagger"},
        "dagger": {"training": {"max_nb_steps_per_episode": 200}},
    }
    wrapper = AlfredTWEnv(config, train_eval="train")
    wrapper.game_files, wrapper.num_games = [str(game)], 1
    environment = wrapper.init_env(batch_size=1)
    actions: list[str] = []
    try:
        _observation, info = environment.reset()
        for _ in range(200):
            plans = info.get("extra.expert_plan") if isinstance(info, dict) else None
            plan = plans[0] if isinstance(plans, list) and plans else None
            if not isinstance(plan, list) or not plan or not isinstance(plan[0], str):
                raise ReferenceRouteError("hand-coded expert did not expose a next action")
            action = plan[0]
            actions.append(action)
            _observation, _reward, done, info = environment.step([action])
            if bool(done[0]):
                return ReferenceRoute(task_id, split, tuple(actions))
    finally:
        environment.close()
    raise ReferenceRouteError("hand-coded expert did not finish within 200 actions")


def midpoint_candidates(route: ReferenceRoute) -> Iterator[tuple[int, tuple[str, ...], tuple[str, ...]]]:
    """Yield deterministic near-midpoint checkpoint candidates.

    Order is midpoint, midpoint + 1, midpoint - 1, and so on.  A prefix is
    always non-empty and leaves at least one continuation action.
    """
    count = len(route.actions)
    if count < 3:
        raise ReferenceRouteError("reference route is too short for a controlled midpoint recovery")
    midpoint = count // 2
    seen: set[int] = set()
    for distance in range(count):
        for index in (midpoint + distance, midpoint - distance):
            if index in seen or not 1 <= index < count:
                continue
            seen.add(index)
            yield index, route.actions[:index], route.actions[index:]
