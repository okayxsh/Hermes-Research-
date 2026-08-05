from __future__ import annotations

from collections.abc import Iterable
from typing import Mapping, Any


def task_success_rate(results: Iterable[Mapping[str, Any]]) -> float | None:
    values = list(results)
    return None if not values else sum(bool(item.get("success")) for item in values) / len(values)


def invalid_action_rate(steps: Iterable[Mapping[str, Any]]) -> float | None:
    values = list(steps)
    return None if not values else sum(not bool(item.get("action_valid")) for item in values) / len(values)


def retrieval_noise_rate(events: Iterable[Mapping[str, Any]]) -> float | None:
    loads = [item for item in events if item.get("event") == "skill_view"]
    return None if not loads else sum(not bool(item.get("relevant")) for item in loads) / len(loads)


def relevant_skill_hit_rate(episodes: Iterable[Mapping[str, Any]]) -> float | None:
    eligible = [item for item in episodes if item.get("relevant_skill_available")]
    return None if not eligible else sum(bool(item.get("relevant_skill_loaded")) for item in eligible) / len(eligible)
