from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Mapping, Any


def task_success_rate(results: Iterable[Mapping[str, Any]]) -> float | None:
    values = list(results)
    return None if not values else sum(bool(item.get("success")) for item in values) / len(values)


def invalid_action_rate(steps: Iterable[Mapping[str, Any]]) -> float | None:
    values = list(steps)
    return None if not values else sum(not bool(item.get("action_valid")) for item in values) / len(values)


def precision_at_k(relevance: Sequence[bool], k: int = 3) -> float | None:
    """Precision@k over binary relevance flags (True = relevant), in rank order.

    Returns ``None`` when there are no retrieved skills (no-retrieval), which
    must be reported separately and never coerced to a precision of 0 or 1.
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    top = list(relevance)[:k]
    if not top:
        return None
    return sum(bool(item) for item in top) / len(top)


def retrieval_noise(precision: float | None) -> float | None:
    """Retrieval Noise = 1 - Precision@3 (or 1 - Precision@k)."""
    return None if precision is None else 1.0 - precision


def retrieval_noise_rate(events: Iterable[Mapping[str, Any]]) -> float | None:
    """DEPRECATED legacy noise metric (irrelevant native skill loads / loads).

    The authoritative RQ1 metric is ``retrieval_noise(precision_at_k(...))``.
    This function is retained only for backward compatibility with the legacy
    Hermes-native skill-event design.
    """
    loads = [
        item
        for item in events
        if item.get("event") in {"skill_view", "skill_selected", "skill_loaded", "skill_managed"}
    ]
    return None if not loads else sum(not bool(item.get("relevant")) for item in loads) / len(loads)


def relevant_skill_hit_rate(episodes: Iterable[Mapping[str, Any]]) -> float | None:
    """DEPRECATED legacy relevant-skill hit rate over native skill events."""
    eligible = [item for item in episodes if item.get("relevant_skill_available")]
    return None if not eligible else sum(bool(item.get("relevant_skill_loaded")) for item in eligible) / len(eligible)
