"""Human relevance labelling support for Precision@3 / Retrieval Noise.

Relevance ground truth is human-defined and binary (RELEVANT / IRRELEVANT).
Library size, cosine score, and evaluation outcome must never determine a
label. Two independent raters are supported; disagreements are resolved by
explicit consensus/adjudication rather than being silently coerced.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

RELEVANT = "RELEVANT"
IRRELEVANT = "IRRELEVANT"
LABELS = (RELEVANT, IRRELEVANT)


class DisagreementError(ValueError):
    """Raised when independent raters disagree and no adjudication was supplied."""


@dataclass(frozen=True)
class RaterRating:
    """One rater's binary label for one retrieved skill."""

    event_id: str
    skill_id: str
    rater_id: str
    label: str

    def __post_init__(self) -> None:
        if self.label not in LABELS:
            raise ValueError(f"label must be one of {LABELS}, got {self.label!r}")


def adjudicate(ratings: Sequence[RaterRating]) -> dict[tuple[str, str], str]:
    """Resolve binary ratings into agreed labels; raise on unresolved disagreement.

    When all raters agree, the shared label is returned. When they disagree the
    caller must supply a consensus decision (e.g., by re-issuing an adjudicated
    ``RaterRating`` with an adjudicator id); this function refuses to guess.
    """
    grouped: dict[tuple[str, str], set[str]] = {}
    for rating in ratings:
        grouped.setdefault((rating.event_id, rating.skill_id), set()).add(rating.label)
    result: dict[tuple[str, str], str] = {}
    for key, labels in grouped.items():
        if len(labels) == 1:
            result[key] = next(iter(labels))
        else:
            raise DisagreementError(
                f"raters disagree on {key}: {sorted(labels)}"
            )
    return result


@dataclass(frozen=True)
class RetrievalLabelSet:
    """Adjudicated binary labels for one retrieval event's top-k, in rank order."""

    event_id: str
    labels: tuple[tuple[str, str], ...]  # (skill_id, label) in rank order

    def precision_at_k(self, k: int = 3) -> float | None:
        top = self.labels[:k]
        if not top:
            return None
        return sum(1 for _skill_id, label in top if label == RELEVANT) / len(top)

    def retrieval_noise(self, k: int = 3) -> float | None:
        precision = self.precision_at_k(k)
        return None if precision is None else 1.0 - precision


def build_label_set(
    event_id: str,
    top_k_skill_ids: Sequence[str],
    adjudicated: dict[tuple[str, str], str],
) -> RetrievalLabelSet:
    """Assemble a rank-ordered label set, requiring a label for every top-k skill."""
    labels: list[tuple[str, str]] = []
    for skill_id in top_k_skill_ids:
        key = (event_id, skill_id)
        if key not in adjudicated:
            raise DisagreementError(f"missing adjudicated label for {key}")
        labels.append((skill_id, adjudicated[key]))
    return RetrievalLabelSet(event_id, tuple(labels))
