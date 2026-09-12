"""Cohen's kappa inter-rater reliability for two raters over nominal categories."""
from __future__ import annotations

from collections import Counter
from typing import Hashable, Sequence


def cohens_kappa(a: Sequence[Hashable], b: Sequence[Hashable]) -> float | None:
    """Return Cohen's kappa for two aligned label sequences, or ``None`` if undefined.

    No pass/fail threshold is imposed; callers report the observed kappa and
    sample size. An empty input and the degenerate single-category perfect-
    agreement case are handled explicitly.
    """
    if len(a) != len(b):
        raise ValueError("rater sequences must have equal length")
    n = len(a)
    if n == 0:
        return None

    observed = sum(x == y for x, y in zip(a, b)) / n
    counts_a = Counter(a)
    counts_b = Counter(b)
    categories = set(counts_a) | set(counts_b)
    expected = sum((counts_a[c] / n) * (counts_b[c] / n) for c in categories)

    denominator = 1.0 - expected
    if denominator == 0.0:
        return 1.0 if observed == 1.0 else 0.0
    return (observed - expected) / denominator


def cohens_kappa_details(a: Sequence[Hashable], b: Sequence[Hashable]) -> dict[str, object]:
    """Return kappa plus observed/expected agreement and sample size for reporting."""
    if len(a) != len(b):
        raise ValueError("rater sequences must have equal length")
    n = len(a)
    if n == 0:
        return {"cohens_kappa": None, "observed_agreement": None, "expected_agreement": None, "sample_size": 0}
    observed = sum(x == y for x, y in zip(a, b)) / n
    counts_a = Counter(a)
    counts_b = Counter(b)
    categories = set(counts_a) | set(counts_b)
    expected = sum((counts_a[c] / n) * (counts_b[c] / n) for c in categories)
    return {
        "cohens_kappa": cohens_kappa(a, b),
        "observed_agreement": observed,
        "expected_agreement": expected,
        "sample_size": n,
    }
