"""Aggregate retrieval-quality statistics: P@3, Retrieval Noise, Cohen's kappa.

No-retrieval episodes are reported separately and never coerced to a noise of 0
or 1. Cohen's kappa is computed from the original independent ratings, not the
adjudicated labels.
"""
from __future__ import annotations

import random
from typing import Any, Sequence

from rq1.analysis.kappa import cohens_kappa
from rq1.analysis.labelling import RetrievalLabelSet

DEFAULT_REPLICATES = 2000
DEFAULT_SEED = 20260806


def _mean(values: Sequence[float]) -> float | None:
    return None if not values else sum(values) / len(values)


def _bootstrap_noise(
    noise_values: Sequence[float],
    *,
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    if not noise_values:
        return {
            "method": "percentile_bootstrap",
            "replicates": replicates,
            "seed": seed,
            "lower": None,
            "upper": None,
        }
    rng = random.Random(seed)
    samples: list[float] = []
    n = len(noise_values)
    for _ in range(replicates):
        sample = [rng.choice(noise_values) for _ in range(n)]
        samples.append(sum(sample) / n)
    ordered = sorted(samples)
    lower = ordered[int(0.025 * (len(ordered) - 1))]
    upper = ordered[int(0.975 * (len(ordered) - 1))]
    return {
        "method": "percentile_bootstrap",
        "replicates": replicates,
        "seed": seed,
        "lower": lower,
        "upper": upper,
    }


def compute_retrieval_quality(
    labels: Sequence[RetrievalLabelSet],
    rater_a: Sequence[str],
    rater_b: Sequence[str],
    *,
    k: int = 3,
    seed: int = DEFAULT_SEED,
    replicates: int = DEFAULT_REPLICATES,
) -> dict[str, Any]:
    """Compute mean P@k, mean retrieval noise, Cohen's kappa, and noise bootstrap CI."""
    precisions = [item.precision_at_k(k) for item in labels]
    noise_values = [1.0 - p for p in precisions if p is not None]
    kappa = cohens_kappa(rater_a, rater_b)
    return {
        "k": k,
        "precision_at_k_mean": _mean([p for p in precisions if p is not None]),
        "retrieval_noise_mean": _mean(noise_values),
        "retrieval_count": len(labels),
        "no_retrieval_count": sum(1 for p in precisions if p is None),
        "cohens_kappa": kappa,
        "kappa_sample_size": len(rater_a),
        "noise_bootstrap": _bootstrap_noise(noise_values, seed=seed, replicates=replicates),
        "per_retrieval": [
            {
                "event_id": item.event_id,
                "precision_at_k": item.precision_at_k(k),
                "retrieval_noise": item.retrieval_noise(k),
            }
            for item in labels
        ],
    }
