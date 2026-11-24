"""Dependency-free statistical primitives (the repository is stdlib-only)."""
from __future__ import annotations

import itertools
import math
import random
from statistics import fmean
from typing import Mapping, Sequence


class StatsError(ValueError):
    pass


def holm(pvalues: Mapping[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values (monotone, capped at 1)."""
    items = sorted(pvalues.items(), key=lambda kv: (kv[1], kv[0]))
    m = len(items)
    adjusted: dict[str, float] = {}
    running = 0.0
    for i, (name, p) in enumerate(items):
        if not 0.0 <= p <= 1.0:
            raise StatsError(f"p-value out of range for {name}: {p}")
        running = max(running, min(1.0, (m - i) * p))
        adjusted[name] = running
    return adjusted


def sign_flip_test(diffs: Sequence[float], *, exact_max: int = 20, draws: int = 100_000, seed: int = 0) -> dict[str, float | int | str]:
    """Two-sided paired sign-flip permutation test of mean(diffs) = 0 (task-level differences).

    Exact enumeration over the non-zero differences when there are at most ``exact_max`` of them; otherwise a seeded
    Monte-Carlo with the (count+1)/(draws+1) correction.  With single binary outcomes per task this is the exact
    McNemar / sign test on discordant tasks.
    """
    values = [float(d) for d in diffs]
    if not values:
        raise StatsError("no paired task differences")
    n = len(values)
    observed = abs(sum(values))
    nonzero = [d for d in values if d != 0.0]
    if not nonzero:
        return {"p_value": 1.0, "method": "degenerate_all_zero", "n": n, "nonzero": 0}
    eps = 1e-12
    if len(nonzero) <= exact_max:
        extreme = total = 0
        for signs in itertools.product((1, -1), repeat=len(nonzero)):
            total += 1
            if abs(sum(s * d for s, d in zip(signs, nonzero))) >= observed - eps:
                extreme += 1
        return {"p_value": extreme / total, "method": "exact_sign_flip", "n": n, "nonzero": len(nonzero)}
    rng = random.Random(seed)
    extreme = 0
    for _ in range(draws):
        if abs(sum(d if rng.random() < 0.5 else -d for d in nonzero)) >= observed - eps:
            extreme += 1
    return {"p_value": (extreme + 1) / (draws + 1), "method": "monte_carlo_sign_flip", "n": n,
            "nonzero": len(nonzero), "draws": draws, "seed": seed}


def mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value for discordant counts b and c (one binary pair per unit only)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def percentile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        raise StatsError("no values")
    pos = (len(sorted_values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] * (1 - (pos - lo)) + sorted_values[hi] * (pos - lo)


def bootstrap_mean_ci(values: Sequence[float], *, seed: int, replicates: int = 10_000, level: float = 0.95) -> list[float] | None:
    vals = list(values)
    if len(vals) < 2:
        return None
    rng = random.Random(seed)
    draws = sorted(fmean(vals[rng.randrange(len(vals))] for _ in vals) for _ in range(replicates))
    alpha = (1 - level) / 2
    return [percentile(draws, alpha), percentile(draws, 1 - alpha)]


# --- Student t via the regularised incomplete beta function (Numerical Recipes continued fraction) -------------------
def _betacf(a: float, b: float, x: float) -> float:
    tiny = 1e-300
    clamp = lambda v: v if abs(v) >= tiny else tiny
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 / clamp(1.0 - qab * x / qap)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 / clamp(1.0 + aa * d)
        c = clamp(1.0 + aa / c)
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 / clamp(1.0 + aa * d)
        c = clamp(1.0 + aa / c)
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 3e-14:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x)
    front = math.exp(lbeta)
    if x < (a + 1) / (a + b + 2):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1 - x) / b


def t_sf_two_sided(t: float, df: float) -> float:
    if df <= 0:
        raise StatsError("degrees of freedom must be positive")
    return betainc(df / 2, 0.5, df / (df + t * t))


def t_quantile(p: float, df: float) -> float:
    """Upper quantile t such that P(T <= t) = p (bisection)."""
    if not 0.5 <= p < 1:
        raise StatsError("t_quantile supports 0.5 <= p < 1")
    lo, hi = 0.0, 1000.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if 1 - t_sf_two_sided(mid, df) / 2 < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def paired_t(diffs: Sequence[float]) -> dict[str, float | None]:
    n = len(diffs)
    if n < 2:
        return {"mean": fmean(diffs) if diffs else None, "p_value": None, "ci_half_width": None}
    mean = fmean(diffs)
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    se = math.sqrt(var / n)
    if se == 0:
        return {"mean": mean, "p_value": 1.0 if mean == 0 else 0.0, "ci_half_width": 0.0}
    return {"mean": mean, "p_value": t_sf_two_sided(mean / se, n - 1), "ci_half_width": t_quantile(0.975, n - 1) * se}


def cohen_kappa(a: Sequence[str], b: Sequence[str]) -> float | None:
    if len(a) != len(b) or not a:
        raise StatsError("kappa needs two equal-length non-empty label sequences")
    categories = sorted(set(a) | set(b))
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n
    expected = sum((a.count(c) / n) * (b.count(c) / n) for c in categories)
    if expected >= 1.0:
        return None
    return (observed - expected) / (1 - expected)
