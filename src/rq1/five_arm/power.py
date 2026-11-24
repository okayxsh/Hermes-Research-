"""Prospective power / precision simulation for task-level paired binary contrasts (PROPOSED; synthetic by definition).

Generative model per task i in goal template j (logit scale):
    reference logit  = logit(p_ref) + u_j + v_i,                u_j ~ N(0, sd_template), v_i ~ N(0, sd_task)
    treatment logit  = reference logit + beta + w_i,             w_i ~ N(0, sd_effect)
``beta`` is calibrated numerically so the MARGINAL success difference equals ``delta`` (probability scale).
Seeds: with probability ``seed_determinism`` all seeds of a task/arm share one Bernoulli draw (temperature-0
near-determinism); otherwise seeds are independent given the task probability.  Each cell is missing completely at
random with probability ``missing_rate``.  Analysis mirrors the SAP: task-level mean over paired seeds, two-sided test.
For speed the simulation uses the paired t-test as an approximation to the sign-flip permutation test (documented).
Power is reported unadjusted and at Bonferroni alpha/3 (a conservative lower bound for Holm across three contrasts).
"""
from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from statistics import fmean
from typing import Any, Sequence

from rq1.five_arm.stats import paired_t

POWER_SCHEMA = "rq1-five-arm-power-v1"


@dataclass(frozen=True)
class Scenario:
    n_tasks: int
    n_templates: int
    seeds: int
    p_ref: float
    delta: float
    sd_template: float = 0.8
    sd_task: float = 1.0
    sd_effect: float = 0.5
    seed_determinism: float = 0.7
    missing_rate: float = 0.02
    alpha: float = 0.05
    multiplicity: int = 3


def _expit(x: float) -> float:
    return 1 / (1 + math.exp(-x)) if x >= 0 else math.exp(x) / (1 + math.exp(x))


def _calibrate_beta(sc: Scenario, rng: random.Random, draws: int = 4000) -> float:
    base = math.log(sc.p_ref / (1 - sc.p_ref))
    latent = [(base + rng.gauss(0, sc.sd_template) + rng.gauss(0, sc.sd_task), rng.gauss(0, sc.sd_effect)) for _ in range(draws)]
    ref = fmean(_expit(l) for l, _ in latent)
    target = min(0.999, max(0.001, ref + sc.delta))
    lo, hi = -15.0, 15.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if fmean(_expit(l + mid + w) for l, w in latent) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _templates(n_tasks: int, n_templates: int) -> list[int]:
    return [i % n_templates for i in range(n_tasks)]


def _draw_seeds(p: float, k: int, rng: random.Random, determinism: float) -> list[int]:
    if rng.random() < determinism:
        return [int(rng.random() < p)] * k
    return [int(rng.random() < p) for _ in range(k)]


def simulate(sc: Scenario, *, reps: int, seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    beta = _calibrate_beta(sc, rng)
    base = math.log(sc.p_ref / (1 - sc.p_ref))
    template_of = _templates(sc.n_tasks, min(sc.n_templates, sc.n_tasks))
    rejections = rejections_adj = 0
    widths, estimates, analysed = [], [], []
    for _ in range(reps):
        u = [rng.gauss(0, sc.sd_template) for _ in range(max(template_of) + 1)]
        diffs = []
        for i in range(sc.n_tasks):
            ref_logit = base + u[template_of[i]] + rng.gauss(0, sc.sd_task)
            p_ref = _expit(ref_logit)
            p_trt = _expit(ref_logit + beta + rng.gauss(0, sc.sd_effect))
            y_ref = _draw_seeds(p_ref, sc.seeds, rng, sc.seed_determinism)
            y_trt = _draw_seeds(p_trt, sc.seeds, rng, sc.seed_determinism)
            pairs = [t - r for t, r in zip(y_trt, y_ref)
                     if rng.random() >= sc.missing_rate and rng.random() >= sc.missing_rate]
            if pairs:
                diffs.append(fmean(pairs))
        analysed.append(len(diffs))
        result = paired_t(diffs)
        if result["p_value"] is not None:
            rejections += result["p_value"] < sc.alpha
            rejections_adj += result["p_value"] < sc.alpha / sc.multiplicity
            widths.append(result["ci_half_width"])
            estimates.append(result["mean"])
    widths.sort()
    return {**asdict(sc), "reps": reps, "calibrated_logit_effect": beta,
            "power_unadjusted": rejections / reps, "power_bonferroni_lower_bound_for_holm": rejections_adj / reps,
            "median_ci95_half_width": widths[len(widths) // 2] if widths else None,
            "mean_estimate": fmean(estimates) if estimates else None, "mean_tasks_analysed": fmean(analysed)}


def default_grid() -> list[Scenario]:
    grid = []
    for n in (20, 40, 60, 79):
        for seeds in (1, 3):
            for p_ref in (0.5, 0.75):
                for delta in (0.05, 0.10, 0.15, 0.20, 0.30):
                    if p_ref + delta < 0.97:
                        grid.append(Scenario(n_tasks=n, n_templates=27 if n == 79 else max(1, round(n * 27 / 79)),
                                             seeds=seeds, p_ref=p_ref, delta=delta))
    for missing in (0.0, 0.10):
        for determinism in (0.0, 1.0):
            grid.append(Scenario(n_tasks=79, n_templates=27, seeds=3, p_ref=0.5, delta=0.10,
                                 missing_rate=missing, seed_determinism=determinism))
    for sd_template in (0.0, 1.5):
        grid.append(Scenario(n_tasks=79, n_templates=27, seeds=3, p_ref=0.5, delta=0.10, sd_template=sd_template))
    return grid


def run_grid(scenarios: Sequence[Scenario], *, reps: int = 400, seed: int = 20261124) -> dict[str, Any]:
    rows = [simulate(sc, reps=reps, seed=seed + index) for index, sc in enumerate(scenarios)]
    return {"schema": POWER_SCHEMA, "synthetic": True, "reps_per_scenario": reps, "seed": seed, "rows": rows,
            "notes": ["synthetic simulation under stated assumptions; not evidence of any effect",
                      "the +33.2 pp engineering difference (six repeated contexts) is NOT used as an expected effect",
                      "paired t approximates the planned sign-flip test; Bonferroni alpha/3 is a conservative Holm bound",
                      "the 79 tasks share 27 templates and three scenes; they are not independent population draws"]}


def minimal_detectable(rows: Sequence[dict[str, Any]], *, key: str = "power_bonferroni_lower_bound_for_holm",
                       target: float = 0.8) -> dict[str, Any]:
    out: dict[str, Any] = {}
    largest: dict[str, float] = {}
    for row in rows:
        if row["missing_rate"] != 0.02 or row["seed_determinism"] != 0.7 or row["sd_template"] != 0.8:
            continue
        name = f"n={row['n_tasks']},seeds={row['seeds']},p_ref={row['p_ref']}"
        out.setdefault(name, None)
        largest[name] = max(largest.get(name, 0.0), row["delta"])
        if row[key] >= target and (out[name] is None or row["delta"] < out[name]):
            out[name] = row["delta"]
    return {name: (value if value is not None else f"> {largest[name]} (not reached in grid)") for name, value in out.items()}
