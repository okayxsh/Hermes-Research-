"""Prospective power / precision simulation for the SINGLE primary comparison D18 - D0 (PROPOSED; synthetic).

Generative model (logit scale) for task i in goal template j and scene s:
    reference logit = logit-calibrated base + u_j + c_s + v_i          u_j ~ N(0, sd_template), c_s ~ N(0, sd_scene),
                                                                       v_i ~ N(0, sd_task)
    treatment logit = reference logit + beta + e_j + w_i              e_j ~ N(0, sd_effect_template), w_i ~ N(0, sd_effect)
``beta`` is calibrated numerically so the MARGINAL (population) success difference equals ``delta`` at marginal
reference success ``p_ref``.  Templates: 27 (balanced 25x3 + 2x2, or a skewed allocation); scenes: 3 (task-level).
Seeds: with probability ``seed_determinism`` all seeds of a task/arm share one draw (temperature-0 near-determinism),
otherwise they are independent given the task probability.  Each cell is missing completely at random with probability
``missing_rate``; analysis is pairwise complete case within task, task mean over paired seeds.

Test (mirrors the SAP): task-level sign-flip, using its conditional normal approximation z = sum d / sqrt(sum d^2) for
speed; template-level sign-flip power is reported as the sensitivity test.  Precision: the Monte-Carlo SD of the
estimate (true sampling SD under the model) and the coverage of Wald intervals using task-level versus template-cluster
standard errors.  Monte-Carlo standard errors are reported for every power estimate.
"""
from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from statistics import fmean, pstdev
from typing import Any, Sequence

from rq1.five_arm.stats import normal_sign_flip_p

POWER_SCHEMA = "rq1-five-arm-power-v2"
N_TEMPLATES, N_SCENES = 27, 3


@dataclass(frozen=True)
class Scenario:
    n_tasks: int
    seeds: int
    p_ref: float
    delta: float
    n_templates: int = N_TEMPLATES
    template_allocation: str = "balanced"
    sd_template: float = 0.8
    sd_scene: float = 0.4
    sd_task: float = 1.0
    sd_effect: float = 0.5
    sd_effect_template: float = 0.3
    seed_determinism: float = 0.7
    missing_rate: float = 0.02
    alpha: float = 0.05
    label: str = "key"


def _expit(x: float) -> float:
    return 1 / (1 + math.exp(-x)) if x >= 0 else math.exp(x) / (1 + math.exp(x))


def allocation(n_tasks: int, n_templates: int, kind: str) -> list[int]:
    """Template index per task.  balanced: as even as possible; skewed: Zipf-like sizes (each template >= 1 task)."""
    n_templates = min(n_templates, n_tasks)
    if kind == "balanced":
        return [i % n_templates for i in range(n_tasks)]
    if kind != "skewed":
        raise ValueError("template_allocation must be balanced or skewed")
    weights = [1 / (k + 1) for k in range(n_templates)]
    sizes = [1] * n_templates
    remaining = n_tasks - n_templates
    total = sum(weights)
    extra = [int(remaining * w / total) for w in weights]
    for k in range(remaining - sum(extra)):
        extra[k] += 1
    sizes = [s + e for s, e in zip(sizes, extra)]
    return [j for j, size in enumerate(sizes) for _ in range(size)]


def _calibrate(sc: Scenario, rng: random.Random, draws: int = 6000) -> tuple[float, float]:
    """(base logit, beta) so that marginal reference success = p_ref and marginal difference = delta."""
    latent = [(rng.gauss(0, sc.sd_template) + rng.gauss(0, sc.sd_scene) + rng.gauss(0, sc.sd_task),
               rng.gauss(0, sc.sd_effect_template) + rng.gauss(0, sc.sd_effect)) for _ in range(draws)]

    def solve(f, target):
        lo, hi = -20.0, 20.0
        for _ in range(70):
            mid = (lo + hi) / 2
            if f(mid) < target:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    base = solve(lambda b: fmean(_expit(b + l) for l, _ in latent), sc.p_ref)
    target = min(0.999, sc.p_ref + sc.delta)
    beta = solve(lambda m: fmean(_expit(base + l + m + w) for l, w in latent), target)
    return base, beta


def _seeds(p: float, k: int, rng: random.Random, determinism: float) -> list[int]:
    if rng.random() < determinism:
        return [int(rng.random() < p)] * k
    return [int(rng.random() < p) for _ in range(k)]


def simulate(sc: Scenario, *, reps: int, seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    base, beta = _calibrate(sc, rng)
    template_of = allocation(sc.n_tasks, sc.n_templates, sc.template_allocation)
    n_t = max(template_of) + 1
    reject_task = reject_template = 0
    estimates, cover_task, cover_cluster, analysed, discord = [], [], [], [], []
    for _ in range(reps):
        u = [rng.gauss(0, sc.sd_template) for _ in range(n_t)]
        e = [rng.gauss(0, sc.sd_effect_template) for _ in range(n_t)]
        c = [rng.gauss(0, sc.sd_scene) for _ in range(N_SCENES)]
        by_template: dict[int, list[float]] = {}
        diffs = []
        pair_disc = []
        for i in range(sc.n_tasks):
            j = template_of[i]
            ref_logit = base + u[j] + c[i % N_SCENES] + rng.gauss(0, sc.sd_task)
            y_ref = _seeds(_expit(ref_logit), sc.seeds, rng, sc.seed_determinism)
            y_trt = _seeds(_expit(ref_logit + beta + e[j] + rng.gauss(0, sc.sd_effect)), sc.seeds, rng, sc.seed_determinism)
            pairs = [t - r for t, r in zip(y_trt, y_ref) if rng.random() >= sc.missing_rate and rng.random() >= sc.missing_rate]
            if pairs:
                d = fmean(pairs)
                diffs.append(d)
                by_template.setdefault(j, []).append(d)
                pair_disc.extend(abs(p) for p in pairs)
        analysed.append(len(diffs))
        if len(diffs) < 2:
            continue
        discord.append(fmean(pair_disc))
        mean = fmean(diffs)
        estimates.append(mean)
        reject_task += normal_sign_flip_p(diffs) < sc.alpha
        reject_template += normal_sign_flip_p([fmean(v) for v in by_template.values()]) < sc.alpha
        n = len(diffs)
        se_task = pstdev(diffs) * math.sqrt(n / (n - 1)) / math.sqrt(n)
        g = len(by_template)
        resid = [sum(v) - len(v) * mean for v in by_template.values()]
        se_cluster = math.sqrt(sum(r * r for r in resid) * g / max(g - 1, 1)) / n
        cover_task.append(abs(mean - sc.delta) <= 1.96 * se_task)
        cover_cluster.append(abs(mean - sc.delta) <= 1.96 * se_cluster)
    valid = len(estimates)
    power = reject_task / reps
    sd = pstdev(estimates) if valid > 1 else None
    return {**asdict(sc), "reps": reps, "calibrated_base_logit": base, "calibrated_logit_effect": beta,
            "power_task_sign_flip": power, "mc_se_power": math.sqrt(power * (1 - power) / reps),
            "power_template_sign_flip": reject_template / reps,
            "mean_estimate": fmean(estimates) if estimates else None,
            "sampling_sd_of_estimate": sd, "precision_half_width_95": 1.96 * sd if sd is not None else None,
            "coverage_wald_task_se": fmean(cover_task) if cover_task else None,
            "coverage_wald_template_cluster_se": fmean(cover_cluster) if cover_cluster else None,
            "mean_tasks_analysed": fmean(analysed), "mean_paired_discordance": fmean(discord) if discord else None}


def key_grid() -> list[Scenario]:
    grid = []
    for p_ref in (0.3, 0.5, 0.7):
        for delta in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30):
            if p_ref + delta <= 0.95:
                for seeds in (1, 3):
                    grid.append(Scenario(n_tasks=79, seeds=seeds, p_ref=p_ref, delta=delta))
    return grid


def sensitivity_grid() -> list[Scenario]:
    grid = []
    for delta in (0.10, 0.15):
        base = dict(n_tasks=79, seeds=3, p_ref=0.5, delta=delta)
        for name, values in (("seed_determinism", (0.0, 1.0)), ("sd_task", (0.5, 2.0)), ("sd_template", (0.0, 1.5)),
                             ("sd_effect_template", (0.0, 0.8)), ("missing_rate", (0.0, 0.10))):
            for value in values:
                grid.append(Scenario(**base, **{name: value}, label=f"sensitivity:{name}={value}"))
        grid.append(Scenario(**base, template_allocation="skewed", label="sensitivity:template_allocation=skewed"))
    for n in (40, 60):
        for delta in (0.10, 0.15, 0.20, 0.25):
            grid.append(Scenario(n_tasks=n, seeds=1, p_ref=0.5, delta=delta, label=f"smaller_n:{n}"))
    return grid


def run_grid(scenarios: Sequence[Scenario], *, reps: int = 2000, seed: int = 20261124) -> dict[str, Any]:
    rows = [simulate(sc, reps=reps, seed=seed + index) for index, sc in enumerate(scenarios)]
    return {"schema": POWER_SCHEMA, "synthetic": True, "reps_per_scenario": reps, "seed": seed, "rows": rows,
            "notes": ["synthetic simulation under stated assumptions; not evidence of any effect",
                      "single primary comparison D18 - D0 at two-sided alpha 0.05 (no multiplicity adjustment)",
                      "the +33.2 pp engineering difference (six repeated contexts) is NOT used as an expected effect",
                      "template allocation is an assumption (balanced 79/27 or skewed); real counts come from the frozen roster",
                      "task sign-flip uses its conditional normal approximation; Monte-Carlo SE = sqrt(p(1-p)/reps)"]}


def minimal_detectable(rows: Sequence[dict[str, Any]], *, target: float = 0.8) -> dict[str, Any]:
    out: dict[str, Any] = {}
    largest: dict[str, float] = {}
    for row in rows:
        if row["label"] != "key":
            continue
        name = f"n={row['n_tasks']},seeds={row['seeds']},p_ref={row['p_ref']}"
        out.setdefault(name, None)
        largest[name] = max(largest.get(name, 0.0), row["delta"])
        if row["power_task_sign_flip"] >= target and (out[name] is None or row["delta"] < out[name]):
            out[name] = row["delta"]
    return {name: (value if value is not None else f"> {largest[name]} (not reached in grid)") for name, value in out.items()}


def episode_counts(*, n_tasks: int, rel_tasks: int, irr_tasks: int, fill_tasks: int, seeds: int,
                   reproducibility_tasks: int = 0, reproducibility_extra_seeds: int = 0) -> dict[str, Any]:
    """Episodes for one confirmatory run: D0 and D18 on every task, Fill/Rel/Irr on their frozen subsets."""
    per_seed = 2 * n_tasks + fill_tasks + rel_tasks + irr_tasks
    main = per_seed * seeds
    reproducibility = reproducibility_tasks * reproducibility_extra_seeds * 2  # D0 and D18 only
    return {"per_seed": per_seed, "main": main, "reproducibility_subset": reproducibility, "total": main + reproducibility}
