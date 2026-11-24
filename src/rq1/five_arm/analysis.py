"""PROPOSED statistical analysis plan implementation (task-level paired inference).  Runs only from saved records.

* Unit of analysis: TASK.  Seeds within a task are repeated measurements; per task and contrast the outcome is the
  mean over seeds where BOTH arms have a VALID outcome (pairwise complete case).
* Primary family (Holm): D18-D0 and D18-Fill over all eligible tasks; Irr-Fill over the frozen Irr subset.
* Test: two-sided task-level sign-flip permutation (exact when feasible); exact McNemar is reported only when every
  task contributes exactly one paired binary outcome.
* Sensitivity: template-level aggregation (27 templates), per-scene and leave-one-scene-out (descriptive only, three
  scenes), missing-cell worst/best bounds, acquisition-template-overlap exclusion.
* Missingness: infrastructure failures are retried (same unit identity, bounded); model failures, budget exhaustion and
  invalid actions are outcomes.  Missing cells are never coerced into failures; denominators are reported by arm.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from statistics import fmean, median
from typing import Any, Mapping, Sequence

from rq1.five_arm.measurements import injected_relevance, retrieved_precision_at_3, validate_record
from rq1.five_arm.protocol import (CONDITIONS, MECHANISM_ARMS, PRIMARY_CONTRASTS, SECONDARY_CONTRASTS,
                                   assert_five_arm_records, contrast_name)
from rq1.five_arm.stats import bootstrap_mean_ci, holm, mcnemar_exact, sign_flip_test

ANALYSIS_SCHEMA = "rq1-five-arm-analysis-v1"


class AnalysisError(ValueError):
    pass


def resolve_units(records: Sequence[Mapping[str, Any]], *, max_infrastructure_retries: int) -> dict[tuple[str, str, int], dict[str, Any]]:
    """Apply the retry rule: at most ``max_infrastructure_retries`` re-attempts, and only after infrastructure failures."""
    assert_five_arm_records(records)
    by_unit: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        problems = validate_record(record)
        if problems:
            raise AnalysisError(f"invalid record {record.get('unit_key')}: {'; '.join(problems)}")
        by_unit[str(record["unit_key"])].append(record)
    cells: dict[tuple[str, str, int], dict[str, Any]] = {}
    for unit_key, attempts in by_unit.items():
        attempts = sorted(attempts, key=lambda r: int(r["attempt"]))
        numbers = [int(r["attempt"]) for r in attempts]
        if numbers != list(range(1, len(numbers) + 1)):
            raise AnalysisError(f"{unit_key}: attempts must be numbered 1..n without gaps or duplicates")
        if len(attempts) > max_infrastructure_retries + 1:
            raise AnalysisError(f"{unit_key}: more attempts than the pre-specified retry limit")
        for earlier in attempts[:-1]:
            if earlier["outcome_status"] != "MISSING_INFRASTRUCTURE":
                raise AnalysisError(f"{unit_key}: only infrastructure failures may be retried (attempt {earlier['attempt']} "
                                    f"was {earlier['outcome_status']})")
        final = dict(attempts[-1])
        final["attempts_used"] = len(attempts)
        key = (str(final["task_id"]), str(final["condition"]), int(final["seed"]))
        if key in cells:
            raise AnalysisError(f"two units report the same task/condition/seed cell {key}")
        cells[key] = final
    return cells


def _planned_cells(design: Mapping[str, Any]) -> set[tuple[str, str, int]]:
    tasks = sorted(design["tasks"])
    subsets = design["mechanism_subsets"]
    planned = set()
    for task in tasks:
        for arm in CONDITIONS:
            if arm in MECHANISM_ARMS and task not in subsets[arm]:
                continue
            for seed in design["seeds"]:
                planned.add((task, arm, int(seed)))
    return planned


def missingness(cells: Mapping[tuple[str, str, int], Mapping[str, Any]], design: Mapping[str, Any]) -> dict[str, Any]:
    planned = _planned_cells(design)
    unexpected = sorted(set(cells) - planned)
    if unexpected:
        raise AnalysisError(f"records for unplanned cells (e.g. a mechanism arm outside its frozen subset): {unexpected[:3]}")
    table: dict[str, Any] = {}
    for arm in CONDITIONS:
        arm_planned = [c for c in planned if c[1] == arm]
        statuses = Counter(cells[c]["outcome_status"] if c in cells else "MISSING_CELL" for c in arm_planned)
        reasons = Counter(cells[c]["termination_reason"] for c in arm_planned if c in cells)
        table[arm] = {"planned": len(arm_planned), "valid": statuses.get("VALID", 0),
                      "by_status": dict(sorted(statuses.items())), "by_termination": dict(sorted(reasons.items()))}
    return table


def _outcome(cells, task, arm, seed):
    cell = cells.get((task, arm, seed))
    return None if cell is None or cell["outcome_status"] != "VALID" else int(bool(cell["success"]))


def _contrast(cells, design, treatment, reference, universe, *, seed, replicates, draws) -> dict[str, Any]:
    seeds = [int(s) for s in design["seeds"]]
    task_diffs: dict[str, float] = {}
    pairs_per_task: dict[str, int] = {}
    excluded: dict[str, str] = {}
    worst: dict[str, float] = {}
    best: dict[str, float] = {}
    for task in universe:
        diffs, w, b = [], [], []
        for s in seeds:
            yt, yr = _outcome(cells, task, treatment, s), _outcome(cells, task, reference, s)
            if yt is not None and yr is not None:
                diffs.append(yt - yr)
            w.append((0 if yt is None else yt) - (1 if yr is None else yr))
            b.append((1 if yt is None else yt) - (0 if yr is None else yr))
        worst[task], best[task] = fmean(w), fmean(b)
        if diffs:
            task_diffs[task], pairs_per_task[task] = fmean(diffs), len(diffs)
        else:
            excluded[task] = "no seed with VALID outcomes in both arms"
    name = contrast_name(treatment, reference)
    result: dict[str, Any] = {"treatment": treatment, "reference": reference, "tasks_in_universe": len(universe),
                              "tasks_analysed": len(task_diffs), "paired_seed_observations": sum(pairs_per_task.values()),
                              "excluded_tasks": excluded}
    if not task_diffs:
        result.update(estimate=None, p_value=None, test=None, ci_95_task_bootstrap=None)
        return result
    values = [task_diffs[t] for t in sorted(task_diffs)]
    test = sign_flip_test(values, draws=draws, seed=seed)
    result.update(estimate=fmean(values), p_value=test["p_value"], test=test,
                  ci_95_task_bootstrap=bootstrap_mean_ci(values, seed=seed, replicates=replicates))
    if all(n == 1 for n in pairs_per_task.values()):
        b = sum(1 for v in values if v > 0)
        c = sum(1 for v in values if v < 0)
        result["mcnemar_exact"] = {"discordant_treatment_better": b, "discordant_reference_better": c, "p_value": mcnemar_exact(b, c)}
    else:
        result["mcnemar_exact"] = "not applicable: tasks contribute several correlated seed outcomes"
    # template-aware sensitivity
    by_template: dict[str, list[float]] = defaultdict(list)
    by_scene: dict[str, list[float]] = defaultdict(list)
    for task, diff in task_diffs.items():
        by_template[design["tasks"][task]["goal_template"]].append(diff)
        by_scene[str(design["tasks"][task]["scene"])].append(diff)
    template_values = [fmean(v) for _, v in sorted(by_template.items())]
    result["template_sensitivity"] = {
        "templates": len(template_values), "estimate": fmean(template_values),
        "sign_flip": sign_flip_test(template_values, draws=draws, seed=seed + 1),
        "ci_95_template_bootstrap": bootstrap_mean_ci(template_values, seed=seed + 1, replicates=replicates)}
    result["scene_descriptive"] = {
        "note": "three scenes: descriptive only; family and scene are confounded",
        "per_scene": {s: {"tasks": len(v), "mean_difference": fmean(v)} for s, v in sorted(by_scene.items())},
        "leave_one_scene_out": {s: fmean([d for t, d in task_diffs.items() if str(design["tasks"][t]["scene"]) != s])
                                if any(str(design["tasks"][t]["scene"]) != s for t in task_diffs) else None
                                for s in sorted(by_scene)}}
    result["missing_cell_bounds"] = {"worst_case_estimate": fmean(worst.values()), "best_case_estimate": fmean(best.values()),
                                     "note": "missing cells set to the least/most favourable outcome for the treatment"}
    return result


def _secondary_effort(cells, design, treatment, reference, universe, field, seed, replicates, draws) -> dict[str, Any]:
    diffs = []
    for task in universe:
        per = []
        for s in design["seeds"]:
            a, b = cells.get((task, treatment, int(s))), cells.get((task, reference, int(s)))
            if not a or not b or a["outcome_status"] != "VALID" or b["outcome_status"] != "VALID":
                continue
            value = lambda r: (r["recovery_actions"] if r["success"] else r["total_action_cap"] - r["prefix_actions"] - r["detour_actions"]) \
                if field == "capped_recovery_actions" else r[field]
            per.append(value(a) - value(b))
        if per:
            diffs.append(fmean(per))
    if not diffs:
        return {"tasks_analysed": 0}
    return {"tasks_analysed": len(diffs), "mean_task_difference": fmean(diffs),
            "sign_flip": sign_flip_test(diffs, draws=draws, seed=seed),
            "ci_95_task_bootstrap": bootstrap_mean_ci(diffs, seed=seed, replicates=replicates)}


def analyze(records: Sequence[Mapping[str, Any]], design: Mapping[str, Any], *,
            labels: Mapping[tuple[str, str], str] | None = None, seed: int = 20261124,
            bootstrap_replicates: int = 10_000, permutation_draws: int = 100_000) -> dict[str, Any]:
    cells = resolve_units(records, max_infrastructure_retries=int(design["max_infrastructure_retries"]))
    miss = missingness(cells, design)
    tasks = sorted(design["tasks"])
    subsets = design["mechanism_subsets"]
    universes = {("D18", "D0"): tasks, ("D18", "Fill"): tasks, ("Irr", "Fill"): sorted(subsets["Irr"]),
                 ("Rel", "Irr"): sorted(set(subsets["Rel"]) & set(subsets["Irr"])), ("Rel", "Fill"): sorted(subsets["Rel"])}
    kw = dict(replicates=bootstrap_replicates, draws=permutation_draws)
    primary = {contrast_name(t, r): _contrast(cells, design, t, r, universes[(t, r)], seed=seed + i, **kw)
               for i, (t, r) in enumerate(PRIMARY_CONTRASTS)}
    available = {k: v["p_value"] for k, v in primary.items() if v["p_value"] is not None}
    adjusted = holm(available) if available else {}
    for name, value in primary.items():
        value["p_value_holm"] = adjusted.get(name)
    secondary = {contrast_name(t, r): _contrast(cells, design, t, r, universes[(t, r)], seed=seed + 10 + i, **kw)
                 for i, (t, r) in enumerate(SECONDARY_CONTRASTS)}
    effort = {f"{contrast_name(t, r)}:{field}": _secondary_effort(cells, design, t, r, universes[(t, r)], field,
                                                                   seed + 20 + i, bootstrap_replicates, permutation_draws)
              for i, (t, r) in enumerate(PRIMARY_CONTRASTS) for field in ("capped_recovery_actions", "invalid_actions")}
    valid = [c for c in cells.values() if c["outcome_status"] == "VALID"]
    latency = {arm: {"episodes_including_failures": len(rows),
                     "wall_clock_seconds_median": median(r["wall_clock_seconds"] for r in rows) if rows else None,
                     "inference_latency_seconds_median": median(r["inference_latency_seconds"] for r in rows) if rows else None,
                     "recovery_actions_median_including_failures": median(r["recovery_actions"] for r in rows) if rows else None}
               for arm in CONDITIONS for rows in [[c for c in valid if c["condition"] == arm]]}
    curves = {}
    for arm in CONDITIONS:
        rows = [c for c in valid if c["condition"] == arm]
        if rows:
            horizon = max(r["total_action_cap"] - r["prefix_actions"] - r["detour_actions"] for r in rows)
            curves[arm] = [sum(1 for r in rows if r["success"] and r["recovery_actions"] <= k) / len(rows) for k in range(horizon + 1)]
    overlap_flagged = sorted(t for t in tasks if design["tasks"][t].get("acquisition_template_overlap"))
    overlap_sensitivity = None
    if overlap_flagged:
        kept = [t for t in tasks if t not in overlap_flagged]
        overlap_sensitivity = {"excluded_tasks": overlap_flagged, "contrasts": {
            contrast_name(t, r): _contrast(cells, design, t, r, [x for x in universes[(t, r)] if x in kept],
                                           seed=seed + 30 + i, **kw) for i, (t, r) in enumerate(PRIMARY_CONTRASTS)}}
    relevance = None
    if labels is not None:
        d18 = [c for c in valid if c["condition"] == "D18"]
        relevance = {"retrieved_precision_at_3": {mode: fmean(retrieved_precision_at_3(c, labels, mode) for c in d18) if d18 else None
                                                  for mode in ("strict", "lenient")},
                     "injected_relevance_by_arm": {
                         arm: _summarise_injected([injected_relevance(c, labels) for c in valid if c["condition"] == arm])
                         for arm in CONDITIONS}}
    synthetic = any(bool(r.get("synthetic")) for r in records)
    return {"schema": ANALYSIS_SCHEMA, "synthetic_input": synthetic, "missingness_by_arm": miss,
            "primary": primary, "primary_family": "Holm over " + ", ".join(primary),
            "secondary": secondary, "secondary_effort": effort, "latency_descriptive": latency,
            "cumulative_success_by_recovery_actions": curves, "acquisition_overlap_sensitivity": overlap_sensitivity,
            "relevance": relevance,
            "status": "analysis output only; not a scientific claim until the SAP, roster and data are approved",
            "settings": {"seed": seed, "bootstrap_replicates": bootstrap_replicates, "permutation_draws": permutation_draws,
                         "unit": "task", "max_infrastructure_retries": design["max_infrastructure_retries"]}}


def _summarise_injected(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    computed = [r for r in rows if r["status"] == "computed"]
    return {"episodes": len(rows), "not_applicable_zero_injected": len(rows) - len(computed),
            "mean_injected_relevant_fraction": fmean(r["injected_relevant_fraction"] for r in computed) if computed else None,
            "mean_injected_irrelevant_count": fmean(r["injected_irrelevant_count"] for r in computed) if computed else None}
