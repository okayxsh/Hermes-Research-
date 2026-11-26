"""PROPOSED statistical analysis plan implementation (Phase 2B).  Runs only from saved five-arm records.

* Unit of analysis: TASK.  Seeds within a task are repeated inference measurements; per task and contrast the outcome
  is the mean over seeds where BOTH arms have a VALID outcome (pairwise complete case).
* ONE primary comparison: D18 - D0 over every roster task.  Estimate: task-weighted mean difference (finite screened
  population).  Test: two-sided task-level sign-flip permutation test of the SHARP null (no effect in any task); exact
  under within-task arm exchangeability, irrespective of shared template/scene effects.  Interval (SD-21): template-
  cluster bootstrap percentile 95% CI (tasks sharing a goal template are dependent); the task bootstrap is reported as a
  sensitivity interval.  Template-level sign-flip (unit = template mean) is the dependence-robust sensitivity test.
* Secondary, EXPLORATORY family (separate Holm adjustment; never pooled with the primary, never confirmatory):
  D18-Fill, Irr-Fill, Rel-Irr, Rel-Fill over their frozen, outcome-blind subsets; recovery effort; invalid actions;
  latency; retrieval relevance; acquisition-template-overlap sensitivity.
* Three scenes: per-scene and leave-one-scene-out estimates are DESCRIPTIVE; family and scene are confounded (the
  family x scene table is reported); no scene-cluster asymptotics.
* Missingness: infrastructure failures are retried (same unit identity, bounded); model failures, budget exhaustion
  and invalid actions are outcomes; missing cells are never coerced into failures; worst/best bounds are reported.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from statistics import fmean, median
from typing import Any, Mapping, Sequence

from rq1.five_arm.measurements import injected_relevance, retrieved_precision_at_3, validate_record
from rq1.five_arm.protocol import (CONDITIONS, FACTORIAL_ESTIMATES, FILLER_ARMS, NATURAL_ARMS, PRIMARY_CONTRAST,
                                   SECONDARY_CONTRASTS, assert_five_arm_records, contrast_name)
from rq1.five_arm.stats import bootstrap_mean_ci, cluster_bootstrap_ci, holm, mcnemar_exact, sign_flip_test

ANALYSIS_SCHEMA = "rq1-seven-condition-analysis-v3"
POPULATION_STATEMENT = (
    "Estimates describe the finite screened task population (and the seeds run).  Inference to a broader population of "
    "ALFWorld tasks assumes the screened tasks are exchangeable with that population, which the screening does not "
    "guarantee; the template-cluster interval reflects dependence among tasks that share a goal template only.")


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


def _arm_tasks(design: Mapping[str, Any], arm: str) -> list[str]:
    tasks = sorted(design["tasks"])
    if arm in NATURAL_ARMS:
        return tasks
    return sorted(design["arm_subsets"][arm])


def _planned_cells(design: Mapping[str, Any]) -> set[tuple[str, str, int]]:
    planned = {(task, arm, int(seed)) for arm in CONDITIONS for task in _arm_tasks(design, arm) for seed in design["seeds"]}
    repro = design.get("reproducibility") or {}
    planned |= {(task, arm, int(seed)) for task in repro.get("tasks", []) for arm in repro.get("conditions", [])
                for seed in repro.get("seeds", [])}
    return planned


def universe(design: Mapping[str, Any], treatment: str, reference: str) -> list[str]:
    return sorted(set(_arm_tasks(design, treatment)) & set(_arm_tasks(design, reference)))


def missingness(cells: Mapping[tuple[str, str, int], Mapping[str, Any]], design: Mapping[str, Any]) -> dict[str, Any]:
    planned = _planned_cells(design)
    unexpected = sorted(set(cells) - planned)
    if unexpected:
        raise AnalysisError(f"records for unplanned cells (e.g. an arm outside its frozen subset): {unexpected[:3]}")
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


def _contrast(cells, design, treatment, reference, tasks_universe, *, seed, replicates, draws, role) -> dict[str, Any]:
    seeds = [int(s) for s in design["seeds"]]
    task_diffs: dict[str, float] = {}
    pairs_per_task: dict[str, int] = {}
    excluded: dict[str, str] = {}
    worst: dict[str, float] = {}
    best: dict[str, float] = {}
    for task in tasks_universe:
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
    result: dict[str, Any] = {"role": role, "treatment": treatment, "reference": reference,
                              "tasks_in_universe": len(tasks_universe), "tasks_analysed": len(task_diffs),
                              "paired_seed_observations": sum(pairs_per_task.values()), "excluded_tasks": excluded}
    if not task_diffs:
        result.update(estimate=None, p_value=None, test=None, ci_95_template_cluster_bootstrap=None, ci_95_task_bootstrap=None)
        return result
    values = [task_diffs[t] for t in sorted(task_diffs)]
    test = sign_flip_test(values, draws=draws, seed=seed) if draws else None      # draws=0: estimation only
    by_template: dict[str, list[float]] = defaultdict(list)
    by_scene: dict[str, list[float]] = defaultdict(list)
    for task, diff in sorted(task_diffs.items()):
        by_template[str(design["tasks"][task]["goal_template"])].append(diff)
        by_scene[str(design["tasks"][task]["scene"])].append(diff)
    result.update(
        estimate=fmean(values), estimate_percentage_points=100 * fmean(values), p_value=test["p_value"] if test else None, test=test,
        null_hypothesis="sharp null: within every task the two arms' outcome distributions are identical (arm labels "
                        "exchangeable); task-level sign flips",
        ci_95_template_cluster_bootstrap=cluster_bootstrap_ci(by_template, seed=seed, replicates=replicates),
        ci_95_task_bootstrap=bootstrap_mean_ci(values, seed=seed, replicates=replicates),
        discordant_tasks={"treatment_better": sum(v > 0 for v in values), "reference_better": sum(v < 0 for v in values),
                          "tied": sum(v == 0 for v in values)})
    if all(n == 1 for n in pairs_per_task.values()):
        b = sum(1 for v in values if v > 0)
        c = sum(1 for v in values if v < 0)
        result["mcnemar_exact"] = {"discordant_treatment_better": b, "discordant_reference_better": c, "p_value": mcnemar_exact(b, c)}
    else:
        result["mcnemar_exact"] = "not applicable: tasks contribute several correlated seed outcomes"
    template_values = [fmean(v) for _, v in sorted(by_template.items())]
    result["template_sensitivity"] = {
        "templates": len(template_values), "template_weighted_estimate": fmean(template_values),
        "sign_flip_template_level": sign_flip_test(template_values, draws=draws, seed=seed + 1) if draws else None,
        "note": "unit = goal-template mean difference; robust to within-template dependence of heterogeneous effects"}
    families: dict[str, Counter] = defaultdict(Counter)
    for task in task_diffs:
        families[str(design["tasks"][task].get("family"))][str(design["tasks"][task]["scene"])] += 1
    result["scene_descriptive"] = {
        "note": "three scenes: descriptive only; family and scene are confounded (see family_by_scene)",
        "per_scene": {s: {"tasks": len(v), "mean_difference": fmean(v)} for s, v in sorted(by_scene.items())},
        "leave_one_scene_out": {s: fmean([d for t, d in task_diffs.items() if str(design["tasks"][t]["scene"]) != s])
                                if any(str(design["tasks"][t]["scene"]) != s for t in task_diffs) else None
                                for s in sorted(by_scene)},
        "family_by_scene": {f: dict(sorted(c.items())) for f, c in sorted(families.items())}}
    result["missing_cell_bounds"] = {"worst_case_estimate": fmean(worst.values()), "best_case_estimate": fmean(best.values()),
                                     "note": "missing cells set to the least/most favourable outcome for the treatment"}
    return result


def _effort(cells, design, treatment, reference, tasks_universe, field, seed, replicates, draws) -> dict[str, Any]:
    diffs = []
    for task in tasks_universe:
        per = []
        for s in design["seeds"]:
            a, b = cells.get((task, treatment, int(s))), cells.get((task, reference, int(s)))
            if not a or not b or a["outcome_status"] != "VALID" or b["outcome_status"] != "VALID":
                continue
            value = (lambda r: r["recovery_actions"] if r["success"] else r["recovery_action_allowance"]) \
                if field == "capped_recovery_actions" else (lambda r: r[field])
            per.append(value(a) - value(b))
        if per:
            diffs.append(fmean(per))
    if not diffs:
        return {"tasks_analysed": 0}
    return {"tasks_analysed": len(diffs), "mean_task_difference": fmean(diffs),
            "sign_flip": sign_flip_test(diffs, draws=draws, seed=seed),
            "ci_95_task_bootstrap": bootstrap_mean_ci(diffs, seed=seed, replicates=replicates),
            "note": "failures counted at the full recovery allowance (restricted mean), never success-only" if field == "capped_recovery_actions" else ""}


def _factorial(cells, design, *, seed, replicates) -> dict[str, Any]:
    """Memory x perturbation estimates on the primary seed(s), ESTIMATION ONLY (intervals, no tests).  D18U shows the frozen
    post-detour D18 block, so the interaction holds memory text fixed across perturbation levels."""
    seeds = [int(s) for s in design["seeds"]]
    tasks = sorted(design["tasks"])
    per_task: dict[str, float] = {}
    excluded: dict[str, str] = {}
    for task in tasks:
        diffs = []
        for s in seeds:
            y = {arm: _outcome(cells, task, arm, s) for arm in ("D0", "D18", "D0U", "D18U")}
            if all(v is not None for v in y.values()):
                diffs.append((y["D18"] - y["D0"]) - (y["D18U"] - y["D0U"]))
        if diffs:
            per_task[task] = fmean(diffs)
        else:
            excluded[task] = "not all four factorial cells VALID"
    by_template: dict[str, list[float]] = defaultdict(list)
    for task, value in per_task.items():
        by_template[str(design["tasks"][task]["goal_template"])].append(value)
    values = [per_task[t] for t in sorted(per_task)]
    interaction = {"definition": "(D18 - D0) - (D18U - D0U), task-weighted", "tasks_analysed": len(values),
                   "excluded_tasks": excluded, "estimate": fmean(values) if values else None,
                   "ci_95_template_cluster_bootstrap": cluster_bootstrap_ci(by_template, seed=seed, replicates=replicates) if values else None,
                   "ci_95_task_bootstrap": bootstrap_mean_ci(values, seed=seed, replicates=replicates) if values else None,
                   "role": "secondary_estimation_only",
                   "interpretation_rule": "no disruption-specific claim unless the template-cluster interval excludes 0"}
    estimates = {}
    for i, (a, b) in enumerate(FACTORIAL_ESTIMATES):
        result = _contrast(cells, design, a, b, universe(design, a, b), seed=seed + 1 + i, replicates=replicates, draws=0,
                           role="secondary_estimation_only")
        for key in ("p_value", "test", "mcnemar_exact"):
            result.pop(key, None)
        result.get("template_sensitivity", {}).pop("sign_flip_template_level", None)
        estimates[contrast_name(a, b)] = result
    return {"estimates": estimates, "interaction": interaction,
            "meaning": {"D18U-D0U": "continuation benefit of memory without perturbation",
                        "D0-D0U": "perturbation effect without memory (manipulation check on the confirmatory roster)",
                        "D18-D18U": "perturbation effect with the same memory block"}}


def interface_report(records: Sequence[Mapping[str, Any]], cells: Mapping[tuple[str, str, int], Mapping[str, Any]]) -> dict[str, Any]:
    """Rejected selections by condition with explicit denominators: episodes, model decisions (one per recovery-trace
    entry) and selection attempts (decisions + rejections); rejection categories; selection-limit terminations.
    Descriptive; the parser is unchanged."""
    out: dict[str, Any] = {}
    for arm in CONDITIONS:
        rows = [c for c in cells.values() if c["condition"] == arm and c["outcome_status"] == "VALID"]
        decisions = sum(len(c["recovery_trace"]) for c in rows)
        rejected = sum(len(step["rejections"]) for c in rows for step in c["recovery_trace"])
        categories = Counter(reason for c in rows for step in c["recovery_trace"] for reason in step["rejections"])
        limit = sum(c["termination_reason"] == "invalid_action_limit" for c in rows)
        out[arm] = {"episodes": len(rows), "decisions": decisions, "selection_attempts": decisions + rejected,
                    "rejected_selections": rejected,
                    "rejected_per_episode": rejected / len(rows) if rows else None,
                    "rejected_per_decision": rejected / decisions if decisions else None,
                    "rejected_per_attempt": rejected / (decisions + rejected) if decisions + rejected else None,
                    "rejection_categories": dict(sorted(categories.items())),
                    "selection_limit_terminations": limit,
                    "selection_limit_termination_rate": limit / len(rows) if rows else None}
    return {"by_condition": out, "note": "descriptive; rejections are part of each outcome and the parser is unchanged"}


def analyze(records: Sequence[Mapping[str, Any]], design: Mapping[str, Any], *,
            labels: Mapping[tuple[str, str], str] | None = None, seed: int = 20261124,
            bootstrap_replicates: int = 10_000, permutation_draws: int = 100_000) -> dict[str, Any]:
    if set(design.get("arm_subsets", {})) != set(FILLER_ARMS):
        raise AnalysisError(f"design must carry frozen arm_subsets for {FILLER_ARMS}")
    cells = resolve_units(records, max_infrastructure_retries=int(design["max_infrastructure_retries"]))
    miss = missingness(cells, design)
    kw = dict(replicates=bootstrap_replicates, draws=permutation_draws)
    t, r = PRIMARY_CONTRAST
    primary = _contrast(cells, design, t, r, universe(design, t, r), seed=seed, role="primary_confirmatory", **kw)
    primary["alpha"] = 0.05
    primary["multiplicity"] = "single primary comparison; no adjustment"
    primary["estimand"] = ("task-weighted mean within-task difference in recovery success, natural Accum-18 retrieval minus "
                           "no memory; TOTAL effect of the natural retrieval intervention (does not isolate relevance)")
    secondary = {contrast_name(a, b): _contrast(cells, design, a, b, universe(design, a, b), seed=seed + 10 + i,
                                                role="secondary_exploratory", **kw)
                 for i, (a, b) in enumerate(SECONDARY_CONTRASTS)}
    available = {k: v["p_value"] for k, v in secondary.items() if v["p_value"] is not None}
    adjusted = holm(available) if available else {}
    for name, value in secondary.items():
        value["p_value_holm_within_secondary_family"] = adjusted.get(name)
    effort = {f"{contrast_name(a, b)}:{field}": _effort(cells, design, a, b, universe(design, a, b), field, seed + 20 + i,
                                                        bootstrap_replicates, permutation_draws)
              for i, (a, b) in enumerate((PRIMARY_CONTRAST, *SECONDARY_CONTRASTS))
              for field in ("capped_recovery_actions", "invalid_actions")}
    factorial = _factorial(cells, design, seed=seed + 40, replicates=bootstrap_replicates)
    interface = interface_report(records, {k: v for k, v in cells.items() if int(k[2]) in {int(x) for x in design["seeds"]}})
    valid = [c for c in cells.values() if c["outcome_status"] == "VALID" and int(c["seed"]) in {int(x) for x in design["seeds"]}]
    latency = {arm: {"episodes_including_failures": len(rows),
                     "wall_clock_seconds_median": median(x["wall_clock_seconds"] for x in rows) if rows else None,
                     "inference_latency_seconds_median": median(x["inference_latency_seconds"] for x in rows) if rows else None,
                     "recovery_actions_median_including_failures": median(x["recovery_actions"] for x in rows) if rows else None}
               for arm in CONDITIONS for rows in [[c for c in valid if c["condition"] == arm]]}
    curves = {}
    for arm in CONDITIONS:
        rows = [c for c in valid if c["condition"] == arm]
        if rows:
            horizon = max(x["recovery_action_allowance"] for x in rows)
            curves[arm] = [sum(1 for x in rows if x["success"] and x["recovery_actions"] <= k) / len(rows) for k in range(horizon + 1)]
    tasks = sorted(design["tasks"])
    overlap_flagged = sorted(x for x in tasks if design["tasks"][x].get("acquisition_template_overlap"))
    overlap_sensitivity = None
    if overlap_flagged:
        kept = set(tasks) - set(overlap_flagged)
        overlap_sensitivity = {"excluded_tasks": overlap_flagged, "contrasts": {
            contrast_name(a, b): _contrast(cells, design, a, b, [x for x in universe(design, a, b) if x in kept],
                                           seed=seed + 30 + i, role="sensitivity", **kw)
            for i, (a, b) in enumerate((PRIMARY_CONTRAST, *SECONDARY_CONTRASTS))}}
    relevance = None
    if labels is not None:
        d18 = [c for c in valid if c["condition"] == "D18"]
        relevance = {"retrieved_precision_at_3": {mode: fmean(retrieved_precision_at_3(c, labels, mode) for c in d18) if d18 else None
                                                  for mode in ("strict", "lenient")},
                     "injected_relevance_by_arm": {
                         arm: _summarise_injected([injected_relevance(c, labels) for c in valid if c["condition"] == arm])
                         for arm in CONDITIONS},
                     "role": "secondary_exploratory_association"}
    synthetic = any(bool(x.get("synthetic")) for x in records)
    return {"schema": ANALYSIS_SCHEMA, "synthetic_input": synthetic, "missingness_by_arm": miss,
            "primary": {contrast_name(t, r): primary},
            "secondary_exploratory": secondary,
            "secondary_multiplicity": "Holm within the secondary family only; exploratory, not confirmatory discoveries",
            "secondary_effort": effort, "latency_descriptive": latency,
            "factorial_estimation": factorial, "interface_by_condition": interface,
            "primary_test_note": ("with one inference seed every task contributes one binary pair, so the task-level "
                                  "sign-flip p-value IS the exact McNemar p-value (reported in mcnemar_exact)"),
            "cumulative_success_by_recovery_actions": curves, "acquisition_overlap_sensitivity": overlap_sensitivity,
            "relevance": relevance, "population_statement": POPULATION_STATEMENT,
            "status": "analysis output only; not a scientific claim until the SAP, roster and data are approved",
            "settings": {"seed": seed, "bootstrap_replicates": bootstrap_replicates, "permutation_draws": permutation_draws,
                         "unit": "task", "max_infrastructure_retries": design["max_infrastructure_retries"]}}


def _summarise_injected(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    computed = [x for x in rows if x["status"] == "computed"]
    return {"episodes": len(rows), "not_applicable_zero_injected": len(rows) - len(computed),
            "mean_injected_relevant_fraction": fmean(x["injected_relevant_fraction"] for x in computed) if computed else None,
            "mean_injected_irrelevant_count": fmean(x["injected_irrelevant_count"] for x in computed) if computed else None}
