"""PILOT-ONLY diagnostics (PROPOSED): failure-manipulation validation, seed variability and the action-budget rule.

All functions run on saved pilot-diagnostic records only (UNDISTURBED / DETOURED continuations without memory; see
``executor``).  They never see a memory arm, so no treatment difference can influence them.  Thresholds are explicit
arguments; ``PROPOSED_DIAGNOSTIC_THRESHOLDS`` is the engineering proposal for SD-16 and is NOT approved.

A 12-task pilot is a diagnostic exercise, not a hypothesis test: every classification is reported with a task-bootstrap
interval.  Phase 2E rules (prespecified, before any diagnostic outcome exists):

* nothing here produces a GO: every class is evidence for author review;
* more recovery actions ALONE never count as meaningful disruption (``EFFORT_ONLY_INCREASE``);
* ``INADEQUATE_MANIPULATION`` needs positive evidence of no disruption (upper interval bounds below the thresholds), as
  distinct from ``INCONCLUSIVE`` (intervals too wide to decide);
* a competence floor (the model rarely completes tasks even WITHOUT the detour) makes the manipulation unjudgeable
  (``UNINFORMATIVE_COMPETENCE_FLOOR``) rather than inadequate;
* ``family_stratified_diagnostic`` separates competence, perturbation difficulty, ceiling/floor, effort and interface
  problems per task family, and derives the prospective floor-family sensitivity set (never a confirmatory exclusion).

A one-action detour is a controlled navigation perturbation; it is not described as a validated long-horizon failure
merely because the oracle can recover from it.
"""
from __future__ import annotations

from collections import defaultdict
from statistics import fmean, median
from typing import Any, Mapping, Sequence

from rq1.five_arm.measurements import MeasurementError, truncate_to_cap, validate_diagnostic_record
from rq1.five_arm.stats import bootstrap_mean_ci

PROPOSED_DIAGNOSTIC_THRESHOLDS: dict[str, Any] = {
    "min_completion_drop": 0.15,
    "min_median_extra_actions_when_both_succeed": 2,
    "min_mean_extra_invalid_selections": 0.5,
    "weak_completion_drop": 0.05,
    "inadequate_max_drop_upper": 0.10,
    "max_mechanical_extra_actions": 1,
    "detoured_ceiling": 0.90,
    "detoured_floor": 0.10,
    "undisturbed_ability_floor": 0.20,
    "max_infrastructure_missing_rate": 0.05,
    "max_mean_invalid_selections_per_episode": 1.0,
    "max_selection_exhaustion_rate": 0.10,
    "budget_candidate_caps": [50, 100],
    "budget_late_success_threshold": 0.20,
    "budget_floor": 0.10,
    "rationale": {
        "min_completion_drop": "15 points is the smallest drop that is at least about one task-level standard error with 12 "
                               "tasks (SE of a proportion near 0.5 with 12 clusters is about 0.14); smaller drops cannot be "
                               "distinguished from noise in the pilot.",
        "min_median_extra_actions_when_both_succeed": "the oracle needs exactly one extra navigation action to undo the "
                               "detour; a median of 2 or more extra actions indicates re-orientation, not a mechanical return.",
        "min_mean_extra_invalid_selections": "half an extra rejected selection per episode is a visible behavioural cost.",
        "weak_completion_drop": "below 5 points with no effort or invalid-action cost the detour is behaviourally inert.",
        "inadequate_max_drop_upper": "positive evidence of an inadequate manipulation requires the UPPER 95% bound of the "
                                     "completion drop to lie below 10 points; otherwise the pilot is inconclusive, not negative.",
        "max_mechanical_extra_actions": "the oracle needs exactly one extra navigation action; a median extra cost of at most "
                                        "one action is a mechanical return, not re-orientation.",
        "effort_only": "extra actions alone never establish meaningful disruption: action counts are sensitive to small "
                       "behavioural changes and say nothing about completion; they are reported as EFFORT_ONLY_INCREASE.",
        "detoured_ceiling": "a no-memory baseline above 0.90 leaves at most 10 points of headroom for any memory effect.",
        "detoured_floor": "below 0.10 recovery is near-infeasible at this budget; memory effects would be floor-limited.",
        "undisturbed_ability_floor": "if the model rarely completes tasks even without disruption, disruption cannot be measured.",
        "max_infrastructure_missing_rate": "more than 5% unrecoverable infrastructure loss threatens complete-case validity.",
        "max_mean_invalid_selections_per_episode": "more than one rejected selection per episode signals an interface problem.",
        "max_selection_exhaustion_rate": "more than 10% of episodes ending in selection exhaustion signals an interface problem.",
        "budget_late_success_threshold": "if more than 20% of baseline successes need more than the cap-50 allowance, cap 50 "
                                         "would censor a substantial share of genuine recoveries.",
    },
    "status": "PROPOSED_NOT_APPROVED",
}


class PilotError(ValueError):
    pass


def _validated(records: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    for record in records:
        problems = validate_diagnostic_record(record)
        if problems:
            raise PilotError(f"not a valid pilot-diagnostic record ({record.get('unit_key')}): {'; '.join(problems)}")
    return list(records)


def _final(records: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str, int], Mapping[str, Any]]:
    final: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    for record in sorted(records, key=lambda r: int(r["attempt"])):
        final[(str(record["task_id"]), str(record["condition"]), int(record["seed"]))] = record
    return final


def _effort(record: Mapping[str, Any]) -> int:
    return int(record["recovery_actions"]) if record["success"] else int(record["recovery_action_allowance"])


def _classify(value: float | None, ci: list[float] | None, threshold: float) -> str:
    if value is None:
        return "not_estimable"
    if ci is None:
        return "meets" if value >= threshold else "below"
    if ci[0] >= threshold:
        return "meets"
    if ci[1] < threshold:
        return "below"
    return "indeterminate"


def disruption_report(records: Sequence[Mapping[str, Any]], thresholds: Mapping[str, Any], *, seed: int = 20261124,
                      replicates: int = 5000) -> dict[str, Any]:
    final = _final(_validated(records))
    units = len(final)
    if not units:
        raise PilotError("no pilot-diagnostic records")
    missing = [r for r in final.values() if r["outcome_status"] != "VALID"]
    pairs = []
    for (task, condition, s), und in final.items():
        if condition != "UNDISTURBED":
            continue
        det = final.get((task, "DETOURED", s))
        if det is None or und["outcome_status"] != "VALID" or det["outcome_status"] != "VALID":
            continue
        pairs.append({"task_id": task, "seed": s, "u": und, "d": det})
    if not pairs:
        raise PilotError("no paired VALID undisturbed/detoured observations")
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for p in pairs:
        by_task[p["task_id"]].append(p)
    task_drop = [fmean(int(p["u"]["success"]) - int(p["d"]["success"]) for p in rows) for rows in by_task.values()]
    task_effort = [fmean(_effort(p["d"]) - _effort(p["u"]) for p in rows) for rows in by_task.values()]
    task_invalid = [fmean(p["d"]["invalid_actions"] - p["u"]["invalid_actions"] for p in rows) for rows in by_task.values()]
    both = [p["d"]["recovery_actions"] - p["u"]["recovery_actions"] for p in pairs if p["u"]["success"] and p["d"]["success"]]
    valid = [r for r in final.values() if r["outcome_status"] == "VALID"]
    rate = lambda cond: fmean(int(r["success"]) for r in valid if r["condition"] == cond) if any(r["condition"] == cond for r in valid) else None
    det_rate, und_rate = rate("DETOURED"), rate("UNDISTURBED")
    drop = fmean(task_drop)
    drop_ci = bootstrap_mean_ci(task_drop, seed=seed, replicates=replicates)
    invalid_extra = fmean(task_invalid)
    invalid_ci = bootstrap_mean_ci(task_invalid, seed=seed + 1, replicates=replicates)
    extra_median = median(both) if both else None
    report: dict[str, Any] = {
        "tasks": len(by_task), "pairs": len(pairs), "units": units,
        "undisturbed_completion": und_rate, "detoured_completion": det_rate,
        "mean_task_completion_drop": drop, "completion_drop_ci95_task_bootstrap": drop_ci,
        "mean_task_extra_capped_effort": fmean(task_effort),
        "median_extra_actions_when_both_succeed": extra_median, "pairs_both_succeed": len(both),
        "mean_task_extra_invalid_selections": invalid_extra, "extra_invalid_ci95_task_bootstrap": invalid_ci,
        "infrastructure_missing_rate": sum(r["outcome_status"] == "MISSING_INFRASTRUCTURE" for r in missing) / units,
        "invalid_measurement_rate": sum(r["outcome_status"] == "INVALID_MEASUREMENT" for r in missing) / units,
        "mean_invalid_selections_per_episode": fmean(r["invalid_actions"] for r in valid) if valid else None,
        "selection_exhaustion_rate": fmean(r["termination_reason"] == "invalid_action_limit" for r in valid) if valid else None,
        "thresholds": dict(thresholds),
        "note": "12-task pilot: diagnostic evidence only; intervals are task-bootstrap and wide by design",
    }
    t = thresholds
    findings: list[str] = []
    if report["infrastructure_missing_rate"] > t["max_infrastructure_missing_rate"]:
        findings.append("INFRASTRUCTURE_UNSTABLE")
    if (report["mean_invalid_selections_per_episode"] or 0) > t["max_mean_invalid_selections_per_episode"] or \
            (report["selection_exhaustion_rate"] or 0) > t["max_selection_exhaustion_rate"]:
        findings.append("INTERFACE_PROBLEM")
    if und_rate is not None and und_rate <= t["undisturbed_ability_floor"]:
        findings.append("TASK_ABILITY_FLOOR")
    if det_rate is not None and det_rate >= t["detoured_ceiling"]:
        findings.append("CEILING")
    if det_rate is not None and det_rate <= t["detoured_floor"]:
        findings.append("FLOOR")
    criteria = {"completion_drop": _classify(drop, drop_ci, t["min_completion_drop"]),
                "extra_effort": ("not_estimable" if extra_median is None else
                                 "meets" if extra_median >= t["min_median_extra_actions_when_both_succeed"] else "below"),
                "extra_invalid": _classify(invalid_extra, invalid_ci, t["min_mean_extra_invalid_selections"])}
    report["criteria"] = criteria
    report["disruption_class"] = _disruption_class(criteria, drop, drop_ci, extra_median, invalid_ci, findings, t)
    report["methodological_findings"] = findings
    report["recommendation"] = RECOMMENDATIONS[report["disruption_class"]]
    report["status"] = "diagnostic evidence for author review; not an approval and never an automatic GO"
    return report


RECOMMENDATIONS = {
    "MEANINGFUL_DISRUPTION": "author review: completion evidence supports a disruptive perturbation; check findings and "
                             "family table before considering the five-arm pilot",
    "EFFORT_ONLY_INCREASE": "author review: only extra actions, no completion or invalid-action evidence; treat the detour as "
                            "a mild perturbation and keep the claim to task completion after a navigation perturbation",
    "INADEQUATE_MANIPULATION": "author review: positive evidence that the detour does not disrupt; strengthen the manipulation "
                               "or narrow the claim to task continuation (no recovery claim)",
    "UNINFORMATIVE_COMPETENCE_FLOOR": "author review: the model rarely completes tasks even without the detour; the "
                                      "manipulation cannot be judged; consider the budget, the population or the claim",
    "INCONCLUSIVE": "author review: intervals too wide to decide; do not treat as evidence of an adequate or an inadequate "
                    "manipulation",
}


def _disruption_class(criteria, drop, drop_ci, extra_median, invalid_ci, findings, t) -> str:
    if criteria["completion_drop"] == "meets" or (criteria["extra_invalid"] == "meets" and drop >= t["weak_completion_drop"]):
        return "MEANINGFUL_DISRUPTION"
    if "TASK_ABILITY_FLOOR" in findings:
        return "UNINFORMATIVE_COMPETENCE_FLOOR"
    drop_upper = drop_ci[1] if drop_ci is not None else drop
    invalid_upper = invalid_ci[1] if invalid_ci is not None else None
    mechanical = extra_median is None or extra_median <= t["max_mechanical_extra_actions"]
    no_invalid_cost = invalid_upper is None or invalid_upper < t["min_mean_extra_invalid_selections"]
    if drop_upper < t["inadequate_max_drop_upper"] and mechanical and no_invalid_cost:
        return "INADEQUATE_MANIPULATION"
    if criteria["extra_effort"] == "meets" and criteria["completion_drop"] == "below" and criteria["extra_invalid"] != "meets":
        return "EFFORT_ONLY_INCREASE"
    return "INCONCLUSIVE"


def family_stratified_diagnostic(records: Sequence[Mapping[str, Any]], thresholds: Mapping[str, Any]) -> dict[str, Any]:
    """Per-family DESCRIPTIVE table (a 12-task pilot has 2-3 tasks per family): basic competence (UNDISTURBED completion),
    perturbation difficulty (completion drop), ceiling/floor, extra capped effort and interface problems.  Families whose
    UNDISTURBED completion is at or below the competence floor form the prospective floor-family SENSITIVITY set for the
    confirmatory analysis (defined from pilot valid_seen tasks only; confirmatory tasks are never excluded on their own
    D18-vs-D0 outcomes, and the primary analysis keeps every roster task)."""
    final = _final(_validated(records))
    t = thresholds
    families: dict[str, dict[str, list[Mapping[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for (task, condition, s), record in final.items():
        if record["outcome_status"] == "VALID":
            families[str(record["family"])][condition].append(record)
    rows = {}
    for family, by in sorted(families.items()):
        und, det = by.get("UNDISTURBED", []), by.get("DETOURED", [])
        rate = lambda rs: fmean(int(r["success"]) for r in rs) if rs else None
        u, d = rate(und), rate(det)
        pairs = {(r["task_id"], r["seed"]): r for r in und}
        matched = [(pairs[(r["task_id"], r["seed"])], r) for r in det if (r["task_id"], r["seed"]) in pairs]
        both = [dd["recovery_actions"] - uu["recovery_actions"] for uu, dd in matched if uu["success"] and dd["success"]]
        episodes = und + det
        invalid = fmean(r["invalid_actions"] for r in episodes) if episodes else None
        flags = []
        if u is not None and u <= t["undisturbed_ability_floor"]:
            flags.append("COMPETENCE_FLOOR")
        if u is not None and d is not None and u >= t["detoured_ceiling"] and d >= t["detoured_ceiling"]:
            flags.append("CEILING")
        if d is not None and d <= t["detoured_floor"] and "COMPETENCE_FLOOR" not in flags:
            flags.append("PERTURBATION_FLOOR")
        if invalid is not None and invalid > t["max_mean_invalid_selections_per_episode"]:
            flags.append("INTERFACE_PROBLEM")
        median_extra = median(both) if both else None
        rows[family] = {"tasks": len({r["task_id"] for r in episodes}), "matched_pairs": len(matched),
                        "competence_undisturbed_completion": u, "detoured_completion": d,
                        "perturbation_completion_drop": (u - d) if u is not None and d is not None else None,
                        "median_extra_actions_when_both_succeed": median_extra,
                        "effort_beyond_mechanical_return": (median_extra is not None
                                                            and median_extra > t["max_mechanical_extra_actions"]),
                        "mean_invalid_selections_per_episode": invalid, "flags": flags,
                        "inference": "descriptive only (too few tasks per family for any test)"}
    sensitivity = sorted(f for f, r in rows.items() if "COMPETENCE_FLOOR" in r["flags"])
    return {"families": rows, "prospective_floor_family_sensitivity_set": sensitivity,
            "rule": "confirmatory primary analysis keeps every roster task; a prespecified sensitivity analysis excludes "
                    "the families listed here (flagged from pilot valid_seen tasks only, before confirmatory outcomes)",
            "overall": disruption_report(records, thresholds, replicates=2000),
            "status": "descriptive diagnostic for author review; not an approval"}


def seed_variability(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Across inference seeds within (task, condition): identical action sequences and outcome discordance."""
    final = _final(_validated(records))
    cells: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for (task, condition, _s), record in final.items():
        if record["outcome_status"] == "VALID":
            cells[(task, condition)].append(record)
    multi = {k: v for k, v in cells.items() if len(v) > 1}
    if not multi:
        return {"cells_with_multiple_seeds": 0, "identical_trajectory_rate": None, "outcome_discordance_rate": None}
    identical = [len({tuple(s["action"] for s in r["recovery_trace"]) for r in v}) == 1 for v in multi.values()]
    discordant = [len({r["success"] for r in v}) > 1 for v in multi.values()]
    return {"cells_with_multiple_seeds": len(multi), "identical_trajectory_rate": fmean(identical),
            "outcome_discordance_rate": fmean(discordant),
            "interpretation": "a high identical-trajectory rate means extra seeds add little information (SD-08)"}


def budget_decision(records: Sequence[Mapping[str, Any]], *, candidate_caps: Sequence[int], late_success_threshold: float,
                    floor: float, cap_invariance_evidence: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """SD-17: uses ONLY DETOURED (no-memory) diagnostic records run at max(candidate_caps).  Choose the smallest cap
    unless (a) more than ``late_success_threshold`` of baseline successes need more recovery actions than that cap allows,
    or (b) baseline success at that cap is <= ``floor`` while higher at the next cap.  Truncation is used only if the
    CPU cap-invariance replay held for every task; otherwise separate cap-specific diagnostic runs are required."""
    caps = sorted(int(c) for c in candidate_caps)
    run_cap = caps[-1]
    detoured = [r for r in _validated(records) if r["condition"] == "DETOURED"]
    final = _final(detoured)
    valid = [r for r in final.values() if r["outcome_status"] == "VALID"]
    if not valid:
        raise PilotError("no valid DETOURED diagnostic records")
    if any(int(r["total_action_cap"]) != run_cap for r in valid):
        raise PilotError(f"diagnostic records must all be run at the largest candidate cap {run_cap}")
    tasks = {r["task_id"] for r in valid}
    held = {e["task_id"] for e in cap_invariance_evidence if e.get("truncation_environment_assumptions_hold") is True}
    if not tasks <= held:
        return {"chosen_cap": None, "decision": "TRUNCATION_NOT_SUPPORTED",
                "action": "run a separate cap-specific diagnostic at each candidate cap; truncation is not valid for "
                          f"{sorted(tasks - held)[:5]}", "pooled_over_arms": "D0 detoured baseline only"}
    rates: dict[int, float] = {}
    truncated: dict[int, list[dict[str, Any]]] = {}
    for cap in caps:
        try:
            truncated[cap] = [truncate_to_cap(r, cap) for r in valid]
        except MeasurementError as exc:
            raise PilotError(str(exc)) from exc
        feasible = [t for t in truncated[cap] if t["feasible"]]
        rates[cap] = fmean(bool(t["success"]) for t in feasible) if feasible else 0.0
    successes = [r for r in valid if r["success"]]
    evaluation = []
    for i, cap in enumerate(caps[:-1]):
        late = (sum(not t["success"] for r, t in zip(valid, truncated[cap]) if r["success"]) / len(successes)) if successes else 0.0
        floored = rates[cap] <= floor and rates[caps[i + 1]] > rates[cap]
        infeasible = sum(not t["feasible"] for t in truncated[cap])
        evaluation.append({"cap": cap, "baseline_success": rates[cap], "late_success_fraction": late,
                           "tasks_infeasible_at_cap": infeasible,
                           "rejected": late > late_success_threshold or floored or infeasible > 0})
        if not evaluation[-1]["rejected"]:
            return {"chosen_cap": cap, "decision": "CHOSEN", "evaluation": evaluation, "baseline_success_by_cap": rates,
                    "pooled_over_arms": "D0 detoured baseline only (no memory arm consulted)"}
    return {"chosen_cap": run_cap, "decision": "CHOSEN", "evaluation": evaluation, "baseline_success_by_cap": rates,
            "pooled_over_arms": "D0 detoured baseline only (no memory arm consulted)"}
