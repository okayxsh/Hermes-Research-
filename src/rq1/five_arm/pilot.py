"""PILOT-ONLY diagnostics (PROPOSED): failure-manipulation validation, seed variability and the action-budget rule.

All functions run on saved pilot-diagnostic records only (UNDISTURBED / DETOURED continuations without memory; see
``executor``).  They never see a memory arm, so no treatment difference can influence them.  Thresholds are explicit
arguments; ``PROPOSED_DIAGNOSTIC_THRESHOLDS`` is the engineering proposal for SD-16 and is NOT approved.

A 12-task pilot is a diagnostic exercise, not a hypothesis test: every classification is reported with a task-bootstrap
interval, and a classification whose interval straddles a threshold is reported as INDETERMINATE (design review), never
as an automatic GO.  A one-action detour is a controlled navigation perturbation; it is not described as a validated
long-horizon failure merely because the oracle can recover from it.
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
    if any(v == "meets" for v in criteria.values()):
        disruption = "MEANINGFUL_DISRUPTION"
    elif any(v == "indeterminate" for v in criteria.values()):
        disruption = "INDETERMINATE"
    elif drop >= t["weak_completion_drop"]:
        disruption = "WEAK_DISRUPTION"
    else:
        disruption = "INSUFFICIENT_DISRUPTION"
    report["disruption_class"] = disruption
    report["methodological_findings"] = findings
    report["recommendation"] = (
        "proceed to five-arm pilot review" if disruption == "MEANINGFUL_DISRUPTION" and not findings else
        "narrow the claim to recovery after a controlled navigation perturbation, or strengthen the manipulation"
        if disruption in ("WEAK_DISRUPTION", "INSUFFICIENT_DISRUPTION") else
        "design review required before any five-arm execution")
    report["status"] = "diagnostic evidence for author review; not an approval"
    return report


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
