"""PILOT-ONLY diagnostics (PROPOSED): failure-manipulation validation and the pre-specified action-budget rule.

Both functions run on saved pilot records only, are pooled over arms (they never look at between-arm differences),
and take their thresholds as explicit arguments -- the thresholds are scientific decisions, not code defaults.

Failure-manipulation diagnostic: for the same pilot task, checkpoint and seed, the model continues from the checkpoint
either UNDISTURBED (no detour; recovery budget = cap - prefix) or after the CONTROLLED DETOUR (budget = cap - prefix - 1),
without memory (D0 presentation), so memory cannot mask the disruption.  Meaningful disruption is evidenced by lower
completion, more actions to success, or more invalid / repeated actions after the detour -- never by the mere presence
of the injected detour.

Budget rule: the pilot runs at the LARGEST candidate cap; the prompt never shows the remaining budget, so outcomes at a
smaller cap are obtained by truncation (success only if reached within that cap's recovery budget).
"""
from __future__ import annotations

from collections import defaultdict
from statistics import fmean, median
from typing import Any, Mapping, Sequence


class PilotError(ValueError):
    pass


def disruption_report(pairs: Sequence[Mapping[str, Any]], *, min_completion_drop: float, min_extra_actions: float,
                      ceiling: float, floor: float) -> dict[str, Any]:
    """``pairs``: one row per (task, seed) with keys undisturbed_success, detour_success, undisturbed_actions,
    detour_actions, undisturbed_invalid, detour_invalid (actions/invalid counted after the checkpoint)."""
    if not pairs:
        raise PilotError("no paired undisturbed/detour pilot observations")
    by_task: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in pairs:
        by_task[str(row["task_id"])].append(row)
    task_drop = {t: fmean(int(r["undisturbed_success"]) - int(r["detour_success"]) for r in rows) for t, rows in by_task.items()}
    both = [r for r in pairs if r["undisturbed_success"] and r["detour_success"]]
    extra = [r["detour_actions"] - r["undisturbed_actions"] for r in both]
    detour_rate = fmean(int(r["detour_success"]) for r in pairs)
    undisturbed_rate = fmean(int(r["undisturbed_success"]) for r in pairs)
    report = {"pairs": len(pairs), "tasks": len(by_task), "undisturbed_completion": undisturbed_rate, "detour_completion": detour_rate,
              "mean_task_completion_drop": fmean(task_drop.values()),
              "median_extra_actions_when_both_succeed": median(extra) if extra else None,
              "mean_extra_invalid_actions": fmean(r["detour_invalid"] - r["undisturbed_invalid"] for r in pairs),
              "thresholds": {"min_completion_drop": min_completion_drop, "min_extra_actions": min_extra_actions,
                             "ceiling": ceiling, "floor": floor}}
    issues = []
    if detour_rate >= ceiling:
        issues.append("ceiling: the detour condition is solved almost always without memory; the manipulation may be too weak")
    if detour_rate <= floor:
        issues.append("floor: the detour condition is almost never solved; recovery may be infeasible at this budget")
    disrupted = report["mean_task_completion_drop"] >= min_completion_drop or (
        report["median_extra_actions_when_both_succeed"] is not None and report["median_extra_actions_when_both_succeed"] >= min_extra_actions)
    if not disrupted:
        issues.append("no meaningful disruption: completion and action costs after the detour match the undisturbed continuation")
    report["meaningful_disruption"] = disrupted and not issues
    report["methodological_issues"] = issues
    return report


def budget_decision(records: Sequence[Mapping[str, Any]], *, candidate_caps: Sequence[int], late_success_threshold: float,
                    floor: float) -> dict[str, Any]:
    """Arm-POOLED rule on pilot records run at max(candidate_caps).  Choose the smallest cap unless (a) more than
    ``late_success_threshold`` of all successes need more recovery actions than that cap allows, or (b) pooled success at
    that cap is <= ``floor`` while it is higher at the next cap.  Never consults the D18-D0 (or any arm) difference."""
    caps = sorted(int(c) for c in candidate_caps)
    run_cap = caps[-1]
    valid = [r for r in records if r.get("outcome_status") == "VALID"]
    if not valid:
        raise PilotError("no valid pilot records")
    if any(int(r["total_action_cap"]) != run_cap for r in valid):
        raise PilotError(f"pilot records must all be run at the largest candidate cap {run_cap}")

    def success_at(r, cap):
        return bool(r["success"]) and r["recovery_actions"] <= cap - r["prefix_actions"] - r["detour_actions"]

    rates = {cap: fmean(success_at(r, cap) for r in valid) for cap in caps}
    successes = [r for r in valid if r["success"]]
    evaluation = []
    for i, cap in enumerate(caps[:-1]):
        late = (sum(not success_at(r, cap) for r in successes) / len(successes)) if successes else 0.0
        floored = rates[cap] <= floor and rates[caps[i + 1]] > rates[cap]
        evaluation.append({"cap": cap, "pooled_success": rates[cap], "late_success_fraction": late,
                           "rejected": late > late_success_threshold or floored})
        if not evaluation[-1]["rejected"]:
            return {"chosen_cap": cap, "evaluation": evaluation, "pooled_success_by_cap": rates, "pooled_over_arms": True}
    return {"chosen_cap": run_cap, "evaluation": evaluation, "pooled_success_by_cap": rates, "pooled_over_arms": True}
