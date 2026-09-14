"""Offline analysis of merged amended-evaluation results (Decision 012).

A pure reader of saved results and human labels: it never runs a model, Hermes, or
ALFWorld, and it computes only the frozen metric definitions.
"""
from __future__ import annotations

import csv
import io
import random
import statistics
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

from rq1.analysis.kappa import cohens_kappa_details
from rq1.analysis.labelling import IRRELEVANT, RELEVANT
from rq1.evaluation.amended_protocol import CONDITIONS
from rq1.skills.library import TASK_FAMILIES

BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20260806


def _rate(values: Iterable[bool]) -> float | None:
    entries = list(values)
    return None if not entries else sum(bool(item) for item in entries) / len(entries)


def _mean(values: Iterable[float | int | None]) -> float | None:
    entries = [float(value) for value in values if value is not None]
    return None if not entries else statistics.fmean(entries)


def _median(values: Iterable[float | int | None]) -> float | None:
    entries = [float(value) for value in values if value is not None]
    return None if not entries else statistics.median(entries)


def _rank(values: Sequence[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(ordered):
        end = index
        while end + 1 < len(ordered) and ordered[end + 1][1] == ordered[index][1]:
            end += 1
        for position in range(index, end + 1):
            ranks[ordered[position][0]] = (index + end + 2) / 2
        index = end + 1
    return ranks


def spearman(pairs: Sequence[tuple[float, float]]) -> float | None:
    if len(pairs) < 2:
        return None
    xs, ys = zip(*pairs)
    rx, ry = _rank(xs), _rank(ys)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    numerator = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    denominator = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return None if denominator == 0 else numerator / denominator


def unit_rows(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """One analysis row per matrix unit from its authoritative terminal record."""
    rows = []
    for record in records:
        unit = record.get("matrix_unit") or {}
        completed = record.get("status") == "completed"
        eligible = completed and record.get("eligible") is True and record.get("post_failure_budget_complete") is True
        retrieval = record.get("retrieval") or {}
        rows.append({
            "unit_key": record.get("run_key"),
            "task_id": record.get("task_id") or unit.get("task_id"),
            "task_family": record.get("task_family") or unit.get("task_family"),
            "seed": record.get("seed") if record.get("seed") is not None else unit.get("seed"),
            "condition": record.get("condition") or unit.get("condition"),
            "infrastructure_failure": record.get("status") == "failed",
            "eligible": eligible,
            "recovery_success": bool(eligible and record.get("recovery_success") is True),
            "post_failure_actions": record.get("post_failure_actions") if eligible else None,
            "recovery_latency_actions": record.get("recovery_latency_actions") if eligible else None,
            "recovery_latency_seconds": record.get("recovery_latency_seconds") if eligible else None,
            "invalid_action_selections": record.get("invalid_action_selections") if eligible else None,
            "retries": record.get("retries") if eligible else None,
            "selection_exhausted": bool(eligible and record.get("selection_exhausted")),
            "retrieval_event_id": retrieval.get("event_id"),
            "retrieved_skill_ids": [item.get("skill_id") for item in retrieval.get("top") or []],
        })
    return rows


def _summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    eligible = [row for row in rows if row["eligible"]]
    successes = [row for row in eligible if row["recovery_success"]]
    return {
        "scheduled_units": len(rows),
        "infrastructure_failures": sum(row["infrastructure_failure"] for row in rows),
        "infrastructure_failure_rate": _rate(row["infrastructure_failure"] for row in rows),
        "eligible_units": len(eligible),
        "recovery_successes": len(successes),
        "conditional_recovery_rate": _rate(row["recovery_success"] for row in eligible),
        "task_completion_rate": (len(successes) / len(rows)) if rows else None,
        "recovery_latency_actions_mean": _mean(row["recovery_latency_actions"] for row in successes),
        "recovery_latency_actions_median": _median(row["recovery_latency_actions"] for row in successes),
        "recovery_latency_seconds_mean": _mean(row["recovery_latency_seconds"] for row in successes),
        "recovery_latency_seconds_median": _median(row["recovery_latency_seconds"] for row in successes),
        "post_failure_actions_mean": _mean(row["post_failure_actions"] for row in eligible),
        "invalid_action_selections_mean": _mean(row["invalid_action_selections"] for row in eligible),
        "invalid_action_selections_total": sum(row["invalid_action_selections"] or 0 for row in eligible),
        "retries_total": sum(row["retries"] or 0 for row in eligible),
        "selection_exhausted_units": sum(row["selection_exhausted"] for row in eligible),
    }


def bootstrap(rows: Sequence[Mapping[str, Any]], *, seed: int = BOOTSTRAP_SEED, replicates: int = BOOTSTRAP_REPLICATES) -> dict[str, Any]:
    """Percentile CIs resampling task-seed cells, paired across conditions."""
    cells: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        cells[(row["task_id"], row["seed"])].append(row)
    keys = sorted(cells)
    rng = random.Random(seed)
    samples: dict[str, list[float]] = defaultdict(list)
    for _ in range(replicates if keys else 0):
        chosen = [cells[rng.choice(keys)] for _ in keys]
        values = {}
        for condition in CONDITIONS:
            eligible = [row for cell in chosen for row in cell if row["condition"] == condition and row["eligible"]]
            rate = _rate(row["recovery_success"] for row in eligible)
            if rate is not None:
                values[condition] = rate
                samples[condition].append(rate)
        for condition in CONDITIONS[1:]:
            if condition in values and "NoLib" in values:
                samples[f"{condition}_minus_NoLib"].append(values[condition] - values["NoLib"])

    def interval(values: list[float]) -> dict[str, Any]:
        ordered = sorted(values)
        return {"lower": ordered[int(0.025 * (len(ordered) - 1))], "upper": ordered[int(0.975 * (len(ordered) - 1))], "replicates": len(ordered)} if ordered else {"lower": None, "upper": None, "replicates": 0}

    return {"method": "percentile_bootstrap_resampling_task_seed_cells", "seed": seed, "replicates": replicates, "cells": len(keys),
            "conditional_recovery_rate": {name: interval(values) for name, values in sorted(samples.items())}}


def read_labels(data: bytes, column: str) -> dict[tuple[str, int], str]:
    labels: dict[tuple[str, int], str] = {}
    for row in csv.DictReader(io.StringIO(data.decode("utf-8-sig"))):
        value = str(row.get(column, "")).strip().upper()
        if value not in {RELEVANT, IRRELEVANT}:
            raise ValueError(f"label for item {row.get('item_id')} rank {row.get('rank')} must be RELEVANT or IRRELEVANT")
        labels[(row["item_id"], int(row["rank"]))] = value
    return labels


def retrieval_quality(
    key_rows: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    *,
    rater_a: Mapping[tuple[str, int], str],
    rater_b: Mapping[tuple[str, int], str],
    adjudicated: Mapping[tuple[str, int], str],
) -> dict[str, Any]:
    aligned = sorted(set(rater_a) & set(rater_b))
    if set(rater_a) != set(rater_b):
        raise ValueError("the two raters labelled different items")
    agreement = cohens_kappa_details([rater_a[key] for key in aligned], [rater_b[key] for key in aligned])
    by_unit = {row["unit_key"]: row for row in rows}
    per_item = []
    for key in key_rows:
        ranks = range(1, int(key["top_count"]) + 1)
        labels = [adjudicated.get((key["item_id"], rank)) for rank in ranks]
        if any(label is None for label in labels):
            raise ValueError(f"missing adjudicated label for item {key['item_id']}")
        precision = sum(label == RELEVANT for label in labels) / len(labels)
        unit = by_unit.get(key["unit_key"]) or {}
        per_item.append({"item_id": key["item_id"], "condition": key["condition"], "task_family": unit.get("task_family"),
                         "precision_at_3": precision, "retrieval_noise": 1 - precision,
                         "recovery_success": unit.get("recovery_success"), "post_failure_actions": unit.get("post_failure_actions"),
                         "recovery_latency_seconds": unit.get("recovery_latency_seconds")})
    by_condition = {}
    for condition in CONDITIONS:
        items = [item for item in per_item if item["condition"] == condition]
        by_condition[condition] = {"retrievals": len(items), "precision_at_3_mean": _mean(item["precision_at_3"] for item in items),
                                   "retrieval_noise_mean": _mean(item["retrieval_noise"] for item in items)}
    by_condition["NoLib"] = {**by_condition["NoLib"], "note": "no retrieval; reported separately, never coerced to a noise value"}

    def association(field: str) -> float | None:
        return spearman([(item["retrieval_noise"], float(item[field])) for item in per_item if item.get(field) is not None])

    return {
        "raters": 2,
        "agreement": {**agreement, "computed_on": "original independent labels"},
        "by_condition": by_condition,
        "association": {"method": "spearman", "causal_claim": False, "descriptive_only": True,
                        "noise_with_recovery_success": association("recovery_success"),
                        "noise_with_post_failure_actions": association("post_failure_actions"),
                        "noise_with_recovery_latency_seconds": association("recovery_latency_seconds")},
        "per_item": per_item,
    }


def analyze(records: Iterable[Mapping[str, Any]], *, quality: Mapping[str, Any] | None = None, seed: int = BOOTSTRAP_SEED,
            replicates: int = BOOTSTRAP_REPLICATES) -> dict[str, Any]:
    rows = unit_rows(records)
    return {
        "schema_version": 1,
        "analysis_kind": "rq1-amended-controlled-recovery",
        "causal_claims": False,
        "units": len(rows),
        "by_condition": {condition: _summary([row for row in rows if row["condition"] == condition]) for condition in CONDITIONS},
        "by_condition_and_family": {condition: {family: _summary([row for row in rows if row["condition"] == condition and row["task_family"] == family])
                                                for family in TASK_FAMILIES} for condition in CONDITIONS},
        "by_condition_and_seed": {condition: {str(seed_value): _summary([row for row in rows if row["condition"] == condition and row["seed"] == seed_value])
                                              for seed_value in sorted({row["seed"] for row in rows})} for condition in CONDITIONS},
        "uncertainty": bootstrap(rows, seed=seed, replicates=replicates),
        "retrieval_quality": dict(quality) if quality is not None else {"status": "PENDING_HUMAN_RELEVANCE_LABELS"},
    }
