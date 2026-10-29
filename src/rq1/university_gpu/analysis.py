"""Offline descriptive and task-cluster analysis for complete six-arm runs."""
from __future__ import annotations

import csv
import io
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping, Sequence

from rq1.university_gpu.manifest import CONDITIONS, validate_manifest
from rq1.university_gpu.models import content_hash, sha256_file

CONTRASTS = (("D18", "D0"), ("Fill", "D0"), ("Rel", "Fill"), ("Irr", "Fill"),
             ("Rel", "Irr"), ("Gate", "D18"), ("Rel", "D18"), ("Irr", "D18"), ("Fill", "D18"))
LABEL_COLUMNS = ("adjudicated", "rater_a", "rater_b")


class AnalysisError(ValueError):
    pass


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AnalysisError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise AnalysisError(f"JSONL row {line_number} must be an object")
            rows.append(dict(value.get("result", value)))
    return rows


def read_relevance(data: bytes) -> dict[tuple[str, str], dict[str, str]]:
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise AnalysisError("relevance labels must be UTF-8") from exc
    reader = csv.DictReader(io.StringIO(text))
    required = {"task_id", "skill_id", *LABEL_COLUMNS}
    if not reader.fieldnames or set(reader.fieldnames) != required:
        raise AnalysisError(f"relevance columns must be exactly {sorted(required)}")
    labels: dict[tuple[str, str], dict[str, str]] = {}
    for line, row in enumerate(reader, 2):
        key = (str(row["task_id"]).strip(), str(row["skill_id"]).strip())
        if not all(key) or key in labels:
            raise AnalysisError(f"missing or duplicate task-skill label at line {line}")
        values = {name: str(row[name]).strip().upper() for name in LABEL_COLUMNS}
        if any(value not in {"RELEVANT", "IRRELEVANT"} for value in values.values()):
            raise AnalysisError(f"invalid relevance rating at line {line}")
        labels[key] = values
    return labels


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise AnalysisError("cannot compute a percentile from no bootstrap draws")
    position = (len(sorted_values) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def _cluster_interval(
    values_by_task: Mapping[str, Sequence[float]], *, seed: int, replicates: int,
) -> dict[str, Any]:
    task_ids = sorted(values_by_task)
    task_means = [fmean(values_by_task[task_id]) for task_id in task_ids]
    estimate = fmean(task_means)
    if len(task_ids) < 2:
        return {"estimate": estimate, "ci_95": None, "task_clusters": len(task_ids), "bootstrap_replicates": 0}
    rng = random.Random(seed)
    draws = []
    for _ in range(replicates):
        sampled = [task_means[rng.randrange(len(task_means))] for _ in task_means]
        draws.append(fmean(sampled))
    draws.sort()
    return {"estimate": estimate,
            "ci_95": [_percentile(draws, 0.025), _percentile(draws, 0.975)],
            "task_clusters": len(task_ids), "bootstrap_replicates": replicates}


def analyze(
    records: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    *,
    relevance_labels: Mapping[tuple[str, str], Mapping[str, str]] | None = None,
    seed: int = 20261026,
    bootstrap_replicates: int = 10_000,
) -> dict[str, Any]:
    problems = validate_manifest(manifest, require_frozen=True)
    if problems:
        raise AnalysisError("frozen manifest is invalid: " + "; ".join(problems))
    if bootstrap_replicates < 1000:
        raise AnalysisError("task-cluster intervals require at least 1,000 bootstrap replicates")
    units = {str(item["unit_key"]): item for item in manifest["units"]}
    rows_by_cell: dict[tuple[str, int], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    seen_units: set[str] = set()
    for raw in records:
        row = dict(raw)
        key = str(row.get("unit_key", ""))
        unit = units.get(key)
        if unit is None or key in seen_units:
            raise AnalysisError(f"unknown or duplicate unit_key in results: {key}")
        seen_units.add(key)
        if row.get("status") != "completed" or not isinstance(row.get("recovery_success"), bool):
            raise AnalysisError(f"unit {key} lacks a completed binary recovery outcome")
        if row.get("task_id") != unit["task_id"] or row.get("condition") != unit["condition"]:
            raise AnalysisError(f"unit {key} result identity differs from its frozen manifest")
        cell = (str(unit["task_id"]), int(unit["seed"]))
        condition = str(unit["condition"])
        if condition in rows_by_cell[cell]:
            raise AnalysisError(f"duplicate paired cell for {cell} / {condition}")
        rows_by_cell[cell][condition] = row
    if seen_units != set(units):
        raise AnalysisError(f"analysis requires all {len(units)} planned units; observed {len(seen_units)}")
    if set(rows_by_cell) != {(str(unit["task_id"]), int(unit["seed"])) for unit in manifest["units"]}:
        raise AnalysisError("results do not cover every frozen task-seed cell")
    for cell, conditions in rows_by_cell.items():
        if set(conditions) != set(CONDITIONS):
            raise AnalysisError(f"incomplete paired condition set for task-seed cell {cell}")

    task_families = {str(task["task_id"]): str(task["task_family"]) for task in manifest["tasks"]}
    condition_summary: dict[str, Any] = {}
    family_summary: dict[str, Any] = {}
    for condition in CONDITIONS:
        outcomes = [bool(row[condition]["recovery_success"]) for row in rows_by_cell.values()]
        condition_summary[condition] = {"episodes": len(outcomes), "successes": sum(outcomes),
                                        "recovery_rate": fmean(outcomes)}
    family_cells: dict[tuple[str, str], list[bool]] = defaultdict(list)
    for (task_id, _seed), condition_rows in rows_by_cell.items():
        family = task_families[task_id]
        for condition in CONDITIONS:
            family_cells[(family, condition)].append(bool(condition_rows[condition]["recovery_success"]))
    for (family, condition), outcomes in sorted(family_cells.items()):
        family_summary.setdefault(family, {})[condition] = {"episodes": len(outcomes), "successes": sum(outcomes),
                                                            "recovery_rate": fmean(outcomes)}

    contrasts: dict[str, Any] = {}
    for index, (treatment, reference) in enumerate(CONTRASTS):
        by_task: dict[str, list[float]] = defaultdict(list)
        for (task_id, _replicate_seed), condition_rows in rows_by_cell.items():
            difference = int(bool(condition_rows[treatment]["recovery_success"])) - int(bool(condition_rows[reference]["recovery_success"]))
            by_task[task_id].append(float(difference))
        contrasts[f"{treatment}-{reference}"] = {
            "treatment": treatment, "reference": reference,
            **_cluster_interval(by_task, seed=seed + index, replicates=bootstrap_replicates),
            "interpretation": "exploratory task-cluster percentile interval; not a power or sufficiency claim",
        }

    noise: dict[str, Any] = {}
    if relevance_labels is not None:
        for column in LABEL_COLUMNS:
            condition_task: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
            for (task_id, _replicate_seed), condition_rows in rows_by_cell.items():
                for condition in CONDITIONS[1:]:
                    retrieved = condition_rows[condition].get("retrieved_skill_ids")
                    if not isinstance(retrieved, list) or len(retrieved) != 3:
                        raise AnalysisError(f"{condition} unit has no exact top-3 retrieval list")
                    hits = []
                    for skill_id in retrieved:
                        label = relevance_labels.get((task_id, str(skill_id)))
                        if label is None or column not in label:
                            raise AnalysisError(f"missing {column} relevance label for {task_id}/{skill_id}")
                        hits.append(label[column] == "RELEVANT")
                    condition_task[condition][task_id].append(1.0 - fmean(hits))
            noise[column] = {
                condition: _cluster_interval(values, seed=seed + 100 + condition_index,
                                             replicates=bootstrap_replicates)
                for condition_index, (condition, values) in enumerate(sorted(condition_task.items()))
            }

    return {
        "schema": "rq1-university-gpu-analysis-v1",
        "stage": manifest["stage"], "run_id": manifest["run_id"],
        "manifest_sha256": manifest["manifest_sha256"],
        "conditions": condition_summary,
        "task_family_descriptive": family_summary,
        "paired_task_cluster_contrasts": contrasts,
        "retrieval_noise_sensitivity": noise,
        "analysis": {"seed": seed, "bootstrap_replicates": bootstrap_replicates,
                     "cluster_unit": "task; all seeds for a task are resampled together",
                     "primary_frozen_analysis_replaced": False, "exploratory": True,
                     "power_or_sample_sufficiency_established": False,
                     "labels_preserved_and_read_only": relevance_labels is not None},
        "records_sha256": content_hash([dict(row) for row in records]),
        "scientific_evidence": False,
        "evidence_note": "Offline analysis alone cannot establish real hardware execution or scientific validity.",
    }


def analyze_files(
    results_path: Path,
    manifest_path: Path,
    *,
    relevance_path: Path | None = None,
    seed: int = 20261026,
    bootstrap_replicates: int = 10_000,
) -> dict[str, Any]:
    from rq1.university_gpu.models import load_json

    labels = read_relevance(relevance_path.read_bytes()) if relevance_path else None
    result = analyze(read_jsonl(results_path), load_json(manifest_path), relevance_labels=labels,
                     seed=seed, bootstrap_replicates=bootstrap_replicates)
    result["source_hashes"] = {
        "results_sha256": sha256_file(results_path), "manifest_sha256": sha256_file(manifest_path),
        "relevance_sha256": sha256_file(relevance_path) if relevance_path else None,
    }
    return result
