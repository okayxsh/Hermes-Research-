"""Fail-closed validation of blinded task-by-skill applicability decisions."""
from __future__ import annotations

import csv
import io
from collections import defaultdict
from typing import Any, Iterable, Mapping

LABELS = {"RELEVANT", "IRRELEVANT"}
REQUIRED_COLUMNS = {"task_id", "skill_id", "rater_a", "rater_b", "adjudicated"}


class RelevanceError(ValueError):
    pass


def parse_relevance_csv(data: bytes, *, allowed_tasks: set[str], library_skill_ids: set[str]) -> dict[str, dict[str, list[str]]]:
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise RelevanceError("relevance CSV must be UTF-8") from exc
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames or set(reader.fieldnames) != REQUIRED_COLUMNS:
        raise RelevanceError(f"relevance CSV columns must be exactly {sorted(REQUIRED_COLUMNS)}")
    rows: dict[tuple[str, str], dict[str, str]] = {}
    for number, raw in enumerate(reader, 2):
        task_id, skill_id = (str(raw.get(key, "")).strip() for key in ("task_id", "skill_id"))
        if task_id not in allowed_tasks or skill_id not in library_skill_ids:
            raise RelevanceError(f"row {number} references a task or skill outside the frozen universe")
        key = (task_id, skill_id)
        if key in rows:
            raise RelevanceError(f"duplicate task-skill row at line {number}")
        labels = {name: str(raw.get(name, "")).strip().upper() for name in ("rater_a", "rater_b", "adjudicated")}
        if any(label not in LABELS for label in labels.values()):
            raise RelevanceError(f"row {number} contains a missing or invalid blinded label")
        rows[key] = labels
    per_task: dict[str, dict[str, set[str]]] = defaultdict(lambda: {"relevant": set(), "irrelevant": set()})
    for (task_id, skill_id), labels in rows.items():
        final = labels["adjudicated"]
        per_task[task_id]["relevant" if final == "RELEVANT" else "irrelevant"].add(skill_id)
    if set(per_task) != allowed_tasks:
        missing = sorted(allowed_tasks - set(per_task))
        raise RelevanceError(f"relevance file does not cover every task: {missing[:5]}")
    result: dict[str, dict[str, list[str]]] = {}
    for task_id in sorted(allowed_tasks):
        task_pairs = {skill_id for row_task, skill_id in rows if row_task == task_id}
        if task_pairs != library_skill_ids:
            raise RelevanceError(f"{task_id} must have blinded ratings for all 18 Accum-18 skills")
        labels = per_task[task_id]
        if not labels["relevant"] or not labels["irrelevant"]:
            raise RelevanceError(f"{task_id} lacks an adjudicated relevant or irrelevant Accum-18 candidate")
        result[task_id] = {key: sorted(values) for key, values in labels.items()}
    return result


def expected_pair_count(task_count: int) -> int:
    return task_count * 18


def count_coverage(rows: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    pairs = {(str(row.get("task_id")), str(row.get("skill_id"))) for row in rows}
    return {"unique_task_skill_pairs": len(pairs)}
