"""The frozen 360-unit evaluation matrix, its balanced interleaved schedule, shards, and merge.

A unit is (task, seed, condition).  Its identity excludes the library hash, so the
matrix and shard manifests can be frozen before the human core review; the library
hashes are bound into every shard's run configuration instead, and resume fails
closed on any drift.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence

from rq1.evaluation.amended_protocol import (
    CONDITIONS,
    EVALUATION_POLICY_VERSION,
    EVALUATION_RUN_PREFIX,
    EVALUATION_SEEDS,
    EVALUATION_TASK_COUNT,
    EVALUATION_UNIT_COUNT,
    SHARD_COUNT,
    TASKS_PER_FAMILY,
)
from rq1.experiment.models import ExperimentUnit, canonical_hash
from rq1.skills.library import TASK_FAMILIES


class MatrixError(ValueError):
    pass


@dataclass(frozen=True)
class EvaluationTask:
    task_id: str
    task_family: str
    order_index: int
    checkpoint_id: str
    checkpoint_digest: str
    post_detour_digest: str
    detour_action: str
    expected_next_action: str
    prefix_actions: tuple[str, ...]
    reference_actions: tuple[str, ...]
    recovery_action_budget: int

    def frozen_perturbation(self) -> dict[str, str]:
        return {"checkpoint_digest": self.checkpoint_digest, "detour_action": self.detour_action, "post_detour_digest": self.post_detour_digest}

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["prefix_actions"] = list(self.prefix_actions)
        value["reference_actions"] = list(self.reference_actions)
        return value


def evaluation_tasks(manifest_tasks: Sequence[Any], controlled_failures: Mapping[str, Any]) -> tuple[EvaluationTask, ...]:
    """Join the frozen task manifest with its oracle-validated controlled failures."""
    definitions = {item["task_id"]: item for item in controlled_failures["tasks"]}
    tasks = []
    for record in sorted(manifest_tasks, key=lambda item: item.order_index):
        definition = definitions.get(record.task_id)
        if definition is None or definition.get("task_family") != record.family or not (definition.get("oracle") or {}).get("validated"):
            raise MatrixError(f"task lacks an oracle-validated controlled failure: {record.task_id}")
        tasks.append(EvaluationTask(
            task_id=record.task_id, task_family=record.family, order_index=record.order_index,
            checkpoint_id=definition["checkpoint_id"], checkpoint_digest=definition["checkpoint_digest"],
            post_detour_digest=definition["post_detour_digest"], detour_action=definition["detour_action"],
            expected_next_action=definition["expected_next_action"], prefix_actions=tuple(definition["prefix_actions"]),
            reference_actions=tuple(definition["reference_actions"]), recovery_action_budget=int(definition["recovery_action_budget"]),
        ))
    if len(definitions) != len(tasks):
        raise MatrixError("controlled failures and the task manifest describe different tasks")
    return tuple(tasks)


def unit_identity(task: EvaluationTask, seed: int, condition: str) -> dict[str, Any]:
    return {
        "phase": "evaluation",
        "policy_version": EVALUATION_POLICY_VERSION,
        "task_id": task.task_id,
        "seed": seed,
        "condition": condition,
        "checkpoint_digest": task.checkpoint_digest,
        "perturbation_digest": task.post_detour_digest,
    }


def build_matrix(tasks: Sequence[EvaluationTask]) -> list[dict[str, Any]]:
    counts = Counter(task.task_family for task in tasks)
    if len(tasks) != EVALUATION_TASK_COUNT or len({task.task_id for task in tasks}) != len(tasks) or any(counts[family] != TASKS_PER_FAMILY for family in TASK_FAMILIES):
        raise MatrixError(f"the matrix needs {EVALUATION_TASK_COUNT} distinct tasks, {TASKS_PER_FAMILY} per family")
    ordered = sorted(tasks, key=lambda task: (TASK_FAMILIES.index(task.task_family), task.order_index))
    units: list[dict[str, Any]] = []
    cells = [(task, seed) for task in ordered for seed in EVALUATION_SEEDS]
    for cell_index, (task, seed) in enumerate(cells):
        rotation = (cell_index // SHARD_COUNT) % len(CONDITIONS)
        order = CONDITIONS[rotation:] + CONDITIONS[:rotation]
        for position, condition in enumerate(order):
            identity = unit_identity(task, seed, condition)
            units.append({
                "global_order": len(units) + 1,
                "unit_key": canonical_hash(identity),
                "identity": identity,
                "cell_index": cell_index,
                "position_in_cell": position,
                "shard": cell_index % SHARD_COUNT + 1,
                "task_id": task.task_id,
                "task_family": task.task_family,
                "seed": seed,
                "condition": condition,
                "checkpoint_id": task.checkpoint_id,
                "recovery_action_budget": task.recovery_action_budget,
            })
    problems = coverage_problems(units)
    if problems:
        raise MatrixError("; ".join(problems))
    return units


def coverage_problems(units: Sequence[Mapping[str, Any]]) -> list[str]:
    problems = []
    if len(units) != EVALUATION_UNIT_COUNT:
        problems.append(f"matrix has {len(units)} units, not {EVALUATION_UNIT_COUNT}")
    keys = [unit["unit_key"] for unit in units]
    if len(set(keys)) != len(keys):
        problems.append("duplicate evaluation unit")
    triples = Counter((unit["task_id"], unit["seed"], unit["condition"]) for unit in units)
    if any(count != 1 for count in triples.values()):
        problems.append("a task-seed-condition triple appears more than once")
    tasks = {unit["task_id"] for unit in units}
    if len(triples) != len(tasks) * len(EVALUATION_SEEDS) * len(CONDITIONS):
        problems.append("the matrix is not the full task x seed x condition product")
    for shard in range(1, SHARD_COUNT + 1):
        members = [unit for unit in units if unit["shard"] == shard]
        per_condition = Counter(unit["condition"] for unit in members)
        if len(members) != EVALUATION_UNIT_COUNT // SHARD_COUNT or len(set(per_condition.values())) != 1 or set(per_condition) != set(CONDITIONS):
            problems.append(f"shard {shard} is not balanced across conditions")
    cells: dict[int, set[int]] = {}
    for unit in units:
        cells.setdefault(unit["cell_index"], set()).add(unit["shard"])
    if any(len(shards) != 1 for shards in cells.values()):
        problems.append("the conditions of a task-seed cell are split across shards")
    return problems


def matrix_sha256(units: Sequence[Mapping[str, Any]]) -> str:
    return canonical_hash([{key: unit[key] for key in ("global_order", "unit_key", "shard", "task_id", "seed", "condition")} for unit in units])


def shard_run_id(shard: int) -> str:
    if not 1 <= shard <= SHARD_COUNT:
        raise MatrixError(f"shard must be 1-{SHARD_COUNT}")
    return f"{EVALUATION_RUN_PREFIX}-shard-{shard}-of-{SHARD_COUNT}"


def shard_units(units: Sequence[Mapping[str, Any]], shard: int) -> list[Mapping[str, Any]]:
    return [unit for unit in units if unit["shard"] == shard]


def shard_manifest(units: Sequence[Mapping[str, Any]], shard: int) -> dict[str, Any]:
    members = shard_units(units, shard)
    return {
        "schema_version": 1,
        "kind": "rq1-evaluation-shard",
        "shard": shard,
        "shard_count": SHARD_COUNT,
        "run_id": shard_run_id(shard),
        "matrix_sha256": matrix_sha256(units),
        "unit_count": len(members),
        "condition_counts": dict(sorted(Counter(unit["condition"] for unit in members).items())),
        "family_counts": dict(sorted(Counter(unit["task_family"] for unit in members).items())),
        "seed_counts": {str(seed): count for seed, count in sorted(Counter(unit["seed"] for unit in members).items())},
        "units": [dict(unit) for unit in members],
        "shard_sha256": canonical_hash([unit["unit_key"] for unit in members]),
    }


def experiment_units(units: Sequence[Mapping[str, Any]], shard: int, library_sizes: Mapping[str, int], library_hashes: Mapping[str, str]) -> list[ExperimentUnit]:
    return [
        ExperimentUnit(
            phase="evaluation",
            task_id=unit["task_id"],
            task_index=index,
            condition=unit["condition"],
            seed=unit["seed"],
            identity=dict(unit["identity"]),
            payload={key: unit[key] for key in ("global_order", "unit_key", "cell_index", "position_in_cell", "shard", "task_family", "checkpoint_id", "recovery_action_budget")},
            library_name=unit["condition"],
            library_size=library_sizes[unit["condition"]],
            library_hash=library_hashes[unit["condition"]],
        )
        for index, unit in enumerate(shard_units(units, shard), 1)
    ]


def merge_shards(units: Sequence[Mapping[str, Any]], shard_records: Mapping[int, Iterable[Mapping[str, Any]]]) -> tuple[list[dict[str, Any]], list[str]]:
    """Deterministically merge terminal shard records into matrix order; report any gap or overlap."""
    by_key = {unit["unit_key"]: unit for unit in units}
    merged: dict[str, dict[str, Any]] = {}
    problems: list[str] = []
    for shard, records in sorted(shard_records.items()):
        for record in records:
            key = record.get("run_key")
            unit = by_key.get(key)
            if unit is None:
                problems.append(f"shard {shard} holds a unit outside the frozen matrix: {key}")
                continue
            if unit["shard"] != shard:
                problems.append(f"unit {key} ran in shard {shard}, not its frozen shard {unit['shard']}")
            if key in merged:
                problems.append(f"unit {key} appears in more than one terminal record")
                continue
            merged[key] = {**dict(record), "matrix_unit": dict(unit)}
    missing = [key for key in by_key if key not in merged]
    if missing:
        problems.append(f"{len(missing)} matrix units have no terminal record")
    return [merged[unit["unit_key"]] for unit in units if unit["unit_key"] in merged], problems
