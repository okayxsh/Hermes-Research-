from __future__ import annotations
import random
from collections import defaultdict
from collections import Counter
from typing import Mapping, Sequence
from rq1.tasks.models import DiscoveryResult, ManifestState, SelectionPolicy, TaskManifest, TaskRecord
from rq1.tasks.validation import manifest_hash
from rq1.utils.time import utc_now

# Frozen RQ1 task/seed/count protocol. These values must not change after
# outcomes are observed.
FROZEN_TASK_FAMILIES = 6
FROZEN_EVALUATION_TASKS = 30
FROZEN_TASKS_PER_FAMILY = 5
FROZEN_SEEDS = (11, 29, 47)
FROZEN_REPETITIONS = len(FROZEN_SEEDS)

ACQUISITION_INITIAL_TASKS = 180
ACQUISITION_TASKS_PER_FAMILY = 30
ACQUISITION_HARD_CAP = 240
ACQUISITION_HARD_CAP_PER_FAMILY = 40


def select_tasks(discovery: DiscoveryResult, policy: SelectionPolicy) -> tuple[tuple[TaskRecord, ...], tuple[dict[str, str], ...]]:
    if policy.requested_count is None or policy.requested_count < 1: raise ValueError("requested_count must be approved and positive")
    groups: dict[str, list[TaskRecord]] = defaultdict(list)
    for record in discovery.records: groups[record.family].append(record)
    rng = random.Random(policy.seed)
    for values in groups.values(): rng.shuffle(values)
    selected: list[TaskRecord] = []; families = sorted(groups)
    while len(selected) < policy.requested_count and any(groups.values()):
        for family in families:
            if groups[family] and len(selected) < policy.requested_count: selected.append(groups[family].pop(0))
    exclusions = tuple({"task_id": value.task_id, "reason": "not_selected"} for values in groups.values() for value in values)
    return tuple(TaskRecord(**{**value.to_dict(), "order_index": index}) for index, value in enumerate(selected, 1)), exclusions


def select_longest_per_family(
    records: Sequence[TaskRecord],
    lengths: Mapping[str, int],
    *,
    tasks_per_family: int = FROZEN_TASKS_PER_FAMILY,
) -> tuple[tuple[TaskRecord, ...], tuple[dict[str, str], ...]]:
    """Select the longest expert trajectories per family, tie-breaking by task ID.

    ``lengths`` maps task_id to expert trajectory length. Selection is
    deterministic: within a family, sort by descending length then ascending
    task ID, and take the first ``tasks_per_family``. If a family has fewer
    eligible tasks than requested, the quota is not satisfied and the caller
    must fail closed (a smaller-than-requested selection is returned, never a
    silent protocol change).
    """
    groups: dict[str, list[TaskRecord]] = defaultdict(list)
    for record in records:
        groups[record.family].append(record)
    selected: list[TaskRecord] = []
    exclusions: list[dict[str, str]] = []
    for family in sorted(groups):
        ordered = sorted(
            groups[family],
            key=lambda item: (-int(lengths.get(item.task_id, 0)), item.task_id),
        )
        chosen = ordered[:tasks_per_family]
        selected.extend(chosen)
        for item in ordered[tasks_per_family:]:
            exclusions.append({"task_id": item.task_id, "reason": "not_longest_in_family"})
    selected.sort(key=lambda item: item.task_id)
    return (
        tuple(TaskRecord(**{**value.to_dict(), "order_index": index}) for index, value in enumerate(selected, 1)),
        tuple(exclusions),
    )


def next_longest_replacement(
    records: Sequence[TaskRecord],
    lengths: Mapping[str, int],
    *,
    family: str,
    excluded_task_id: str,
) -> TaskRecord | None:
    """Deterministic replacement when a candidate task cannot support perturbation.

    Returns the next-longest eligible task in the same family (tie-broken by
    task ID), excluding the failing task and any already-frozen task IDs the
    caller passes in ``lengths``/``records``. Returns ``None`` when no eligible
    replacement exists. Replacement is never based on agent performance.
    """
    candidates = [
        record
        for record in records
        if record.family == family and record.task_id != excluded_task_id
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-int(lengths.get(item.task_id, 0)), item.task_id))
    return candidates[0]


def propose_manifest(kind: str, discovery: DiscoveryResult, policy: SelectionPolicy, *, alfworld_version: str | None, repository_commit: str | None) -> TaskManifest:
    selected, exclusions = select_tasks(discovery, policy)
    value = {
        "schema_version": 1, "manifest_type": kind, "status": ManifestState.PROPOSED.value, "split": discovery.split,
        "alfworld_version": alfworld_version, "data_root_identity": discovery.data_root_identity, "repository_commit": repository_commit,
        "selection_policy": policy.to_dict(), "requested_count": policy.requested_count, "actual_count": len(selected),
        "family_counts": dict(sorted(Counter(item.family for item in selected).items())), "tasks": [item.to_dict() for item in selected],
        "exclusions": [*discovery.exclusions, *exclusions], "duplicate_resolution": [], "generated_at": utc_now(),
        "approved_at": None, "approval_reference": None, "manifest_sha256": "",
    }
    value["manifest_sha256"] = manifest_hash(value)
    return TaskManifest(**{**value, "tasks": selected, "exclusions": tuple(value["exclusions"]), "duplicate_resolution": ()})
