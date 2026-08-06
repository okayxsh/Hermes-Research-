from __future__ import annotations
import random
from collections import defaultdict
from collections import Counter
from rq1.tasks.models import DiscoveryResult, ManifestState, SelectionPolicy, TaskManifest, TaskRecord
from rq1.tasks.validation import manifest_hash
from rq1.utils.time import utc_now

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
