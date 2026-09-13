"""Balanced acquisition extension core (Decision 011): parent, starting pool, queue, gates.

The completed parent acquisition is read-only evidence.  Nothing here writes to
its results, checkpoint, or skill pool; the extension is a separate run whose
pool starts from the parent's exact final pool and appends new skills only.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from rq1.acquisition.extension_protocol import (
    EXTENSION_ENVIRONMENT_FREEZE,
    EXTENSION_FROZEN_DIR,
    EXTENSION_MANIFEST_TYPE,
    EXTENSION_POLICY_VERSION,
    EXTENSION_PROTOCOL_FREEZE,
    EXTENSION_STATE_DIR,
    EXTENSION_TASK_COUNT,
    EXTENSION_TASKS_PER_FAMILY,
    FROZEN_MODEL_DIGEST,
    PARENT,
    ParentReference,
    extension_protocol_definition,
    extension_protocol_sha256,
    extension_selection_policy,
)
from rq1.acquisition.gates import AcquisitionGate, _approved, load_task_manifest, queue_identity_sha256
from rq1.acquisition.launch import attempt_lineage, run_configuration
from rq1.acquisition.protocol import (
    ACQUISITION_ACTION_BUDGET,
    ACQUISITION_MODEL,
    TASK_SELECTION_BALANCING,
    TASK_SELECTION_SEED,
    TASK_SELECTION_VERSION,
    protocol_sha256,
)
from rq1.acquisition.skill_pool import SNAPSHOT_NAME, PoolSkill, SkillPoolError, pool_hash, rebuild_pool, verify_snapshot
from rq1.experiment.persistence import ExperimentStateError, ExperimentStore
from rq1.freeze.validation import git_state, read_freeze
from rq1.hermes.episode_driver import INFERENCE_SEED, MODEL_QUANTIZATION, provider_settings
from rq1.skills.library import TASK_FAMILIES
from rq1.tasks.models import DiscoveryResult, ManifestState, SelectionPolicy, TaskManifest, TaskRecord
from rq1.tasks.reporting import write_immutable
from rq1.tasks.selection import select_tasks
from rq1.tasks.validation import manifest_hash, validate_manifest
from rq1.utils.hashing import sha256_file
from rq1.utils.time import utc_now

PARENT_CHECK_MESSAGES = {
    "parent_results_readable": "parent results are unreadable",
    "parent_exact_completed_units": "parent run does not have exactly the recorded completed and successful units",
    "parent_retry_lineage_authorized": "parent run has an unauthorized retry or a duplicate completed unit",
    "parent_primary_checkpoint_completed": "parent primary checkpoint is not finalized as completed",
    "parent_skill_pool_consistent": "parent skill pool is inconsistent with its results or snapshot",
    "parent_pool_size_exact": "parent final skill pool size differs from the recorded size",
    "parent_pool_hash_exact": "parent final skill pool hash differs from the recorded hash",
    "parent_frozen_queue_exact": "parent frozen task manifest differs from the recorded parent queue",
    "parent_results_follow_frozen_queue": "parent results do not follow the parent frozen queue order",
    "parent_closeout_manifest_valid": "parent closeout manifest is missing, differs from its recorded SHA-256, or does not certify the parent",
    "parent_files_unchanged_since_closeout": "parent authoritative files changed since closeout",
}
# Gate reasons that only mean the human-approved extension freezes do not exist yet.
EXTENSION_APPROVAL_PENDING_REASONS = frozenset({
    "exactly one frozen acquisition extension task manifest is required (found 0)",
    "invalid acquisition-extension-environment freeze: FileNotFoundError",
    "invalid acquisition-extension-protocol freeze: FileNotFoundError",
})
CLOSEOUT_AUTHORITATIVE_FILES = ("results.jsonl", "checkpoint.json", "skill_pool.json", "run_manifest.json")


@dataclass(frozen=True)
class ParentState:
    checks: dict[str, bool]
    records: tuple[dict[str, Any], ...] = ()
    pool: tuple[PoolSkill, ...] = ()
    manifest: TaskManifest | None = None
    closeout: dict[str, Any] | None = None

    @property
    def problems(self) -> list[str]:
        return [PARENT_CHECK_MESSAGES[name] for name, passed in self.checks.items() if not passed]

    @property
    def queue_task_ids(self) -> set[str]:
        return {task.task_id for task in self.manifest.tasks} if self.manifest is not None else set()


def parent_directory(root: Path, parent: ParentReference = PARENT) -> Path:
    return root / parent.results_directory


def load_parent(root: Path, parent: ParentReference = PARENT) -> ParentState:
    """Verify the completed parent run read-only; it is never repaired or rewritten."""
    directory = parent_directory(root, parent)
    store = ExperimentStore(root, parent.run_id, base=directory.parent)
    checks = dict.fromkeys(PARENT_CHECK_MESSAGES, False)
    try:
        rows = [row for row in store.read_results(repair_tail=False) if row.get("phase") == "acquisition"]
        latest = store.terminal_results(phase="acquisition", repair_tail=False)
        checks["parent_results_readable"] = bool(rows)
    except (OSError, ExperimentStateError):
        rows, latest = [], {}
    records = tuple(sorted(latest.values(), key=lambda row: int(row.get("task_index", 0))))
    checks["parent_exact_completed_units"] = (
        len(records) == parent.completed_units
        and all(row.get("status") == "completed" for row in records)
        and sum(row.get("success") is True for row in records) == parent.successful_units
    )
    lineage = attempt_lineage(rows)
    checks["parent_retry_lineage_authorized"] = bool(rows) and lineage["authorized"] and lineage["duplicate_completed_units"] == 0
    checkpoint, source = store.load_checkpoint()
    checkpoint = checkpoint or {}
    checks["parent_primary_checkpoint_completed"] = (
        source == "primary"
        and checkpoint.get("status") == "completed"
        and checkpoint.get("completed_run_count") == parent.completed_units
        and not checkpoint.get("blocking_error")
    )
    pool: tuple[PoolSkill, ...] = ()
    try:
        pool = rebuild_pool(records)
        verify_snapshot(store.directory, pool)
        snapshot = json.loads((store.directory / SNAPSHOT_NAME).read_text(encoding="utf-8"))
        checks["parent_skill_pool_consistent"] = snapshot.get("pool_hash") == pool_hash(pool) and snapshot.get("pool_size") == len(pool)
    except (OSError, ValueError, SkillPoolError):
        pool = ()
    checks["parent_pool_size_exact"] = checks["parent_skill_pool_consistent"] and len(pool) == parent.pool_size
    checks["parent_pool_hash_exact"] = checks["parent_skill_pool_consistent"] and pool_hash(pool) == parent.pool_hash
    manifest: TaskManifest | None
    try:
        manifest = load_task_manifest(root / parent.task_manifest)
    except (OSError, TypeError, ValueError):
        manifest = None
    if manifest is not None:
        ordered = [task.task_id for task in sorted(manifest.tasks, key=lambda item: item.order_index)]
        checks["parent_frozen_queue_exact"] = (
            manifest.manifest_type == "acquisition"
            and manifest.status == ManifestState.FROZEN.value
            and manifest.manifest_sha256 == parent.task_manifest_sha256
            and queue_identity_sha256(manifest) == parent.queue_sha256
            and not validate_manifest(manifest, require_frozen=True)
        )
        checks["parent_results_follow_frozen_queue"] = (
            [row.get("task_id") for row in records] == ordered
            and [row.get("task_index") for row in records] == list(range(1, len(ordered) + 1))
        )
    closeout: dict[str, Any] | None = None
    closeout_path = root / parent.closeout_manifest
    if closeout_path.is_file() and sha256_file(closeout_path) == parent.closeout_manifest_sha256:
        try:
            value = json.loads(closeout_path.read_text(encoding="utf-8"))
        except ValueError:
            value = None
        closeout = value if isinstance(value, dict) else None
    if closeout is not None:
        counts = closeout.get("counts") or {}
        checks["parent_closeout_manifest_valid"] = (
            closeout.get("closeout_passed") is True
            and closeout.get("run_id") == parent.run_id
            and closeout.get("frozen_git_sha") == parent.repository_commit
            and closeout.get("queue_hash") == parent.queue_sha256
            and closeout.get("final_pool_hash") == parent.pool_hash
            and counts.get("total") == parent.completed_units
            and counts.get("successful") == parent.successful_units
            and counts.get("final_skill_count") == parent.pool_size
        )
        authoritative = closeout.get("authoritative_artifacts") or {}
        checks["parent_files_unchanged_since_closeout"] = all(
            bool((authoritative.get(name) or {}).get("sha256"))
            and (store.directory / name).is_file()
            and sha256_file(store.directory / name) == authoritative[name]["sha256"]
            for name in CLOSEOUT_AUTHORITATIVE_FILES
        )
    return ParentState(checks, records, pool, manifest, closeout)


def starting_pool_path(root: Path, parent: ParentReference = PARENT) -> Path:
    return root / EXTENSION_STATE_DIR / parent.run_id / "starting-pool.json"


def starting_pool_payload(root: Path, pool: Sequence[PoolSkill], parent: ParentReference = PARENT) -> dict[str, Any]:
    directory = parent_directory(root, parent)
    return {
        "schema_version": 1,
        "kind": "rq1-acquisition-extension-starting-pool",
        "label": "IMMUTABLE STARTING POOL OF THE ACQUISITION EXTENSION: a copy of the parent final pool; the parent results remain the authority",
        "source_run_id": parent.run_id,
        "source_results": {"path": f"{parent.results_directory}/results.jsonl", "sha256": sha256_file(directory / "results.jsonl")},
        "source_pool_snapshot": {"path": f"{parent.results_directory}/{SNAPSHOT_NAME}", "sha256": sha256_file(directory / SNAPSHOT_NAME)},
        "source_closeout_manifest": {"path": parent.closeout_manifest, "sha256": parent.closeout_manifest_sha256},
        "pool_size": len(pool),
        "pool_hash": pool_hash(pool),
        "per_family": {family: sum(skill.task_family == family for skill in pool) for family in TASK_FAMILIES},
        "skills": [skill.to_dict() for skill in pool],
    }


def ensure_starting_pool(root: Path, state: ParentState, parent: ParentReference = PARENT) -> Path:
    """Write the immutable starting-pool snapshot once; later calls only verify it."""
    if state.problems:
        raise SkillPoolError("the parent run is not verified: " + "; ".join(state.problems))
    path = starting_pool_path(root, parent)
    if not path.exists():
        write_immutable(path, starting_pool_payload(root, state.pool, parent))
    problems = starting_pool_problems(root, state, parent)
    if problems:
        raise SkillPoolError("; ".join(problems))
    return path


def starting_pool_problems(root: Path, state: ParentState, parent: ParentReference = PARENT) -> list[str]:
    path = starting_pool_path(root, parent)
    try:
        recorded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ["extension starting-pool snapshot is missing or unreadable"]
    if state.problems or recorded != starting_pool_payload(root, state.pool, parent):
        return ["extension starting-pool snapshot differs from the verified parent final pool"]
    return []


def continuation_selection(
    discovery: DiscoveryResult, parent_manifest: TaskManifest,
) -> tuple[tuple[TaskRecord, ...], tuple[dict[str, str], ...], list[str]]:
    """Recompute the frozen selection at the combined count; its tail is the extension."""
    parent_tasks = sorted(parent_manifest.tasks, key=lambda item: item.order_index)
    policy = SelectionPolicy(TASK_SELECTION_VERSION, TASK_SELECTION_SEED, len(parent_tasks) + EXTENSION_TASK_COUNT, TASK_SELECTION_BALANCING)
    selected, exclusions = select_tasks(discovery, policy)
    problems: list[str] = []
    if discovery.split != "train" or discovery.data_root_identity != parent_manifest.data_root_identity:
        problems.append("ALFWorld TRAIN data identity differs from the parent frozen queue")
    if [task.to_dict() for task in selected[: len(parent_tasks)]] != [task.to_dict() for task in parent_tasks]:
        problems.append("recomputed frozen selection does not reproduce the parent queue as its prefix")
    extension = tuple(
        TaskRecord(**{**task.to_dict(), "order_index": index})
        for index, task in enumerate(selected[len(parent_tasks):], 1)
    )
    if len(extension) != EXTENSION_TASK_COUNT:
        problems.append(f"TRAIN metadata cannot supply {EXTENSION_TASK_COUNT} continuation tasks")
    return extension, exclusions, problems


def continuation_problems(discovery: DiscoveryResult, parent_manifest: TaskManifest | None, manifest: TaskManifest) -> list[str]:
    if parent_manifest is None:
        return ["parent frozen queue is unavailable"]
    extension, _exclusions, problems = continuation_selection(discovery, parent_manifest)
    ordered = sorted(manifest.tasks, key=lambda item: item.order_index)
    if [task.to_dict() for task in extension] != [task.to_dict() for task in ordered]:
        problems.append("extension queue is not the continuation of the frozen selection")
    if manifest.data_root_identity != discovery.data_root_identity:
        problems.append("extension queue ALFWorld data identity differs from the live TRAIN data")
    return problems


def extension_lineage(parent: ParentReference, tasks: Sequence[TaskRecord], parent_task_ids: set[str]) -> dict[str, Any]:
    ordered = sorted(tasks, key=lambda item: item.order_index)
    identifiers = [task.task_id for task in ordered]
    return {
        "schema_version": 1,
        "kind": "rq1-acquisition-extension",
        "policy_version": EXTENSION_POLICY_VERSION,
        "parent_run_id": parent.run_id,
        "parent_completed_units": parent.completed_units,
        "parent_repository_commit": parent.repository_commit,
        "parent_queue_sha256": parent.queue_sha256,
        "parent_task_manifest": parent.task_manifest,
        "parent_task_manifest_sha256": parent.task_manifest_sha256,
        "parent_closeout_manifest": parent.closeout_manifest,
        "parent_closeout_manifest_sha256": parent.closeout_manifest_sha256,
        "starting_pool_size": parent.pool_size,
        "starting_pool_hash": parent.pool_hash,
        "logical_index_offset": parent.completed_units,
        "combined_task_count_after_completion": parent.completed_units + len(ordered),
        "units": [
            {
                "extension_queue_index": task.order_index,
                "logical_acquisition_index": parent.completed_units + task.order_index,
                "task_id": task.task_id,
                "family": task.family,
            }
            for task in ordered
        ],
        "proof": {
            "overlap_with_parent_queue": sorted(set(identifiers) & parent_task_ids),
            "internal_duplicate_task_ids": sorted(item for item, count in Counter(identifiers).items() if count > 1),
        },
    }


def propose_extension_manifest(
    discovery: DiscoveryResult,
    parent_manifest: TaskManifest,
    parent: ParentReference = PARENT,
    *,
    alfworld_version: str | None,
    repository_commit: str | None,
) -> TaskManifest:
    extension, exclusions, problems = continuation_selection(discovery, parent_manifest)
    if problems:
        raise ValueError("; ".join(problems))
    parent_tasks = sorted(parent_manifest.tasks, key=lambda item: item.order_index)
    value: dict[str, Any] = {
        "schema_version": 1,
        "manifest_type": EXTENSION_MANIFEST_TYPE,
        "status": ManifestState.PROPOSED.value,
        "split": "train",
        "alfworld_version": alfworld_version,
        "data_root_identity": discovery.data_root_identity,
        "repository_commit": repository_commit,
        "selection_policy": extension_selection_policy(parent),
        "requested_count": EXTENSION_TASK_COUNT,
        "actual_count": len(extension),
        "family_counts": dict(sorted(Counter(task.family for task in extension).items())),
        "tasks": [task.to_dict() for task in extension],
        "exclusions": [
            *discovery.exclusions,
            *({"task_id": task.task_id, "reason": "selected_in_parent_acquisition_queue"} for task in parent_tasks),
            *exclusions,
        ],
        "duplicate_resolution": [],
        "generated_at": utc_now(),
        "approved_at": None,
        "approval_reference": None,
        "manifest_sha256": "",
        "lineage": extension_lineage(parent, extension, {task.task_id for task in parent_tasks}),
    }
    value["manifest_sha256"] = manifest_hash(value)
    return TaskManifest(**{**value, "tasks": extension, "exclusions": tuple(value["exclusions"]), "duplicate_resolution": ()})


def validate_extension_queue_manifest(
    manifest: TaskManifest,
    parent_manifest: TaskManifest | None,
    parent: ParentReference = PARENT,
    *,
    require_frozen: bool,
) -> list[str]:
    errors = list(validate_manifest(manifest, require_frozen=require_frozen))
    if manifest.manifest_type != EXTENSION_MANIFEST_TYPE:
        errors.append("task manifest is not an acquisition extension manifest")
    if manifest.split != "train" or any(task.split != "train" or not task.task_id.startswith("train:") for task in manifest.tasks):
        errors.append("extension queue must contain TRAIN tasks only")
    if manifest.actual_count != EXTENSION_TASK_COUNT or len(manifest.tasks) != EXTENSION_TASK_COUNT:
        errors.append(f"extension queue must contain exactly {EXTENSION_TASK_COUNT} tasks")
    if dict(manifest.family_counts) != {family: EXTENSION_TASKS_PER_FAMILY for family in TASK_FAMILIES}:
        errors.append(f"extension queue must contain exactly {EXTENSION_TASKS_PER_FAMILY} tasks per family")
    if dict(manifest.selection_policy) != extension_selection_policy(parent):
        errors.append("extension queue selection policy differs from the frozen continuation rule")
    identifiers = [task.task_id for task in manifest.tasks]
    if len(set(identifiers)) != len(identifiers):
        errors.append("extension queue contains duplicate task IDs")
    if parent_manifest is None:
        errors.append("parent frozen queue is unavailable")
    else:
        parent_ids = {task.task_id for task in parent_manifest.tasks}
        if set(identifiers) & parent_ids:
            errors.append("extension queue overlaps the parent acquisition queue")
        if manifest.lineage != extension_lineage(parent, manifest.tasks, parent_ids):
            errors.append("extension queue lineage differs from the parent reference")
    return errors


def validate_extension_gates(
    root: Path, *, task_manifest_path: Path | None = None, parent: ParentReference = PARENT,
) -> AcquisitionGate:
    """Launch gate of the scientific extension: verified parent plus approved extension freezes."""
    reasons: list[str] = []
    commit, clean, error = git_state(root)
    if error:
        reasons.append(error)
    elif not clean:
        reasons.append("repository working tree is not clean")
    state = load_parent(root, parent)
    reasons.extend(state.problems)
    reasons.extend(starting_pool_problems(root, state, parent))

    path = task_manifest_path
    if path is None:
        candidates = sorted((root / EXTENSION_FROZEN_DIR).glob(f"{EXTENSION_MANIFEST_TYPE}-*.json"))
        if len(candidates) == 1:
            path = candidates[0]
        else:
            reasons.append(f"exactly one frozen acquisition extension task manifest is required (found {len(candidates)})")
    manifest: TaskManifest | None = None
    queue_sha: str | None = None
    if path is not None:
        try:
            manifest = load_task_manifest(path)
        except (OSError, TypeError, ValueError) as exc:
            reasons.append(f"frozen acquisition extension task manifest is unreadable: {type(exc).__name__}")
        else:
            reasons.extend(validate_extension_queue_manifest(manifest, state.manifest, parent, require_frozen=True))
            queue_sha = queue_identity_sha256(manifest)
            if manifest.repository_commit != commit:
                reasons.append("frozen acquisition extension queue was frozen at a different commit")
            if not manifest.approved_at or not manifest.approval_reference:
                reasons.append("frozen acquisition extension queue lacks approval metadata")

    environment, errors = read_freeze(root / EXTENSION_ENVIRONMENT_FREEZE, "acquisition-extension-environment")
    reasons.extend(errors)
    protocol, errors = read_freeze(root / EXTENSION_PROTOCOL_FREEZE, "acquisition-extension-protocol")
    reasons.extend(errors)
    for freeze in (environment, protocol):
        if freeze is None:
            continue
        if freeze.repository_commit != commit or freeze.inputs.get("repository_commit") != commit:
            reasons.append(f"repository commit changed since {freeze.kind} freeze")
        if not _approved(freeze.approval):
            reasons.append(f"{freeze.kind} freeze lacks human approval")
        if queue_sha is not None and freeze.inputs.get("task_queue_sha256") != queue_sha:
            reasons.append(f"{freeze.kind} freeze references a different extension queue")
        if freeze.inputs.get("inference_seed") != INFERENCE_SEED:
            reasons.append(f"{freeze.kind} freeze inference seed differs from {INFERENCE_SEED}")
        if freeze.inputs.get("parent_run_id") != parent.run_id:
            reasons.append(f"{freeze.kind} freeze references a different parent run")
    if protocol is not None:
        inputs = protocol.inputs
        if (
            inputs.get("protocol") != extension_protocol_definition(parent)
            or inputs.get("protocol_sha256") != extension_protocol_sha256(parent)
            or inputs.get("inherited_protocol_sha256") != protocol_sha256()
        ):
            reasons.append("acquisition extension protocol freeze differs from the repository protocol")
        if inputs.get("acquisition_action_budget") != ACQUISITION_ACTION_BUDGET:
            reasons.append("acquisition extension protocol freeze action budget differs from the frozen budget")
        expected_parent = {
            "parent_pool_size": parent.pool_size,
            "parent_pool_hash": parent.pool_hash,
            "parent_queue_sha256": parent.queue_sha256,
            "parent_closeout_manifest_sha256": parent.closeout_manifest_sha256,
        }
        if any(inputs.get(key) != value for key, value in expected_parent.items()):
            reasons.append("acquisition extension protocol freeze parent identity differs from the parent reference")
        pool_path = starting_pool_path(root, parent)
        if not pool_path.is_file() or inputs.get("starting_pool_sha256") != sha256_file(pool_path):
            reasons.append("acquisition extension protocol freeze starting pool differs from the starting-pool snapshot")
    if environment is not None:
        inputs = environment.inputs
        if (
            inputs.get("model_tag") != ACQUISITION_MODEL
            or inputs.get("model_quantization") != MODEL_QUANTIZATION
            or inputs.get("model_digest") != FROZEN_MODEL_DIGEST
        ):
            reasons.append("acquisition extension environment freeze model identity differs from the frozen model")
        if inputs.get("provider_settings") != provider_settings():
            reasons.append("acquisition extension environment freeze provider settings differ from the frozen settings")
        if manifest is not None and inputs.get("alfworld_data_identity") != manifest.data_root_identity:
            reasons.append("environment freeze ALFWorld data identity differs from the frozen extension queue")
    if environment is not None and protocol is not None and environment.inputs.get("prompt_hashes") != protocol.inputs.get("prompt_hashes"):
        reasons.append("environment/protocol freeze prompt hashes differ")
    return AcquisitionGate(not reasons, tuple(reasons), manifest, path, environment, protocol)


def extension_run_configuration(
    root: Path,
    *,
    queue_sha256: str,
    scientific: bool,
    parent: ParentReference = PARENT,
    freezes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """The initial acquisition's configuration (identical settings) bound to the parent lineage."""
    value = run_configuration(root, queue_sha256=queue_sha256, scientific=scientific, freezes=freezes)
    value["extension"] = {
        "policy_version": EXTENSION_POLICY_VERSION,
        "protocol_sha256": extension_protocol_sha256(parent),
        "parent_run_id": parent.run_id,
        "starting_pool_size": parent.pool_size,
        "starting_pool_hash": parent.pool_hash,
        "logical_index_offset": parent.completed_units,
    }
    return value
