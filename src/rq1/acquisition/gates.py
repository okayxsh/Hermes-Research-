"""Acquisition launch gate: approved task, environment, and protocol freezes.

The final-evaluation freeze requires library hashes that cannot exist before
acquisition, so acquisition is gated by its own approved freezes, built with the
same ``FreezeManifest`` format and human-approval rules.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rq1.acquisition.protocol import (
    ACQUISITION_ACTION_BUDGET,
    ACQUISITION_MODEL,
    TASK_SELECTION_BALANCING,
    TASK_SELECTION_SEED,
    TASK_SELECTION_VERSION,
    protocol_definition,
    protocol_sha256,
)
from rq1.experiment.models import canonical_hash
from rq1.freeze.models import FreezeManifest
from rq1.freeze.validation import git_state, read_freeze
from rq1.hermes.episode_driver import INFERENCE_SEED, MODEL_QUANTIZATION
from rq1.skills.library import TASK_FAMILIES
from rq1.tasks.models import TaskManifest, TaskRecord
from rq1.tasks.selection import ACQUISITION_INITIAL_TASKS, ACQUISITION_TASKS_PER_FAMILY
from rq1.tasks.validation import validate_manifest

FROZEN_TASK_DIR = Path("artifacts") / "task_manifests" / "frozen"
ENVIRONMENT_FREEZE = Path("artifacts") / "freezes" / "acquisition-environment-freeze.json"
PROTOCOL_FREEZE = Path("artifacts") / "freezes" / "acquisition-protocol-freeze.json"


def load_task_manifest(path: Path) -> TaskManifest:
    value = json.loads(path.read_text(encoding="utf-8"))
    value["tasks"] = tuple(TaskRecord(**item) for item in value.get("tasks", []))
    value["exclusions"] = tuple(value.get("exclusions", []))
    value["duplicate_resolution"] = tuple(value.get("duplicate_resolution", []))
    return TaskManifest(**value)


def queue_identity_sha256(manifest: TaskManifest) -> str:
    """Queue hash that is invariant under freezing (status, approval, timestamps)."""
    return canonical_hash(
        {
            "manifest_type": manifest.manifest_type,
            "split": manifest.split,
            "data_root_identity": manifest.data_root_identity,
            "selection_policy": dict(manifest.selection_policy),
            "tasks": [task.to_dict() for task in sorted(manifest.tasks, key=lambda item: item.order_index)],
        }
    )


def validate_queue_manifest(manifest: TaskManifest, *, require_frozen: bool) -> list[str]:
    errors = list(validate_manifest(manifest, require_frozen=require_frozen))
    if manifest.manifest_type != "acquisition":
        errors.append("task manifest is not an acquisition manifest")
    if manifest.split != "train" or any(task.split != "train" or not task.task_id.startswith("train:") for task in manifest.tasks):
        errors.append("acquisition queue must contain TRAIN tasks only")
    if manifest.actual_count != ACQUISITION_INITIAL_TASKS:
        errors.append(f"acquisition queue must contain exactly {ACQUISITION_INITIAL_TASKS} tasks")
    expected = {family: ACQUISITION_TASKS_PER_FAMILY for family in TASK_FAMILIES}
    if dict(manifest.family_counts) != expected:
        errors.append(f"acquisition queue must contain exactly {ACQUISITION_TASKS_PER_FAMILY} tasks per family")
    policy = {
        "version": TASK_SELECTION_VERSION,
        "seed": TASK_SELECTION_SEED,
        "requested_count": ACQUISITION_INITIAL_TASKS,
        "balancing": TASK_SELECTION_BALANCING,
    }
    if dict(manifest.selection_policy) != policy:
        errors.append("acquisition queue selection policy differs from the frozen protocol")
    return errors


@dataclass(frozen=True)
class AcquisitionGate:
    valid: bool
    reasons: tuple[str, ...]
    task_manifest: TaskManifest | None = None
    task_manifest_path: Path | None = None
    environment: FreezeManifest | None = None
    protocol: FreezeManifest | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "reasons": list(self.reasons),
            "task_manifest_path": str(self.task_manifest_path) if self.task_manifest_path else None,
            "task_manifest_sha256": self.task_manifest.manifest_sha256 if self.task_manifest else None,
            "task_queue_sha256": queue_identity_sha256(self.task_manifest) if self.task_manifest else None,
            "family_counts": dict(self.task_manifest.family_counts) if self.task_manifest else None,
            "environment_freeze_fingerprint": self.environment.input_fingerprint if self.environment else None,
            "protocol_freeze_fingerprint": self.protocol.input_fingerprint if self.protocol else None,
        }


def _approved(meta: Any) -> bool:
    return isinstance(meta, dict) and bool(meta.get("approved_by")) and bool(meta.get("approved_at")) and meta.get("status", "APPROVED") == "APPROVED"


def validate_acquisition_gates(root: Path, *, task_manifest_path: Path | None = None) -> AcquisitionGate:
    reasons: list[str] = []
    commit, clean, error = git_state(root)
    if error:
        reasons.append(error)
    elif not clean:
        reasons.append("repository working tree is not clean")

    path = task_manifest_path
    if path is None:
        candidates = sorted((root / FROZEN_TASK_DIR).glob("acquisition-*.json"))
        if len(candidates) == 1:
            path = candidates[0]
        else:
            reasons.append(f"exactly one frozen acquisition task manifest is required (found {len(candidates)})")
    manifest: TaskManifest | None = None
    queue_sha: str | None = None
    if path is not None:
        try:
            manifest = load_task_manifest(path)
        except (OSError, TypeError, ValueError) as exc:
            reasons.append(f"frozen acquisition task manifest is unreadable: {type(exc).__name__}")
        else:
            reasons.extend(validate_queue_manifest(manifest, require_frozen=True))
            queue_sha = queue_identity_sha256(manifest)
            if manifest.repository_commit != commit:
                reasons.append("frozen acquisition queue was frozen at a different commit")
            if not manifest.approved_at or not manifest.approval_reference:
                reasons.append("frozen acquisition queue lacks approval metadata")

    environment, errors = read_freeze(root / ENVIRONMENT_FREEZE, "acquisition-environment")
    reasons.extend(errors)
    protocol, errors = read_freeze(root / PROTOCOL_FREEZE, "acquisition-protocol")
    reasons.extend(errors)
    for freeze in (environment, protocol):
        if freeze is None:
            continue
        if freeze.repository_commit != commit or freeze.inputs.get("repository_commit") != commit:
            reasons.append(f"repository commit changed since {freeze.kind} freeze")
        if not _approved(freeze.approval):
            reasons.append(f"{freeze.kind} freeze lacks human approval")
        if queue_sha is not None and freeze.inputs.get("task_queue_sha256") != queue_sha:
            reasons.append(f"{freeze.kind} freeze references a different acquisition queue")
        if freeze.inputs.get("inference_seed") != INFERENCE_SEED:
            reasons.append(f"{freeze.kind} freeze inference seed differs from {INFERENCE_SEED}")
    if protocol is not None:
        if protocol.inputs.get("protocol") != protocol_definition() or protocol.inputs.get("protocol_sha256") != protocol_sha256():
            reasons.append("acquisition protocol freeze differs from the repository protocol")
        if protocol.inputs.get("acquisition_action_budget") != ACQUISITION_ACTION_BUDGET:
            reasons.append("acquisition protocol freeze action budget differs from the frozen budget")
    if environment is not None and environment.inputs.get("model_tag") != ACQUISITION_MODEL:
        reasons.append("acquisition environment freeze model differs from the frozen model")
    if environment is not None and environment.inputs.get("model_quantization") != MODEL_QUANTIZATION:
        reasons.append("acquisition environment freeze model quantization differs from the frozen quantization")
    if environment is not None and protocol is not None and environment.inputs.get("prompt_hashes") != protocol.inputs.get("prompt_hashes"):
        reasons.append("environment/protocol freeze prompt hashes differ")
    if environment is not None and manifest is not None and environment.inputs.get("alfworld_data_identity") != manifest.data_root_identity:
        reasons.append("environment freeze ALFWorld data identity differs from the frozen queue")
    return AcquisitionGate(not reasons, tuple(reasons), manifest, path, environment, protocol)
