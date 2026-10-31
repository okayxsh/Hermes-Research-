"""Frozen six-arm manifests for the prospective selective-memory follow-up."""
from __future__ import annotations

import os
import random
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import Any, Mapping, Sequence

from rq1.experiment.models import canonical_hash
from rq1.evaluation.amended_protocol import SBERT_SNAPSHOT_SHA256
from rq1.skills.library import TASK_FAMILIES
from rq1.university_gpu.models import atomic_write_json, content_hash

SCHEMA = "rq1-university-gpu-manifest-v1"
PROTOCOL_VERSION = "rq1-selective-memory-six-arm-v1"
CONDITIONS = ("D0", "D18", "Rel", "Irr", "Fill", "Gate")
SEEDS = (11, 29, 47)
PILOT_SEED = 11
SHARD_COUNT = 6
MODEL_TAG = "gemma4:12b"
MODEL_DIGEST = "4eb23ef187e2c5462566d6a1d3bbbc2f1346d0b4327cbb66d58fffbcc9b2b05c"
MODEL_QUANTIZATION = "Q4_K_M"
SBERT_MODEL = "sentence-transformers/all-mpnet-base-v2"
SBERT_REVISION = "e8c3b32edf5434bc2275fc9bab85f82640a19130"
ARMS = {"pilot": 36, "confirmatory": 540}


class ManifestError(ValueError):
    pass


_SAFE_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


def validate_run_id(run_id: object) -> str:
    """Preserve safe historical labels while excluding path syntax and dot segments."""
    if not isinstance(run_id, str) or not _SAFE_RUN_ID.fullmatch(run_id):
        raise ManifestError("run_id must be a 1-128 character path-safe identifier starting with a letter or digit")
    if run_id.endswith((".", " ")):
        raise ManifestError("run_id cannot end in a Windows-normalized dot or space")
    windows_device = run_id.split(".", 1)[0].upper()
    if (windows_device in {"CON", "PRN", "AUX", "NUL"}
            or re.fullmatch(r"(?:COM|LPT)[1-9]", windows_device)
            or windows_device in {"COM¹", "COM²", "COM³", "LPT¹", "LPT²", "LPT³"}):
        raise ManifestError("run_id cannot use a reserved Windows device name")
    return run_id


def resolve_run_root(output_root: Path, run_id: object) -> Path:
    """Resolve a run directory as one contained child of its configured output root."""
    safe_id = validate_run_id(run_id)
    expanded_root = output_root.expanduser()
    windows = PureWindowsPath(str(expanded_root))
    if ".." in expanded_root.parts or ".." in windows.parts or (os.name != "nt" and windows.drive):
        raise ManifestError("configured output root cannot contain lexical '..' components")
    absolute_root = expanded_root.absolute()
    if any(path.is_symlink() for path in (absolute_root, *absolute_root.parents)):
        raise ManifestError("configured output root cannot be a symlink")
    resolved_root = expanded_root.resolve()
    candidate = resolved_root / safe_id
    if candidate.is_symlink():
        raise ManifestError("run directory cannot be a symlink")
    resolved_candidate = candidate.resolve(strict=False)
    if resolved_candidate.parent != resolved_root:
        raise ManifestError("run directory escapes the configured output root")
    return candidate


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value.lower()):
        raise ManifestError(f"{label} must be a SHA-256 hex digest")
    return value.lower()


def _validate_tasks(stage: str, tasks: Sequence[Mapping[str, Any]]) -> None:
    expected_count = 6 if stage == "pilot" else 30
    expected_split = "valid_seen" if stage == "pilot" else "valid_unseen"
    if len(tasks) != expected_count:
        raise ManifestError(f"{stage} requires exactly {expected_count} frozen tasks")
    ids = [str(task.get("task_id", "")) for task in tasks]
    if any(not task_id.startswith(f"{expected_split}:") for task_id in ids):
        raise ManifestError(f"{stage} tasks must all come from {expected_split}")
    if len(set(ids)) != len(ids):
        raise ManifestError("task IDs must be unique")
    family_counts = Counter(str(task.get("task_family", "")) for task in tasks)
    if set(family_counts) != set(TASK_FAMILIES):
        raise ManifestError("tasks must cover all six frozen ALFWorld families")
    expected_per_family = 1 if stage == "pilot" else 5
    if any(family_counts[family] != expected_per_family for family in TASK_FAMILIES):
        raise ManifestError(f"{stage} requires {expected_per_family} task(s) per family")
    from rq1.university_gpu.pilot_readiness import budget_problems
    for task in tasks:
        issues = budget_problems(dict(task))
        if issues:
            raise ManifestError(f"{task.get('task_id')}: " + "; ".join(issues))
        if not task.get("checkpoint_id") or not task.get("prefix_actions"):
            raise ManifestError(f"task lacks frozen checkpoint/prefix: {task.get('task_id')}")
        _digest(task.get("checkpoint_digest"), f"{task.get('task_id')} checkpoint_digest")
        _digest(task.get("post_detour_digest"), f"{task.get('task_id')} post_detour_digest")
        if int(task.get("recovery_action_budget", 0)) < 1:
            raise ManifestError(f"task has no valid recovery action budget: {task.get('task_id')}")


def _select_skill_ids(
    tasks: Sequence[Mapping[str, Any]],
    eligibility: Mapping[str, Mapping[str, Sequence[str]]],
    library_skill_ids: set[str],
    randomization_seed: int,
) -> dict[str, dict[str, str]]:
    selected: dict[str, dict[str, str]] = {}
    for task in tasks:
        task_id = str(task["task_id"])
        candidates = eligibility.get(task_id)
        if not isinstance(candidates, Mapping):
            raise ManifestError(f"no blinded relevance eligibility record for {task_id}")
        relevant = sorted(set(str(item) for item in candidates.get("relevant", ())))
        irrelevant = sorted(set(str(item) for item in candidates.get("irrelevant", ())))
        if not relevant or not irrelevant:
            raise ManifestError(f"{task_id} needs at least one relevant and one irrelevant Accum-18 skill")
        if (set(relevant) | set(irrelevant)) - library_skill_ids:
            raise ManifestError(f"eligibility file references a skill outside frozen Accum-18 for {task_id}")
        if set(relevant) & set(irrelevant):
            raise ManifestError(f"a skill cannot be both relevant and irrelevant for {task_id}")
        pair: dict[str, str] = {}
        for label, candidates_for_label in (("relevant", relevant), ("irrelevant", irrelevant)):
            salt = canonical_hash({"seed": randomization_seed, "task": task_id, "label": label})
            chooser = random.Random(int(salt[:16], 16))
            pair[label] = chooser.choice(candidates_for_label)
        selected[task_id] = pair
    return selected


def build_manifest(
    *,
    stage: str,
    run_id: str,
    tasks: Sequence[Mapping[str, Any]],
    relevance: Mapping[str, Mapping[str, Sequence[str]]],
    library_skill_ids: Sequence[str],
    randomization_seed: int,
    source_commit: str,
    task_manifest_sha256: str,
    controlled_failure_sha256: str,
    library_sha256: str,
    alfworld_data_sha256: str,
    relevance_sha256: str,
    tokenizer_sha256: str,
    tokenizer_revision: str,
    filler_sha256: str,
    gate_scores_sha256: str,
    gate_labels_sha256: str,
    gate_threshold: float,
    gate_calibration_sha256: str,
    gate_calibration_split: str = "valid_seen",
) -> dict[str, Any]:
    if stage not in {"pilot", "confirmatory"}:
        raise ManifestError("stage must be pilot or confirmatory")
    validate_run_id(run_id)
    _validate_tasks(stage, tasks)
    if not isinstance(source_commit, str) or len(source_commit) != 40:
        raise ManifestError("source_commit must be a full Git commit SHA")
    hashes = {
        "task_manifest_sha256": _digest(task_manifest_sha256, "task_manifest_sha256"),
        "controlled_failure_sha256": _digest(controlled_failure_sha256, "controlled_failure_sha256"),
        "library_sha256": _digest(library_sha256, "library_sha256"),
        "alfworld_data_sha256": _digest(alfworld_data_sha256, "alfworld_data_sha256"),
        "relevance_sha256": _digest(relevance_sha256, "relevance_sha256"),
        "tokenizer_sha256": _digest(tokenizer_sha256, "tokenizer_sha256"),
        "filler_sha256": _digest(filler_sha256, "filler_sha256"),
        "gate_scores_sha256": _digest(gate_scores_sha256, "gate_scores_sha256"),
        "gate_labels_sha256": _digest(gate_labels_sha256, "gate_labels_sha256"),
        "gate_calibration_sha256": _digest(gate_calibration_sha256, "gate_calibration_sha256"),
    }
    if not -1 <= float(gate_threshold) <= 1:
        raise ManifestError("cosine gate_threshold must lie in [-1, 1]")
    if gate_calibration_split != "valid_seen":
        raise ManifestError("gate threshold must be frozen from valid_seen only")
    if not tokenizer_revision or "unverified" in tokenizer_revision.lower() or tokenizer_revision.startswith("SET_TO_"):
        raise ManifestError("an exact pinned Gemma tokenizer revision must be frozen")
    library_ids = set(str(item) for item in library_skill_ids)
    if len(library_ids) != 18:
        raise ManifestError("the frozen Accum-18 candidate universe must contain 18 unique skill IDs")
    selected = _select_skill_ids(tasks, relevance, library_ids, randomization_seed)
    seeds = (PILOT_SEED,) if stage == "pilot" else SEEDS
    ordered_tasks = sorted(tasks, key=lambda item: (TASK_FAMILIES.index(str(item["task_family"])), int(item.get("order_index", 0))))
    units: list[dict[str, Any]] = []
    cell_index = 0
    for task in ordered_tasks:
        for seed in seeds:
            offset = cell_index % len(CONDITIONS)
            condition_order = CONDITIONS[offset:] + CONDITIONS[:offset]
            for position, condition in enumerate(condition_order):
                identity = {
                    "protocol": PROTOCOL_VERSION,
                    "stage": stage,
                    "task_id": task["task_id"],
                    "seed": seed,
                    "condition": condition,
                    "checkpoint_digest": task["checkpoint_digest"],
                    "post_detour_digest": task["post_detour_digest"],
                }
                unit = {
                    "global_order": len(units) + 1,
                    "unit_key": canonical_hash(identity),
                    "identity": identity,
                    "cell_index": cell_index,
                    "position_in_cell": position,
                    "shard": cell_index % SHARD_COUNT + 1,
                    "task_id": task["task_id"],
                    "task_family": task["task_family"],
                    "seed": seed,
                    "condition": condition,
                    "checkpoint_id": task["checkpoint_id"],
                    "checkpoint_digest": task["checkpoint_digest"],
                    "post_detour_digest": task["post_detour_digest"],
                    "recovery_action_budget": int(task["recovery_action_budget"]),
                }
                if condition in {"Rel", "Irr"}:
                    unit["selected_skill_id"] = selected[str(task["task_id"])]["relevant" if condition == "Rel" else "irrelevant"]
                units.append(unit)
            cell_index += 1
    counts = Counter(unit["condition"] for unit in units)
    expected = ARMS[stage]
    if len(units) != expected or set(counts) != set(CONDITIONS) or any(count != expected // len(CONDITIONS) for count in counts.values()):
        raise ManifestError("generated matrix does not have the exact balanced episode count")
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "protocol_version": PROTOCOL_VERSION,
        "status": "proposed",
        "run_id": run_id,
        "stage": stage,
        "source_commit": source_commit,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": {"tag": MODEL_TAG, "digest": MODEL_DIGEST, "quantization": MODEL_QUANTIZATION},
        "retrieval": {"model": SBERT_MODEL, "revision": SBERT_REVISION,
                      "snapshot_sha256": SBERT_SNAPSHOT_SHA256, "top_k": 3},
        "inputs": hashes,
        "tokenizer": {"revision": str(tokenizer_revision), "sha256": hashes["tokenizer_sha256"]},
        "gate": {"threshold": float(gate_threshold), "calibration_split": gate_calibration_split},
        "randomization_seed": int(randomization_seed),
        "condition_order": "balanced-left-rotation-by-task-seed-cell-v1",
        "conditions": list(CONDITIONS),
        "seeds": list(seeds),
        "task_count": len(ordered_tasks),
        "task_seed_cells": cell_index,
        "episode_count": len(units),
        "condition_counts": dict(sorted(counts.items())),
        "selected_skill_ids": selected,
        "library_skill_ids": sorted(library_ids),
        "tasks": [dict(task) for task in ordered_tasks],
        "units": units,
    }
    payload["manifest_sha256"] = content_hash({key: value for key, value in payload.items() if key not in {"generated_at", "manifest_sha256", "status", "approval"}})
    problems = validate_manifest(payload)
    if problems:
        raise ManifestError("generated invalid matrix: " + "; ".join(problems))
    return payload


def validate_manifest(manifest: Mapping[str, Any], *, require_frozen: bool = False) -> list[str]:
    if not isinstance(manifest, Mapping):
        return ["manifest must be a JSON object"]
    problems: list[str] = []
    if manifest.get("schema") != SCHEMA or manifest.get("protocol_version") != PROTOCOL_VERSION:
        problems.append("unsupported manifest schema or protocol version")
    if require_frozen and manifest.get("status") != "frozen":
        problems.append("manifest is not frozen")
    elif manifest.get("status") not in {"proposed", "frozen"}:
        problems.append("manifest status must be proposed or frozen")
    try:
        validate_run_id(manifest.get("run_id"))
    except ManifestError as exc:
        problems.append(str(exc))
    expected_model = {"tag": MODEL_TAG, "digest": MODEL_DIGEST, "quantization": MODEL_QUANTIZATION}
    if manifest.get("model") != expected_model:
        problems.append("manifest model identity differs from the frozen Gemma 4 12B model")
    if manifest.get("retrieval") != {"model": SBERT_MODEL, "revision": SBERT_REVISION,
                                     "snapshot_sha256": SBERT_SNAPSHOT_SHA256, "top_k": 3}:
        problems.append("manifest retrieval identity differs from the frozen Sentence-BERT top-3 protocol")
    stage = manifest.get("stage")
    expected_count = ARMS.get(str(stage))
    units = manifest.get("units")
    if expected_count is None or not isinstance(units, list):
        return problems + ["stage or unit list is invalid"]
    if any(not isinstance(unit, Mapping) for unit in units):
        return problems + ["every unit must be a JSON object"]
    integer_fields = ("global_order", "cell_index", "position_in_cell", "shard", "seed")
    for index, unit in enumerate(units):
        for field in integer_fields:
            value = unit.get(field)
            if isinstance(value, bool) or not isinstance(value, int):
                problems.append(f"unit {index + 1} has invalid integer field: {field}")
        if not isinstance(unit.get("condition"), str) or not isinstance(unit.get("task_id"), str):
            problems.append(f"unit {index + 1} has invalid task or condition identity")
    if problems and any("unit " in item and ("integer field" in item or "task or condition identity" in item) for item in problems):
        return problems
    if len(units) != expected_count or manifest.get("episode_count") != expected_count:
        problems.append(f"expected {expected_count} units")
    keys = [unit.get("unit_key") for unit in units if isinstance(unit, Mapping)]
    if any(not isinstance(key, str) or not key for key in keys) or len(keys) != len(units) or len(set(keys)) != len(keys):
        problems.append("unit identities are missing or duplicated")
    expected_per_arm = expected_count // len(CONDITIONS)
    counts = Counter(str(unit.get("condition")) for unit in units if isinstance(unit, Mapping))
    if counts != Counter({condition: expected_per_arm for condition in CONDITIONS}):
        problems.append("condition counts are not balanced")
    if manifest.get("condition_counts") != dict(sorted(counts.items())):
        problems.append("manifest condition-count summary differs from its unit list")
    expected_tasks = 6 if stage == "pilot" else 30
    tasks = manifest.get("tasks")
    if not isinstance(tasks, list):
        problems.append("task list must be an array")
        tasks = []
    task_ids = {str(task.get("task_id", "")) for task in tasks if isinstance(task, Mapping)}
    if len(task_ids) != expected_tasks:
        problems.append("task list has incorrect count")
    try:
        _validate_tasks(str(stage), tasks)
    except (ManifestError, KeyError, TypeError, ValueError) as exc:
        problems.append(f"frozen task definitions are invalid: {exc}")
    expected_seeds = [PILOT_SEED] if stage == "pilot" else list(SEEDS)
    if manifest.get("seeds") != expected_seeds:
        problems.append("manifest seed list differs from the frozen stage design")
    if manifest.get("conditions") != list(CONDITIONS):
        problems.append("manifest condition list differs from the six-arm design")
    if not isinstance(manifest.get("source_commit"), str) or len(str(manifest.get("source_commit"))) != 40 or any(
        char not in "0123456789abcdef" for char in str(manifest.get("source_commit", "")).lower()
    ):
        problems.append("source_commit must be a full Git commit SHA")
    input_hashes = manifest.get("inputs")
    required_hashes = {"task_manifest_sha256", "controlled_failure_sha256", "library_sha256", "alfworld_data_sha256",
                       "relevance_sha256", "tokenizer_sha256", "filler_sha256", "gate_scores_sha256",
                       "gate_labels_sha256", "gate_calibration_sha256"}
    if not isinstance(input_hashes, Mapping) or set(input_hashes) != required_hashes:
        problems.append("manifest input hash map is incomplete or contains unexpected entries")
    else:
        for name, digest in input_hashes.items():
            try:
                _digest(digest, name)
            except ManifestError as exc:
                problems.append(str(exc))
    gate = manifest.get("gate")
    if not isinstance(gate, Mapping) or gate.get("calibration_split") != "valid_seen":
        problems.append("Gate threshold must be calibrated on valid_seen")
    else:
        try:
            threshold = float(gate.get("threshold"))
            if not -1 <= threshold <= 1:
                problems.append("Gate threshold is outside cosine range [-1, 1]")
        except (TypeError, ValueError):
            problems.append("Gate threshold is not numeric")
    library_ids = manifest.get("library_skill_ids")
    if (not isinstance(library_ids, list) or len(library_ids) != 18
            or any(not isinstance(item, str) or not item for item in library_ids)
            or len(set(library_ids)) != 18):
        problems.append("manifest must carry exactly 18 unique Accum-18 skill identities")
    tokenizer = manifest.get("tokenizer")
    if (not isinstance(tokenizer, Mapping) or not tokenizer.get("revision")
            or "unverified" in str(tokenizer.get("revision", "")).lower()
            or tokenizer.get("sha256") != (input_hashes or {}).get("tokenizer_sha256")):
        problems.append("tokenizer revision/hash metadata is missing or mismatched")
    selected = manifest.get("selected_skill_ids")
    if not isinstance(selected, Mapping) or set(selected) != task_ids:
        problems.append("manifest relevance selections must cover every task exactly once")
    elif isinstance(library_ids, list):
        for task_id, pair in selected.items():
            if not isinstance(pair, Mapping) or set(pair) != {"relevant", "irrelevant"}:
                problems.append(f"{task_id} lacks one relevant and one irrelevant frozen skill selection")
            elif any(skill_id not in library_ids for skill_id in pair.values()) or pair["relevant"] == pair["irrelevant"]:
                problems.append(f"{task_id} selected an invalid or duplicate relevance skill")
    cells: dict[int, list[Mapping[str, Any]]] = {}
    for unit in units:
        cells.setdefault(unit["cell_index"], []).append(unit)
    expected_cells = expected_tasks * len(expected_seeds)
    declared_cells = manifest.get("task_seed_cells")
    if isinstance(declared_cells, bool) or not isinstance(declared_cells, int):
        problems.append("task_seed_cells must be an integer")
    if (len(cells) != expected_cells or len(cells) != declared_cells
            or set(cells) != set(range(expected_cells))
            or any(len(cell) != len(CONDITIONS) for cell in cells.values())):
        problems.append("task-seed cells are incomplete")
    task_by_id = {str(task.get("task_id")): task for task in tasks if isinstance(task, Mapping)}
    seen_pairs: set[tuple[str, int]] = set()
    expected_order = 1
    for cell_index, cell in cells.items():
        if {unit.get("condition") for unit in cell} != set(CONDITIONS):
            problems.append(f"cell {cell_index} does not include every condition exactly once")
        first = cell[0]
        pair = (first["task_id"], first["seed"])
        if pair in seen_pairs or pair[0] not in task_ids or pair[1] not in expected_seeds:
            problems.append(f"cell {cell_index} duplicates or contains an unknown task-seed pair")
        seen_pairs.add(pair)
        expected_ordered_conditions = CONDITIONS[cell_index % len(CONDITIONS):] + CONDITIONS[:cell_index % len(CONDITIONS)]
        task = task_by_id.get(pair[0], {})
        for position, unit in enumerate(sorted(cell, key=lambda item: item["position_in_cell"])):
            expected_condition = expected_ordered_conditions[position]
            identity = {"protocol": PROTOCOL_VERSION, "stage": stage, "task_id": pair[0], "seed": pair[1],
                        "condition": expected_condition, "checkpoint_digest": task.get("checkpoint_digest"),
                        "post_detour_digest": task.get("post_detour_digest")}
            if (unit.get("global_order") != expected_order or unit.get("position_in_cell") != position
                    or unit.get("condition") != expected_condition or unit.get("task_id") != pair[0]
                    or unit.get("task_family") != task.get("task_family") or unit.get("seed") != pair[1]
                    or unit.get("checkpoint_digest") != task.get("checkpoint_digest")
                    or unit.get("post_detour_digest") != task.get("post_detour_digest")
                    or unit.get("unit_key") != canonical_hash(identity)
                    or unit.get("shard") != cell_index % SHARD_COUNT + 1):
                problems.append(f"unit order or identity mismatch at global order {expected_order}")
            if expected_condition in {"Rel", "Irr"} and isinstance(selected, Mapping):
                label = "relevant" if expected_condition == "Rel" else "irrelevant"
                if unit.get("selected_skill_id") != (selected.get(pair[0]) or {}).get(label):
                    problems.append(f"unit relevance selection mismatch at global order {expected_order}")
            expected_order += 1
    if len(seen_pairs) != expected_cells:
        problems.append("manifest task-seed cells do not cover the full paired design")
    try:
        stored_hash = str(manifest.get("manifest_sha256", ""))
        actual_hash = content_hash({key: value for key, value in manifest.items() if key not in {"generated_at", "manifest_sha256", "status", "approval"}})
        # A frozen manifest's approval/status fields are deliberately outside the scientific identity.
        if stored_hash != actual_hash:
            problems.append("manifest hash does not match contents")
    except (TypeError, ValueError):
        problems.append("manifest contents are not canonical JSON")
    if require_frozen:
        approval = manifest.get("approval")
        if not isinstance(approval, Mapping) or approval.get("approved") is not True or not approval.get("approved_by") or not approval.get("approval_reference"):
            problems.append("frozen manifest lacks its human approval record")
        if stage == "confirmatory" and (not isinstance(approval, Mapping)
                                         or approval.get("previous_task_outcomes_acknowledged") is not True):
            problems.append("confirmatory manifest lacks historical outcome exposure acknowledgment")
        if stage == "confirmatory" and isinstance(approval, Mapping):
            for field in ("eligibility_verified", "blinded_relevance_verified", "leakage_review_complete",
                          "pilot_go_approved", "precision_power_review_complete"):
                if approval.get(field) is not True:
                    problems.append(f"confirmatory approval lacks {field}")
            for name in ("power_report_sha256", "pilot_validation_sha256"):
                try:
                    _digest(approval.get(name), name)
                except ManifestError as exc:
                    problems.append(str(exc))
    return problems


def freeze_manifest(manifest: Mapping[str, Any], approval: Mapping[str, Any], destination: Path) -> dict[str, Any]:
    problems = validate_manifest(manifest)
    if problems:
        raise ManifestError("cannot freeze invalid manifest: " + "; ".join(problems))
    if approval.get("approved") is not True or not approval.get("approved_by") or not approval.get("approval_reference"):
        raise ManifestError("explicit human approval, approver, and reference are required")
    if approval.get("manifest_sha256") != manifest.get("manifest_sha256"):
        raise ManifestError("approval is not bound to this exact manifest hash")
    if manifest.get("stage") == "confirmatory" and approval.get("previous_task_outcomes_acknowledged") is not True:
        raise ManifestError("confirmatory approval must acknowledge that these task outcomes were observed historically")
    confirmatory_fields = ("eligibility_verified", "blinded_relevance_verified", "leakage_review_complete",
                           "pilot_go_approved", "precision_power_review_complete")
    if manifest.get("stage") == "confirmatory":
        missing = [name for name in confirmatory_fields if approval.get(name) is not True]
        if missing:
            raise ManifestError("confirmatory approval lacks: " + ", ".join(missing))
        _digest(approval.get("power_report_sha256"), "power_report_sha256")
        _digest(approval.get("pilot_validation_sha256"), "pilot_validation_sha256")
    frozen = dict(manifest)
    frozen["status"] = "frozen"
    frozen["approval"] = {
        "approved": True,
        "approved_by": str(approval["approved_by"]),
        "approval_reference": str(approval["approval_reference"]),
        "approved_at": str(approval.get("approved_at") or datetime.now(timezone.utc).isoformat()),
        "previous_task_outcomes_acknowledged": approval.get("previous_task_outcomes_acknowledged") is True,
    }
    if manifest.get("stage") == "confirmatory":
        frozen["approval"].update({field: True for field in confirmatory_fields})
        frozen["approval"]["power_report_sha256"] = approval["power_report_sha256"]
        frozen["approval"]["pilot_validation_sha256"] = approval["pilot_validation_sha256"]
    atomic_write_json(destination, frozen, overwrite=False)
    return frozen
