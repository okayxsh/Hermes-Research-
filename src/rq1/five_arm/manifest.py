"""PROPOSED five-arm and pilot-diagnostic unit manifests, built from frozen inputs only.

Eligibility is defined per arm and frozen before any outcome exists (SD-23):

* D0 and D18 are scheduled for EVERY roster task -- they use no filler and no labels, so a filler problem can never
  remove a task's primary (D18 vs D0) pair;
* Fill is scheduled unless its neutral content misses the approved token tolerance (cell BLOCKED, recorded);
* Rel / Irr are scheduled only where the frozen selection rule yields a skill and the cell is not blocked.

The resulting ``arm_subsets`` are part of the hashed manifest.  Every task must carry an oracle replay at EXACTLY the
manifest's total action cap.  Freezing requires a human approval bound to the manifest hash and the matching readiness
level.  The pilot-diagnostic manifest (UNDISTURBED / DETOURED) is a separate schema and run.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from typing import Any, Mapping, Sequence

from rq1.five_arm import readiness
from rq1.five_arm.measurements import DIAGNOSTIC_CONDITIONS, DIAGNOSTIC_PROTOCOL_VERSION
from rq1.five_arm.protocol import CONDITIONS, FILLER_ARMS, NATURAL_ARMS, PROTOCOL_STATUS, PROTOCOL_VERSION

SCHEMA = "rq1-five-arm-manifest-v2"
DIAGNOSTIC_SCHEMA = "rq1-five-arm-diagnostic-manifest-v1"
REQUIRED_INPUT_HASHES = ("roster_sha256", "labels_sha256", "ranking_sha256", "mechanism_subsets_sha256",
                         "filler_corpus_sha256", "runtime_config_sha256", "oracle_evidence_sha256", "exposure_statement_sha256")
DIAGNOSTIC_INPUT_HASHES = ("roster_sha256", "runtime_config_sha256", "oracle_evidence_sha256", "diagnostic_protocol_sha256",
                           "exposure_statement_sha256")
STAGE_LEVEL = {"pilot_diagnostic": "d0_failure_manipulation_pilot", "pilot": "five_arm_scientific_pilot",
               "confirmatory": "confirmatory"}
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_TASK_FIELDS = ("goal_template", "scene", "family", "oracle_sha256", "checkpoint_digest", "post_detour_digest")


class FiveArmManifestError(ValueError):
    pass


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def manifest_hash(manifest: Mapping[str, Any]) -> str:
    return _hash({k: v for k, v in manifest.items() if k not in ("status", "approval", "manifest_sha256")})


def _common_checks(stage: str, run_id: str, source_commit: str, input_hashes: Mapping[str, str], required: Sequence[str],
                   seeds: Sequence[int], tasks: Sequence[Mapping[str, Any]], cap: int, split: str) -> list[int]:
    if not _RUN_ID.match(run_id):
        raise FiveArmManifestError("run_id must be path-safe")
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit or ""):
        raise FiveArmManifestError("source_commit must be a full Git SHA")
    if set(input_hashes) != set(required) or not all(_HEX64.match(v or "") for v in input_hashes.values()):
        raise FiveArmManifestError(f"input_hashes must be exactly {tuple(required)} as SHA-256 digests")
    seeds = [int(s) for s in seeds]
    if not seeds or len(set(seeds)) != len(seeds):
        raise FiveArmManifestError("seeds must be a non-empty list of unique integers")
    ids = [str(t["task_id"]) for t in tasks]
    if len(set(ids)) != len(ids) or not ids:
        raise FiveArmManifestError("task IDs must be unique and non-empty")
    for task in tasks:
        task_id = str(task["task_id"])
        if not task_id.startswith(split + ":"):
            raise FiveArmManifestError(f"{stage} tasks must come from {split}: {task_id}")
        for field in _TASK_FIELDS:
            if not task.get(field):
                raise FiveArmManifestError(f"{task_id} lacks {field}")
        if int(task["detour_actions"]) != 1:
            raise FiveArmManifestError(f"{task_id}: the frozen controlled disruption is exactly one detour action")
        allowance = cap - int(task["prefix_actions"]) - 1
        if allowance < int(task["continuation_actions"]):
            raise FiveArmManifestError(f"{task_id}: oracle continuation does not fit the total action cap {cap}")
        if int(task.get("oracle_replay_total_action_limit") or 0) != cap:
            raise FiveArmManifestError(f"{task_id}: no oracle replay at exactly the total action cap {cap}")
    return seeds


def _unit(identity: Mapping[str, Any], order: int, cell: int, position: int, task: Mapping[str, Any], cap: int,
          extra: Mapping[str, Any]) -> dict[str, Any]:
    return {"unit_key": _hash(dict(identity)), "identity": dict(identity), "global_order": order, "cell_index": cell,
            "position_in_cell": position, "task_id": identity["task_id"], "seed": identity["seed"],
            "condition": identity["condition"], "total_action_cap": cap,
            "recovery_action_allowance": cap - int(task["prefix_actions"]) - 1, **extra}


def build_manifest(*, stage: str, run_id: str, tasks: Sequence[Mapping[str, Any]], selections: Mapping[str, Mapping[str, Any]],
                   memory_audits: Mapping[str, Mapping[str, Mapping[str, Any]]], seeds: Sequence[int], total_action_cap: int,
                   source_commit: str, input_hashes: Mapping[str, str], synthetic: bool = False) -> dict[str, Any]:
    if stage not in ("pilot", "confirmatory"):
        raise FiveArmManifestError("stage must be pilot or confirmatory")
    split = "valid_seen" if stage == "pilot" else "valid_unseen"
    seeds = _common_checks(stage, run_id, source_commit, input_hashes, REQUIRED_INPUT_HASHES, seeds, tasks, total_action_cap, split)
    for task in tasks:
        if str(task["task_id"]) not in selections:
            raise FiveArmManifestError(f"{task['task_id']} has no frozen Rel/Irr selection record")
    units, blocked = [], []
    subsets: dict[str, list[str]] = {arm: [] for arm in FILLER_ARMS}
    cell_index = 0
    for task in sorted(tasks, key=lambda t: str(t["task_id"])):
        task_id = str(task["task_id"])
        sel = selections[task_id]
        runnable = []
        for arm in CONDITIONS:
            if arm in ("Rel", "Irr") and sel.get(arm.lower()) is None:
                continue
            audit = memory_audits.get(task_id, {}).get(arm)
            if audit is None:
                raise FiveArmManifestError(f"{task_id}/{arm} has no pre-computed memory audit")
            if arm in NATURAL_ARMS:
                runnable.append(arm)
            elif audit.get("within_token_tolerance") is True and audit.get("filler_corpus_approved") is True:
                runnable.append(arm)
                subsets[arm].append(task_id)
            else:
                reason = ("neutral content outside the approved token tolerance" if audit.get("within_token_tolerance") is not True
                          else "filler corpus not approved")
                blocked.append({"task_id": task_id, "condition": arm, "reason": reason,
                                "residual": audit.get("token_residual_vs_anchor")})
        for seed in seeds:
            offset = cell_index % len(runnable)
            for position, arm in enumerate(runnable[offset:] + runnable[:offset]):
                identity = {"protocol": PROTOCOL_VERSION, "stage": stage, "task_id": task_id, "seed": seed, "condition": arm,
                            "checkpoint_digest": task["checkpoint_digest"], "post_detour_digest": task["post_detour_digest"],
                            "total_action_cap": total_action_cap}
                units.append(_unit(identity, len(units) + 1, cell_index, position, task, total_action_cap,
                                   {"selected_skill_id": (sel.get(arm.lower()) or {}).get("skill_id") if arm in ("Rel", "Irr") else None,
                                    "memory_block_sha256": memory_audits[task_id][arm]["memory_block_sha256"]}))
            cell_index += 1
    payload = {"schema": SCHEMA, "protocol_version": PROTOCOL_VERSION, "protocol_status": PROTOCOL_STATUS, "status": "proposed",
               "synthetic": bool(synthetic), "run_id": run_id, "stage": stage, "split": split, "source_commit": source_commit,
               "input_hashes": dict(sorted(input_hashes.items())), "seeds": seeds, "total_action_cap": total_action_cap,
               "condition_order": "within-cell left rotation over the task's runnable arms",
               "tasks": [dict(t) for t in sorted(tasks, key=lambda t: str(t["task_id"]))],
               "selections": {k: dict(v) for k, v in sorted(selections.items())},
               "arm_subsets": {arm: sorted(v) for arm, v in subsets.items()},
               "units": units, "blocked_cells": blocked,
               "condition_counts": dict(sorted(Counter(u["condition"] for u in units).items()))}
    payload["manifest_sha256"] = manifest_hash(payload)
    return payload


def build_diagnostic_manifest(*, run_id: str, tasks: Sequence[Mapping[str, Any]], seeds: Sequence[int], total_action_cap: int,
                              source_commit: str, input_hashes: Mapping[str, str], synthetic: bool = False) -> dict[str, Any]:
    """PILOT ONLY: UNDISTURBED and DETOURED continuations without memory, same tasks, checkpoints, seeds and allowance."""
    seeds = _common_checks("pilot_diagnostic", run_id, source_commit, input_hashes, DIAGNOSTIC_INPUT_HASHES, seeds, tasks,
                           total_action_cap, "valid_seen")
    units = []
    cell_index = 0
    for task in sorted(tasks, key=lambda t: str(t["task_id"])):
        for seed in seeds:
            order = DIAGNOSTIC_CONDITIONS if cell_index % 2 == 0 else tuple(reversed(DIAGNOSTIC_CONDITIONS))
            for position, condition in enumerate(order):
                identity = {"protocol": DIAGNOSTIC_PROTOCOL_VERSION, "stage": "pilot_diagnostic", "task_id": str(task["task_id"]),
                            "seed": seed, "condition": condition, "checkpoint_digest": task["checkpoint_digest"],
                            "total_action_cap": total_action_cap}
                units.append(_unit(identity, len(units) + 1, cell_index, position, task, total_action_cap, {}))
            cell_index += 1
    payload = {"schema": DIAGNOSTIC_SCHEMA, "protocol_version": DIAGNOSTIC_PROTOCOL_VERSION, "protocol_status": PROTOCOL_STATUS,
               "status": "proposed", "synthetic": bool(synthetic), "run_id": run_id, "stage": "pilot_diagnostic",
               "split": "valid_seen", "source_commit": source_commit, "input_hashes": dict(sorted(input_hashes.items())),
               "seeds": seeds, "total_action_cap": total_action_cap,
               "allowance_rule": "identical post-checkpoint model-action allowance cap - prefix - 1 (SD-15)",
               "tasks": [dict(t) for t in sorted(tasks, key=lambda t: str(t["task_id"]))], "units": units,
               "isolation": "pilot diagnostic only; never merged into five-arm pilot or confirmatory data"}
    payload["manifest_sha256"] = manifest_hash(payload)
    return payload


def validate_manifest(manifest: Mapping[str, Any]) -> list[str]:
    problems = []
    if (manifest.get("schema"), manifest.get("protocol_version")) not in ((SCHEMA, PROTOCOL_VERSION),
                                                                          (DIAGNOSTIC_SCHEMA, DIAGNOSTIC_PROTOCOL_VERSION)):
        problems.append("not a five-arm or pilot-diagnostic manifest")
    if manifest.get("manifest_sha256") != manifest_hash(manifest):
        problems.append("manifest hash does not match its contents")
    keys = [u["unit_key"] for u in manifest.get("units", [])]
    if len(keys) != len(set(keys)):
        problems.append("duplicate unit identities")
    for cell in manifest.get("blocked_cells", []):
        if cell.get("condition") not in FILLER_ARMS:
            problems.append(f"blocked cell for {cell.get('condition')}: only filler-bearing arms may be blocked")
    return problems


def freeze_manifest(manifest: Mapping[str, Any], approval: Mapping[str, Any], readiness_state: Mapping[str, Any],
                    artifacts: Mapping[str, str], *, real: bool = True) -> dict[str, Any]:
    problems = validate_manifest(manifest)
    if real and manifest.get("synthetic"):
        problems.append("a synthetic manifest can never be frozen for real execution")
    if not readiness.valid_approval(approval, manifest.get("manifest_sha256"), real=real):
        problems.append("a human approval bound to this exact manifest hash is required")
    if problems:
        raise FiveArmManifestError("; ".join(problems))
    readiness.require(STAGE_LEVEL[manifest["stage"]], readiness_state, artifacts, real=real)
    frozen = dict(manifest)
    frozen["status"] = "frozen"
    frozen["approval"] = dict(approval)
    return frozen
