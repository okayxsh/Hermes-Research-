"""PROPOSED five-arm unit manifest.  Built from frozen inputs only (roster, rankings, labels, selections, memory audits).

D0 / D18 / Fill are scheduled for every task; Rel / Irr only where the frozen selection rule yields a skill.  Cells whose
neutral content misses the approved token tolerance are BLOCKED (recorded), and a manifest with blocked cells cannot be
frozen.  Freezing additionally requires a human approval bound to the manifest hash and the matching readiness level.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from typing import Any, Mapping, Sequence

from rq1.five_arm import readiness
from rq1.five_arm.protocol import ALWAYS_SCHEDULED, CONDITIONS, PROTOCOL_STATUS, PROTOCOL_VERSION

SCHEMA = "rq1-five-arm-manifest-v1"
REQUIRED_INPUT_HASHES = ("roster_sha256", "labels_sha256", "ranking_sha256", "mechanism_subsets_sha256",
                         "filler_corpus_sha256", "runtime_config_sha256", "oracle_evidence_sha256")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class FiveArmManifestError(ValueError):
    pass


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def build_manifest(*, stage: str, run_id: str, tasks: Sequence[Mapping[str, Any]], selections: Mapping[str, Mapping[str, Any]],
                   memory_audits: Mapping[str, Mapping[str, Mapping[str, Any]]], seeds: Sequence[int], total_action_cap: int,
                   source_commit: str, input_hashes: Mapping[str, str]) -> dict[str, Any]:
    if stage not in ("pilot", "confirmatory"):
        raise FiveArmManifestError("stage must be pilot or confirmatory")
    if not _RUN_ID.match(run_id):
        raise FiveArmManifestError("run_id must be path-safe")
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit or ""):
        raise FiveArmManifestError("source_commit must be a full Git SHA")
    if set(input_hashes) != set(REQUIRED_INPUT_HASHES) or not all(_HEX64.match(v or "") for v in input_hashes.values()):
        raise FiveArmManifestError(f"input_hashes must be exactly {REQUIRED_INPUT_HASHES} as SHA-256 digests")
    seeds = [int(s) for s in seeds]
    if not seeds or len(set(seeds)) != len(seeds):
        raise FiveArmManifestError("seeds must be a non-empty list of unique integers")
    split = "valid_seen" if stage == "pilot" else "valid_unseen"
    ids = [str(t["task_id"]) for t in tasks]
    if len(set(ids)) != len(ids) or not ids:
        raise FiveArmManifestError("task IDs must be unique and non-empty")
    for task in tasks:
        task_id = str(task["task_id"])
        if not task_id.startswith(split + ":"):
            raise FiveArmManifestError(f"{stage} tasks must come from {split}: {task_id}")
        for field in ("goal_template", "scene", "family", "oracle_sha256", "checkpoint_digest", "post_detour_digest"):
            if not task.get(field):
                raise FiveArmManifestError(f"{task_id} lacks {field}")
        budget = total_action_cap - int(task["prefix_actions"]) - int(task["detour_actions"])
        if budget < int(task["continuation_actions"]):
            raise FiveArmManifestError(f"{task_id}: oracle continuation does not fit the total action cap {total_action_cap}")
        if task_id not in selections:
            raise FiveArmManifestError(f"{task_id} has no frozen Rel/Irr selection record")
    units, blocked = [], []
    cell_index = 0
    for task in sorted(tasks, key=lambda t: str(t["task_id"])):
        task_id = str(task["task_id"])
        sel = selections[task_id]
        arms = [a for a in CONDITIONS if a in ALWAYS_SCHEDULED or sel.get({"Rel": "rel", "Irr": "irr"}[a]) is not None]
        runnable = []
        for arm in arms:
            audit = memory_audits.get(task_id, {}).get(arm)
            if audit is None:
                raise FiveArmManifestError(f"{task_id}/{arm} has no pre-computed memory audit")
            if audit.get("within_token_tolerance") is False:
                blocked.append({"task_id": task_id, "condition": arm, "reason": "neutral content outside the approved token tolerance",
                                "residual": audit.get("token_residual_vs_anchor")})
            else:
                runnable.append(arm)
        for seed in seeds:
            offset = cell_index % len(runnable)
            for position, arm in enumerate(runnable[offset:] + runnable[:offset]):
                identity = {"protocol": PROTOCOL_VERSION, "stage": stage, "task_id": task_id, "seed": seed, "condition": arm,
                            "checkpoint_digest": task["checkpoint_digest"], "post_detour_digest": task["post_detour_digest"],
                            "total_action_cap": total_action_cap}
                units.append({"unit_key": _hash(identity), "identity": identity, "global_order": len(units) + 1,
                              "cell_index": cell_index, "position_in_cell": position, "task_id": task_id, "seed": seed,
                              "condition": arm, "selected_skill_id": (sel.get(arm.lower()) or {}).get("skill_id") if arm in ("Rel", "Irr") else None,
                              "memory_block_sha256": memory_audits[task_id][arm]["memory_block_sha256"],
                              "recovery_action_budget": total_action_cap - int(task["prefix_actions"]) - int(task["detour_actions"])})
            cell_index += 1
    payload = {"schema": SCHEMA, "protocol_version": PROTOCOL_VERSION, "protocol_status": PROTOCOL_STATUS, "status": "proposed",
               "run_id": run_id, "stage": stage, "split": split, "source_commit": source_commit,
               "input_hashes": dict(sorted(input_hashes.items())), "seeds": seeds, "total_action_cap": total_action_cap,
               "condition_order": "within-cell left rotation over the task's runnable arms",
               "tasks": [dict(t) for t in sorted(tasks, key=lambda t: str(t["task_id"]))],
               "selections": {k: dict(v) for k, v in sorted(selections.items())},
               "units": units, "blocked_cells": blocked,
               "condition_counts": dict(sorted(Counter(u["condition"] for u in units).items()))}
    payload["manifest_sha256"] = _hash({k: v for k, v in payload.items() if k not in ("status", "approval")})
    return payload


def validate_manifest(manifest: Mapping[str, Any]) -> list[str]:
    problems = []
    if manifest.get("schema") != SCHEMA or manifest.get("protocol_version") != PROTOCOL_VERSION:
        problems.append("not a five-arm manifest")
    if manifest.get("manifest_sha256") != _hash({k: v for k, v in manifest.items() if k not in ("status", "approval", "manifest_sha256")}):
        problems.append("manifest hash does not match its contents")
    keys = [u["unit_key"] for u in manifest.get("units", [])]
    if len(keys) != len(set(keys)):
        problems.append("duplicate unit identities")
    return problems


def freeze_manifest(manifest: Mapping[str, Any], approval: Mapping[str, Any], readiness_state: Mapping[str, Any]) -> dict[str, Any]:
    problems = validate_manifest(manifest)
    if manifest.get("blocked_cells"):
        problems.append(f"{len(manifest['blocked_cells'])} blocked cell(s) must be resolved before freezing")
    if not readiness.valid_approval(approval) or approval.get("artifact_sha256") != manifest.get("manifest_sha256"):
        problems.append("a human approval bound to this exact manifest hash is required")
    if problems:
        raise FiveArmManifestError("; ".join(problems))
    readiness.require("scientific_pilot" if manifest["stage"] == "pilot" else "confirmatory", readiness_state)
    frozen = dict(manifest)
    frozen["status"] = "frozen"
    frozen["approval"] = dict(approval)
    return frozen
