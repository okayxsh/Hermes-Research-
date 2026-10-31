"""Versioned, non-scientific 200-action feasibility execution profile.

This profile is deliberately isolated from the approved 50-action scientific
manifest, launch, and identity constants. Candidate inputs remain unapproved
until separately reviewed evidence and an engineering-only authorization are
supplied on the university workstation.
"""
from __future__ import annotations

import copy
import json
import math
import tarfile
import re
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import Any, Callable, Mapping, Sequence
from uuid import uuid4

from rq1.bridge.adapters.task_index import build_task_index
from rq1.evaluation.amended_protocol import SBERT_MODEL, SBERT_REVISION, SBERT_SNAPSHOT_SHA256
from rq1.skills.library import TASK_FAMILIES
from rq1.tasks.discovery import discover_tasks
from rq1.university_gpu.interventions import LocalGemmaTokenizer, TokenizerIdentity
from rq1.university_gpu.models import atomic_write_json, content_hash, load_json, sha256_file
from rq1.utils.process_safety import require_posix_process_management

PROFILE_ID = "rq1-feasibility-200-v1"
SCHEMA = "rq1-feasibility-execution-manifest-v1"
RESULT_SCHEMA = "rq1-feasibility-episode-result-v1"
POLICY_VERSION = "rq1-feasibility-policy-v1"
TOTAL_ACTION_BUDGET = 200
CONDITIONS = ("D0", "D18")
PAIRED_EPISODE_SEED = 11
MODEL_TAG = "gemma4:12b"
MODEL_DIGEST = "6114515d63c17436a7c0417d82820ac65ad643e2806c5a3c89cb62846436ed0b"
MODEL_QUANTIZATION = "Q4_K_M"
HISTORICAL_MODEL_DIGEST = "4eb23ef187e2c5462566d6a1d3bbbc2f1346d0b4327cbb66d58fffbcc9b2b05c"
MODEL_SETTINGS = {
    "temperature": 0,
    "think": False,
    "num_predict": 2048,
    "num_ctx": 32768,
    "episode_seed_policy": "paired-unit-seed-11-passed-as-options.seed",
    "smoke_seed": 42,
}
EXPECTED_DISCOVERY_IDENTITY = "43be7c24c054c3b2299b5054f293c21eaca08734554310094c5654783a8d7aa1"
EXPECTED_ADAPTER_IDENTITY = "5fd5b9193ea361d62e877e93157ab567e63a95cf2dece0859db36cf4f534a0c0"
EXPECTED_ARCHIVE_SHA256 = "11c8ca453ef4ceaef5dce189c976ffc09636fdc218e40e22a0803deac9e429e2"
HISTORICAL_EVALUATION_ARCHIVE_SHA256 = "d52ed82e01a073f8c3382711d03f65916cb41ca89bc0b773e6845817bd37e4fd"
HISTORICAL_LIBRARY_MEMBER = "rq1-evaluation-360-final/repo/artifacts/evaluation-libraries/libraries-0030f89e76c6bc4c.json"
_RUN_ID = re.compile(r"rq1-feasibility-200-v1-[A-Za-z0-9][A-Za-z0-9._-]{1,79}[A-Za-z0-9_-]\Z")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class FeasibilityError(ValueError):
    """A feasibility manifest or evidence gate is invalid or incomplete."""


def _manifest_digest(value: Mapping[str, Any]) -> str:
    return content_hash({key: item for key, item in value.items() if key != "manifest_sha256"})


def _unit_rows(tasks: Sequence[Mapping[str, Any]], library_sha256: str | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task_index, task in enumerate(tasks):
        ordered_conditions = CONDITIONS if task_index % 2 == 0 else tuple(reversed(CONDITIONS))
        for position, condition in enumerate(ordered_conditions):
            identity = {
                "profile_id": PROFILE_ID,
                "task_id": task["task_id"],
                "condition": condition,
                "seed": PAIRED_EPISODE_SEED,
                "checkpoint_digest": task["checkpoint_digest"],
                "post_detour_digest": task["post_detour_digest"],
                "total_action_budget": TOTAL_ACTION_BUDGET,
                "model_digest": MODEL_DIGEST,
                "sbert_snapshot_sha256": SBERT_SNAPSHOT_SHA256,
                "accum18_library_sha256": library_sha256,
            }
            rows.append({
                "global_order": len(rows) + 1,
                "unit_key": content_hash(identity),
                "identity": identity,
                "task_id": task["task_id"],
                "task_family": task["task_family"],
                "seed": PAIRED_EPISODE_SEED,
                "condition": condition,
                "profile_id": PROFILE_ID,
                "checkpoint_id": task["checkpoint_id"],
                "checkpoint_digest": task["checkpoint_digest"],
                "post_detour_digest": task["post_detour_digest"],
                "total_action_budget": TOTAL_ACTION_BUDGET,
                "recovery_action_budget": task["recovery_action_budget"],
                "position_in_pair": position + 1,
                "scientific_evidence": False,
            })
    return rows


def build_candidate_manifest(
    candidate_path: Path,
    original_proposal_path: Path,
    transferred_audit_path: Path,
    model_proposal_path: Path,
) -> dict[str, Any]:
    """Build an unapproved 12-unit matrix while preserving candidate task evidence."""
    candidates = load_json(candidate_path)
    proposal = load_json(original_proposal_path)
    audit = load_json(transferred_audit_path)
    model = load_json(model_proposal_path)
    source_tasks = candidates.get("tasks")
    if candidates.get("status") != "candidate_unapproved" or candidates.get("model_calls") != 0:
        raise FeasibilityError("candidate source must remain unapproved and model-call-free")
    if not isinstance(source_tasks, list) or len(source_tasks) != 6:
        raise FeasibilityError("the transferred candidate source must contain exactly six tasks")
    audit_rows = audit.get("rows")
    if not isinstance(audit_rows, list) or len(audit_rows) != 6 or audit.get("archive_sha256") != EXPECTED_ARCHIVE_SHA256:
        raise FeasibilityError("the transferred audit does not match the six-task source archive")
    if proposal.get("actual_count") != 6 or proposal.get("status") not in {"proposed", "candidate_unapproved"}:
        raise FeasibilityError("the original six-task proposal is missing or has an unexpected status")
    if (model.get("model_tag") != MODEL_TAG or model.get("model_digest") != MODEL_DIGEST
            or model.get("quantization") != MODEL_QUANTIZATION or model.get("approved") is not False):
        raise FeasibilityError("prospective model identity differs from the unapproved workstation proposal")

    audit_by_task = {str(row.get("task_id")): row for row in audit_rows}
    tasks: list[dict[str, Any]] = []
    seen: set[str] = set()
    for original in source_tasks:
        task = copy.deepcopy(original)
        task_id = str(task.get("task_id", ""))
        if not task_id or task_id in seen:
            raise FeasibilityError("candidate task IDs must be nonempty and unique")
        seen.add(task_id)
        evidence = task.get("source_evidence") or {}
        audit_row = audit_by_task.get(task_id)
        if audit_row is None or evidence.get("archive_sha256") != EXPECTED_ARCHIVE_SHA256:
            raise FeasibilityError(f"task is not tied to the transferred audit: {task_id}")
        if (not _HEX64.fullmatch(str(evidence.get("member_sha256", "")))
                or evidence.get("member") != audit_row.get("evidence")):
            raise FeasibilityError(f"task evidence reference differs from the transferred audit: {task_id}")
        prefix = task.get("prefix_actions")
        route = task.get("reference_actions")
        if not isinstance(prefix, list) or not isinstance(route, list):
            raise FeasibilityError(f"route and checkpoint prefix must be preserved as arrays: {task_id}")
        if len(route) != task.get("reference_route_length") or route[:len(prefix)] != prefix or len(prefix) >= len(route):
            raise FeasibilityError(f"reference route does not match its frozen prefix: {task_id}")
        if (task.get("split") != "valid_seen" or not task_id.startswith("valid_seen:")
                or task.get("scientifically_frozen") is not False):
            raise FeasibilityError(f"candidate is not an unfrozen valid_seen task: {task_id}")
        original_budget = {
            "total_action_budget": task.get("total_action_budget"),
            "recovery_action_budget": task.get("recovery_action_budget"),
        }
        recovery_budget = TOTAL_ACTION_BUDGET - len(prefix) - 1
        if recovery_budget <= 0 or len(route) + 1 > TOTAL_ACTION_BUDGET:
            raise FeasibilityError(f"the preserved expert witness cannot fit under 200 actions: {task_id}")
        task["original_protocol_budget"] = original_budget
        task["total_action_budget"] = TOTAL_ACTION_BUDGET
        task["recovery_action_budget"] = recovery_budget
        task["oracle_replay_required"] = True
        task["prior_model_outcome_exposure"] = (
            "historical_gemma_memory_no_memory_outcomes" if "pick_and_place_simple-Pencil-None-Desk-310" in task_id
            else "transferred_oracle_validation_only"
        )
        tasks.append(task)

    families = [str(task.get("task_family", "")) for task in tasks]
    if len(set(families)) != 6 or set(families) != set(TASK_FAMILIES):
        raise FeasibilityError("the six candidates must preserve one task per frozen family")
    source_expectations = {str(row.get("task_id")): row for row in audit_rows}
    for task in tasks:
        expected = source_expectations[str(task["task_id"])]
        if (len(task["reference_actions"]) != expected.get("route_length")
                or len(task["prefix_actions"]) != expected.get("prefix_length")
                or not _HEX64.fullmatch(str(task.get("checkpoint_digest", "")))
                or not _HEX64.fullmatch(str(task.get("post_detour_digest", "")))
                or not task.get("checkpoint_id")
                or task.get("detour_action") != expected.get("detour")
                or task.get("oracle", {}).get("actual_total_actions") != expected.get("actual_oracle_actions")
                or int(task["reference_route_length"]) + 1 != int(expected.get("actual_oracle_actions", -1))):
            raise FeasibilityError(f"candidate route differs from the audited route summary: {task['task_id']}")

    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "profile_id": PROFILE_ID,
        "status": "candidate_unapproved",
        "run_id": None,
        "source_commit": None,
        "execution_scope": "engineering_feasibility_only",
        "scientific_evidence": False,
        "scientific_protocol_approved": False,
        "scientific_input_freeze": False,
        "option_b_author_decision_status": "pending",
        "total_action_budget": TOTAL_ACTION_BUDGET,
        "action_accounting": "total = checkpoint_prefix + 1 controlled detour + recovery budget",
        "conditions": list(CONDITIONS),
        "condition_order_policy": "alternate D0/D18 order by task row; three of each order",
        "episode_seed": PAIRED_EPISODE_SEED,
        "model": {"tag": MODEL_TAG, "digest": MODEL_DIGEST, "quantization": MODEL_QUANTIZATION,
                  "historical_digest": HISTORICAL_MODEL_DIGEST,
                  "identity_status": "USER_REPORTED_PENDING_WORKSTATION_VERIFICATION",
                  "identity_source": model.get("identity_source")},
        "inference_settings": dict(MODEL_SETTINGS),
        "retrieval": {"model": SBERT_MODEL, "revision": SBERT_REVISION,
                      "snapshot_sha256": SBERT_SNAPSHOT_SHA256, "top_k": 3,
                      "status": "EXPECTED_IDENTITY_PENDING_WORKSTATION_REHASH"},
        "accum18_library": {"condition": "D18", "skill_count": 18, "sha256": None,
                            "status": "MISSING_FROM_CURRENT_CHECKOUT_REQUIRES_FROZEN_REFERENCE"},
        "tokenizer": {"sha256": None, "revision": None,
                      "status": "PENDING_UNIVERSITY_MODEL_AND_TOKENIZER_EVIDENCE"},
        "data_identity_expectations": {
            "discovery_valid_seen_sha256": EXPECTED_DISCOVERY_IDENTITY,
            "adapter_default_train_valid_seen_sha256": EXPECTED_ADAPTER_IDENTITY,
            "status": "TWO_REPORTED_IDENTITIES_REQUIRE_REPRODUCTION_AND_REVIEW",
        },
        "candidate_source": {
            "candidate_file": candidate_path.name,
            "candidate_sha256": sha256_file(candidate_path),
            "original_proposal_file": original_proposal_path.name,
            "original_proposal_sha256": sha256_file(original_proposal_path),
            "original_proposal_manifest_sha256": proposal.get("manifest_sha256"),
            "original_proposal_repository_commit": proposal.get("repository_commit"),
            "transferred_audit_sha256": sha256_file(transferred_audit_path),
            "archive_sha256": EXPECTED_ARCHIVE_SHA256,
            "model_proposal_file": model_proposal_path.name,
            "model_proposal_sha256": sha256_file(model_proposal_path),
        },
        "task_count": len(tasks),
        "episode_count": 2 * len(tasks),
        "planned_matrix_action_allowance": 2 * len(tasks) * TOTAL_ACTION_BUDGET,
        "planned_matrix_allowance_scope": (
            "one planned attempt per unit; every retry adds up to 200 actions and is recorded as a distinct attempt"
        ),
        "condition_counts": {"D0": len(tasks), "D18": len(tasks)},
        "task_selection": {"source": "the six identities in the existing proposed valid_seen manifest",
                           "task_selection_performed": False, "replacement_permitted": False},
        "tasks": tasks,
        "units": _unit_rows(tasks, None),
        "required_before_engineering_freeze": [
            "Reconcile both ALFWorld identity methods against the actual university dataset and obtain human review",
            "Replay each exact candidate prefix, detour, and unchanged continuation under total cap 200",
            "Verify the frozen Accum-18 file against its previously frozen library reference",
            "Rehash the pinned Sentence-BERT snapshot and verify exact prospective Gemma manifest, blobs, and tokenizer provenance",
            "Confirm current source commit is clean and all four GPU workers can be isolated and placed",
            "Obtain a separate human engineering-only GPU authorization; scientific Option B remains pending",
        ],
    }
    manifest["manifest_sha256"] = _manifest_digest(manifest)
    problems = validate_feasibility_manifest(manifest)
    if problems:
        raise FeasibilityError("generated candidate manifest failed validation: " + "; ".join(problems))
    return manifest


def validate_feasibility_manifest(manifest: Mapping[str, Any], *, require_frozen: bool = False) -> list[str]:
    problems: list[str] = []
    if manifest.get("schema") != SCHEMA or manifest.get("profile_id") != PROFILE_ID:
        return ["unsupported feasibility schema or profile"]
    status = manifest.get("status")
    allowed = {"candidate_unapproved", "engineering_frozen"}
    if status not in allowed:
        problems.append("feasibility status must be candidate_unapproved or engineering_frozen")
    if require_frozen and status != "engineering_frozen":
        problems.append("candidate manifest is not engineering_frozen")
    if manifest.get("execution_scope") != "engineering_feasibility_only":
        problems.append("execution scope must remain engineering_feasibility_only")
    for field in ("scientific_evidence", "scientific_protocol_approved", "scientific_input_freeze"):
        if manifest.get(field) is not False:
            problems.append(f"{field} must remain false for the feasibility profile")
    if manifest.get("option_b_author_decision_status") != "pending":
        problems.append("the author decision for scientific Option B must remain pending")
    if manifest.get("total_action_budget") != TOTAL_ACTION_BUDGET:
        problems.append("feasibility total_action_budget must be exactly 200")
    if manifest.get("conditions") != list(CONDITIONS):
        problems.append("feasibility conditions must be exactly D0 and D18")
    if manifest.get("episode_seed") != PAIRED_EPISODE_SEED:
        problems.append("paired feasibility episode seed must be 11")
    if manifest.get("task_count") != 6 or manifest.get("episode_count") != 12:
        problems.append("engineering feasibility matrix must remain six tasks by two arms")
    if (manifest.get("planned_matrix_action_allowance") != 2400
            or manifest.get("planned_matrix_allowance_scope") != (
                "one planned attempt per unit; every retry adds up to 200 actions and is recorded as a distinct attempt")):
        problems.append("planned 12-unit one-attempt allowance must remain 2400 with retry costs excluded")
    if manifest.get("model") != {
        "tag": MODEL_TAG, "digest": MODEL_DIGEST, "quantization": MODEL_QUANTIZATION,
        "historical_digest": HISTORICAL_MODEL_DIGEST,
        "identity_status": manifest.get("model", {}).get("identity_status"),
        "identity_source": manifest.get("model", {}).get("identity_source"),
    }:
        problems.append("prospective Gemma identity differs from its separate versioned profile")
    if manifest.get("model", {}).get("digest") == HISTORICAL_MODEL_DIGEST:
        problems.append("prospective profile must never spoof the historical model digest")
    if manifest.get("inference_settings") != MODEL_SETTINGS:
        problems.append("inference settings or paired seed policy differ from the reviewed profile")
    if manifest.get("retrieval") != {"model": SBERT_MODEL, "revision": SBERT_REVISION,
                                      "snapshot_sha256": SBERT_SNAPSHOT_SHA256,
                                      "top_k": 3, "status": "EXPECTED_IDENTITY_PENDING_WORKSTATION_REHASH"}:
        if status == "candidate_unapproved":
            problems.append("candidate retrieval identity differs from the pinned Sentence-BERT profile")
        elif (manifest.get("retrieval", {}).get("model") != SBERT_MODEL
              or manifest.get("retrieval", {}).get("revision") != SBERT_REVISION
              or manifest.get("retrieval", {}).get("snapshot_sha256") != SBERT_SNAPSHOT_SHA256
              or manifest.get("retrieval", {}).get("top_k") != 3):
            problems.append("frozen retrieval identity differs from the pinned Sentence-BERT profile")

    tasks = manifest.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 6:
        problems.append("feasibility profile requires exactly six existing candidate tasks")
        tasks = []
    ids: set[str] = set()
    families: list[str] = []
    for task in tasks:
        if not isinstance(task, Mapping):
            problems.append("every task definition must be an object")
            continue
        task_id = str(task.get("task_id", ""))
        if not task_id or task_id in ids:
            problems.append("task IDs must be unique and nonempty")
        ids.add(task_id)
        families.append(str(task.get("task_family", "")))
        prefix, route = task.get("prefix_actions"), task.get("reference_actions")
        if task.get("split") != "valid_seen" or not task_id.startswith("valid_seen:"):
            problems.append(f"{task_id}: only existing valid_seen candidates are allowed")
        if (not isinstance(prefix, list) or not prefix or not isinstance(route, list)
                or not route or route[:len(prefix)] != prefix or len(prefix) >= len(route)):
            problems.append(f"{task_id}: route and prefix are invalid or changed")
            continue
        if len(route) != task.get("reference_route_length"):
            problems.append(f"{task_id}: reference_route_length differs from frozen actions")
        expected_recovery = TOTAL_ACTION_BUDGET - len(prefix) - 1
        if expected_recovery <= 0 or task.get("total_action_budget") != TOTAL_ACTION_BUDGET:
            problems.append(f"{task_id}: invalid 200-action total budget")
        if type(task.get("recovery_action_budget")) is not int or task.get("recovery_action_budget") != expected_recovery:
            problems.append(f"{task_id}: recovery budget must equal 200 - prefix length - 1")
        if len(route) + 1 > TOTAL_ACTION_BUDGET:
            problems.append(f"{task_id}: preserved detoured expert witness exceeds 200 actions")
        if type(task.get("original_protocol_budget")) is not dict or task["original_protocol_budget"].get("total_action_budget") != 50:
            problems.append(f"{task_id}: original 50-action budget provenance is not preserved")
        if task.get("scientifically_frozen") is not False:
            problems.append(f"{task_id}: candidate inputs must not be marked scientifically frozen")
        for field in ("checkpoint_digest", "post_detour_digest"):
            if not isinstance(task.get(field), str) or not _HEX64.fullmatch(task[field]):
                problems.append(f"{task_id}: missing or malformed {field}")
        oracle = task.get("oracle") if isinstance(task.get("oracle"), Mapping) else {}
        if oracle.get("validated") is not False or oracle.get("cap_50_replay_performed") is not False:
            problems.append(f"{task_id}: historical oracle fields must remain explicitly unvalidated")
    if len(ids) == 6 and (len(set(families)) != 6 or set(families) != set(TASK_FAMILIES)):
        problems.append("candidate tasks must retain one task in each frozen family")
    expected_units = _unit_rows(tasks, manifest.get("accum18_library", {}).get("sha256")) if len(tasks) == 6 else []
    units = manifest.get("units")
    if not isinstance(units, list) or len(units) != 12:
        problems.append("two-arm feasibility matrix must contain exactly 12 episodes")
    elif units != expected_units:
        problems.append("unit matrix differs from the deterministic paired D0/D18 construction")
    counts = Counter(str(row.get("condition")) for row in units if isinstance(row, Mapping)) if isinstance(units, list) else Counter()
    if counts != Counter({"D0": 6, "D18": 6}):
        problems.append("matrix must contain six D0 and six D18 units")
    if not isinstance(manifest.get("manifest_sha256"), str) or manifest.get("manifest_sha256") != _manifest_digest(manifest):
        problems.append("feasibility manifest content hash is missing or invalid")
    if status == "engineering_frozen":
        problems.extend(_validate_engineering_freeze(manifest))
    return problems


def _validate_engineering_freeze(manifest: Mapping[str, Any]) -> list[str]:
    problems: list[str] = []
    if not isinstance(manifest.get("run_id"), str) or not _RUN_ID.fullmatch(manifest["run_id"]):
        problems.append("engineering-frozen manifest needs a run_id")
    commit = manifest.get("source_commit")
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        problems.append("engineering-frozen manifest needs the full clean source commit")
    inputs = manifest.get("verified_inputs")
    required = ("data_reconciliation_report_sha256", "data_identity_review_sha256", "oracle_replay_report_sha256",
                "accum18_library_sha256", "accum18_reference_manifest_sha256", "sbert_snapshot_sha256",
                "tokenizer_sha256", "tokenizer_identity_report_sha256", "tokenizer_identity_review_sha256",
                "model_identity_report_sha256")
    if not isinstance(inputs, Mapping):
        problems.append("engineering-frozen manifest lacks verified_inputs")
    else:
        for key in required:
            value = inputs.get(key)
            if not isinstance(value, str) or not _HEX64.fullmatch(value):
                problems.append(f"engineering-frozen manifest has no verified {key}")
    candidate_sha = manifest.get("candidate_manifest_sha256")
    if not isinstance(candidate_sha, str) or not _HEX64.fullmatch(candidate_sha):
        problems.append("engineering-frozen manifest is not bound to its candidate manifest")
    candidate = manifest.get("candidate_manifest")
    if not isinstance(candidate, Mapping):
        problems.append("engineering-frozen manifest lacks original candidate records")
    elif (candidate.get("status") != "candidate_unapproved"
          or _manifest_digest(candidate) != candidate_sha
          or candidate.get("manifest_sha256") != candidate_sha
          or validate_feasibility_manifest(candidate)
          or candidate.get("tasks") != manifest.get("tasks")):
        problems.append("frozen candidate ancestry or original task records differ")
    if manifest.get("tokenizer", {}).get("status") != "REVIEWED_PROSPECTIVE_TOKENIZER":
        problems.append("engineering-frozen manifest lacks reviewed prospective tokenizer provenance")
    if manifest.get("model", {}).get("identity_status") != "REVIEWED_PROSPECTIVE_IDENTITY":
        problems.append("engineering-frozen manifest lacks reviewed prospective model identity")
    if manifest.get("accum18_library", {}).get("status") != "VERIFIED_AGAINST_FROZEN_REFERENCE":
        problems.append("engineering-frozen manifest lacks verified Accum-18 reference identity")
    approval = manifest.get("engineering_approval")
    if not isinstance(approval, Mapping) or approval.get("approval_scope") != "engineering_feasibility_only":
        problems.append("an explicit engineering-only authorization is required")
    elif (approval.get("gpu_inference") is not True or approval.get("scientific_run") is not False
          or approval.get("not_for_scientific_claims") is not True or not approval.get("approved_by")
          or not approval.get("approval_reference") or approval.get("candidate_manifest_sha256") != manifest.get("candidate_manifest_sha256")
          or approval.get("prior_pick_place_exposure_acknowledged") is not True):
        problems.append("engineering authorization is incomplete or attempts to authorize a scientific run")
    return problems


def _hash_without(value: Mapping[str, Any], field: str) -> str:
    return content_hash({key: item for key, item in value.items() if key != field})


def _require_report_hash(report: Mapping[str, Any], field: str, label: str) -> str:
    expected = report.get(field)
    if not isinstance(expected, str) or not _HEX64.fullmatch(expected) or expected != _hash_without(report, field):
        raise FeasibilityError(f"{label} content hash is missing or invalid")
    return expected


def freeze_engineering_manifest(
    root: Path,
    candidate: Mapping[str, Any],
    *,
    run_id: str,
    data_report: Mapping[str, Any],
    data_review: Mapping[str, Any],
    oracle_report: Mapping[str, Any],
    identity_report: Mapping[str, Any],
    identity_review: Mapping[str, Any],
    accum18_library_path: Path,
    accum18_reference_path: Path,
    sbert_cache: Path,
    tokenizer_path: Path,
    engineering_approval: Mapping[str, Any],
) -> dict[str, Any]:
    """Freeze only an engineering run after all external evidence and human reviews bind."""
    candidate_problems = validate_feasibility_manifest(candidate)
    if candidate_problems:
        raise FeasibilityError("candidate manifest failed validation: " + "; ".join(candidate_problems))
    if candidate.get("status") != "candidate_unapproved":
        raise FeasibilityError("only the immutable candidate manifest can be engineering-frozen")
    candidate_sha = str(candidate["manifest_sha256"])
    data_sha = _require_report_hash(data_report, "report_sha256", "data reconciliation report")
    if data_report.get("schema") != "rq1-alfworld-identity-reconciliation-v1" or data_report.get("status") != "REVIEW_REQUIRED":
        raise FeasibilityError("ALFWorld identities must reproduce their expected scopes before human review")
    if (data_report.get("reported_identities_reproduced") is not True
            or data_report.get("selected_candidate_sources_match_both_valid_seen_indexes") is not True):
        raise FeasibilityError("ALFWorld expected identities and selected source hashes must reproduce exactly")
    if (data_review.get("schema") != "rq1-data-identity-review-v1"
            or data_review.get("reviewed_report_sha256") != data_sha
            or data_review.get("decision") != "ACCEPT_SCOPED_DIFFERENCES"
            or not data_review.get("reviewed_by") or not data_review.get("review_reference")):
        raise FeasibilityError("data identity review is missing, unsigned, or bound to a different report")
    data_review_sha = content_hash(data_review)
    if validate_oracle_report(oracle_report, candidate):
        raise FeasibilityError("six candidate cap-200 oracle replays are not verified")
    oracle_sha = _require_report_hash(oracle_report, "report_sha256", "oracle replay report")
    identity_sha = _require_report_hash(identity_report, "content_sha256", "prospective identity report")
    if (identity_report.get("schema") != "rq1-prospective-identity-evidence-v1"
            or identity_report.get("status") != "CANDIDATE_REVIEW_REQUIRED"
            or identity_report.get("approved") is not False
            or identity_report.get("launch_authorized") is not False
            or identity_report.get("model_tag") != MODEL_TAG
            or identity_report.get("model_digest") != MODEL_DIGEST
            or identity_report.get("quantization") != MODEL_QUANTIZATION
            or not identity_report.get("checked_blobs")):
        raise FeasibilityError("prospective identity report is incomplete or appears to authorize launch")
    tokenizer_sha = sha256_file(tokenizer_path)
    if tokenizer_sha != identity_report.get("tokenizer_sha256"):
        raise FeasibilityError("local tokenizer bytes differ from the reviewed identity report")
    if (identity_review.get("schema") != "rq1-prospective-identity-review-v1"
            or identity_review.get("reviewed_report_sha256") != identity_sha
            or identity_review.get("model_tag") != MODEL_TAG
            or identity_review.get("model_digest") != MODEL_DIGEST
            or identity_review.get("quantization") != MODEL_QUANTIZATION
            or identity_review.get("tokenizer_sha256") != tokenizer_sha
            or identity_review.get("tokenizer_revision") != identity_report.get("tokenizer_revision")
            or not re.fullmatch(r"[0-9a-f]{40}", str(identity_review.get("tokenizer_revision", "")))
            or identity_review.get("model_tokenizer_link_confirmed") is not True
            or not identity_review.get("reviewed_by") or not identity_review.get("review_reference")):
        raise FeasibilityError("prospective model/tokenizer review does not bind the exact proposed identity")
    identity_review_sha = content_hash(identity_review)

    library_path = accum18_library_path.expanduser().resolve()
    reference_path = accum18_reference_path.expanduser().resolve()
    library_sha = sha256_file(library_path)
    library = load_json(library_path)
    libraries = library.get("libraries", library) if isinstance(library, Mapping) else {}
    d18 = libraries.get("Accum-18") if isinstance(libraries, Mapping) else None
    skills = d18.get("skills") if isinstance(d18, Mapping) else None
    if not isinstance(skills, list) or len(skills) != 18:
        raise FeasibilityError("the genuine Accum-18 library must contain exactly 18 skill entries")
    skill_ids = [str(skill.get("skill_id", "")) for skill in skills if isinstance(skill, Mapping)]
    if len(skill_ids) != 18 or len(set(skill_ids)) != 18 or any(not item for item in skill_ids):
        raise FeasibilityError("Accum-18 skill IDs must be 18 distinct nonempty values")
    reference = load_json(reference_path)
    if (not isinstance(reference, Mapping)
            or reference.get("schema") != "rq1-accum18-frozen-reference-v1"
            or reference.get("status") != "verified_frozen"
            or reference.get("reference_sha256") != _hash_without(reference, "reference_sha256")
            or reference.get("library_file_sha256") != library_sha
            or reference.get("skill_count") != 18
            or reference.get("skill_ids") != sorted(skill_ids)):
        raise FeasibilityError("Accum-18 does not match its independently supplied frozen reference manifest")
    verify_accum18_reference(library_path, reference)
    reference_sha = sha256_file(reference_path)

    from rq1.acquisition.environment import sbert_snapshot_sha256
    observed_sbert = sbert_snapshot_sha256(sbert_cache.expanduser().resolve())
    if observed_sbert != SBERT_SNAPSHOT_SHA256:
        raise FeasibilityError("local Sentence-BERT snapshot is missing or differs from the pinned revision")

    if not isinstance(run_id, str) or not _RUN_ID.fullmatch(run_id):
        raise FeasibilityError("run_id must be unique and scoped to rq1-feasibility-200-v1")
    try:
        commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True,
                                text=True, check=True, timeout=15).stdout.strip()
        status = subprocess.run(["git", "-C", str(root), "status", "--porcelain"], capture_output=True,
                                text=True, check=True, timeout=15).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise FeasibilityError(f"clean committed source provenance is unavailable: {exc}") from exc
    if not re.fullmatch(r"[0-9a-f]{40}", commit) or status.strip():
        raise FeasibilityError("source checkout must be clean at the reviewed feasibility implementation commit")
    if (engineering_approval.get("approval_scope") != "engineering_feasibility_only"
            or engineering_approval.get("gpu_inference") is not True
            or engineering_approval.get("scientific_run") is not False
            or engineering_approval.get("not_for_scientific_claims") is not True
            or engineering_approval.get("candidate_manifest_sha256") != candidate_sha
            or engineering_approval.get("prior_pick_place_exposure_acknowledged") is not True
            or not engineering_approval.get("approved_by")
            or not engineering_approval.get("approval_reference")):
        raise FeasibilityError("a separate engineering-only GPU approval bound to the candidate is required")
    from rq1.university_gpu.authorization import verify_engineering_authorization
    try:
        verify_engineering_authorization(engineering_approval, candidate_sha256=candidate_sha,
            model_tag=str(candidate.get("model", {}).get("tag", "")),
            model_digest=str(candidate.get("model", {}).get("digest", "")), profile_id=PROFILE_ID,
            action_budget=TOTAL_ACTION_BUDGET, run_id=run_id, model_quantization=MODEL_QUANTIZATION)
    except RuntimeError as exc:
        raise FeasibilityError(str(exc)) from exc

    frozen = copy.deepcopy(dict(candidate))
    frozen["status"] = "engineering_frozen"
    frozen["run_id"] = run_id
    frozen["source_commit"] = commit
    frozen["candidate_manifest_sha256"] = candidate_sha
    frozen["candidate_manifest"] = copy.deepcopy(dict(candidate))
    frozen["model"]["identity_status"] = "REVIEWED_PROSPECTIVE_IDENTITY"
    frozen["retrieval"]["status"] = "VERIFIED_LOCAL_SNAPSHOT"
    frozen["accum18_library"] = {"condition": "D18", "skill_count": 18, "sha256": library_sha,
                                  "status": "VERIFIED_AGAINST_FROZEN_REFERENCE",
                                  "reference_manifest_sha256": reference_sha, "skill_ids": sorted(skill_ids)}
    frozen["tokenizer"] = {"sha256": tokenizer_sha, "revision": identity_review.get("tokenizer_revision"),
                           "status": "REVIEWED_PROSPECTIVE_TOKENIZER"}
    frozen["verified_inputs"] = {
        "data_reconciliation_report_sha256": data_sha,
        "data_identity_review_sha256": data_review_sha,
        "oracle_replay_report_sha256": oracle_sha,
        "accum18_library_sha256": library_sha,
        "accum18_reference_manifest_sha256": reference_sha,
        "sbert_snapshot_sha256": observed_sbert,
        "tokenizer_sha256": tokenizer_sha,
        "tokenizer_identity_report_sha256": identity_sha,
        "tokenizer_identity_review_sha256": identity_review_sha,
        "model_identity_report_sha256": identity_sha,
    }
    frozen["engineering_approval"] = dict(engineering_approval)
    frozen["units"] = _unit_rows(frozen["tasks"], library_sha)
    frozen["manifest_sha256"] = _manifest_digest(frozen)
    problems = validate_feasibility_manifest(frozen, require_frozen=True)
    if problems:
        raise FeasibilityError("engineering-frozen manifest failed validation: " + "; ".join(problems))
    return frozen


def build_accum18_reference(library_path: Path, frozen_manifest_path: Path | None = None, *,
                            archive_path: Path | None = None, archive_manifest_path: Path | None = None) -> dict[str, Any]:
    """Bind Accum-18 only to the pinned historical archive and member ledger."""
    if archive_path is not None or archive_manifest_path is not None:
        if frozen_manifest_path is not None or archive_path is None or archive_manifest_path is None:
            raise FeasibilityError("supply either frozen manifest or both historical archive and member ledger")
        return _historical_accum18_reference(library_path, archive_path, archive_manifest_path)
    raise FeasibilityError("Accum-18 provenance requires the pinned historical archive and original member ledger")


def verify_accum18_reference(library_path: Path, reference: Mapping[str, Any]) -> None:
    if reference.get("source_kind") == "historical_archive":
        actual = build_accum18_reference(library_path, archive_path=Path(str(reference.get("archive_path", ""))),
                                         archive_manifest_path=Path(str(reference.get("archive_manifest_path", ""))))
    else:
        raise FeasibilityError("Accum-18 reference lacks authenticated source ancestry")
    if actual != reference:
        raise FeasibilityError("Accum-18 reference differs from reverified source ancestry")


def _historical_accum18_reference(library_path: Path, archive_path: Path, ledger_path: Path) -> dict[str, Any]:
    """Verify exact library bytes against pinned historical archive and its member ledger."""
    if any(path.is_symlink() or not path.is_file() for path in (library_path, archive_path, ledger_path)):
        raise FeasibilityError("historical library/archive/ledger must be existing regular files")
    if sha256_file(archive_path) != HISTORICAL_EVALUATION_ARCHIVE_SHA256:
        raise FeasibilityError("historical evaluation archive hash differs from frozen reference")
    ledger = load_json(ledger_path)
    if ledger.get("archive_sha256") != HISTORICAL_EVALUATION_ARCHIVE_SHA256:
        raise FeasibilityError("historical archive ledger identity differs")
    rows = [r for r in ledger.get("members", []) if r.get("path") == HISTORICAL_LIBRARY_MEMBER]
    if len(rows) != 1:
        raise FeasibilityError("historical ledger must contain exactly one original library member")
    with tarfile.open(archive_path, "r:gz") as archive:
        members = [m for m in archive.getmembers() if m.name == HISTORICAL_LIBRARY_MEMBER]
        if len(members) != 1 or not members[0].isfile():
            raise FeasibilityError("historical library member is missing, duplicate or not regular")
        stream = archive.extractfile(members[0])
        if stream is None:
            raise FeasibilityError("historical library member is unreadable")
        data = stream.read()
    from rq1.university_gpu.models import sha256_bytes
    digest = sha256_bytes(data)
    if (digest != rows[0].get("sha256") or len(data) != rows[0].get("size")
            or digest != sha256_file(library_path) or data != library_path.read_bytes()):
        raise FeasibilityError("library bytes differ from authenticated historical archive/member ledger")
    payload = json.loads(data)
    skills = payload.get("libraries", payload).get("Accum-18", {}).get("skills", [])
    ids = [s.get("skill_id") for s in skills if isinstance(s, Mapping)]
    if (len(skills) != 18 or len(ids) != 18 or any(not isinstance(i, str) or not i for i in ids)
            or len(set(ids)) != 18 or any(not str(s.get("text", "")).strip() for s in skills)):
        raise FeasibilityError("authenticated historical Accum-18 must have 18 distinct IDs and nonempty texts")
    result = {"schema": "rq1-accum18-frozen-reference-v1", "status": "verified_frozen",
              "source_kind": "historical_archive", "library_file_sha256": digest,
              "skill_count": 18, "skill_ids": sorted(ids), "archive_sha256": HISTORICAL_EVALUATION_ARCHIVE_SHA256,
              "archive_member": HISTORICAL_LIBRARY_MEMBER, "archive_path": str(archive_path.resolve()),
              "archive_manifest_path": str(ledger_path.resolve()), "archive_manifest_sha256": sha256_file(ledger_path)}
    result["reference_sha256"] = content_hash(result)
    return result


def check_selected_task_sources(data_root: Path, tasks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Rehash only the six selected traj/game pairs and compare to archived candidates."""
    root = data_root.expanduser().resolve()
    rows: list[dict[str, Any]] = []
    for task in tasks:
        source = task.get("source_task") if isinstance(task.get("source_task"), Mapping) else {}
        relative = str(source.get("source_identifier", ""))
        if not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
            rows.append({"task_id": task.get("task_id"), "status": "FAIL", "reason": "unsafe source identifier"})
            continue
        base = root / "json_2.1.1" / "valid_seen" / Path(relative)
        trajectory, game = base / "traj_data.json", base / "game.tw-pddl"
        try:
            observed_source, observed_game = sha256_file(trajectory), sha256_file(game)
            expected_source, expected_game = source.get("source_sha256"), source.get("game_sha256")
            equal = observed_source == expected_source and observed_game == expected_game
            rows.append({"task_id": task.get("task_id"), "source_sha256": observed_source,
                         "expected_source_sha256": expected_source, "game_sha256": observed_game,
                         "expected_game_sha256": expected_game, "status": "PASS" if equal else "FAIL"})
        except OSError as exc:
            rows.append({"task_id": task.get("task_id"), "status": "BLOCKED", "error": f"{type(exc).__name__}: {exc}"})
    return {"status": "PASS" if len(rows) == len(tasks) and all(item["status"] == "PASS" for item in rows) else "BLOCKED",
            "selected_task_count": len(tasks), "rows": rows}


def reconcile_data_identity(data_root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Compute both existing identities, compare scopes/records, and expose every mismatch."""
    discovery = discover_tasks(data_root, "valid_seen")
    adapter_default = build_task_index(data_root)
    adapter_seen = build_task_index(data_root, splits=("valid_seen",))
    discovery_rows = {row.task_id: {"task_family": row.family, "source_sha256": row.source_sha256,
                                    "game_sha256": row.game_sha256, "source_identifier": row.source_identifier}
                      for row in discovery.records}

    def adapter_rows(index: Any) -> dict[str, dict[str, Any]]:
        return {row.task_id: {"task_family": row.task_family, "source_sha256": row.source_sha256,
                              "game_sha256": row.game_sha256,
                              "source_identifier": row.task_id.split(":", 1)[1]}
                for row in index.entries}

    default_rows = adapter_rows(adapter_default)
    seen_rows = adapter_rows(adapter_seen)

    def compare(left: Mapping[str, Mapping[str, Any]], right: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
        left_ids, right_ids = set(left), set(right)
        common = left_ids & right_ids
        mismatches = [task_id for task_id in sorted(common)
                      if any(left[task_id].get(field) != right[task_id].get(field)
                             for field in ("source_sha256", "game_sha256", "task_family", "source_identifier"))]
        file_hash_mismatches = [task_id for task_id in sorted(common)
                                if left[task_id].get("source_sha256") != right[task_id].get("source_sha256")
                                or left[task_id].get("game_sha256") != right[task_id].get("game_sha256")]
        return {"left_count": len(left_ids), "right_count": len(right_ids),
                "only_left": sorted(left_ids - right_ids), "only_right": sorted(right_ids - left_ids),
                "file_hash_mismatches": [{"task_id": task_id, "left": dict(left[task_id]), "right": dict(right[task_id])}
                                          for task_id in file_hash_mismatches],
                "record_mismatches": [{"task_id": task_id, "left": dict(left[task_id]), "right": dict(right[task_id])}
                                       for task_id in mismatches],
                "same_task_ids": left_ids == right_ids,
                "same_file_hashes_for_common_ids": not file_hash_mismatches,
                "same_full_records_for_common_ids": not mismatches}

    expected = manifest.get("data_identity_expectations") or {}
    candidate_tasks = manifest.get("tasks") or []
    selected_sources = check_selected_task_sources(data_root, candidate_tasks)
    expected_discovery = expected.get("discovery_valid_seen_sha256")
    expected_adapter = expected.get("adapter_default_train_valid_seen_sha256")
    reported_match = discovery.data_root_identity == expected_discovery and adapter_default.identity == expected_adapter
    selected_match = selected_sources.get("status") == "PASS" and all(
        task.get("task_id") in discovery_rows and task.get("task_id") in seen_rows
        and discovery_rows[task["task_id"]]["source_sha256"] == task["source_task"]["source_sha256"]
        and discovery_rows[task["task_id"]]["game_sha256"] == task["source_task"]["game_sha256"]
        and seen_rows[task["task_id"]]["source_sha256"] == task["source_task"]["source_sha256"]
        and seen_rows[task["task_id"]]["game_sha256"] == task["source_task"]["game_sha256"]
        for task in candidate_tasks
    )
    discovery_adapter_seen = compare(discovery_rows, seen_rows)
    adapter_default_vs_discovery = compare(default_rows, discovery_rows)
    if reported_match and selected_match:
        code = "SCOPED_IDENTITIES_REPRODUCED"
        status = "REVIEW_REQUIRED"
    else:
        code = "UNRESOLVED_IDENTITY_MISMATCH"
        status = "BLOCKED"
    report: dict[str, Any] = {
        "schema": "rq1-alfworld-identity-reconciliation-v1",
        "status": status,
        "reconciliation_code": code,
        "authoritative_freeze": False,
        "human_review_required": True,
        "input_root_recorded": False,
        "expected_identities": {"discovery_valid_seen_sha256": expected_discovery,
                                "adapter_default_train_valid_seen_sha256": expected_adapter},
        "observed_identities": {"discovery_valid_seen_sha256": discovery.data_root_identity,
                                "adapter_default_train_valid_seen_sha256": adapter_default.identity,
                                "adapter_valid_seen_only_sha256": adapter_seen.identity},
        "identity_digest_equal": discovery.data_root_identity == adapter_default.identity,
        "discovery_and_valid_seen_adapter_share_ids": discovery_adapter_seen["same_task_ids"],
        "discovery_and_valid_seen_adapter_hashes_match": discovery_adapter_seen["same_file_hashes_for_common_ids"],
        "reported_identities_reproduced": reported_match,
        "selected_candidate_sources_match_both_valid_seen_indexes": selected_match,
        "identity_algorithms": {
            "discovery": "valid_seen task_id\\0traj_sha256\\0game_sha256\\n, sorted by task_id; includes task/game pairs and does not filter game.solvable",
            "adapter": "relative_task_path\\0traj_sha256\\0game_sha256\\n over train and valid_seen by default; filters missing/unsolvable games and unmapped families",
            "serialization_note": "for valid_seen-only matching rows, discovery uses split-prefixed task_id while adapter uses split-relative path; the byte streams and resulting hashes are intentionally reported separately",
        },
        "comparisons": {"discovery_vs_adapter_valid_seen_scope": discovery_adapter_seen,
                        "discovery_valid_seen_vs_adapter_default_scope": adapter_default_vs_discovery},
        "adapter_default_split_counts": dict(sorted(Counter(row.split for row in adapter_default.entries).items())),
        "discovery_exclusions_missing_game": list(discovery.exclusions),
        "selected_source_hash_checks": selected_sources,
        "limitations": [
            "The identities are not coerced into equality; scope, eligibility, path key, and every record mismatch remain visible.",
            "This report does not mark the task candidates scientifically frozen or authorize execution.",
            "A human reviewer must inspect the full record differences and bind this report hash to the engineering run record.",
        ],
    }
    report["report_sha256"] = content_hash(report)
    return report


def write_reconciliation_report(data_root: Path, manifest: Mapping[str, Any], output: Path) -> dict[str, Any]:
    report = reconcile_data_identity(data_root, manifest)
    atomic_write_json(output, report, overwrite=False)
    return report


def validate_oracle_report(report: Mapping[str, Any], manifest: Mapping[str, Any]) -> list[str]:
    problems: list[str] = []
    if report.get("schema") != "rq1-feasibility-oracle-replay-v1" or report.get("manifest_sha256") != (manifest.get("candidate_manifest_sha256") if manifest.get("status") == "engineering_frozen" else manifest.get("manifest_sha256")):
        problems.append("oracle report schema or candidate manifest binding differs")
    rows = report.get("tasks")
    expected = {str(task["task_id"]): task for task in manifest.get("tasks", [])}
    if not isinstance(rows, list) or len(rows) != 6:
        return problems + ["oracle replay report must contain exactly six task rows"]
    actual_ids = {str(row.get("task_id")) for row in rows if isinstance(row, Mapping)}
    if actual_ids != set(expected):
        problems.append("oracle replay rows do not exactly cover the six candidate task IDs")
    for row in rows:
        if not isinstance(row, Mapping):
            problems.append("oracle replay row is not an object")
            continue
        task = expected.get(str(row.get("task_id")))
        if task is None:
            continue
        if row.get("validated") is not True or row.get("total_action_limit") != TOTAL_ACTION_BUDGET:
            problems.append(f"{task['task_id']}: cap-200 oracle replay did not pass")
        expected_actions = (list(task.get("prefix_actions", [])) + [str(task.get("detour_action"))]
                            + list(task.get("reference_actions", [])[len(task.get("prefix_actions", [])):]))
        if (row.get("checkpoint_digest") != task.get("checkpoint_digest")
                or row.get("post_detour_digest") != task.get("post_detour_digest")
                or row.get("detour_action") != task.get("detour_action")
                or row.get("oracle_checkpoint_digest") != task.get("checkpoint_digest")
                or row.get("oracle_post_detour_digest") != task.get("post_detour_digest")
                or row.get("oracle_success") is not True
                or row.get("action_sequence_sha256") != content_hash(expected_actions)
                or not _HEX64.fullmatch(str(row.get("oracle_event_log_sha256", "")))
                or row.get("split") != "valid_seen" or row.get("seed") != PAIRED_EPISODE_SEED
                or row.get("total_actions") != len(expected_actions)
                or row.get("total_actions", TOTAL_ACTION_BUDGET + 1) > TOTAL_ACTION_BUDGET):
            problems.append(f"{task['task_id']}: oracle state/action evidence differs from the exact candidate")
    if (report.get("status") != "PASS" or report.get("model_calls") != 0
            or report.get("gpu_inference") is not False or report.get("scientific_evidence") is not False):
        problems.append("oracle replay must be model-free and non-scientific")
    return problems


def replay_candidate_oracles(
    root: Path,
    data_root: Path,
    manifest: Mapping[str, Any],
    *,
    evidence_dir: Path,
    output: Path,
    driver_factory: Callable[..., Any] | None = None,
    harness_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Run six actual ALFWorld reference replays without model inference or GPU use."""
    problems = validate_feasibility_manifest(manifest)
    if problems:
        raise FeasibilityError("candidate manifest failed validation: " + "; ".join(problems))
    root = root.expanduser().resolve()
    evidence_dir = evidence_dir.expanduser().resolve()
    output = output.expanduser().resolve()
    for path, label in ((evidence_dir, "oracle raw logs"), (output, "oracle report")):
        try:
            path.relative_to(root)
        except ValueError:
            continue
        raise FeasibilityError(f"{label} must be stored outside the Git checkout")
    if evidence_dir.exists() or output.exists():
        raise FileExistsError("oracle evidence and report paths must be new; historical outputs are never overwritten")
    source_check = check_selected_task_sources(data_root, manifest["tasks"])
    if source_check["status"] != "PASS":
        raise FeasibilityError("selected ALFWorld source hashes do not match candidate definitions")
    if driver_factory is None:
        from rq1.hermes.episode_driver import RealEpisodeDriver
        driver_factory = RealEpisodeDriver
    if harness_factory is None:
        from rq1.pilot.real_runtime.harnesses import RealRecoveryHarness
        harness_factory = RealRecoveryHarness
    from rq1.pilot.real_runtime.harnesses import bridge_state_digest
    evidence_dir.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = []
    driver = driver_factory(root, data_dir=data_root, model_name=MODEL_TAG, ollama_url="http://127.0.0.1:9",
                            bridge_log_root=evidence_dir / "bridge")
    # The driver bridge only hosts the ALFWorld tool. Oracle mode never issues Ollama requests.
    driver.__enter__()
    try:
        for index, task in enumerate(manifest["tasks"], 1):
            row: dict[str, Any] = {"task_id": task["task_id"], "task_family": task["task_family"],
                                   "total_action_limit": TOTAL_ACTION_BUDGET, "model_calls": 0,
                                   "scientific_evidence": False, "validated": False}
            task_log = evidence_dir / f"task-{index:02d}-{content_hash(task['task_id'])[:10]}"
            try:
                harness = harness_factory(
                    driver, output_dir=task_log, run_id=f"{PROFILE_ID}-oracle-{index:02d}",
                    attempt_id=uuid4().hex, reference_actions=task["reference_actions"],
                    total_action_limit=TOTAL_ACTION_BUDGET, allowed_split="valid_seen",
                    profile_prefix=PROFILE_ID,
                    frozen_perturbation={"checkpoint_digest": task["checkpoint_digest"],
                                         "detour_action": task["detour_action"],
                                         "post_detour_digest": task["post_detour_digest"]},
                )
                try:
                    harness.start_and_replay(task["task_id"], "valid_seen", PAIRED_EPISODE_SEED, task["prefix_actions"])
                    perturbation = harness.apply_controlled_action_perturbation(task["checkpoint_id"])
                    actual_checkpoint = harness.checkpoint_digest
                    actual_post = perturbation.post_state_digest
                    if (perturbation.action != task["detour_action"]
                            or actual_checkpoint != task["checkpoint_digest"]
                            or actual_post != task["post_detour_digest"]):
                        raise FeasibilityError("candidate checkpoint or detour state failed final-runtime replay")
                finally:
                    harness.close()

                # Re-run the unchanged expert continuation from the same task, seed,
                # checkpoint, and detour. This explicitly records the successful
                # witness and its full prefix + detour + recovery action count.
                oracle_dir = task_log / "solvability-oracle"
                oracle_attempt = uuid4().hex
                with driver.session(output_dir=oracle_dir, run_id=f"{PROFILE_ID}-oracle-witness-{index:02d}",
                                    attempt_id=oracle_attempt, profile=f"{PROFILE_ID}-oracle-witness") as oracle:
                    oracle.start(task["task_id"], "valid_seen", PAIRED_EPISODE_SEED, TOTAL_ACTION_BUDGET)
                    oracle.apply_scripted(task["prefix_actions"], phase="oracle_checkpoint_replay")
                    oracle_checkpoint = bridge_state_digest(oracle.state or {})
                    detour_result = oracle.step(task["detour_action"], phase="oracle_controlled_detour")
                    oracle_post = bridge_state_digest(oracle.state or {})
                    continuation = task["reference_actions"][len(task["prefix_actions"]):]
                    for action_index, action in enumerate(continuation):
                        result = oracle.step(action, phase="oracle_reference_completion")
                        if result.get("action_valid") is not True:
                            raise FeasibilityError("unchanged expert continuation contains an invalid action")
                        if result.get("done") and action_index != len(continuation) - 1:
                            raise FeasibilityError("expert witness terminated before its final action")
                    expected_actions = list(task["prefix_actions"]) + [task["detour_action"]] + list(continuation)
                    actual_actions = [record.action for record in oracle.records]
                    oracle_success = bool(oracle.state and oracle.state.get("done") is True
                                          and oracle.state.get("success") is True)
                    checks = {
                        "detour_valid": detour_result.get("action_valid") is True and detour_result.get("done") is not True,
                        "checkpoint_matches": oracle_checkpoint == task["checkpoint_digest"],
                        "post_detour_matches": oracle_post == task["post_detour_digest"],
                        "action_sequence_matches": actual_actions == expected_actions,
                        "terminal_success": oracle_success,
                        "within_total_action_cap": len(actual_actions) <= TOTAL_ACTION_BUDGET,
                        "actual_actions": len(actual_actions),
                        "expected_actions": len(expected_actions),
                        "terminal_state": dict(oracle.state or {}),
                    }
                    if not all(checks.values()):
                        raise FeasibilityError("exact expert witness failed cap-200 checks: "
                                               + json.dumps(checks, sort_keys=True))
                    total_actions = len(actual_actions)
                oracle_event_log = oracle_dir / "episode-events.jsonl"
                if not oracle_event_log.is_file():
                    raise FeasibilityError("oracle witness completed without a durable raw event log")
                row.update({"checkpoint_digest": actual_checkpoint, "post_detour_digest": actual_post,
                            "detour_action": perturbation.action,
                            "oracle_checkpoint_digest": oracle_checkpoint,
                            "oracle_post_detour_digest": oracle_post,
                            "oracle_success": oracle_success,
                            "split": "valid_seen", "seed": PAIRED_EPISODE_SEED,
                            "total_actions": total_actions,
                            "reference_route_length": len(task["reference_actions"]),
                            "action_sequence_sha256": content_hash(actual_actions),
                            "oracle_event_log_sha256": sha256_file(oracle_event_log),
                            "status": "PASS", "validated": True,
                            "evidence_dir": task_log.name})
            except Exception as exc:
                row.update({"status": "BLOCKED", "error_type": type(exc).__name__, "error": str(exc),
                            "evidence_dir": task_log.name})
            rows.append(row)
    finally:
        driver.__exit__(None, None, None)
    report: dict[str, Any] = {
        "schema": "rq1-feasibility-oracle-replay-v1",
        "profile_id": PROFILE_ID,
        "status": "PASS" if all(row.get("validated") is True for row in rows) else "BLOCKED",
        "manifest_sha256": manifest["manifest_sha256"],
        "data_source_check": source_check,
        "task_count": len(rows),
        "total_action_limit": TOTAL_ACTION_BUDGET,
        "tasks": rows,
        "model_calls": 0,
        "gpu_inference": False,
        "scientific_evidence": False,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    report["report_sha256"] = content_hash({key: value for key, value in report.items() if key != "report_sha256"})
    validation = validate_oracle_report(report, manifest)
    if validation:
        report["validation_problems"] = validation
        report["status"] = "BLOCKED"
    atomic_write_json(output, report, overwrite=False)
    return report


def _report_file_matches(path: Path, expected: str, field: str, label: str) -> dict[str, Any]:
    value = load_json(path)
    observed = _require_report_hash(value, field, label)
    if observed != expected:
        raise FeasibilityError(f"{label} differs from the hash bound into the frozen run manifest")
    return value


def verify_frozen_feasibility_assets(root: Path, config: Mapping[str, Any], manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Recheck all local, non-GPU inputs immediately before a feasibility run."""
    problems = validate_feasibility_manifest(manifest, require_frozen=True)
    if problems:
        raise FeasibilityError("frozen feasibility manifest is invalid: " + "; ".join(problems))
    inputs = manifest["verified_inputs"]
    data_root = Path(str(config["alfworld_data_dir"]))
    data_report = _report_file_matches(Path(str(config["data_reconciliation_report_path"])),
                                       inputs["data_reconciliation_report_sha256"], "report_sha256", "data reconciliation report")
    review = load_json(Path(str(config["data_identity_review_path"])))
    if content_hash(review) != inputs["data_identity_review_sha256"] or review.get("reviewed_report_sha256") != data_report["report_sha256"]:
        raise FeasibilityError("data identity review differs from the human-reviewed frozen evidence")
    oracle_report = _report_file_matches(Path(str(config["oracle_replay_report_path"])),
                                         inputs["oracle_replay_report_sha256"], "report_sha256", "oracle replay report")
    if validate_oracle_report(oracle_report, manifest):
        raise FeasibilityError("the cap-200 oracle replay report failed revalidation")
    identity_report = _report_file_matches(Path(str(config["model_identity_report_path"])),
                                           inputs["model_identity_report_sha256"], "content_sha256", "prospective identity report")
    identity_review = load_json(Path(str(config["model_identity_review_path"])))
    if (content_hash(identity_review) != inputs["tokenizer_identity_review_sha256"]
            or identity_review.get("reviewed_report_sha256") != identity_report.get("content_sha256")):
        raise FeasibilityError("model/tokenizer identity review differs from frozen evidence")

    library_path = Path(str(config["accum18_library_path"]))
    library_ref_path = Path(str(config["accum18_reference_manifest_path"]))
    observed_library = sha256_file(library_path)
    if observed_library != inputs["accum18_library_sha256"]:
        raise FeasibilityError("Accum-18 file hash differs from frozen run identity")
    reference = load_json(library_ref_path)
    verify_accum18_reference(library_path, reference)
    if (sha256_file(library_ref_path) != inputs["accum18_reference_manifest_sha256"]
            or reference.get("library_file_sha256") != observed_library
            or reference.get("skill_ids") != manifest["accum18_library"].get("skill_ids")):
        raise FeasibilityError("Accum-18 reference manifest differs from the frozen run identity")
    from rq1.acquisition.environment import sbert_snapshot_sha256
    observed_sbert = sbert_snapshot_sha256(Path(str(config["sbert_cache"])))
    if observed_sbert != inputs["sbert_snapshot_sha256"] or observed_sbert != SBERT_SNAPSHOT_SHA256:
        raise FeasibilityError("local Sentence-BERT snapshot differs from the reviewed frozen identity")

    tokenizer_path = Path(str(config["tokenizer_path"]))
    observed_tokenizer = sha256_file(tokenizer_path)
    if observed_tokenizer != inputs["tokenizer_sha256"] or observed_tokenizer != manifest["tokenizer"].get("sha256"):
        raise FeasibilityError("local tokenizer file differs from the reviewed frozen identity")
    revision = str(config.get("tokenizer_revision", ""))
    if (revision != manifest["tokenizer"].get("revision")
            or revision != identity_report.get("tokenizer_revision")
            or revision != identity_review.get("tokenizer_revision")):
        raise FeasibilityError("tokenizer revision differs from the reviewed frozen identity")
    tokenizer = LocalGemmaTokenizer(TokenizerIdentity(tokenizer_path, observed_tokenizer, revision))
    provenance = load_json(Path(str(config["tokenizer_provenance_path"])))
    if provenance.get("revision") != revision:
        raise FeasibilityError("tokenizer provenance revision differs from runtime/review/report")
    from rq1.university_gpu.prospective_identity import verify_candidate
    evidence_report = verify_candidate(
        Path(str(config["ollama_manifest_path"])), Path(str(config["ollama_blobs_dir"])), tokenizer_path,
        provenance, tokenizer.encode, Path(str(config["model_tokenizer_link_evidence_path"])),
        expected_model_digest=MODEL_DIGEST,
    )
    if evidence_report.get("content_sha256") != identity_report.get("content_sha256"):
        raise FeasibilityError("recomputed model/tokenizer blob verification differs from the reviewed report")

    # Recompute scoped data identity and selected task hashes to expose drift.
    actual_data_report = reconcile_data_identity(data_root, manifest)
    if actual_data_report.get("report_sha256") != data_report.get("report_sha256"):
        raise FeasibilityError("ALFWorld identity or task records changed since review")
    source_check = check_selected_task_sources(data_root, manifest["tasks"])
    if source_check.get("status") != "PASS":
        raise FeasibilityError("one or more selected task/game source hashes differ")
    provenance = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True,
                                text=True, check=True, timeout=15).stdout.strip()
    status = subprocess.run(["git", "-C", str(root), "status", "--porcelain"], capture_output=True,
                            text=True, check=True, timeout=15).stdout
    if provenance != manifest.get("source_commit") or status.strip():
        raise FeasibilityError("launch source must be the exact clean commit bound to the manifest")
    return {"status": "PASS", "profile_id": PROFILE_ID, "scientific_evidence": False,
            "verified_inputs": dict(inputs), "selected_task_sources": source_check,
            "data_identity_report_sha256": data_report["report_sha256"],
            "model_identity_report_sha256": identity_report["content_sha256"],
            "accum18_library_sha256": observed_library, "sbert_snapshot_sha256": observed_sbert,
            "tokenizer_sha256": observed_tokenizer, "source_commit": provenance}


def _feasibility_smoke(endpoints: Sequence[str], approval: Mapping[str, Any]) -> dict[str, Any]:
    if (approval.get("approval_scope") != "engineering_feasibility_only"
            or approval.get("gpu_inference") is not True or approval.get("scientific_run") is not False
            or approval.get("not_for_scientific_claims") is not True):
        raise FeasibilityError("four-worker smoke inference requires explicit engineering-only GPU approval")
    if len(endpoints) != 4:
        raise FeasibilityError("the GPU benchmark requires exactly four isolated endpoints")
    from urllib.request import Request, urlopen
    def smoke_one(endpoint: str) -> dict[str, Any]:
        request = Request(endpoint.rstrip("/") + "/api/chat", data=json.dumps({
            "model": MODEL_TAG,
            "messages": [{"role": "user", "content": "Reply exactly: RQ1_READY"}],
            "stream": False,
            "think": False,
            "options": {"temperature": 0, "seed": MODEL_SETTINGS["smoke_seed"], "num_predict": 8,
                        "num_ctx": MODEL_SETTINGS["num_ctx"]},
        }).encode("utf-8"), method="POST", headers={"Content-Type": "application/json"})
        started = time.monotonic()
        try:
            with urlopen(request, timeout=180) as response:
                payload = json.loads(response.read().decode("utf-8"))
            text = str((payload.get("message") or {}).get("content") or "").strip()
            return {"endpoint": endpoint, "status": "PASS" if text == "RQ1_READY" else "FAIL",
                    "latency_seconds": round(time.monotonic() - started, 3), "response": text}
        except Exception as exc:
            return {"endpoint": endpoint, "status": "FAIL", "error_type": type(exc).__name__,
                    "error": str(exc), "latency_seconds": round(time.monotonic() - started, 3)}
    with ThreadPoolExecutor(max_workers=min(4, len(endpoints))) as pool:
        rows = list(pool.map(smoke_one, endpoints))
    return {"schema": "rq1-feasibility-worker-smoke-v1",
            "status": "PASS" if len(rows) == 4 and all(row["status"] == "PASS" for row in rows) else "FAIL",
            "profile_id": PROFILE_ID, "scientific_evidence": False,
            "approval_reference": approval.get("approval_reference"), "inference_count": len(rows), "workers": rows}


def _validate_engineering_approval(manifest: Mapping[str, Any], approval: Mapping[str, Any], *, yes: bool) -> None:
    if not yes:
        raise FeasibilityError("GPU-consuming feasibility command requires explicit --yes")
    if (not isinstance(manifest.get("engineering_approval"), Mapping)
            or dict(manifest["engineering_approval"]) != dict(approval)
            or approval.get("candidate_manifest_sha256") != manifest.get("candidate_manifest_sha256")
            or approval.get("approval_scope") != "engineering_feasibility_only"
            or approval.get("gpu_inference") is not True or approval.get("scientific_run") is not False
            or approval.get("not_for_scientific_claims") is not True
            or approval.get("prior_pick_place_exposure_acknowledged") is not True
            or not approval.get("approved_by") or not approval.get("approval_reference")):
        raise FeasibilityError("GPU command requires the exact candidate-bound engineering-only approval")
    from rq1.university_gpu.authorization import verify_engineering_authorization, verify_frozen_manifest_anchor
    try:
        verify_engineering_authorization(approval, candidate_sha256=str(manifest.get("candidate_manifest_sha256", "")),
            model_tag=str(manifest.get("model", {}).get("tag", "")),
            model_digest=str(manifest.get("model", {}).get("digest", "")), profile_id=PROFILE_ID,
            action_budget=TOTAL_ACTION_BUDGET, run_id=str(manifest.get("run_id", "")),
            model_quantization=MODEL_QUANTIZATION)
        verify_frozen_manifest_anchor(manifest)
    except RuntimeError as exc:
        raise FeasibilityError(str(exc)) from exc


def run_feasibility_benchmark(
    root: Path,
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    approval: Mapping[str, Any],
    yes: bool,
    output: Path,
    machine_report_fn: Callable[..., Mapping[str, Any]] | None = None,
    server_factory: Callable[..., Any] | None = None,
    probe_fn: Callable[..., Mapping[str, Any]] | None = None,
    smoke_fn: Callable[..., Mapping[str, Any]] | None = None,
    placement_fn: Callable[..., Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Perform four approved, concurrent model smoke requests without starting episodes."""
    _validate_engineering_approval(manifest, approval, yes=yes)
    verified = verify_frozen_feasibility_assets(root, config, manifest)
    from rq1.university_gpu.hardware import machine_report as default_machine_report
    from rq1.university_gpu.models import GPUDevice
    from rq1.university_gpu.ollama import OllamaServer, probe_endpoint as default_probe, worker_specs
    from rq1.university_gpu.ops import capture_placement
    machine_report_fn = machine_report_fn or default_machine_report
    server_factory = server_factory or OllamaServer
    probe_fn = probe_fn or default_probe
    smoke_fn = smoke_fn or _feasibility_smoke
    placement_fn = placement_fn or capture_placement
    output = output.expanduser().resolve()
    try:
        output.relative_to(root.resolve())
    except ValueError:
        pass
    else:
        raise FeasibilityError("benchmark evidence must be stored outside the Git checkout")
    if output.exists():
        raise FileExistsError(f"benchmark output is immutable and already exists: {output}")
    evidence_dir = output.parent / (output.stem + "-runtime")
    if evidence_dir.exists():
        raise FileExistsError(f"benchmark runtime evidence path already exists: {evidence_dir}")
    evidence_dir.mkdir(parents=True)
    devices_report = machine_report_fn(output_dir=evidence_dir, gpu_count=4,
                                       min_gpu_free_mib=int(config.get("min_gpu_free_mib", 16_384)),
                                       ollama_base_port=int(config.get("ollama_base_port", 11_534)))
    if devices_report.get("status") != "PASS" or len(devices_report.get("devices", [])) != 4:
        raise FeasibilityError("four-GPU readiness failed before the approved benchmark")
    devices = [GPUDevice(str(item["uuid"]), int(item["index"]), str(item["name"]),
                         int(item["memory_total_mib"]), int(item["memory_free_mib"]))
               for item in devices_report["devices"]]
    specs = worker_specs(devices, count=4, base_port=int(config.get("ollama_base_port", 11_534)),
                         root=evidence_dir / "workers")
    servers: list[Any] = []
    launch_id = uuid4().hex
    failure: BaseException | None = None
    try:
        servers = [server_factory(spec, Path(str(config["ollama_models_dir"])),
                                  binary=str(config.get("ollama_binary", "ollama"))) for spec in specs]
        for server in servers:
            server.start()
        probes = [probe_fn(spec.endpoint, expected_model=MODEL_TAG, expected_digest=MODEL_DIGEST,
                           expected_quantization=MODEL_QUANTIZATION) for spec in specs]
        if len(probes) != 4 or any(item.get("status") != "PASS" for item in probes):
            raise FeasibilityError("benchmark endpoint model identity did not pass on all four workers")
        smoke = smoke_fn([spec.endpoint for spec in specs], approval)
        if smoke.get("status") != "PASS" or smoke.get("inference_count") != 4:
            raise FeasibilityError("four-way benchmark smoke test failed")
        placement = placement_fn({server.spec.worker_id: int(server.process.pid) for server in servers
                                  if server.process is not None})
        expected = {spec.worker_id: spec.gpu_uuid for spec in specs}
        if placement.get("passed") is not True or placement.get("assignments") != expected:
            raise FeasibilityError("four-worker benchmark could not verify distinct GPU UUID placement")
        report = {"schema": "rq1-feasibility-four-gpu-benchmark-v1", "status": "PASS",
                  "profile_id": PROFILE_ID, "scientific_evidence": False, "episodes_started": 0,
                  "manifest_sha256": manifest["manifest_sha256"],
                  "candidate_manifest_sha256": manifest["candidate_manifest_sha256"],
                  "approval": dict(approval), "input_verification": verified, "machine": dict(devices_report),
                  "model_probes": probes, "smoke_test": dict(smoke), "gpu_placement": dict(placement),
                  "worker_ids": [spec.worker_id for spec in specs], "recorded_at": datetime.now(timezone.utc).isoformat()}
    except BaseException as exc:
        report = {"schema": "rq1-feasibility-four-gpu-benchmark-v1", "status": "BLOCKED",
                  "profile_id": PROFILE_ID, "scientific_evidence": False, "episodes_started": 0,
                  "manifest_sha256": manifest.get("manifest_sha256"), "error_type": type(exc).__name__,
                  "error": str(exc), "recorded_at": datetime.now(timezone.utc).isoformat()}
        failure = exc
    finally:
        cleanup_errors = []
        for server in reversed(servers):
            try:
                server.stop()
            except BaseException as exc:
                cleanup_errors.append({"worker_id": getattr(getattr(server, "spec", None), "worker_id", None),
                                       "error_type": type(exc).__name__, "error": str(exc)})
        if cleanup_errors:
            report = locals().get("report", {
                "schema": "rq1-feasibility-four-gpu-benchmark-v1", "status": "BLOCKED",
                "profile_id": PROFILE_ID, "scientific_evidence": False, "episodes_started": 0,
                "manifest_sha256": manifest.get("manifest_sha256"),
                "error_type": "WorkerShutdownError", "error": "one or more Ollama workers did not stop cleanly",
                "recorded_at": datetime.now(timezone.utc).isoformat(),
            })
            report["cleanup_errors"] = cleanup_errors
            report["status"] = "BLOCKED"
            failure = failure or FeasibilityError("one or more Ollama workers did not stop cleanly")
    report["report_sha256"] = content_hash(report)
    atomic_write_json(output, report, overwrite=False)
    if failure is not None:
        raise FeasibilityError(f"four-GPU benchmark failed; failure evidence saved to {output}: {failure}") from failure
    return report


def bounded_run_root(output_root: Path, run_id: str) -> Path:
    if not isinstance(run_id, str) or not _RUN_ID.fullmatch(run_id):
        raise FeasibilityError("unsafe feasibility run_id")
    expanded = output_root.expanduser()
    absolute = expanded.absolute()
    windows = PureWindowsPath(str(expanded))
    if (".." in expanded.parts or ".." in windows.parts or (sys.platform != "win32" and windows.drive)
            or any(path.is_symlink() for path in (absolute, *absolute.parents))):
        raise FeasibilityError("unsafe or symlinked feasibility output root")
    parent = expanded.resolve()
    raw = parent / run_id
    resolved = raw.resolve()
    if raw.is_symlink() or parent not in resolved.parents:
        raise FeasibilityError("run directory escapes approved output root")
    return resolved


def checked_run_path(output_root: Path, run_root: Path, manifest_path: Path | None = None, *,
                     require_existing: bool = True) -> Path:
    """Check an operator-selected run against its explicit output root and frozen identity."""
    if output_root.expanduser().is_symlink() or not output_root.expanduser().is_dir():
        raise FeasibilityError("approved output root must be an existing non-symlink directory")
    raw = run_root.expanduser()
    if ".." in raw.parts or raw.is_symlink():
        raise FeasibilityError("unsafe or symlinked feasibility run path")
    expected = bounded_run_root(output_root, raw.name)
    if raw.resolve() != expected:
        raise FeasibilityError("run path is outside the explicitly approved output root")
    if expected.exists() and not expected.is_dir():
        raise FeasibilityError("feasibility run path is not a directory")
    if require_existing and not expected.is_dir():
        raise FeasibilityError("feasibility run directory does not exist")
    selected = manifest_path if manifest_path is not None else expected / "manifest.json"
    if selected.is_symlink() or not selected.is_file():
        raise FeasibilityError("frozen manifest must be an existing non-symlink file")
    manifest = load_json(selected)
    if (not isinstance(manifest, Mapping) or validate_feasibility_manifest(manifest, require_frozen=True)
            or manifest.get("run_id") != raw.name):
        raise FeasibilityError("frozen manifest does not validate or match the selected run")
    recorded = expected / "manifest.json"
    if expected.is_dir():
        if recorded.is_symlink() or not recorded.is_file() or load_json(recorded) != manifest:
            raise FeasibilityError("run's immutable manifest differs from the selected manifest")
    return expected


def run_feasibility_launch(
    root: Path,
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    approval: Mapping[str, Any],
    yes: bool,
    resume: bool = False,
    retry_failed: bool = False,
    cutoff_epoch: float | None = None,
    machine_report_fn: Callable[..., Mapping[str, Any]] | None = None,
    server_factory: Callable[..., Any] | None = None,
    worker_factory: Callable[..., Any] | None = None,
    probe_fn: Callable[..., Mapping[str, Any]] | None = None,
    smoke_fn: Callable[..., Mapping[str, Any]] | None = None,
    placement_fn: Callable[..., Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run only an explicitly approved 12-episode engineering feasibility batch."""
    if not yes:
        raise FeasibilityError("feasibility launch requires the explicit --yes acknowledgement")
    if manifest.get("status") != "engineering_frozen":
        raise FeasibilityError("unapproved candidate manifests cannot be launched")
    if validate_feasibility_manifest(manifest, require_frozen=True):
        raise FeasibilityError("engineering manifest failed validation")
    _validate_engineering_approval(manifest, approval, yes=yes)
    if cutoff_epoch is None or type(cutoff_epoch) not in (int, float) or not math.isfinite(cutoff_epoch) or cutoff_epoch <= time.time() + 1800:
        raise FeasibilityError("a real university access cutoff at least 30 minutes away is required")
    if int(config.get("worker_count", 0)) != 4:
        raise FeasibilityError("feasibility launch requires four isolated GPU workers")
    require_posix_process_management("GPU feasibility launch/resume/stop")
    verified = verify_frozen_feasibility_assets(root, config, manifest)

    from rq1.university_gpu.hardware import machine_report as default_machine_report
    from rq1.university_gpu.models import GPUDevice
    from rq1.university_gpu.ollama import OllamaServer, probe_endpoint as default_probe, worker_specs
    from rq1.university_gpu.ops import OperatorError, capture_placement
    from rq1.university_gpu.scheduler import Coordinator, RunLock, RunStore
    from rq1.university_gpu.worker_process import WorkerProcessClient

    machine_report_fn = machine_report_fn or default_machine_report
    server_factory = server_factory or OllamaServer
    worker_factory = worker_factory or WorkerProcessClient
    probe_fn = probe_fn or default_probe
    smoke_fn = smoke_fn or _feasibility_smoke
    placement_fn = placement_fn or capture_placement
    output_root = Path(str(config["run_output_dir"])).expanduser()
    try:
        output_root.resolve().relative_to(root.resolve())
    except ValueError:
        pass
    else:
        raise FeasibilityError("run_output_dir must be outside the Git checkout")
    run_root = bounded_run_root(output_root, str(manifest["run_id"]))
    if run_root.exists() and any(run_root.iterdir()) and not resume:
        raise FeasibilityError("existing run data requires explicit --resume; it will never be overwritten")
    run_root.mkdir(parents=True, exist_ok=True)
    machine = machine_report_fn(output_dir=run_root, gpu_count=4,
                                min_gpu_free_mib=int(config.get("min_gpu_free_mib", 16_384)),
                                ollama_base_port=int(config.get("ollama_base_port", 11_534)))
    if machine.get("status") != "PASS" or len(machine.get("devices", [])) != 4:
        raise FeasibilityError("all four workstation GPUs and free local ports must pass readiness")
    devices = [GPUDevice(str(item["uuid"]), int(item["index"]), str(item["name"]),
                         int(item["memory_total_mib"]), int(item["memory_free_mib"])) for item in machine["devices"]]
    specs = worker_specs(devices, count=4, base_port=int(config.get("ollama_base_port", 11_534)), root=run_root / "workers")
    store = RunStore(run_root, manifest)
    lock = RunLock(run_root / "RUN.lock", resume=resume)
    lock.__enter__()
    try:
        store.initialize()
    except BaseException:
        lock.__exit__(None, None, None)
        raise
    servers: list[Any] = []
    clients: list[Any] = []
    launch_id = uuid4().hex
    try:
        servers = [server_factory(spec, Path(str(config["ollama_models_dir"])),
                                  binary=str(config.get("ollama_binary", "ollama"))) for spec in specs]
        for server in servers:
            server.start()
        probes = [probe_fn(spec.endpoint, expected_model=MODEL_TAG, expected_digest=MODEL_DIGEST,
                           expected_quantization=MODEL_QUANTIZATION) for spec in specs]
        if len(probes) != 4 or any(item.get("status") != "PASS" for item in probes):
            raise FeasibilityError("one or more isolated endpoints do not serve the exact prospective model identity")
        smoke = smoke_fn([spec.endpoint for spec in specs], approval)
        if smoke.get("status") != "PASS" or smoke.get("scientific_evidence") is not False:
            raise FeasibilityError("engineering smoke inference or approval boundary failed")
        placement = placement_fn({server.spec.worker_id: int(server.process.pid) for server in servers
                                  if server.process is not None})
        expected_placement = {spec.worker_id: spec.gpu_uuid for spec in specs}
        if placement.get("passed") is not True or placement.get("assignments") != expected_placement:
            raise FeasibilityError("live Ollama process-to-physical-GPU UUID placement did not verify")
        atomic_write_json(run_root / "launches" / f"{launch_id}.json", {
            "schema": "rq1-feasibility-launch-evidence-v1", "launch_id": launch_id,
            "recorded_at": datetime.now(timezone.utc).isoformat(), "profile_id": PROFILE_ID,
            "manifest_sha256": manifest["manifest_sha256"], "candidate_manifest_sha256": manifest["candidate_manifest_sha256"],
            "approval": dict(approval), "input_verification": verified, "machine": dict(machine),
            "model_probes": probes, "smoke_test": dict(smoke), "gpu_placement": dict(placement),
            "scientific_evidence": False,
        }, overwrite=False)

        from rq1.university_gpu.ops import _library_payload
        _ids, skill_texts, library_payload = _library_payload(Path(str(config["accum18_library_path"])))
        d18 = library_payload.get("libraries", library_payload)["Accum-18"]
        libraries = {"D0": {"skills": []}, "D18": {"skills": list(d18["skills"])}
                     }
        worker_configs = run_root / "worker-configs"
        worker_configs.mkdir(parents=True, exist_ok=True)
        task_rows = list(manifest["tasks"])
        provenance = {
            "manifest_sha256": manifest["manifest_sha256"], "candidate_manifest_sha256": manifest["candidate_manifest_sha256"],
            "profile_id": PROFILE_ID, "policy_version": POLICY_VERSION, "source_commit": manifest["source_commit"],
            "model_digest": MODEL_DIGEST, "model_quantization": MODEL_QUANTIZATION,
            "sbert_snapshot_sha256": manifest["verified_inputs"]["sbert_snapshot_sha256"],
            "accum18_library_sha256": manifest["verified_inputs"]["accum18_library_sha256"],
            "tokenizer_sha256": manifest["verified_inputs"]["tokenizer_sha256"],
            "tokenizer_revision": manifest["tokenizer"]["revision"],
            "data_reconciliation_report_sha256": manifest["verified_inputs"]["data_reconciliation_report_sha256"],
            "oracle_replay_report_sha256": manifest["verified_inputs"]["oracle_replay_report_sha256"],
            "inference_settings": dict(MODEL_SETTINGS), "scientific_evidence": False,
            "placement": {"passed": True, "assignments": dict(placement["assignments"])},
        }
        for spec in specs:
            worker_config = {
                "profile_id": PROFILE_ID, "scientific_evidence": False, "worker_id": spec.worker_id,
                "gpu_uuid": spec.gpu_uuid, "ollama_endpoint": spec.endpoint, "worker_output_dir": str(spec.worker_dir),
                "run_id": manifest["run_id"], "repository_root": str(root.resolve()), "split": "valid_seen",
                "alfworld_data_dir": str(config["alfworld_data_dir"]), "sbert_cache": str(config["sbert_cache"]),
                "sbert_model": SBERT_MODEL, "sbert_revision": SBERT_REVISION,
                "sbert_snapshot_sha256": SBERT_SNAPSHOT_SHA256,
                "model_tag": MODEL_TAG, "model_digest": MODEL_DIGEST, "model_quantization": MODEL_QUANTIZATION,
                "inference_settings": dict(MODEL_SETTINGS),
                "tokenizer_path": str(config["tokenizer_path"]), "tokenizer_sha256": manifest["tokenizer"]["sha256"],
                "tokenizer_revision": manifest["tokenizer"]["revision"],
                "accum18_library_path": str(config["accum18_library_path"]),
                "accum18_library_sha256": manifest["accum18_library"]["sha256"],
                "accum18_skill_ids": manifest["accum18_library"]["skill_ids"],
                "tasks": task_rows, "libraries": libraries, "skill_texts": skill_texts,
                "provenance": provenance,
            }
            config_path = worker_configs / f"{spec.worker_id}.json"
            if config_path.exists():
                if load_json(config_path) != worker_config:
                    raise FeasibilityError(f"existing worker config differs from this immutable manifest: {config_path}")
            else:
                atomic_write_json(config_path, worker_config, overwrite=False)
            client = worker_factory(spec.worker_id, config_path, cwd=root, gpu_uuid=spec.gpu_uuid,
                                   module="rq1.university_gpu.feasibility_worker")
            client.start()
            clients.append(client)
        coordinator = Coordinator(store, clients)
        outcome = coordinator.run(resume=resume, retry_failed=retry_failed, cutoff_epoch=float(cutoff_epoch), lock_held=True)
        return {"input_verification": verified, "smoke_test": smoke, "gpu_placement": placement,
                "run": outcome, "scientific_evidence": False}
    except BaseException as exc:
        try:
            atomic_write_json(run_root / "launches" / f"{launch_id}-failure.json", {
                "schema": "rq1-feasibility-launch-failure-v1", "launch_id": launch_id,
                "recorded_at": datetime.now(timezone.utc).isoformat(), "error_type": type(exc).__name__,
                "error": str(exc), "profile_id": PROFILE_ID, "scientific_evidence": False,
            }, overwrite=False)
        except OSError:
            pass
        raise
    finally:
        cleanup_errors = []
        for client in reversed(clients):
            try:
                client.stop()
            except BaseException as exc:
                cleanup_errors.append({"worker_id": getattr(client, "worker_id", None),
                                       "kind": "episode_worker", "error_type": type(exc).__name__,
                                       "error": str(exc)})
        for server in reversed(servers):
            try:
                server.stop()
            except BaseException as exc:
                cleanup_errors.append({"worker_id": getattr(getattr(server, "spec", None), "worker_id", None),
                                       "kind": "ollama_server", "error_type": type(exc).__name__,
                                       "error": str(exc)})
        try:
            lock.__exit__(None, None, None)
        except BaseException as exc:
            cleanup_errors.append({"kind": "run_lock", "error_type": type(exc).__name__, "error": str(exc)})
        if cleanup_errors:
            try:
                atomic_write_json(run_root / "launches" / f"{launch_id}-cleanup-failure.json", {
                    "schema": "rq1-feasibility-launch-cleanup-failure-v1", "launch_id": launch_id,
                    "recorded_at": datetime.now(timezone.utc).isoformat(), "profile_id": PROFILE_ID,
                    "cleanup_errors": cleanup_errors, "scientific_evidence": False,
                }, overwrite=False)
            except OSError:
                pass
            if sys.exc_info()[0] is None:
                raise FeasibilityError("feasibility launch cleanup was incomplete; inspect cleanup-failure evidence")
