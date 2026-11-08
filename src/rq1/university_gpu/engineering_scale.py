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

BASELINE_CANDIDATE_SHA256 = "61f01fe25a78d45d4e51c8dcf958a2f3103fa8665030ec28ced9e4e98490bff0"
PROFILE_ID = "rq1-engineering-500-v1"
SCHEMA = "rq1-engineering-scale-manifest-v1"
RESULT_SCHEMA = "rq1-engineering-scale-result-v1"
POLICY_VERSION = "rq1-engineering-scale-policy-v1"
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
_RUN_ID = re.compile(r"rq1-engineering-500-v1-[A-Za-z0-9][A-Za-z0-9._-]{1,79}[A-Za-z0-9_-]\Z")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class FeasibilityError(ValueError):
    """A feasibility manifest or evidence gate is invalid or incomplete."""


def _manifest_digest(value: Mapping[str, Any]) -> str:
    return content_hash({key: item for key, item in value.items() if key != "manifest_sha256"})


def _unit_rows(tasks: Sequence[Mapping[str, Any]], library_sha256: str | None, episode_count: int = 500) -> list[dict[str, Any]]:
    from rq1.university_gpu.feasibility import _unit_rows as legacy_rows
    rows = []
    for repetition in range(1, 43):
        for task_index, task in enumerate(tasks):
            if repetition == 42 and task_index >= 4:
                continue
            pair_id = content_hash({"profile_id": PROFILE_ID, "task_id": task["task_id"], "seed": 11, "repetition_id": repetition})
            pair = legacy_rows([task], library_sha256)
            if (repetition - 1 + task_index) % 2:
                pair.reverse()
            for position, row in enumerate(pair, 1):
                row["identity"].update(profile_id=PROFILE_ID, repetition_id=repetition, pair_id=pair_id)
                row.update(profile_id=PROFILE_ID, repetition_id=repetition, pair_id=pair_id,
                           position_in_pair=position, global_order=len(rows)+1, unit_key=content_hash(row["identity"]))
                rows.append(row)
                if len(rows) == episode_count:
                    return rows
    return rows


def build_scale_candidate(legacy: Mapping[str, Any], *, acceptance: bool = False, additional_attempts: int = 0) -> dict[str, Any]:
    from rq1.university_gpu.feasibility import validate_feasibility_manifest as legacy_validate
    if type(additional_attempts) is not int or not 0 <= additional_attempts <= 20:
        raise FeasibilityError("additional attempt allowance must be 0..20, separately approved")
    if legacy_validate(legacy) or legacy.get("status") != "candidate_unapproved" or legacy.get("manifest_sha256") != BASELINE_CANDIDATE_SHA256:
        raise FeasibilityError("a validated original unapproved six-task candidate is required")
    value = copy.deepcopy(dict(legacy))
    value.update(schema=SCHEMA, profile_id=PROFILE_ID, purpose="acceptance" if acceptance else "engineering_500",
                 episode_count=4 if acceptance else 500, baseline_candidate=copy.deepcopy(dict(legacy)),
                 baseline_source_commit="6af25622464142beef25d2c248655946389e4f8b",
                 dependency_files={name: sha256_file(Path(__file__).resolve().parents[3] / name) for name in ("pyproject.toml", "uv.lock")},
                 pause_every=4 if acceptance else 50, cutoff_reserve_seconds=4500,
                 maximum_attempts=(4 if acceptance else 500) + additional_attempts, minimum_free_bytes=50_000_000_000)
    value["planned_matrix_action_allowance"] = value["episode_count"] * 200
    value["condition_counts"] = {"D0": value["episode_count"] // 2, "D18": value["episode_count"] // 2}
    value["units"] = _unit_rows(value["tasks"], value.get("accum18_library", {}).get("sha256"), value["episode_count"])
    value["manifest_sha256"] = _manifest_digest(value)
    issues = validate_feasibility_manifest(value)
    if issues:
        raise FeasibilityError("; ".join(issues))
    return value


def validate_feasibility_manifest(manifest: Mapping[str, Any], *, require_frozen: bool = False) -> list[str]:
    problems: list[str] = []
    from rq1.university_gpu.feasibility import validate_feasibility_manifest as legacy_validate
    baseline = manifest.get("baseline_candidate", {})
    if legacy_validate(baseline) or baseline.get("manifest_sha256") != BASELINE_CANDIDATE_SHA256 or baseline.get("status") != "candidate_unapproved" or baseline.get("tasks") != manifest.get("tasks"):
        problems.append("original candidate ancestry differs")
    if (manifest.get("purpose") not in {"acceptance", "engineering_500"}
            or manifest.get("pause_every") != (4 if manifest.get("purpose") == "acceptance" else 50)
            or manifest.get("cutoff_reserve_seconds") != 4500
            or type(manifest.get("maximum_attempts")) is not int
            or not manifest.get("episode_count",0) <= manifest["maximum_attempts"] <= manifest.get("episode_count",0) + 20
            or manifest.get("minimum_free_bytes") != 50_000_000_000):
        problems.append("scale execution policy differs")
    if manifest.get("schema") != SCHEMA or manifest.get("profile_id") != PROFILE_ID:
        return ["unsupported feasibility schema or profile"]
    if (manifest.get("baseline_source_commit") != "6af25622464142beef25d2c248655946389e4f8b"
            or set(manifest.get("dependency_files", {})) != {"pyproject.toml", "uv.lock"}
            or any(not _HEX64.fullmatch(str(v)) for v in manifest.get("dependency_files", {}).values())):
        problems.append("baseline/dependency provenance missing")
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
    if manifest.get("task_count") != 6 or manifest.get("episode_count") != (4 if manifest.get("purpose") == "acceptance" else 500):
        problems.append("scale matrix task/count identity differs")
    if (manifest.get("planned_matrix_action_allowance") != manifest.get("episode_count", 0) * 200
            or manifest.get("planned_matrix_allowance_scope") != (
                "one planned attempt per unit; every retry adds up to 200 actions and is recorded as a distinct attempt")):
        problems.append("one-attempt action allowance differs from the versioned matrix")
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
    expected_units = _unit_rows(tasks, manifest.get("accum18_library", {}).get("sha256"), manifest.get("episode_count", 500)) if len(tasks) == 6 else []
    units = manifest.get("units")
    if not isinstance(units, list) or len(units) != manifest.get("episode_count"):
        problems.append("scale matrix length differs from approved episode_count")
    elif units != expected_units:
        problems.append("unit matrix differs from the deterministic paired D0/D18 construction")
    counts = Counter(str(row.get("condition")) for row in units if isinstance(row, Mapping)) if isinstance(units, list) else Counter()
    if counts != Counter({"D0": manifest.get("episode_count", 0)//2, "D18": manifest.get("episode_count", 0)//2}) or manifest.get("condition_counts") != dict(counts):
        problems.append("scale matrix must contain equal D0 and D18 counts")
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
    dependencies = manifest.get("runtime_dependencies")
    if (not isinstance(dependencies, dict) or not isinstance(dependencies.get("packages"), dict)
            or not dependencies.get("python")
            or content_hash(dependencies) != (inputs or {}).get("runtime_dependencies_sha256")):
        problems.append("frozen runtime dependency inventory is missing or inconsistent")
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
    if (not isinstance(approval, Mapping) or approval.get("schema") != "rq1-engineering-scale-approval-v1"
            or approval.get("source_commit") != commit or approval.get("episode_count") != manifest.get("episode_count")
            or approval.get("maximum_attempts") != manifest.get("maximum_attempts")
            or approval.get("backup_destination") != manifest.get("backup_destination")
            or approval.get("backup_independence_confirmed") is not True
            or type(approval.get("access_end_epoch")) not in (int, float) or not math.isfinite(approval["access_end_epoch"])):
        problems.append("scale authorization lacks source/count/storage/access bindings")
    if not isinstance(approval, Mapping) or approval.get("approval_scope") != "engineering_feasibility_only":
        problems.append("an explicit engineering-only authorization is required")
    elif (approval.get("gpu_inference") is not True or approval.get("scientific_run") is not False
          or approval.get("not_for_scientific_claims") is not True or not approval.get("approved_by")
          or not approval.get("approval_reference") or approval.get("candidate_manifest_sha256") != manifest.get("candidate_manifest_sha256")
          or approval.get("prior_pick_place_exposure_acknowledged") is not True):
        problems.append("engineering authorization is incomplete or attempts to authorize a scientific run")
    if manifest.get("purpose") == "engineering_500" and not _HEX64.fullmatch(str((approval or {}).get("acceptance_export_manifest_sha256", ""))):
        problems.append("500-run approval must bind the separate acceptance export")
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
        raise FeasibilityError("run_id must be unique and scoped to rq1-engineering-500-v1")
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

    if (engineering_approval.get("source_commit") != commit
            or engineering_approval.get("episode_count") != candidate["episode_count"]
            or engineering_approval.get("maximum_attempts") != candidate["maximum_attempts"]
            or not engineering_approval.get("backup_destination")):
        raise FeasibilityError("fresh scale authorization must bind source, counts and backup destination")
    frozen = copy.deepcopy(dict(candidate))
    frozen["backup_destination"] = engineering_approval["backup_destination"]
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
    frozen["runtime_dependencies"] = runtime_dependencies()
    frozen["verified_inputs"]["runtime_dependencies_sha256"] = content_hash(frozen["runtime_dependencies"])
    frozen["engineering_approval"] = dict(engineering_approval)
    frozen["units"] = _unit_rows(frozen["tasks"], library_sha, frozen["episode_count"])
    frozen["manifest_sha256"] = _manifest_digest(frozen)
    problems = validate_feasibility_manifest(frozen, require_frozen=True)
    if problems:
        raise FeasibilityError("engineering-frozen manifest failed validation: " + "; ".join(problems))
    return frozen


from rq1.university_gpu.feasibility import (
    build_accum18_reference, verify_accum18_reference, check_selected_task_sources,
    reconcile_data_identity, write_reconciliation_report,
)


def validate_oracle_report(report: Mapping[str, Any], manifest: Mapping[str, Any]) -> list[str]:
    problems: list[str] = []
    if report.get("schema") != "rq1-feasibility-oracle-replay-v1" or report.get("manifest_sha256") != manifest.get("baseline_candidate", {}).get("manifest_sha256"):
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


def _report_file_matches(path: Path, expected: str, field: str, label: str) -> dict[str, Any]:
    value = load_json(path)
    observed = _require_report_hash(value, field, label)
    if observed != expected:
        raise FeasibilityError(f"{label} differs from the hash bound into the frozen run manifest")
    return value


def verify_frozen_feasibility_assets(root: Path, config: Mapping[str, Any], manifest: Mapping[str, Any], *, expected_source_commit: str | None = None) -> dict[str, Any]:
    """Recheck all local, non-GPU inputs immediately before a feasibility run."""
    problems = validate_feasibility_manifest(manifest, require_frozen=True)
    if problems:
        raise FeasibilityError("frozen feasibility manifest is invalid: " + "; ".join(problems))
    if Path(str(config.get("run_output_dir", ""))).expanduser().resolve() == root.resolve():
        raise FeasibilityError("external run storage is required")
    import shutil
    from rq1.university_gpu.scale_protection import approved_destination
    backup = approved_destination(manifest)
    run_storage = Path(str(config["run_output_dir"])).expanduser().resolve()
    if (backup == root.resolve() or root.resolve() in backup.parents or backup in root.resolve().parents
            or backup == run_storage or run_storage in backup.parents or backup in run_storage.parents):
        raise FeasibilityError("backup must be disjoint from source and run storage")
    for location in (run_storage, backup):
        if not location.is_dir() or shutil.disk_usage(location).free < manifest["minimum_free_bytes"]:
            raise FeasibilityError("existing approved storage lacks the 50 GB reserve")
    for name, digest in manifest["dependency_files"].items():
        if sha256_file(root / name) != digest:
            raise FeasibilityError("dependency manifest changed: " + name)
    if runtime_dependencies() != manifest.get("runtime_dependencies"):
        raise FeasibilityError("installed Python/package versions changed since reviewed freeze")
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

    storage_projection = {"required_free_bytes": manifest["minimum_free_bytes"], "basis": "minimum reserve"}
    if manifest["purpose"] == "engineering_500":
        acceptance_check = verify_acceptance_export(config, manifest)
        required = projected_storage_bytes(acceptance_check["run_bytes"], 4, manifest["maximum_attempts"])
        storage_projection = {"required_free_bytes": required, "acceptance_run_bytes": acceptance_check["run_bytes"],
                              "basis": "observed bytes per execution * physical attempt allowance * 12; floor 50 GB; estimate, not upper bound"}
        if any(shutil.disk_usage(location).free < required for location in (run_storage, backup)):
            raise FeasibilityError("storage is below the acceptance-based projection")

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
    if provenance != (expected_source_commit or manifest.get("source_commit")) or status.strip():
        raise FeasibilityError("launch source must be the exact clean commit bound to the manifest")
    return {"status": "PASS", "profile_id": PROFILE_ID, "scientific_evidence": False,
            "storage_projection": storage_projection, "verified_inputs": dict(inputs), "selected_task_sources": source_check,
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
    if (approval.get("source_commit") != manifest.get("source_commit")
            or approval.get("episode_count") != manifest.get("episode_count")
            or approval.get("maximum_attempts") != manifest.get("maximum_attempts")
            or approval.get("backup_destination") != manifest.get("backup_destination")
            or not manifest.get("backup_destination")):
        raise FeasibilityError("scale approval source/count/backup scope differs")
    if type(approval.get("access_end_epoch")) not in (int, float) or not math.isfinite(approval["access_end_epoch"]) or approval["access_end_epoch"] <= time.time() + 4500:
        raise FeasibilityError("approved access deadline must leave at least 75 minutes")
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
    from rq1.university_gpu.feasibility_placement import capture_feasibility_placement
    machine_report_fn = machine_report_fn or default_machine_report
    server_factory = server_factory or OllamaServer
    probe_fn = probe_fn or default_probe
    smoke_fn = smoke_fn or _feasibility_smoke
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
    if not scale_machine_matches(devices_report):
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
        placement = (placement_fn({server.spec.worker_id: int(server.process.pid) for server in servers
                                   if server.process is not None}) if placement_fn is not None
                     else capture_feasibility_placement(servers))
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
        if 'placement' in locals():
            report['gpu_placement'] = dict(placement)
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
    continuation: Mapping[str, Any] | None = None,
    continuation_start: bool = True,
) -> dict[str, Any]:
    """Run only an explicitly approved 12-episode engineering feasibility batch."""
    if not yes:
        raise FeasibilityError("feasibility launch requires the explicit --yes acknowledgement")
    if manifest.get("status") != "engineering_frozen":
        raise FeasibilityError("unapproved candidate manifests cannot be launched")
    if validate_feasibility_manifest(manifest, require_frozen=True):
        raise FeasibilityError("engineering manifest failed validation")
    if continuation is None:
        _validate_engineering_approval(manifest, approval, yes=yes)
    else:
        from rq1.university_gpu.scale_continuation import verify_authorization
        if not resume or retry_failed:
            raise FeasibilityError("continuation permits resume without failed retries only")
        verify_authorization(root, manifest, approval, continuation, continuation["checkpoint"], cutoff_epoch=cutoff_epoch)
    if cutoff_epoch is None or type(cutoff_epoch) not in (int, float) or not math.isfinite(cutoff_epoch) or cutoff_epoch <= time.time() + 4500:
        raise FeasibilityError("a real university access cutoff at least 30 minutes away is required")
    if cutoff_epoch > float((continuation or approval).get("access_end_epoch", 0)):
        raise FeasibilityError("cutoff exceeds approved access deadline")
    if int(config.get("worker_count", 0)) != 4:
        raise FeasibilityError("feasibility launch requires four isolated GPU workers")
    require_posix_process_management("GPU feasibility launch/resume/stop")
    verified = verify_frozen_feasibility_assets(root, config, manifest, expected_source_commit=(continuation or {}).get("continuation_source_commit"))

    from rq1.university_gpu.hardware import machine_report as default_machine_report
    from rq1.university_gpu.models import GPUDevice
    from rq1.university_gpu.ollama import OllamaServer, probe_endpoint as default_probe, worker_specs
    from rq1.university_gpu.ops import OperatorError
    from rq1.university_gpu.feasibility_placement import capture_feasibility_placement
    from rq1.university_gpu.scheduler import Coordinator, RunLock, RunStore
    from rq1.university_gpu.worker_process import WorkerProcessClient

    machine_report_fn = machine_report_fn or default_machine_report
    server_factory = server_factory or OllamaServer
    worker_factory = worker_factory or WorkerProcessClient
    probe_fn = probe_fn or default_probe
    smoke_fn = smoke_fn or _feasibility_smoke
    output_root = Path(str(config["run_output_dir"])).expanduser()
    try:
        output_root.resolve().relative_to(root.resolve())
    except ValueError:
        pass
    else:
        raise FeasibilityError("run_output_dir must be outside the Git checkout")
    run_root = bounded_run_root(output_root, str(manifest["run_id"]))
    if continuation and (not run_root.is_dir() or run_root.is_symlink()):
        from rq1.university_gpu.scale_continuation import ContinuationError
        raise ContinuationError("continuation requires the existing, non-symlink run directory")
    if run_root.exists() and any(run_root.iterdir()) and not resume:
        raise FeasibilityError("existing run data requires explicit --resume; it will never be overwritten")
    run_root.mkdir(parents=True, exist_ok=True)
    machine = machine_report_fn(output_dir=run_root, gpu_count=4,
                                min_gpu_free_mib=int(config.get("min_gpu_free_mib", 16_384)),
                                ollama_base_port=int(config.get("ollama_base_port", 11_534)))
    if not scale_machine_matches(machine):
        raise FeasibilityError("all four workstation GPUs and free local ports must pass readiness")
    devices = [GPUDevice(str(item["uuid"]), int(item["index"]), str(item["name"]),
                         int(item["memory_total_mib"]), int(item["memory_free_mib"])) for item in machine["devices"]]
    launch_id = uuid4().hex
    runtime_root = run_root / "continuation-runtime" / launch_id if continuation else run_root
    specs = worker_specs(devices, count=4, base_port=int(config.get("ollama_base_port", 11_534)), root=runtime_root / "workers")
    store = RunStore(run_root, manifest)
    lock = RunLock(run_root / "RUN.lock", resume=resume)
    lock.__enter__()
    try:
        store.initialize()
        if continuation:
            from rq1.university_gpu.scale_continuation import checkpoint_identity, verify_authorization, verify_authorized_checkpoint_preserved
            current_identity = checkpoint_identity(run_root, manifest)
            verify_authorized_checkpoint_preserved(run_root, manifest, continuation["checkpoint"])
            verify_authorization(root, manifest, approval, continuation,
                current_identity if continuation_start else continuation["checkpoint"], cutoff_epoch=cutoff_epoch)
    except BaseException:
        lock.__exit__(None, None, None)
        raise
    servers: list[Any] = []
    clients: list[Any] = []
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
        placement = (placement_fn({server.spec.worker_id: int(server.process.pid) for server in servers
                                   if server.process is not None}) if placement_fn is not None
                     else capture_feasibility_placement(servers))
        expected_placement = {spec.worker_id: spec.gpu_uuid for spec in specs}
        if placement.get("passed") is not True or placement.get("assignments") != expected_placement:
            raise FeasibilityError("live Ollama process-to-physical-GPU UUID placement did not verify")
        atomic_write_json(run_root / "launches" / f"{launch_id}.json", {
            "schema": "rq1-feasibility-launch-evidence-v1", "launch_id": launch_id,
            "recorded_at": datetime.now(timezone.utc).isoformat(), "profile_id": PROFILE_ID,
            "manifest_sha256": manifest["manifest_sha256"], "candidate_manifest_sha256": manifest["candidate_manifest_sha256"],
            "approval": dict(approval), "input_verification": verified, "machine": dict(machine),
            "model_probes": probes, "smoke_test": dict(smoke), "gpu_placement": dict(placement),
            **({"continuation_authorization": dict(continuation), "continuation_source_commit": continuation["continuation_source_commit"],
                "original_source_commit": manifest["source_commit"]} if continuation else {}),
            "scientific_evidence": False,
        }, overwrite=False)

        from rq1.university_gpu.ops import _library_payload
        _ids, skill_texts, library_payload = _library_payload(Path(str(config["accum18_library_path"])))
        d18 = library_payload.get("libraries", library_payload)["Accum-18"]
        libraries = {"D0": {"skills": []}, "D18": {"skills": list(d18["skills"])}
                     }
        worker_configs = runtime_root / "worker-configs"
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
        if continuation:
            provenance.update(source_commit=continuation["continuation_source_commit"], original_source_commit=manifest["source_commit"],
                              continuation_source_commit=continuation["continuation_source_commit"],
                              continuation_authorization_sha256=content_hash(dict(continuation)))
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
                                   module="rq1.university_gpu.engineering_scale_worker")
            client.start()
            clients.append(client)
        coordinator = Coordinator(store, clients)
        prior_attempts = sum(e["event"] == "attempt_started" for e in store.event_rows())
        remaining_allowance = manifest["maximum_attempts"] - prior_attempts
        next_boundary = min(manifest["episode_count"], (len(store.completed()) // manifest["pause_every"] + 1) * manifest["pause_every"])
        outcome = coordinator.run(resume=resume, retry_failed=retry_failed, cutoff_epoch=float(cutoff_epoch), lock_held=True,
                                  max_attempts=(None if len(store.completed()) == manifest["episode_count"] else min(remaining_allowance, next_boundary - len(store.completed()))),
                                  minimum_start_seconds=4500, storage_minimum_bytes=manifest["minimum_free_bytes"],
                                  monitor=lambda: monitor_resources(runtime_root, servers, manifest))
        if outcome["status"] == "paused" and ((run_root / "STOP").exists() or outcome["cutoff_reached"]):
            from rq1.university_gpu.snapshot import backup_checkpoint
            backup_checkpoint(store)
        return {"input_verification": verified, "smoke_test": smoke, "gpu_placement": placement,
                "run": outcome, "scientific_evidence": False}
    except BaseException as exc:
        try:
            atomic_write_json(run_root / "launches" / f"{launch_id}-failure.json", {
                "schema": "rq1-feasibility-launch-failure-v1", "launch_id": launch_id,
                "recorded_at": datetime.now(timezone.utc).isoformat(), "error_type": type(exc).__name__,
                "error": str(exc), "profile_id": PROFILE_ID, "scientific_evidence": False,
                **({"original_source_commit": manifest["source_commit"],
                    "continuation_source_commit": continuation["continuation_source_commit"],
                    "continuation_authorization_sha256": content_hash(dict(continuation))} if continuation else {}),
                "gpu_placement": dict(placement) if 'placement' in locals() else None,
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


def monitor_resources(run_root: Path, servers: Sequence[Any], manifest: Mapping[str, Any]) -> None:
    import csv
    import io
    import shutil
    query = subprocess.run(["nvidia-smi", "--query-gpu=uuid,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=20, check=True)
    rows = list(csv.reader(io.StringIO(query.stdout), skipinitialspace=True))
    expected = {s.spec.gpu_uuid for s in servers}
    if len(rows) != 4 or {r[0] for r in rows} != expected or any(s.process is None or s.process.poll() is not None for s in servers):
        raise FeasibilityError("GPU inventory or owned worker process lost")
    free = shutil.disk_usage(run_root).free
    if free < manifest["minimum_free_bytes"]:
        raise FeasibilityError("local evidence reserve exhausted")
    stamp = datetime.now(timezone.utc).isoformat()
    record = {"timestamp": stamp, "coordinator_alive": True, "free_bytes": free, "gpu_rows": rows,
              "workers": [{"worker_id": s.spec.worker_id, "pid": s.process.pid} for s in servers],
              "note": "Coordinator liveness is not episode progress; inspect raw action logs and attempt heartbeats."}
    atomic_write_json(run_root / "monitor" / (uuid4().hex + ".json"), record, overwrite=False)
    atomic_write_json(run_root / "status-live.json", record)


def _run_supervised(root: Path, config: Mapping[str, Any], manifest: Mapping[str, Any], **kwargs: Any) -> dict[str, Any]:
    """Only clean segment pauses may auto-resume, after destination readback."""
    from rq1.university_gpu.scale_protection import protect_segment, verify_last_protection
    run_root = bounded_run_root(Path(str(config["run_output_dir"])), str(manifest["run_id"]))
    manifest_path = run_root / "manifest.json"
    if kwargs.get("resume"):
        verify_last_protection(run_root, manifest)
    while True:
        value = run_feasibility_launch(root, config, manifest, **kwargs)
        outcome = value["run"]
        if outcome["failures"] or outcome["status"] not in {"paused", "completed"}:
            return value
        if outcome["status"] == "completed":
            finalize_run(run_root, manifest)
        value["protection"] = protect_segment(root, run_root, manifest_path)
        if (outcome["status"] == "completed" or outcome["cutoff_reached"] or (run_root / "STOP").exists()
                or outcome["attempts_started"] == 0 or time.time() + 4500 >= kwargs["cutoff_epoch"]):
            return value
        kwargs["resume"] = True
        kwargs["retry_failed"] = False
        if kwargs.get("continuation") is not None:
            kwargs["continuation_start"] = False
        verify_last_protection(run_root, manifest)


def run_supervised(root: Path, config: Mapping[str, Any], manifest: Mapping[str, Any], **kwargs: Any) -> dict[str, Any]:
    from rq1.university_gpu.scheduler import RunLock
    continuation = kwargs.get("continuation")
    if continuation is None:
        _validate_engineering_approval(manifest, kwargs["approval"], yes=kwargs.get("yes", False))
    elif not kwargs.get("yes") or not kwargs.get("resume") or kwargs.get("retry_failed"):
        raise FeasibilityError("continuation requires explicit resume acknowledgement and forbids retries")
    issues = validate_feasibility_manifest(manifest, require_frozen=True)
    if issues:
        raise FeasibilityError("invalid scale manifest: " + "; ".join(issues))
    output_root = Path(str(config["run_output_dir"]))
    if continuation is None:
        output_root.mkdir(parents=True, exist_ok=True)
    elif not output_root.is_dir() or output_root.is_symlink():
        raise FeasibilityError("continuation requires existing, non-symlink run storage")
    target = bounded_run_root(output_root, manifest["run_id"])
    if continuation is not None and (not target.is_dir() or target.is_symlink()):
        raise FeasibilityError("continuation requires an existing, non-symlink run directory")
    if target.exists():
        from rq1.university_gpu.scheduler import RunStore
        store = RunStore(target, manifest)
        events = store.event_rows()
        terminal = {e["attempt_id"] for e in events if e["event"] in {"unit_completed", "attempt_failed", "recovered_authoritative_result"}}
        accepted = store.completed()
        unknown = [e for e in events if e["event"] == "attempt_started" and e["attempt_id"] not in terminal
                   and (e["unit_key"] not in accepted or accepted[e["unit_key"]].get("attempt_id") != e["attempt_id"])]
        latest = {}
        for event in events:
            if event["event"] in {"attempt_started", "attempt_failed", "unit_completed", "recovered_authoritative_result"}:
                latest[event["unit_key"]] = event
        failed = [e for e in latest.values() if e["event"] == "attempt_failed"]
        if unknown or (failed and (not kwargs.get("retry_failed") or any(e.get("safe_to_continue") is not True or e.get("mutation_state_known") is not True for e in failed))):
            raise FeasibilityError("unknown/unsafe attempts require manual recovery; no workers started")
        if len(store.completed()) < manifest["episode_count"] and sum(e["event"] == "attempt_started" for e in events) >= manifest["maximum_attempts"]:
            raise FeasibilityError("approved total attempt allowance exhausted")
    with RunLock(output_root / (manifest["run_id"] + ".supervisor.lock"), resume=kwargs.get("resume", False)):
        if continuation is not None:
            from rq1.university_gpu.scale_continuation import checkpoint_identity, verify_authorization
            with RunLock(target / "RUN.lock", resume=True):
                identity = checkpoint_identity(target, manifest)
                if not 218 <= identity["completed_count"] < 500:
                    raise FeasibilityError("continuation checkpoint must contain 218 to 499 verified executions")
                digest = verify_authorization(root, manifest, kwargs["approval"], continuation, identity,
                                              cutoff_epoch=kwargs.get("cutoff_epoch"))
                verify_frozen_feasibility_assets(root, config, manifest,
                    expected_source_commit=continuation["continuation_source_commit"])
                from rq1.university_gpu.scheduler import RunStore
                evidence = RunStore(target, manifest).path(Path("continuation-authorizations") / (digest + ".json"))
                if evidence.exists():
                    if evidence.is_symlink() or load_json(evidence) != continuation:
                        raise FeasibilityError("existing continuation authorization evidence conflicts")
                else:
                    atomic_write_json(evidence, dict(continuation), overwrite=False)
        return _run_supervised(root, config, manifest, **kwargs)


def verify_acceptance_export(config: Mapping[str, Any], manifest: Mapping[str, Any]) -> dict[str, Any]:
    from rq1.university_gpu.snapshot import verify_progress
    acceptance = Path(str(config["acceptance_export_path"]))
    if sha256_file(acceptance / "EXPORT-MANIFEST.json") != manifest["engineering_approval"].get("acceptance_export_manifest_sha256"):
        raise FeasibilityError("acceptance export is not bound by the fresh scale approval")
    accepted = verify_progress(acceptance)
    am = load_json(acceptance / "run" / "manifest.json")
    if (not accepted["complete"] or accepted["completed"] != 4 or am.get("purpose") != "acceptance"
            or am.get("source_commit") != manifest["source_commit"] or am.get("verified_inputs") != manifest["verified_inputs"]
            or am.get("tasks") != manifest["tasks"]):
        raise FeasibilityError("separate acceptance run does not match reviewed source/assets/tasks")
    envelopes = [load_json(p) for p in (acceptance / "run" / "results" / "units").glob("*.json")]
    if len({e.get("worker_id") for e in envelopes}) != 4:
        raise FeasibilityError("acceptance did not execute on all four workers")

    export_record = load_json(acceptance / "EXPORT-MANIFEST.json")
    source = export_record.get("analysis_source", {})
    if source.get("clean") is not True or source.get("commit") != manifest["source_commit"] or source.get("matches_run_commit") is not True:
        raise FeasibilityError("acceptance export did not preserve exact clean source provenance")
    launches = [load_json(p) for p in (acceptance / "run" / "launches").glob("*.json")]
    placements = [r.get("gpu_placement", {}) for r in launches if r.get("manifest_sha256") == am["manifest_sha256"]]
    if not any(p.get("passed") is True and len(p.get("assignments", {})) == 4
               and len(set(p["assignments"].values())) == 4
               and {e["worker_id"] for e in envelopes} == set(p["assignments"]) for p in placements):
        raise FeasibilityError("acceptance lacks matching four-GPU placement evidence")
    return {"status": "PASS", "acceptance_manifest_sha256": am["manifest_sha256"], "completed": 4,
            "run_bytes": sum(item["bytes"] for item in export_record["files"] if item["path"].startswith("run/"))}


def descriptive_summary(run_root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    from rq1.university_gpu.scheduler import Coordinator, RunStore
    store = RunStore(run_root, manifest)
    validation = Coordinator(store, [type("Offline", (), {"worker_id": "summary"})()]).validate()
    if validation["problems"]:
        raise FeasibilityError("invalid evidence cannot be summarized")
    results = [e["result"] for e in store.completed().values()]
    groups = {}
    for result in results:
        key = result["task_id"] + ":" + result["condition"]
        group = groups.setdefault(key, {"task_id": result["task_id"], "condition": result["condition"], "completed": 0, "recovery_successes": 0, "recovery_actions": 0})
        group["completed"] += 1
        group["recovery_successes"] += int(result["recovery_success"])
        group["recovery_actions"] += result["actions"]
    pairs = {}
    for result in results:
        pairs.setdefault(result["pair_id"], {})[result["condition"]] = result
    paired = [{"pair_id": key, "task_id": values["D0"]["task_id"], "repetition_id": values["D0"]["repetition_id"],
               "d18_minus_d0_success": int(values["D18"]["recovery_success"]) - int(values["D0"]["recovery_success"])}
              for key, values in sorted(pairs.items()) if set(values) == {"D0", "D18"}]
    events = store.event_rows()
    conditions = {}
    for condition in CONDITIONS:
        rows = [r for r in results if r["condition"] == condition]
        successes = sum(r["recovery_success"] for r in rows)
        conditions[condition] = {"completed": len(rows), "recovery_successes": successes,
                                 "recovery_rate": successes / len(rows) if rows else None}
    return {"schema": "rq1-engineering-scale-descriptive-v1", "scientific_evidence": False,
            "manifest_sha256": manifest["manifest_sha256"], "planned": manifest["episode_count"], "completed": len(results),
            "physical_attempts": sum(e["event"] == "attempt_started" for e in events),
            "failed_attempts": sum(e["event"] == "attempt_failed" for e in events),
            "conditions": conditions, "task_condition_results": list(groups.values()), "paired_results": paired,
            "limitations": "Repeated fixed-seed executions on six task contexts; no independent-sample or confirmatory inference."}


def finalize_run(run_root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    from rq1.university_gpu.scheduler import Coordinator, RunStore
    result = Coordinator(RunStore(run_root, manifest), [type("Offline", (), {"worker_id": "merge"})()]).merge()
    summary = descriptive_summary(run_root, manifest)
    target = run_root / "ENGINEERING-SUMMARY.json"
    if target.exists() and load_json(target) != summary:
        raise FeasibilityError("existing engineering summary differs; refusing overwrite")
    if not target.exists():
        atomic_write_json(target, summary, overwrite=False)
    result["engineering_summary"] = str(target)
    return result


def scale_machine_matches(report: Mapping[str, Any]) -> bool:
    devices = report.get("devices", [])
    return (report.get("status") == "PASS" and len(devices) == 4
            and len({d.get("uuid") for d in devices}) == 4
            and all("rtx5000ada" in "".join(str(d.get("name", "")).lower().split())
                    and int(d.get("memory_total_mib", 0)) >= 30 * 1024 for d in devices))


def runtime_dependencies() -> dict[str, Any]:
    import importlib.metadata
    import platform
    if sys.version_info[:2] != (3, 11):
        raise FeasibilityError("engineering runtime requires Python 3.11")
    packages = {}
    for distribution in importlib.metadata.distributions():
        name = str(distribution.metadata.get("Name", "")).lower().replace("_", "-")
        if not name:
            raise FeasibilityError("installed dependency metadata lacks a name")
        version = distribution.version
        if name in packages and packages[name] != version:
            raise FeasibilityError("conflicting installed versions: " + name)
        packages[name] = version
    return {"python": platform.python_version(), "packages": dict(sorted(packages.items()))}


def projected_storage_bytes(observed_run_bytes: int, observed_executions: int, maximum_attempts: int) -> int:
    if any(type(value) is not int or value <= 0 for value in (observed_run_bytes, observed_executions, maximum_attempts)):
        raise FeasibilityError("positive measured storage and execution counts are required")
    return max(50_000_000_000, (observed_run_bytes * maximum_attempts * 12 + observed_executions - 1) // observed_executions)
