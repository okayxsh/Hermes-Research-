"""Fail-closed validation for experiment freezes.

Freeze files are evidence, not configuration generators: every scientific choice
comes from a human-approved approval file.  Final evaluation uses the
``environment``/``protocol`` freezes after a real Phase 7 pilot; acquisition uses
the acquisition-scoped freezes after a passed non-scientific acquisition check.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from rq1.freeze.models import FreezeManifest, FreezeValidation
from rq1.utils.time import utc_now

ENVIRONMENT_REQUIRED = {
    "python_version", "dependency_lock_sha256", "alfworld_version", "alfworld_data_sha256",
    "hermes_version", "ollama_version", "model_tag", "model_digest", "gpu_driver",
    "prompt_hashes", "task_manifest_hashes", "checkpoint_policy_sha256",
    "perturbation_policy_sha256", "solvability_policy_sha256", "action_limits",
    "timeout_policy", "snapshot_policy", "repetition_count",
    "seeds", "library_hashes", "retriever_model",
}
PROTOCOL_REQUIRED = {"checkpoint_policy_sha256", "perturbation_policy_sha256", "solvability_policy_sha256", "action_limits", "timeout_policy", "snapshot_policy", "repetition_count", "seeds", "retriever_model"}
ACQUISITION_ENVIRONMENT_REQUIRED = {
    "repository_commit", "branch", "hostname", "gpu", "gpu_driver", "python_version",
    "python_executable", "python_environment", "dependency_lock_sha256", "packages",
    "alfworld_version", "alfworld_data_identity", "hermes_version", "hermes_commit",
    "ollama_version", "model_tag", "model_digest", "inference_seed", "sbert_model",
    "sbert_revision", "sbert_snapshot_sha256", "task_queue_sha256", "prompt_hashes", "config_hashes",
}
ACQUISITION_PROTOCOL_REQUIRED = {
    "repository_commit", "protocol", "protocol_sha256", "task_queue_sha256",
    "acquisition_action_budget", "inference_seed", "prompt_hashes", "decision_record_sha256",
}
REQUIRED_INPUTS = {
    "environment": ENVIRONMENT_REQUIRED,
    "protocol": PROTOCOL_REQUIRED,
    "acquisition-environment": ACQUISITION_ENVIRONMENT_REQUIRED,
    "acquisition-protocol": ACQUISITION_PROTOCOL_REQUIRED,
}
ACQUISITION_EVIDENCE_MODE = "non_scientific_acquisition_check"


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def git_state(root: Path) -> tuple[str | None, bool, str | None]:
    try:
        commit = subprocess.run(("git", "rev-parse", "HEAD"), cwd=root, text=True, capture_output=True, check=False).stdout.strip()
        status = subprocess.run(("git", "status", "--porcelain"), cwd=root, text=True, capture_output=True, check=False)
    except OSError as exc:
        return None, False, type(exc).__name__
    if not commit:
        return None, False, "repository commit is unavailable"
    return commit, status.returncode == 0 and not status.stdout.strip(), None


def read_freeze(path: Path, kind: str) -> tuple[FreezeManifest | None, list[str]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        manifest = FreezeManifest(**data)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return None, [f"invalid {kind} freeze: {type(exc).__name__}"]
    if manifest.kind != kind:
        return None, [f"freeze kind must be {kind}"]
    if manifest.input_fingerprint != _sha(manifest.inputs):
        return None, [f"{kind} freeze input fingerprint mismatch"]
    return manifest, []


_read_manifest = read_freeze


def build_freeze(root: Path, kind: str, approval: dict[str, Any], pilot_report: dict[str, Any]) -> FreezeManifest:
    if kind not in REQUIRED_INPUTS:
        raise ValueError(f"unsupported freeze kind: {kind}")
    inputs = approval.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("approval file must contain an inputs object")
    missing = sorted(REQUIRED_INPUTS[kind] - set(inputs))
    if missing:
        raise ValueError("approval file lacks frozen inputs: " + ", ".join(missing))
    if approval.get("approval_kind") not in (None, kind):
        raise ValueError("approval file kind does not match the requested freeze")
    commit, clean, error = git_state(root)
    if error or not clean or not commit:
        raise ValueError("freeze requires a clean repository with a resolved commit")
    if kind.startswith("acquisition-"):
        if (
            pilot_report.get("mode") != ACQUISITION_EVIDENCE_MODE
            or pilot_report.get("passed") is not True
            or pilot_report.get("scientific_evidence") is not False
        ):
            raise ValueError("acquisition freeze requires a passed non-scientific acquisition check report")
        if pilot_report.get("repository_commit") != commit or inputs.get("repository_commit") != commit:
            raise ValueError("acquisition freeze evidence and inputs must match the current commit")
        pilot_run_id = str(pilot_report.get("run_id", ""))
    else:
        if pilot_report.get("mode") != "real" or pilot_report.get("experimental_ready") is not True or pilot_report.get("go_no_go", {}).get("decision") != "go":
            raise ValueError("freeze requires an approved real Phase 7 go report with experimental_ready=true")
        pilot_run_id = str(pilot_report.get("pilot_run_id", ""))
    if not pilot_run_id:
        raise ValueError("pilot report lacks a run identifier")
    approval_meta = approval.get("approval")
    if not isinstance(approval_meta, dict) or not approval_meta.get("approved_by") or not approval_meta.get("approved_at"):
        raise ValueError("approval file requires approved_by and approved_at metadata")
    if approval_meta.get("status", "APPROVED") != "APPROVED":
        raise ValueError("approval status is not APPROVED")
    return FreezeManifest(1, kind, utc_now(), commit, pilot_run_id, _sha(pilot_report), inputs, _sha(inputs), approval_meta)


def write_freeze(root: Path, manifest: FreezeManifest) -> Path:
    path = root / "artifacts" / "freezes" / f"{manifest.kind}-freeze.json"
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite immutable freeze: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def validate_final_gates(root: Path) -> FreezeValidation:
    environment, errors = read_freeze(root / "artifacts" / "freezes" / "environment-freeze.json", "environment")
    protocol, protocol_errors = read_freeze(root / "artifacts" / "freezes" / "protocol-freeze.json", "protocol")
    errors.extend(protocol_errors)
    commit, clean, error = git_state(root)
    if error:
        errors.append(error)
    elif not clean:
        errors.append("repository working tree changed since freeze")
    for manifest in (environment, protocol):
        if manifest and commit != manifest.repository_commit:
            errors.append(f"repository commit changed since {manifest.kind} freeze")
    if environment and protocol:
        for key in PROTOCOL_REQUIRED:
            if environment.inputs.get(key) != protocol.inputs.get(key):
                errors.append(f"environment/protocol freeze mismatch for {key}")
    return FreezeValidation(not errors, tuple(errors), environment, protocol)
