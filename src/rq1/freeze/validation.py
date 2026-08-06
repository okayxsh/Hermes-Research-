"""Fail-closed validation for final-stage freezes.

Freeze files are evidence, not configuration generators: all scientific choices
must be supplied in the manual approval file after a real Phase 7 pilot.
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
}
PROTOCOL_REQUIRED = {"checkpoint_policy_sha256", "perturbation_policy_sha256", "solvability_policy_sha256", "action_limits", "timeout_policy", "snapshot_policy", "repetition_count"}


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


def _read_manifest(path: Path, kind: str) -> tuple[FreezeManifest | None, list[str]]:
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


def build_freeze(root: Path, kind: str, approval: dict[str, Any], pilot_report: dict[str, Any]) -> FreezeManifest:
    required = ENVIRONMENT_REQUIRED if kind == "environment" else PROTOCOL_REQUIRED
    inputs = approval.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("approval file must contain an inputs object")
    missing = sorted(required - set(inputs))
    if missing:
        raise ValueError("approval file lacks frozen inputs: " + ", ".join(missing))
    commit, clean, error = git_state(root)
    if error or not clean or not commit:
        raise ValueError("freeze requires a clean repository with a resolved commit")
    if pilot_report.get("mode") != "real" or pilot_report.get("experimental_ready") is not True or pilot_report.get("go_no_go", {}).get("decision") != "go":
        raise ValueError("freeze requires an approved real Phase 7 go report with experimental_ready=true")
    pilot_run_id = str(pilot_report.get("pilot_run_id", ""))
    if not pilot_run_id:
        raise ValueError("pilot report lacks pilot_run_id")
    approval_meta = approval.get("approval")
    if not isinstance(approval_meta, dict) or not approval_meta.get("approved_by") or not approval_meta.get("approved_at"):
        raise ValueError("approval file requires approved_by and approved_at metadata")
    return FreezeManifest(1, kind, utc_now(), commit, pilot_run_id, _sha(pilot_report), inputs, _sha(inputs), approval_meta)


def write_freeze(root: Path, manifest: FreezeManifest) -> Path:
    path = root / "artifacts" / "freezes" / f"{manifest.kind}-freeze.json"
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite immutable freeze: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def validate_final_gates(root: Path) -> FreezeValidation:
    environment, errors = _read_manifest(root / "artifacts" / "freezes" / "environment-freeze.json", "environment")
    protocol, protocol_errors = _read_manifest(root / "artifacts" / "freezes" / "protocol-freeze.json", "protocol")
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
