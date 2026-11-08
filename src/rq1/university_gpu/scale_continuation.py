"""Owner-reviewed continuation of an existing protected engineering run."""
from __future__ import annotations

import hmac
import math
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

from rq1.university_gpu.models import content_hash, load_json, sha256_file

ORIGINAL_COMMIT = "e27188554f354d096700287a1b3c3f62cb5d11e6"
SCHEMA = "rq1-engineering-scale-resume-authorization-v1"
PIN_ENV = "RQ1_OWNER_REVIEWED_RESUME_AUTHORIZATION_SHA256"
ALLOWED_CHANGES = frozenset({
    "src/rq1/university_gpu/engineering_scale.py", "src/rq1/university_gpu/scale_continuation.py",
    "src/rq1/university_gpu/cli.py", "src/rq1/university_gpu/snapshot.py",
    "tests/test_engineering_scale_continuation.py", "university-gpu/docs/ENGINEERING_500_CONTINUATION.md",
})


class ContinuationError(ValueError):
    pass


def check_source(root: Path, original: str, continuation: str) -> None:
    if original != ORIGINAL_COMMIT or not re.fullmatch(r"[0-9a-f]{40}", str(continuation)) or continuation == original:
        raise ContinuationError("invalid original or continuation source identity")
    def git(*args: str) -> str:
        return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True, timeout=30).stdout.strip()
    if git("rev-parse", "HEAD") != continuation or git("status", "--porcelain"):
        raise ContinuationError("continuation checkout must be clean and match its approved commit")
    subprocess.run(["git", "-C", str(root), "merge-base", "--is-ancestor", original, continuation],
                   check=True, capture_output=True, timeout=30)
    changes = set(git("diff", "--name-only", "--no-renames", original, continuation, "--").splitlines())
    if not changes or not changes <= ALLOWED_CHANGES:
        raise ContinuationError("continuation commit changes files outside the reviewed scope")


def checkpoint_identity(run_root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    from rq1.university_gpu.scheduler import Coordinator, RunStore
    from rq1.university_gpu.snapshot import verify_unit_checkpoint
    from rq1.university_gpu.scale_protection import verify_last_protection
    if not run_root.is_dir() or run_root.is_symlink():
        raise ContinuationError("existing run directory is missing or unsafe")
    if (run_root / "STOP").exists() or (run_root / "STOP").is_symlink():
        raise ContinuationError("existing STOP marker requires manual review and must not be removed automatically")
    store = RunStore(run_root, manifest)
    if not store.manifest_path.is_file() or load_json(store.manifest_path) != manifest:
        raise ContinuationError("existing run manifest differs")
    completed = store.completed()
    rows = store.event_rows()
    starts = [r for r in rows if r["event"] == "attempt_started"]
    terminals = [r for r in rows if r["event"] == "unit_completed"]
    if (not completed or any(r["event"] in {"attempt_failed", "recovered_authoritative_result"} for r in rows)
            or len(starts) != len(completed) or len(terminals) != len(completed)
            or {r["attempt_id"] for r in starts} != {r["attempt_id"] for r in terminals}
            or {r["attempt_id"] for r in starts} != {x["attempt_id"] for x in completed.values()}
            or {r["unit_key"] for r in terminals} != set(completed)
            or any(completed.get(r["unit_key"], {}).get("attempt_id") != r["attempt_id"] for r in starts)
            or any(completed.get(r["unit_key"], {}).get("attempt_id") != r["attempt_id"]
                   or sha256_file(store.results_dir / (str(r["unit_key"]) + ".json")) != r.get("result_sha256") for r in terminals)):
        raise ContinuationError("attempt history is failed, unknown, duplicated or inconsistent")
    check = Coordinator(store, [type("OfflineWorker", (), {"worker_id": "offline"})()]).validate()
    if check["problems"]:
        raise ContinuationError("invalid existing results or event records")
    checkpoints = {}
    for key in sorted(completed):
        verify_unit_checkpoint(store, key)
        checkpoints[key] = sha256_file(store.path(Path("backups/units") / key / "CHECKPOINT-MANIFEST.json"))
    receipt_path = store.path(Path("protection") / f"{len(completed):04d}.json")
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise ContinuationError("protected backup receipt missing or unsafe")
    receipt = verify_last_protection(run_root, dict(manifest))
    if receipt.get("status") != "VERIFIED_AT_APPROVED_DESTINATION" or receipt.get("unit_keys") != sorted(completed):
        raise ContinuationError("protected backup does not match accepted units")
    return {"completed_count": len(completed), "completed_unit_keys_sha256": content_hash(sorted(completed)),
            "checkpoint_manifests_sha256": content_hash(checkpoints), "receipt_sha256": sha256_file(receipt_path),
            "receipt_archive_sha256": receipt["archive_sha256"]}


def verify_authorized_checkpoint_preserved(run_root: Path, manifest: Mapping[str, Any], identity: Mapping[str, Any]) -> None:
    """Ensure subsequent segments retain the exact reviewed starting authority."""
    from rq1.university_gpu.scheduler import RunStore
    from rq1.university_gpu.scale_protection import approved_destination
    count = identity.get("completed_count")
    if type(count) is not int or not 218 <= count < 500:
        raise ContinuationError("invalid reviewed starting checkpoint count")
    store = RunStore(run_root, manifest)
    path = store.path(Path("protection") / f"{count:04d}.json")
    if path.is_symlink() or not path.is_file() or sha256_file(path) != identity.get("receipt_sha256"):
        raise ContinuationError("reviewed starting protection receipt changed")
    receipt = load_json(path)
    keys = receipt.get("unit_keys")
    if (not isinstance(keys, list) or len(keys) != count or len(set(keys)) != count
            or content_hash(sorted(keys)) != identity.get("completed_unit_keys_sha256")
            or not set(keys) <= set(store.completed())):
        raise ContinuationError("reviewed starting completed-unit set changed")
    checkpoints = {key: sha256_file(store.path(Path("backups/units") / key / "CHECKPOINT-MANIFEST.json")) for key in sorted(keys)}
    if content_hash(checkpoints) != identity.get("checkpoint_manifests_sha256"):
        raise ContinuationError("reviewed starting checkpoints changed")
    copy = Path(receipt["destination"])
    archive = copy / "export.tar.gz"
    if (copy.is_symlink() or copy.parent != approved_destination(dict(manifest)) or archive.is_symlink()
            or not archive.is_file() or sha256_file(archive) != identity.get("receipt_archive_sha256")):
        raise ContinuationError("reviewed starting protected archive changed")


def verify_authorization(root: Path, manifest: Mapping[str, Any], approval: Mapping[str, Any],
                         authorization: Mapping[str, Any], identity: Mapping[str, Any], *, cutoff_epoch: float) -> str:
    from rq1.university_gpu.authorization import verify_engineering_authorization, verify_frozen_manifest_anchor
    from rq1.university_gpu.engineering_scale import MODEL_QUANTIZATION, PROFILE_ID, TOTAL_ACTION_BUDGET
    if (os.environ.get("RQ1_ENGINEERING_AUTHORITY_MODE") != "owner_reviewed"
            or manifest.get("profile_id") != PROFILE_ID or manifest.get("purpose") != "engineering_500"
            or manifest.get("episode_count") != 500 or manifest.get("maximum_attempts") != 500
            or manifest.get("source_commit") != ORIGINAL_COMMIT or dict(approval) != manifest.get("engineering_approval")):
        raise ContinuationError("original engineering scope, source or approval differs")
    try:
        verify_engineering_authorization(approval, candidate_sha256=manifest["candidate_manifest_sha256"],
            model_tag=manifest["model"]["tag"], model_digest=manifest["model"]["digest"], profile_id=PROFILE_ID,
            action_budget=TOTAL_ACTION_BUDGET, run_id=manifest["run_id"], model_quantization=MODEL_QUANTIZATION)
        verify_frozen_manifest_anchor(manifest)
    except RuntimeError as exc:
        raise ContinuationError(str(exc)) from exc
    if (authorization.get("schema") != SCHEMA or authorization.get("review_mode") != "owner_reviewed"
            or authorization.get("independent_review") is not False or authorization.get("reviewer_id") != "Yash"
            or not isinstance(authorization.get("approval_reference"), str) or not authorization["approval_reference"].strip()
            or "owner-reviewed" not in str(authorization.get("approval_provenance", "")).lower()
            or authorization.get("engineering_only") is not True or authorization.get("scientific_run") is not False
            or authorization.get("gpu_inference") is not True or authorization.get("test_only") is not False
            or authorization.get("run_id") != manifest["run_id"]
            or authorization.get("manifest_sha256") != manifest["manifest_sha256"]
            or authorization.get("original_source_commit") != manifest["source_commit"]
            or authorization.get("backup_destination") != manifest["backup_destination"]
            or authorization.get("checkpoint") != dict(identity)):
        raise ContinuationError("resume authorization identity, review or checkpoint scope differs")
    deadline = authorization.get("access_end_epoch")
    if (type(deadline) not in (int, float) or not math.isfinite(deadline)
            or type(cutoff_epoch) not in (int, float) or not math.isfinite(cutoff_epoch)
            or cutoff_epoch > deadline or cutoff_epoch <= time.time() + 4500):
        raise ContinuationError("resume deadline/cutoff must be valid and leave 75 minutes")
    digest = content_hash(dict(authorization))
    pin = os.environ.get(PIN_ENV, "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", pin) or not hmac.compare_digest(pin, digest):
        raise ContinuationError("owner-reviewed resume integrity pin is missing or differs")
    try:
        check_source(root, manifest["source_commit"], str(authorization.get("continuation_source_commit", "")))
    except (OSError, subprocess.SubprocessError) as exc:
        raise ContinuationError("cannot verify continuation source ancestry") from exc
    return digest
