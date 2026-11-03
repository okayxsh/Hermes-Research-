"""Readback-verified copies to a human-approved independent destination."""
from __future__ import annotations
import os
import shutil
import tarfile
from pathlib import Path
from uuid import uuid4
from rq1.university_gpu.models import load_json, atomic_write_json, content_hash, sha256_file


def approved_destination(manifest: dict) -> Path:
    raw = Path(manifest["backup_destination"]).expanduser()
    if not raw.is_absolute() or ".." in raw.parts or any(p.is_symlink() for p in (raw, *raw.parents)):
        raise ValueError("approved destination must be absolute and free of traversal/symlinks")
    return raw.resolve()


def verify_last_protection(run_root: Path, manifest: dict) -> dict:
    from rq1.university_gpu.scheduler import RunStore
    from rq1.university_gpu.snapshot import verify_progress
    from rq1.university_gpu.evidence_anchor import verify_anchor_dir
    store = RunStore(run_root, manifest)
    completed = store.completed()
    if not completed:
        # This does not permit retrying unknown attempts; coordinator checks those.
        return {"status": "EMPTY"}
    receipt = load_json(run_root / "protection" / f"{len(completed):04d}.json")
    destination = approved_destination(manifest)
    copy = Path(receipt["destination"])
    if (copy.parent != destination or copy.is_symlink() or receipt["manifest_sha256"] != manifest["manifest_sha256"]
            or receipt["unit_keys"] != sorted(completed)):
        raise ValueError("protection receipt conflicts with approved destination or accepted units")
    if sha256_file(copy / "export.tar.gz") != receipt["archive_sha256"]:
        raise ValueError("independent archive missing or corrupt")
    result = verify_progress(copy / "export")
    anchors = verify_anchor_dir(copy / "export" / "run", copy / "export" / "run" / "manifest.json", copy / "anchors")
    source_anchors = verify_anchor_dir(run_root, run_root / "manifest.json", copy / "anchors")
    if result["completed"] != len(completed) or anchors["unanchored_units"] or source_anchors["unanchored_units"]:
        raise ValueError("independent copy coverage differs")
    return receipt


def protect_segment(root: Path, run_root: Path, manifest_path: Path) -> dict:
    from rq1.university_gpu.scheduler import RunStore, RunLock
    from rq1.university_gpu.snapshot import export_progress, verify_progress, _fsync_checkpoint_tree, _fsync_path
    from rq1.university_gpu.evidence_anchor import create_anchors, verify_anchor_dir
    manifest = load_json(manifest_path)
    destination = approved_destination(manifest)
    if not destination.is_dir() or destination == root.resolve() or root.resolve() in destination.parents or run_root.resolve() in destination.parents:
        raise ValueError("approved independent destination must exist outside source and run")
    count = len(RunStore(run_root, manifest).completed())
    receipt_path = run_root / "protection" / f"{count:04d}.json"
    if receipt_path.exists():
        return verify_last_protection(run_root, manifest)
    ident = f"{manifest['run_id']}-{count:04d}-{uuid4().hex}"
    staging = destination / (".incomplete-" + ident)
    final = destination / ident
    staging.mkdir(exist_ok=False)
    # Export takes the coordinator lock; create anchors after it returns, while
    # retaining our separate supervisor lock for the complete transfer operation.
    with RunLock(run_root / "PROTECT.lock", resume=False):
        export_progress(root, run_root, manifest_path, staging / "export")
        verify_progress(staging / "export")
        create_anchors(staging / "export" / "run", staging / "export" / "run" / "manifest.json", staging / "anchors")
        check = verify_anchor_dir(staging / "export" / "run", staging / "export" / "run" / "manifest.json", staging / "anchors")
        if check["unanchored_units"]:
            raise ValueError("anchor coverage incomplete")
        archive = staging / "export.tar.gz"
        with archive.open("xb") as stream:
            with tarfile.open(fileobj=stream, mode="w:gz") as tar:
                tar.add(staging / "export", arcname="export", recursive=True)
            stream.flush()
            os.fsync(stream.fileno())
        digest = sha256_file(archive)
        # Check every archived file against the copied export, without extracting.
        with tarfile.open(archive, "r:gz") as tar:
            for member in tar:
                if member.isfile():
                    import hashlib
                    if hashlib.sha256(tar.extractfile(member).read()).hexdigest() != sha256_file(staging / member.name):
                        raise ValueError("archive readback mismatch")
        atomic_write_json(staging / "CHECKSUM.json", {"archive_sha256": digest}, overwrite=False)
        _fsync_checkpoint_tree(staging)
        if final.exists():
            raise FileExistsError(final)
        staging.rename(final)
        _fsync_path(destination)
        receipt = {"status": "VERIFIED_AT_APPROVED_DESTINATION", "destination": str(final), "archive_sha256": digest,
                   "manifest_sha256": manifest["manifest_sha256"], "unit_keys": sorted(RunStore(run_root, manifest).completed()),
                   "independence": "destination independence is an explicit human approval assertion; byte integrity verified"}
        if len(receipt["unit_keys"]) != count:
            raise ValueError("source changed during protection; preserve copy and stop")
        atomic_write_json(receipt_path, receipt, overwrite=False)
    return verify_last_protection(run_root, manifest)
