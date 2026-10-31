"""Verified offline export of a stopped complete or incomplete run."""
from __future__ import annotations

import json
import errno
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from uuid import uuid4

from rq1.university_gpu.models import atomic_write_bytes, atomic_write_json, content_hash, load_json, sha256_file
from rq1.university_gpu.scheduler import Coordinator, RunLock, RunStore, SchedulerError, _valid_unit_key


class SnapshotError(ValueError):
    pass


def _fsync_path(path: Path) -> None:
    """Flush a file or directory when the host platform supports it."""
    if path.is_file():
        with path.open("r+b") as handle:
            os.fsync(handle.fileno())
        return
    if os.name == "nt":
        return
    try:
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        fd = os.open(path, flags)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError as exc:
        unsupported = {errno.EINVAL, errno.EBADF}
        for name in ("ENOTSUP", "EOPNOTSUPP"):
            if hasattr(errno, name):
                unsupported.add(getattr(errno, name))
        if exc.errno not in unsupported:
            raise


def _fsync_checkpoint_tree(root: Path) -> None:
    for path in sorted(root.rglob("*")):
        if path.is_file():
            _fsync_path(path)
    for path in sorted((p for p in root.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
        _fsync_path(path)
    _fsync_path(root)


def _copy_immutable(source: Path, target: Path, rows: list[dict], root: Path) -> None:
    try:
        source.resolve(strict=True).relative_to(root.resolve())
    except (OSError, ValueError) as exc:
        raise SnapshotError(f"checkpoint source escapes or is missing: {source}") from exc
    cursor = root.resolve()
    for part in source.relative_to(root).parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise SnapshotError(f"checkpoint source contains symlink: {source}")
    if source.is_symlink() or not source.is_file():
        raise SnapshotError(f"invalid checkpoint source: {source}")
    rel = source.relative_to(root)
    destination = target / rel
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = sha256_file(source)
    shutil.copy2(source, destination)
    _fsync_path(destination)
    if sha256_file(source) != digest or sha256_file(destination) != digest:
        raise SnapshotError(f"checkpoint copy changed: {rel}")
    rows.append({"path": rel.as_posix(), "sha256": digest, "bytes": destination.stat().st_size})


def verify_unit_checkpoint(store: RunStore, unit_key: str) -> dict:
    """Verify the create-only local checkpoint for one accepted result."""
    if not _valid_unit_key(unit_key):
        raise SnapshotError(f"unsafe unit key for checkpoint: {unit_key}")
    try:
        checkpoint = store.path(Path("backups/units") / unit_key)
        report_path = store.path(Path("backups/units") / unit_key / "CHECKPOINT-MANIFEST.json")
    except SchedulerError as exc:
        raise SnapshotError(f"unsafe per-unit checkpoint path: {unit_key}") from exc
    if checkpoint.is_symlink() or not report_path.is_file() or report_path.is_symlink():
        raise SnapshotError(f"per-unit checkpoint is missing or unsafe: {unit_key}")
    report = load_json(report_path)
    if (not isinstance(report, dict) or report.get("schema") != "rq1-unit-checkpoint-v1" or report.get("run_id") != store.run_id
            or report.get("manifest_sha256") != store.manifest.get("manifest_sha256")
            or report.get("unit_key") != unit_key):
        raise SnapshotError(f"per-unit checkpoint identity mismatch: {unit_key}")
    if not isinstance(report.get("files"), list) or report.get("files_sha256") != content_hash(report.get("files")):
        raise SnapshotError(f"per-unit checkpoint metadata hash mismatch: {unit_key}")
    listed = set()
    for item in report["files"]:
        try:
            rel = PurePosixPath(item["path"])
            if rel.is_absolute() or not rel.parts or any(part in {".", ".."} for part in rel.parts):
                raise ValueError("unsafe relative path")
            target = checkpoint.joinpath(*rel.parts)
            if target.is_symlink() or any(parent.is_symlink() for parent in target.parents if parent != checkpoint.parent):
                raise ValueError("symlink in checkpoint path")
            target.resolve(strict=True).relative_to(checkpoint.resolve())
            if item["path"] in listed:
                raise ValueError("duplicate checkpoint path")
            listed.add(item["path"])
            if (target.is_symlink() or not target.is_file()
                    or target.stat().st_size != item["bytes"] or sha256_file(target) != item["sha256"]):
                raise ValueError("missing or corrupt checkpoint file")
        except (KeyError, TypeError, ValueError, OSError) as exc:
            raise SnapshotError(f"per-unit checkpoint file missing, unsafe or corrupt: {unit_key}") from exc
    entries = list(checkpoint.rglob("*"))
    if any(path.is_symlink() for path in entries):
        raise SnapshotError(f"per-unit checkpoint contains a symlink: {unit_key}")
    actual_files = {path.relative_to(checkpoint).as_posix() for path in entries
                    if path.is_file() and path != report_path}
    if actual_files != listed:
        raise SnapshotError(f"per-unit checkpoint file inventory is incomplete or contains extras: {unit_key}")
    result_path = checkpoint / "results" / "units" / f"{unit_key}.json"
    envelope = load_json(result_path)
    if (not isinstance(envelope, dict) or not isinstance(report.get("attempt_id"), str)
            or not report["attempt_id"].isalnum()
            or ("results/units/" + unit_key + ".json") not in listed
            or envelope.get("unit_key") != unit_key
            or envelope.get("attempt_id") != report.get("attempt_id")
            or sha256_file(result_path) != report.get("result_sha256")):
        raise SnapshotError(f"per-unit checkpoint result identity mismatch: {unit_key}")
    result = envelope.get("result")
    if (not isinstance(result, dict) or envelope.get("result_sha256") != content_hash(result)
            or result.get("attempt_id") != report.get("attempt_id")):
        raise SnapshotError(f"per-unit checkpoint embedded result is invalid: {unit_key}")
    if store.manifest.get("schema") == "rq1-feasibility-execution-manifest-v1":
        matches = list((checkpoint / "attempts" / unit_key).glob("*-" + str(report["attempt_id"])))
        if len(matches) != 1 or matches[0].is_symlink():
            raise SnapshotError(f"checkpoint raw attempt is missing or ambiguous: {unit_key}")
        task = next((task for task in store.manifest.get("tasks", [])
                     if task.get("task_id") == result.get("task_id")), None)
        if task is None:
            raise SnapshotError(f"checkpoint task is absent from the frozen manifest: {unit_key}")
        from rq1.university_gpu.evidence import validate_attempt_evidence
        issues = validate_attempt_evidence(matches[0], result, task, store.run_id)
        if issues:
            raise SnapshotError("checkpoint raw evidence is invalid: " + "; ".join(issues))
    current = store.path(Path("results/units") / f"{unit_key}.json")
    if not current.is_file() or current.is_symlink():
        raise SnapshotError(f"per-unit checkpoint has no authoritative result: {unit_key}")
    if sha256_file(current) != sha256_file(result_path):
        raise SnapshotError(f"authoritative result conflicts with per-unit checkpoint: {unit_key}")
    return {"status": "PASS", "unit_key": unit_key, "attempt_id": envelope.get("attempt_id"),
            "files": len(report["files"]), "checkpoint": str(checkpoint)}


def backup_completed_unit(store: RunStore, unit_key: str) -> Path:
    """Publish one completed result and its closed raw traces without copying prior units."""
    completed = store.completed()
    if unit_key not in completed:
        raise SnapshotError(f"cannot checkpoint a unit without validated authority: {unit_key}")
    envelope = completed[unit_key]
    attempt_id = str(envelope["attempt_id"])
    try:
        attempt_parent = store.path(Path("attempts") / unit_key)
        result_source = store.path(Path("results/units") / f"{unit_key}.json")
        parent = store.path("backups/units")
        target = store.path(Path("backups/units") / unit_key)
    except SchedulerError as exc:
        raise SnapshotError(f"unsafe per-unit checkpoint path: {unit_key}") from exc
    matches = list(attempt_parent.glob("*-" + attempt_id))
    if len(matches) != 1 or matches[0].is_symlink():
        raise SnapshotError(f"completed attempt is missing or ambiguous: {unit_key}")
    try:
        matches[0].resolve().relative_to(store.attempts_dir.resolve())
    except ValueError as exc:
        raise SnapshotError(f"completed attempt escapes run: {unit_key}") from exc
    parent = store.path("backups/units")
    parent.mkdir(parents=True, exist_ok=True)
    target = store.path(Path("backups/units") / unit_key)
    if target.exists():
        verify_unit_checkpoint(store, unit_key)
        _fsync_checkpoint_tree(target)
        _fsync_path(parent)
        return target
    temporary = parent / f".partial-{unit_key}-{uuid4().hex}"
    temporary.mkdir()
    try:
        files: list[dict] = []
        _copy_immutable(result_source, temporary, files, store.root)
        for source in sorted(matches[0].rglob("*")):
            if source.is_symlink():
                raise SnapshotError("completed attempt contains symlink")
            if source.is_file():
                _copy_immutable(source, temporary, files, store.root)
        report = {"schema": "rq1-unit-checkpoint-v1", "run_id": store.run_id,
                  "manifest_sha256": store.manifest["manifest_sha256"], "unit_key": unit_key,
                  "attempt_id": attempt_id, "result_sha256": sha256_file(result_source),
                  "files": sorted(files, key=lambda row: row["path"]),
                  "files_sha256": content_hash(sorted(files, key=lambda row: row["path"])),
                  "independent_backup": "not_verified"}
        atomic_write_json(temporary / "CHECKPOINT-MANIFEST.json", report, overwrite=False)
        if sha256_file(result_source) != report["result_sha256"] or unit_key not in store.completed():
            raise SnapshotError("authoritative result changed during unit checkpoint")
        _fsync_checkpoint_tree(temporary)
        temporary.rename(target)
        _fsync_path(parent)
        verify_unit_checkpoint(store, unit_key)
        return target
    except BaseException:
        # Leave a clearly named partial directory for diagnosis. It is never a checkpoint.
        raise


def backup_checkpoint(store: RunStore) -> Path:
    """Copy immutable authority files while the coordinator is the sole writer.

    Interrupted copies stay under a visibly incomplete `.partial-*` name for
    diagnosis. They are never promoted to a checkpoint and are excluded from
    verified exports.
    """
    try:
        parent = store.path("backups")
    except SchedulerError as exc:
        raise SnapshotError("backup destination escapes or traverses the run root") from exc
    parent.mkdir(exist_ok=True)
    token = uuid4().hex
    temporary = store.path(Path("backups") / f".partial-{token}")
    temporary.mkdir()
    files = []
    # Only the coordinator publishes terminal events. Active workers may still
    # append traces, so do not traverse their directories at all.
    events = store.event_rows()
    closed_ids = {str(row["attempt_id"]) for row in events
                  if row.get("event") in {"unit_completed", "recovered_authoritative_result"}}
    completed = store.completed()
    closed_ids.update(str(row["attempt_id"]) for row in completed.values())
    starts = {str(row["attempt_id"]): row for row in events if row.get("event") == "attempt_started"}
    closed_sources = []
    for attempt_id in sorted(closed_ids):
        start = starts.get(attempt_id)
        if start is None or not start.get("attempt_dir"):
            # Historical authority-only fixtures may not carry raw attempt paths.
            continue
        unit_key = str(start.get("unit_key", ""))
        matches = [p for p in (store.attempts_dir / unit_key).iterdir()
                   if p.name == attempt_id or p.name.endswith("-" + attempt_id)]
        if len(matches) != 1:
            raise SnapshotError("closed attempt directory missing or ambiguous")
        attempt_dir = matches[0]
        if (attempt_dir.is_symlink() or store.attempts_dir.resolve() not in attempt_dir.resolve().parents):
            raise SnapshotError("closed attempt directory escapes the run")
        for path in sorted(attempt_dir.rglob("*")):
            if path.is_symlink():
                raise SnapshotError("closed attempt directory contains a symlink")
            if path.is_file():
                closed_sources.append(path)
    sources = [store.manifest_path, *sorted(store.events_dir.glob("*.json")),
               *sorted(store.results_dir.glob("*.json")), *closed_sources]
    for source in sources:
        if source.is_symlink() or not source.is_file():
            raise SnapshotError(f"invalid checkpoint source: {source}")
        rel = source.relative_to(store.root)
        target = temporary / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        digest = sha256_file(source)
        shutil.copy2(source, target)
        if sha256_file(source) != digest or sha256_file(target) != digest:
            raise SnapshotError(f"checkpoint copy changed: {rel}")
        files.append({"path": rel.as_posix(), "sha256": digest, "bytes": target.stat().st_size})
    report = {"schema": "rq1-authority-checkpoint-v1", "run_id": store.run_id,
              "completed": len(completed), "files": files, "files_sha256": content_hash(files),
              "attempt_evidence_scope": "completed_attempts_only",
              "limitations": ["Failed attempts may still have live timed-out writers; retained in stopped export only"],
              "closed_attempt_ids": sorted(closed_ids),
              "omitted_open_attempt_ids": sorted(set(starts) - closed_ids)}
    atomic_write_json(temporary / "CHECKPOINT-MANIFEST.json", report, overwrite=False)
    destination = store.path(Path("backups") / f"checkpoint-{report['completed']:04d}-{token}")
    _fsync_checkpoint_tree(temporary)
    temporary.rename(destination)
    _fsync_path(parent)
    return destination


def _attempt_states(store: RunStore) -> list[dict]:
    """Describe unresolved attempts without claiming mid-episode restoration."""
    rows = store.event_rows()
    states = {}
    for row in rows:
        attempt_id = str(row["attempt_id"])
        if row["event"] == "attempt_started":
            states[attempt_id] = {"attempt_id": attempt_id, "unit_key": str(row["unit_key"]),
                                  "state": "nonterminal_at_export"}
        elif row["event"] in {"attempt_failed", "unit_completed", "recovered_authoritative_result"}:
            states[attempt_id] = {"attempt_id": attempt_id, "unit_key": str(row["unit_key"]),
                                  "state": "failed" if row["event"] == "attempt_failed" else "completed"}
    for row in store.completed().values():
        states[str(row["attempt_id"])] = {"attempt_id": str(row["attempt_id"]),
                                            "unit_key": str(row["unit_key"]), "state": "completed"}
    return [states[key] for key in sorted(states)]


def export_progress(root: Path, run_root: Path, manifest_path: Path, destination: Path, relevance_path: Path | None = None) -> dict:
    root, run_root, destination = root.resolve(), run_root.resolve(), destination.resolve()
    if not run_root.is_dir():
        raise SnapshotError("run directory does not exist")
    if destination.exists() or destination == run_root or run_root in destination.parents:
        raise SnapshotError("destination must be new and outside the run directory")
    manifest = load_json(manifest_path)
    store = RunStore(run_root, manifest)
    if not store.manifest_path.is_file() or load_json(store.manifest_path) != manifest:
        raise SnapshotError("run manifest differs from the supplied frozen manifest")
    worker = type("OfflineWorker", (), {"worker_id": "snapshot"})()
    coordinator = Coordinator(store, [worker])
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(f".{destination.name}.incomplete-{uuid4().hex}")
    with RunLock(run_root / "RUN.lock", resume=False):
        validation = coordinator.validate()
        if validation["problems"]:
            raise SnapshotError("run contains invalid results: " + "; ".join(validation["problems"]))
        store.event_rows()
        completed = store.completed()
        from rq1.university_gpu.ops import status as run_status
        checkpoint_status = run_status(run_root, manifest_path)["local_unit_checkpoints"]
        anchor_setting = os.environ.get("RQ1_EVIDENCE_ANCHOR_DIR")
        anchor_report = store.evidence_anchor_status(completed)
        staging.mkdir()
        files = []

        def copy_checked(source: Path, relative: Path) -> None:
            if source.is_symlink() or not source.is_file():
                raise SnapshotError(f"source is not a regular file: {source}")
            before = sha256_file(source)
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            if sha256_file(source) != before or sha256_file(target) != before:
                raise SnapshotError(f"source changed or copy was corrupted: {source}")
            files.append({"path": relative.as_posix(), "sha256": before, "bytes": target.stat().st_size})

        for source in sorted(run_root.rglob("*")):
            rel = source.relative_to(run_root)
            if rel.parts[0] == "backups" or rel.as_posix() == "RUN.lock":
                continue
            if source.is_symlink():
                raise SnapshotError(f"run contains a symlink: {rel}")
            if source.is_file():
                copy_checked(source, Path("run") / rel)
        if anchor_setting and manifest.get("schema") == "rq1-feasibility-execution-manifest-v1":
            for unit in anchor_report["matched_units"]:
                copy_checked(Path(anchor_setting) / f"{unit}.json", Path("anchor-records") / f"{unit}.json")
        for source in sorted((root / "src" / "rq1").rglob("*.py")):
            copy_checked(source, Path("analysis-src/rq1") / source.relative_to(root / "src" / "rq1"))
        copy_checked(root / "pyproject.toml", Path("analysis-src/pyproject.toml"))
        if relevance_path:
            if sha256_file(relevance_path) != manifest["inputs"]["relevance_sha256"]:
                raise SnapshotError("relevance file differs from frozen manifest")
            copy_checked(relevance_path, Path("inputs/relevance.csv"))
        rows = [completed[str(unit["unit_key"])]["result"] for unit in manifest["units"]
                if str(unit["unit_key"]) in completed]
        data = "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows).encode("utf-8")
        atomic_write_bytes(staging / "results.jsonl", data, overwrite=False)
        files.append({"path": "results.jsonl", "sha256": sha256_file(staging / "results.jsonl"), "bytes": len(data)})
        report = {"schema": "rq1-progress-export-v1", "created_at": datetime.now(timezone.utc).isoformat(),
                  "run_id": store.run_id, "manifest_sha256": manifest["manifest_sha256"],
                  "complete": validation["passed"], "planned": validation["expected"],
                  "execution_status": "completed" if validation["passed"] else "incomplete",
                  "export_status": "verified_local_copy",
                  "independent_copy_status": "not_verified",
                  "source_checkpoint_status_at_export": checkpoint_status,
                  "evidence_anchor_status": anchor_report,
                  "attempt_states": _attempt_states(store),
                  "completed": validation["completed"], "missing_unit_keys": validation["missing"],
                  "source_commit": manifest["source_commit"], "files": sorted(files, key=lambda row: row["path"])}
        try:
            revision = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10, check=True).stdout.strip()
            status = subprocess.run(["git", "-C", str(root), "status", "--porcelain"], capture_output=True, text=True, timeout=10, check=True).stdout
            report["analysis_source"] = {"commit": revision, "clean": not bool(status.strip()),
                                         "matches_run_commit": revision == manifest["source_commit"] and not status.strip()}
        except (OSError, subprocess.SubprocessError):
            report["analysis_source"] = {"commit": None, "clean": None, "matches_run_commit": False}
        report["files_sha256"] = content_hash(report["files"])
        atomic_write_json(staging / "EXPORT-MANIFEST.json", report, overwrite=False)
        staging.rename(destination)
    return {"status": "EXPORTED", "destination": str(destination), "complete": report["complete"],
            "completed": report["completed"], "planned": report["planned"], "files": len(files),
            "execution_status": report["execution_status"], "export_status": report["export_status"],
            "independent_copy_status": "not_verified", "evidence_anchor_status": anchor_report,
            "source_checkpoint_status_at_export": checkpoint_status}


def verify_progress(destination: Path) -> dict:
    destination = destination.resolve()
    report = load_json(destination / "EXPORT-MANIFEST.json")
    if report.get("schema") != "rq1-progress-export-v1" or report.get("files_sha256") != content_hash(report.get("files")):
        raise SnapshotError("export manifest is invalid")
    seen = set()
    for item in report["files"]:
        rel = PurePosixPath(item["path"])
        if rel.is_absolute() or ".." in rel.parts or str(rel) in seen:
            raise SnapshotError("unsafe or duplicate export path")
        seen.add(str(rel))
        target = destination.joinpath(*rel.parts)
        if (target.is_symlink() or destination not in target.resolve().parents or not target.is_file()
                or target.stat().st_size != item["bytes"] or sha256_file(target) != item["sha256"]):
            raise SnapshotError(f"export file missing or corrupt: {rel}")
    actual = {path.relative_to(destination).as_posix() for path in destination.rglob("*") if path.is_file()}
    if actual != seen | {"EXPORT-MANIFEST.json"}:
        raise SnapshotError("export contains missing or unlisted files")
    manifest = load_json(destination / "run" / "manifest.json")
    store = RunStore(destination / "run", manifest)
    store.event_rows()
    validation = Coordinator(store, [type("OfflineWorker", (), {"worker_id": "verify"})()]).validate()
    if (validation["problems"] or validation["passed"] != report["complete"]
            or validation["completed"] != report["completed"] or validation["missing"] != report["missing_unit_keys"]
            or manifest["manifest_sha256"] != report["manifest_sha256"]):
        raise SnapshotError("export contents disagree with the recorded run status")
    if report.get("attempt_states") is not None and report["attempt_states"] != _attempt_states(store):
        raise SnapshotError("export attempt states differ from preserved event evidence")
    rows = [json.loads(line) for line in (destination / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    completed = store.completed()
    expected = [completed[str(unit["unit_key"])]["result"] for unit in manifest["units"]
                if str(unit["unit_key"]) in completed]
    if rows != expected:
        raise SnapshotError("analysis-ready results differ from authoritative unit records")
    return {"status": "PASS", "complete": report["complete"], "completed": report["completed"],
            "planned": report["planned"], "missing": len(report["missing_unit_keys"]), "files": len(seen),
            "evidence_anchor_status": report.get("evidence_anchor_status", {"classification": "internally_validated", "independent_protection": "not_verified"}),
            "source_checkpoint_status_at_export": report.get("source_checkpoint_status_at_export", "not_recorded"),
            "external_anchor_comparison": "performed" if (os.environ.get("RQ1_EVIDENCE_ANCHOR_DIR") and manifest.get("schema") == "rq1-feasibility-execution-manifest-v1") else "not_performed"}
