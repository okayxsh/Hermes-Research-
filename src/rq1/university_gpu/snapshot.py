"""Verified offline export of a stopped complete or incomplete run."""
from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from uuid import uuid4

from rq1.university_gpu.models import atomic_write_bytes, atomic_write_json, content_hash, load_json, sha256_file
from rq1.university_gpu.scheduler import Coordinator, RunLock, RunStore, SchedulerError


class SnapshotError(ValueError):
    pass


def backup_checkpoint(store: RunStore) -> Path:
    """Copy immutable authority files while the coordinator is the sole writer."""
    parent = store.root / "backups"
    parent.mkdir(exist_ok=True)
    token = uuid4().hex
    temporary = parent / f".partial-{token}"
    temporary.mkdir()
    files = []
    sources = [store.manifest_path, *sorted(store.events_dir.glob("*.json")), *sorted(store.results_dir.glob("*.json"))]
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
              "completed": len(store.completed()), "files": files, "files_sha256": content_hash(files)}
    atomic_write_json(temporary / "CHECKPOINT-MANIFEST.json", report, overwrite=False)
    destination = parent / f"checkpoint-{report['completed']:04d}-{token}"
    temporary.rename(destination)
    return destination


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
            "completed": report["completed"], "planned": report["planned"], "files": len(files)}


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
    rows = [json.loads(line) for line in (destination / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    completed = store.completed()
    expected = [completed[str(unit["unit_key"])]["result"] for unit in manifest["units"]
                if str(unit["unit_key"]) in completed]
    if rows != expected:
        raise SnapshotError("analysis-ready results differ from authoritative unit records")
    return {"status": "PASS", "complete": report["complete"], "completed": report["completed"],
            "planned": report["planned"], "missing": len(report["missing_unit_keys"]), "files": len(seen)}
