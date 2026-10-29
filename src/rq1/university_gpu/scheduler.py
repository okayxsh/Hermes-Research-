"""Single-writer coordinator with immutable attempts and resumable unit results."""
from __future__ import annotations

import json
import os
import shutil
import socket
import time
from contextlib import nullcontext
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence
from uuid import uuid4

from rq1.university_gpu.manifest import validate_manifest
from rq1.university_gpu.models import atomic_write_bytes, atomic_write_json, content_hash, load_json, sha256_file


class SchedulerError(RuntimeError):
    pass


class EpisodeWorker(Protocol):
    worker_id: str

    def execute(self, unit: Mapping[str, Any], attempt_dir: Path, attempt_id: str) -> Mapping[str, Any]: ...


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


class RunLock:
    def __init__(self, path: Path, *, resume: bool) -> None:
        self.path = path
        self.resume = resume
        self.owned = False

    def __enter__(self) -> "RunLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"pid": os.getpid(), "host": socket.gethostname(), "created_at": _utc()}
        while True:
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    json.dump(payload, handle, sort_keys=True)
                    handle.flush()
                    os.fsync(handle.fileno())
                self.owned = True
                return self
            except FileExistsError:
                if not self.resume:
                    raise SchedulerError(f"run is already locked: {self.path}")
                try:
                    existing = load_json(self.path)
                    same_host = existing.get("host") == socket.gethostname()
                    pid = int(existing.get("pid", -1))
                except (OSError, ValueError, TypeError):
                    raise SchedulerError("existing run lock is malformed; manual inspection is required")
                if not same_host or _pid_alive(pid):
                    raise SchedulerError("existing run owner may still be active; refusing to steal its lock")
                stale = self.path.with_name(f"{self.path.name}.stale-{pid}-{uuid4().hex[:8]}")
                try:
                    os.replace(self.path, stale)
                except FileNotFoundError:
                    continue

    def __exit__(self, *_: object) -> None:
        if self.owned:
            try:
                current = load_json(self.path)
                if int(current.get("pid", -1)) == os.getpid() and current.get("host") == socket.gethostname():
                    self.path.unlink(missing_ok=True)
            except (OSError, ValueError, TypeError):
                pass
            self.owned = False


class RunStore:
    def __init__(self, root: Path, manifest: Mapping[str, Any]) -> None:
        self.root = root.resolve()
        self.manifest = dict(manifest)
        self.run_id = str(manifest.get("run_id", ""))
        if not self.run_id:
            raise SchedulerError("manifest lacks run_id")
        self.manifest_path = self.root / "manifest.json"
        self.events_dir = self.root / "events"
        self.results_dir = self.root / "results" / "units"
        self.attempts_dir = self.root / "attempts"
        self._next_event_sequence: int | None = None

    def initialize(self) -> None:
        problems = validate_manifest(self.manifest, require_frozen=True)
        if problems:
            raise SchedulerError("frozen manifest failed validation: " + "; ".join(problems))
        self.root.mkdir(parents=True, exist_ok=True)
        if self.manifest_path.exists():
            existing = load_json(self.manifest_path)
            if existing != self.manifest:
                raise SchedulerError("run ID already exists with a different frozen manifest or approval record")
        else:
            atomic_write_json(self.manifest_path, self.manifest, overwrite=False)
        self.events_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.attempts_dir.mkdir(parents=True, exist_ok=True)

    def event_rows(self) -> list[dict[str, Any]]:
        rows = []
        known_units = {str(unit["unit_key"]) for unit in self.manifest.get("units", [])}
        started_attempts: set[str] = set()
        for path in sorted(self.events_dir.glob("*.json")):
            value = load_json(path)
            if not isinstance(value, dict) or not isinstance(value.get("sequence"), int):
                raise SchedulerError(f"invalid event file: {path}")
            if value.get("event") not in {"attempt_started", "attempt_heartbeat", "attempt_failed", "unit_completed",
                                           "recovered_authoritative_result"}:
                raise SchedulerError(f"unknown run event type: {path}")
            if not value.get("unit_key") or not value.get("attempt_id"):
                raise SchedulerError(f"run event lacks unit or attempt identity: {path}")
            if str(value["unit_key"]) not in known_units:
                raise SchedulerError(f"run event references a unit outside the frozen manifest: {path}")
            if value.get("event") == "attempt_started":
                attempt_id = str(value["attempt_id"])
                if attempt_id in started_attempts:
                    raise SchedulerError(f"duplicate physical attempt identity: {attempt_id}")
                started_attempts.add(attempt_id)
            rows.append(value)
        rows.sort(key=lambda item: int(item["sequence"]))
        if [row["sequence"] for row in rows] != list(range(1, len(rows) + 1)):
            raise SchedulerError("event sequence has a gap or duplicate")
        self._next_event_sequence = len(rows) + 1
        return rows

    def append_event(self, value: Mapping[str, Any]) -> Path:
        if self._next_event_sequence is None:
            self.event_rows()
        assert self._next_event_sequence is not None
        sequence = self._next_event_sequence
        payload = {"sequence": sequence, "timestamp": _utc(), **dict(value)}
        path = self.events_dir / f"{sequence:08d}-{uuid4().hex}.json"
        atomic_write_json(path, payload, overwrite=False)
        self._next_event_sequence += 1
        return path

    def completed(self) -> dict[str, dict[str, Any]]:
        completed: dict[str, dict[str, Any]] = {}
        known = {unit["unit_key"] for unit in self.manifest["units"]}
        for path in sorted(self.results_dir.glob("*.json")):
            value = load_json(path)
            if not isinstance(value, dict):
                raise SchedulerError(f"authoritative result is not a JSON object: {path}")
            unit_key = str(value.get("unit_key", ""))
            if unit_key not in known or path.stem != unit_key:
                raise SchedulerError(f"authoritative result has unknown unit identity: {path}")
            if unit_key in completed:
                raise SchedulerError(f"duplicate authoritative result: {unit_key}")
            payload = value.get("result")
            if not isinstance(payload, dict) or value.get("result_sha256") != content_hash(payload):
                raise SchedulerError(f"result hash mismatch: {path}")
            completed[unit_key] = value
        return completed

    def record_result(self, unit: Mapping[str, Any], result: Mapping[str, Any], *, attempt_id: str, worker_id: str) -> Path:
        unit_key = str(unit["unit_key"])
        expected_units = {str(item["unit_key"]): item for item in self.manifest["units"]}
        expected = expected_units.get(unit_key)
        if expected is None:
            raise SchedulerError(f"assigned unit is outside the frozen manifest: {unit_key}")
        path = self.results_dir / f"{unit_key}.json"
        if path.exists():
            raise SchedulerError(f"completed unit cannot be overwritten: {unit_key}")
        body = dict(result)
        if body.get("unit_key") != unit_key:
            raise SchedulerError("worker result unit_key differs from the assigned unit")
        if (body.get("status") != "completed" or body.get("task_id") != expected.get("task_id")
                or body.get("condition") != expected.get("condition") or body.get("seed") != expected.get("seed")
                or body.get("task_family") != expected.get("task_family")
                or not isinstance(body.get("recovery_success"), bool)):
            raise SchedulerError("worker result status or scientific identity differs from the frozen assignment")
        envelope = {"unit_key": unit_key, "attempt_id": attempt_id, "worker_id": worker_id,
                    "result_sha256": content_hash(body), "result": body}
        atomic_write_json(path, envelope, overwrite=False)
        return path


class Coordinator:
    """Dynamically assigns one episode per worker and writes all authority centrally."""

    def __init__(self, store: RunStore, workers: Sequence[EpisodeWorker]) -> None:
        if not workers or len(workers) > 4:
            raise SchedulerError("one to four workers are supported")
        ids = [worker.worker_id for worker in workers]
        if len(set(ids)) != len(ids):
            raise SchedulerError("worker IDs must be unique")
        self.store = store
        self.workers = list(workers)
        self.stop_file = store.root / "STOP"

    def request_stop(self, reason: str = "operator requested safe stop") -> None:
        self.stop_file.write_text(reason.strip() + "\n", encoding="utf-8")

    def run(
        self,
        *,
        resume: bool = False,
        retry_failed: bool = False,
        cutoff_epoch: float | None = None,
        max_attempts: int | None = None,
        lock_held: bool = False,
        storage_minimum_bytes: int = 1_000_000_000,
        minimum_start_seconds: float = 1800,
        heartbeat_seconds: float = 60,
    ) -> dict[str, Any]:
        if heartbeat_seconds <= 0:
            raise SchedulerError("heartbeat interval must be positive")
        if max_attempts is not None and max_attempts < 1:
            raise SchedulerError("max_attempts must be positive")
        if storage_minimum_bytes < 0:
            raise SchedulerError("storage minimum cannot be negative")
        self.store.initialize()
        with (nullcontext() if lock_held else RunLock(self.store.root / "RUN.lock", resume=resume)):
            if self.stop_file.exists() and not resume:
                raise SchedulerError("STOP file exists; resume only after reviewing the run state")
            if resume:
                self.stop_file.unlink(missing_ok=True)
            rows = self.store.event_rows()
            attempts: dict[str, int] = {}
            latest: dict[str, str] = {}
            for row in rows:
                if row.get("event") == "attempt_started":
                    key = str(row["unit_key"])
                    attempts[key] = attempts.get(key, 0) + 1
                    latest[key] = "running"
                elif row.get("event") in {"attempt_failed", "unit_completed"}:
                    latest[str(row["unit_key"])] = str(row["event"])
            completed = self.store.completed()
            # A crash after the atomic result write but before the completion event is recoverable.
            for key, item in completed.items():
                if latest.get(key) != "unit_completed":
                    self.store.append_event({"event": "recovered_authoritative_result", "unit_key": key,
                                             "attempt_id": item.get("attempt_id"), "result_sha256": item.get("result_sha256")})
                    latest[key] = "unit_completed"
            known = {unit["unit_key"] for unit in self.store.manifest["units"]}
            if set(completed) - known:
                raise SchedulerError("completed results include a unit outside the frozen manifest")
            pending: list[Mapping[str, Any]] = []
            for unit in self.store.manifest["units"]:
                key = str(unit["unit_key"])
                if key in completed:
                    continue
                state = latest.get(key)
                if state == "attempt_failed" and not retry_failed:
                    continue
                pending.append(unit)
            active: dict[Future[Mapping[str, Any]], tuple[EpisodeWorker, Mapping[str, Any], str, Path, float]] = {}
            free_workers = list(self.workers)
            attempted = 0
            interrupted_by_cutoff = False
            failures: list[dict[str, Any]] = []
            backup_failed = False
            storage_failed = False
            with ThreadPoolExecutor(max_workers=len(self.workers), thread_name_prefix="rq1-gpu-scheduler") as pool:
                while pending or active:
                    if pending and not storage_failed:
                        try:
                            free_bytes = shutil.disk_usage(self.store.root).free
                            if free_bytes < storage_minimum_bytes:
                                raise OSError(f"free storage {free_bytes} below reserve {storage_minimum_bytes}")
                        except OSError as exc:
                            storage_failed = True
                            failures.append({"failure_class": "storage", "error_type": type(exc).__name__, "error": str(exc)})
                            try:
                                self.request_stop(f"storage check failed: {exc}")
                            except OSError:
                                pass
                    stop_requested = self.stop_file.exists() or backup_failed or storage_failed
                    before_cutoff = cutoff_epoch is None or time.time() + minimum_start_seconds < cutoff_epoch
                    while pending and free_workers and not stop_requested and before_cutoff and (max_attempts is None or attempted < max_attempts):
                        worker = free_workers.pop(0)
                        unit = pending.pop(0)
                        key = str(unit["unit_key"])
                        attempt_id = uuid4().hex
                        attempt_number = attempts.get(key, 0) + 1
                        attempts[key] = attempt_number
                        attempt_dir = self.store.attempts_dir / key / f"{attempt_number:03d}-{attempt_id}"
                        attempt_dir.mkdir(parents=True, exist_ok=False)
                        self.store.append_event({"event": "attempt_started", "unit_key": key, "attempt_id": attempt_id,
                                                 "attempt_number": attempt_number, "worker_id": worker.worker_id,
                                                 "task_id": unit.get("task_id"), "task_family": unit.get("task_family"),
                                                 "seed": unit.get("seed"), "condition": unit.get("condition"),
                                                 "attempt_dir": str(attempt_dir)})
                        future = pool.submit(worker.execute, unit, attempt_dir, attempt_id)
                        active[future] = (worker, unit, attempt_id, attempt_dir, time.monotonic())
                        attempted += 1
                    if not active:
                        if pending and cutoff_epoch is not None and not before_cutoff:
                            interrupted_by_cutoff = True
                        break
                    done, _ = wait(active, timeout=0.5, return_when=FIRST_COMPLETED)
                    now = time.monotonic()
                    for future, (worker, unit, attempt_id, attempt_dir, last_heartbeat) in list(active.items()):
                        if future not in done and now - last_heartbeat >= heartbeat_seconds:
                            self.store.append_event({"event": "attempt_heartbeat", "unit_key": str(unit["unit_key"]),
                                                     "attempt_id": attempt_id, "worker_id": worker.worker_id,
                                                     "attempt_dir": str(attempt_dir)})
                            active[future] = (worker, unit, attempt_id, attempt_dir, now)
                    for future in done:
                        worker, unit, attempt_id, attempt_dir, _ = active.pop(future)
                        key = str(unit["unit_key"])
                        try:
                            result = dict(future.result())
                            if result.get("unit_key") != key:
                                raise SchedulerError("worker returned a different unit identity")
                            output_path = self.store.record_result(unit, result, attempt_id=attempt_id, worker_id=worker.worker_id)
                            self.store.append_event({"event": "unit_completed", "unit_key": key, "attempt_id": attempt_id,
                                                     "worker_id": worker.worker_id, "result_sha256": sha256_file(output_path)})
                            free_workers.append(worker)
                            if not backup_failed and len(self.store.completed()) % 10 == 0:
                                try:
                                    from rq1.university_gpu.snapshot import backup_checkpoint
                                    backup_checkpoint(self.store)
                                except Exception as exc:
                                    backup_failed = True
                                    failures.append({"failure_class": "backup", "error_type": type(exc).__name__, "error": str(exc)})
                                    try:
                                        self.request_stop(f"automatic backup failed: {type(exc).__name__}: {exc}")
                                    except OSError:
                                        pass  # A full or failed disk must still stop new assignments.
                        except BaseException as exc:
                            info = {"unit_key": key, "attempt_id": attempt_id, "worker_id": worker.worker_id,
                                    "task_id": unit.get("task_id"), "task_family": unit.get("task_family"),
                                    "seed": unit.get("seed"), "condition": unit.get("condition"),
                                    "error_type": type(exc).__name__, "error": str(exc), "attempt_dir": str(attempt_dir),
                                    "failure_details": dict(getattr(exc, "details", {}) or {}),
                                    "safe_to_continue": getattr(exc, "safe_to_continue", None),
                                    "mutation_state_known": getattr(exc, "mutation_state_known", None)}
                            failures.append(info)
                            self.store.append_event({"event": "attempt_failed", **info})
                            if not getattr(worker, "quarantine_on_failure", False):
                                free_workers.append(worker)
                        free_workers.sort(key=lambda item: item.worker_id)
                    if (self.stop_file.exists() or backup_failed or storage_failed) and pending:
                        interrupted_by_cutoff = False
                        pending.clear()
            completed_now = self.store.completed()
            return {"run_id": self.store.run_id, "status": "completed" if len(completed_now) == len(self.store.manifest["units"]) else "paused" if not failures else "incomplete",
                    "planned_units": len(self.store.manifest["units"]), "completed_units": len(completed_now), "attempts_started": attempted,
                    "failures": failures, "cutoff_reached": interrupted_by_cutoff, "active_workers_at_return": len(active),
                    "worker_ids": [worker.worker_id for worker in self.workers]}

    def validate(self) -> dict[str, Any]:
        problems = validate_manifest(self.store.manifest, require_frozen=True)
        completed = self.store.completed()
        expected = {str(unit["unit_key"]): unit for unit in self.store.manifest["units"]}
        if set(completed) - set(expected):
            problems.append("authoritative result outside manifest")
        bad = []
        for key, envelope in completed.items():
            result = envelope["result"]
            if result.get("unit_key") != key or result.get("status") != "completed" or not isinstance(result.get("recovery_success"), bool):
                bad.append(key)
            if (result.get("task_id") != expected[key].get("task_id")
                    or result.get("task_family") != expected[key].get("task_family")
                    or result.get("condition") != expected[key].get("condition")
                    or result.get("seed") != expected[key].get("seed")):
                bad.append(key)
        if bad:
            problems.append(f"invalid result identity/status for {len(set(bad))} units")
        return {"passed": not problems and len(completed) == len(expected), "problems": problems,
                "expected": len(expected), "completed": len(completed), "missing": sorted(set(expected) - set(completed))}

    def merge(self) -> dict[str, Any]:
        validation = self.validate()
        if not validation["passed"]:
            raise SchedulerError("cannot merge incomplete or invalid run: " + "; ".join(validation["problems"]))
        completed = self.store.completed()
        lines = []
        for unit in self.store.manifest["units"]:
            result = completed[str(unit["unit_key"])]
            lines.append(json.dumps(result["result"], sort_keys=True, ensure_ascii=False, separators=(",", ":")))
        data = ("\n".join(lines) + "\n").encode("utf-8")
        digest = __import__("hashlib").sha256(data).hexdigest()
        path = self.store.root / "merged-results.jsonl"
        if path.exists():
            if sha256_file(path) != digest:
                raise SchedulerError("existing merged results differ; refusing overwrite")
        else:
            atomic_write_bytes(path, data, overwrite=False)
        return {"path": str(path), "sha256": digest, "units": len(lines)}
