"""Single-writer coordinator with immutable attempts and resumable unit results."""
from __future__ import annotations

import json
import math
import os
import shutil
import socket
import time
from contextlib import nullcontext
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import Any, Mapping, Protocol, Sequence
from uuid import uuid4

from rq1.university_gpu.manifest import validate_manifest
from rq1.university_gpu.models import atomic_write_bytes, atomic_write_json, content_hash, load_json, sha256_file
from rq1.utils.process_safety import require_posix_process_management


class SchedulerError(RuntimeError):
    pass


def _contained_run_path(root: Path, relative: str | Path) -> Path:
    """Resolve a run-relative path and reject traversal or symlinked components."""
    root = root.resolve()
    rel = Path(relative)
    windows = PureWindowsPath(str(relative))
    if (rel.is_absolute() or rel.drive or windows.is_absolute() or windows.drive or not rel.parts
            or any(part in {".", ".."} for part in rel.parts)
            or any(part in {".", ".."} for part in windows.parts)):
        raise SchedulerError(f"unsafe run-relative path: {relative}")
    candidate = root / rel
    cursor = root
    for part in rel.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise SchedulerError(f"run path contains a symlink: {cursor}")
    try:
        candidate.resolve(strict=False).relative_to(root)
    except ValueError as exc:
        raise SchedulerError(f"run path escapes the run root: {relative}") from exc
    return candidate


def _valid_unit_key(value: str) -> bool:
    return (bool(value) and len(value) <= 128
            and all(ch.isascii() and (ch.isalnum() or ch in "-_") for ch in value))


def _validate_run_manifest(manifest: Mapping[str, Any]) -> list[str]:
    if manifest.get("schema") == "rq1-feasibility-execution-manifest-v1":
        from rq1.university_gpu.feasibility import validate_feasibility_manifest

        return validate_feasibility_manifest(manifest, require_frozen=True)
    return validate_manifest(manifest, require_frozen=True)


def _validate_feasibility_result(result: Mapping[str, Any], expected: Mapping[str, Any],
                                 manifest: Mapping[str, Any]) -> list[str]:
    """Keep engineering-only records from being reinterpreted as scientific results."""
    if manifest.get("schema") != "rq1-feasibility-execution-manifest-v1":
        return []
    profile = "rq1-feasibility-200-v1"
    measurements = result.get("measurements")
    problems = []
    if (result.get("scientific_evidence") is not False or result.get("profile_id") != profile
            or result.get("model_tag") != manifest.get("model", {}).get("tag")
            or result.get("model_digest") != manifest.get("model", {}).get("digest")
            or result.get("model_quantization") != manifest.get("model", {}).get("quantization")
            or result.get("total_action_budget") != expected.get("total_action_budget")):
        problems.append("feasibility result top-level identity differs from its non-scientific manifest")
    if not isinstance(measurements, Mapping):
        problems.append("feasibility result lacks recorded episode measurements")
    else:
        if (measurements.get("scientific_evidence") is not False
                or measurements.get("evaluation_profile") != profile
                or measurements.get("result_schema") != "rq1-feasibility-episode-result-v1"
                or measurements.get("model") != manifest.get("model", {}).get("tag")
                or measurements.get("model_digest") != manifest.get("model", {}).get("digest")
                or measurements.get("condition") != expected.get("condition")
                or measurements.get("seed") != expected.get("seed")
                or measurements.get("inference_seed") != expected.get("seed")
                or measurements.get("inference_settings") != manifest.get("inference_settings")
                or measurements.get("total_action_budget") != expected.get("total_action_budget")
                or measurements.get("recovery_action_budget") != expected.get("recovery_action_budget")
                or measurements.get("checkpoint_digest") != expected.get("checkpoint_digest")
                or measurements.get("perturbation_digest") != expected.get("post_detour_digest")):
            problems.append("feasibility episode measurements do not preserve model, action, or perturbation identity")
    retrieved = result.get("retrieved_skill_ids")
    scores = result.get("retrieval_similarity_scores")
    if not isinstance(retrieved, list) or (expected.get("condition") == "D0" and retrieved) \
            or (expected.get("condition") == "D18" and len(retrieved) != 3):
        problems.append("feasibility result retrieval does not match D0 or top-3 D18")
    if not isinstance(scores, list) or not isinstance(retrieved, list) or len(scores) != len(retrieved):
        problems.append("feasibility result retrieval score records do not match retrieved skill IDs")
    if isinstance(retrieved, list):
        allowed = set(manifest.get("accum18_library", {}).get("skill_ids", []))
        if any(not isinstance(x, str) for x in retrieved) or len(set(map(str, retrieved))) != len(retrieved) or any(x not in allowed for x in retrieved):
            problems.append("retrieved IDs must be unique frozen-library members")
    if isinstance(scores, list) and any(type(x) not in (int, float) or not math.isfinite(x) or not -1.0000001 <= x <= 1.0000001 for x in scores):
        problems.append("retrieval scores must be finite cosine values")
    counts = [result.get(k) for k in ("actions", "steps", "invalid_actions")]
    if any(type(x) is not int or x < 0 for x in counts):
        problems.append("invalid recovery counts")
    elif isinstance(measurements, Mapping):
        actions, steps, invalid = counts
        budget = expected["recovery_action_budget"]
        if (actions > budget or not actions <= steps <= actions + 1 or invalid > steps
                or measurements.get("post_failure_actions") != actions
                or measurements.get("checkpoint_prefix_length") != expected["total_action_budget"] - budget - 1
                or not isinstance(measurements.get("recovery_actions"), list)
                or len(measurements.get("recovery_actions", [])) != actions):
            problems.append("recovery action/step accounting differs from frozen budget")
        success = result.get("recovery_success")
        reason = measurements.get("termination_reason")
        if (type(success) is not bool or result.get("success") is not success
                or measurements.get("recovery_success") is not success
                or measurements.get("task_completed") is not success
                or reason not in {"success", "action_selection_exhausted", "recovery_budget_exhausted", "environment_terminated_without_success"}
                or (reason == "success") != success
                or (reason == "recovery_budget_exhausted" and actions != budget)
                or measurements.get("retrieval_events") != (0 if expected["condition"] == "D0" else 1)
                or measurements.get("no_retrieval") is not (expected["condition"] == "D0")):
            problems.append("unsupported termination/retrieval state")
    return problems


class EpisodeWorker(Protocol):
    worker_id: str

    def execute(self, unit: Mapping[str, Any], attempt_dir: Path, attempt_id: str) -> Mapping[str, Any]: ...


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    require_posix_process_management("run-lock resume")
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return True  # An unexpected probe error does not establish that the owner is dead.
    return True


class RunLock:
    def __init__(self, path: Path, *, resume: bool) -> None:
        self.path = path
        self.resume = resume
        self.owned = False

    def __enter__(self) -> "RunLock":
        if self.path.is_symlink() or self.path.parent.is_symlink():
            raise SchedulerError("run lock path or its parent is a symlink")
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
                require_posix_process_management("run-lock resume")
                takeover = self.path.with_name(self.path.name + ".takeover")
                try:
                    takeover_fd = os.open(takeover, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                except FileExistsError as exc:
                    raise SchedulerError("run-lock takeover is already in progress or needs manual inspection") from exc
                try:
                    try:
                        existing = load_json(self.path)
                        if not isinstance(existing, Mapping):
                            raise ValueError("lock is not an object")
                        same_host = existing.get("host") == socket.gethostname()
                        pid = int(existing.get("pid", -1))
                        if pid <= 0:
                            raise ValueError("invalid lock PID")
                    except FileNotFoundError:
                        continue
                    except (OSError, ValueError, TypeError) as exc:
                        raise SchedulerError("existing run lock is malformed; manual inspection is required") from exc
                    if not same_host or _pid_alive(pid):
                        raise SchedulerError("existing run owner may still be active; refusing to steal its lock")
                    stale = self.path.with_name(f"{self.path.name}.stale-{pid}-{uuid4().hex[:8]}")
                    try:
                        os.replace(self.path, stale)
                    except FileNotFoundError:
                        continue
                    try:
                        fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                    except FileExistsError as exc:
                        raise SchedulerError("another run owner acquired the lock during takeover") from exc
                    with os.fdopen(fd, "w", encoding="utf-8") as handle:
                        json.dump(payload, handle, sort_keys=True)
                        handle.flush()
                        os.fsync(handle.fileno())
                    self.owned = True
                    return self
                finally:
                    os.close(takeover_fd)
                    takeover.unlink(missing_ok=True)

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
        absolute_root = root.expanduser().absolute()
        windows_root = PureWindowsPath(str(root))
        if (".." in root.parts or ".." in windows_root.parts or (os.name != "nt" and windows_root.drive)
                or any(path.is_symlink() for path in (absolute_root, *absolute_root.parents))):
            raise SchedulerError("run root cannot contain traversal or symlink components")
        self.root = root.resolve()
        self.manifest = dict(manifest)
        self.run_id = str(manifest.get("run_id", ""))
        if not self.run_id:
            raise SchedulerError("manifest lacks run_id")
        self.manifest_path = _contained_run_path(self.root, "manifest.json")
        self.events_dir = _contained_run_path(self.root, "events")
        self.results_dir = _contained_run_path(self.root, "results/units")
        self.attempts_dir = _contained_run_path(self.root, "attempts")
        self._next_event_sequence: int | None = None

    def path(self, relative: str | Path) -> Path:
        return _contained_run_path(self.root, relative)

    def initialize(self) -> None:
        problems = _validate_run_manifest(self.manifest)
        if problems:
            raise SchedulerError("frozen manifest failed validation: " + "; ".join(problems))
        self.root.mkdir(parents=True, exist_ok=True)
        for rel in ("manifest.json", "events", "results", "results/units", "attempts"):
            _contained_run_path(self.root, rel)
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
            try:
                self.path(path.relative_to(self.root))
            except SchedulerError as exc:
                raise SchedulerError(f"unsafe event path: {path}") from exc
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
        path = self.path(f"events/{sequence:08d}-{uuid4().hex}.json")
        atomic_write_json(path, payload, overwrite=False)
        self._next_event_sequence += 1
        return path

    def completed(self) -> dict[str, dict[str, Any]]:
        completed: dict[str, dict[str, Any]] = {}
        known = {unit["unit_key"] for unit in self.manifest["units"]}
        for path in sorted(self.results_dir.glob("*.json")):
            try:
                self.path(path.relative_to(self.root))
            except SchedulerError:
                raise
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
            if self.manifest.get("schema") == "rq1-feasibility-execution-manifest-v1":
                expected = next(u for u in self.manifest["units"] if u["unit_key"] == unit_key)
                issues = _validate_feasibility_result(payload, expected, self.manifest)
                issues.extend(self.validate_evidence(payload, str(value.get("attempt_id", ""))))
                if issues:
                    raise SchedulerError("invalid authoritative evidence: " + "; ".join(issues))
            completed[unit_key] = value
        anchor_dir = os.environ.get("RQ1_EVIDENCE_ANCHOR_DIR")
        if anchor_dir and self.manifest.get("schema") == "rq1-feasibility-execution-manifest-v1":
            from rq1.university_gpu.evidence_anchor import verify_anchor_dir
            try:
                verify_anchor_dir(self.root, self.manifest_path, Path(anchor_dir))
            except (OSError, ValueError, KeyError) as exc:
                raise SchedulerError("external evidence anchor is missing, unsafe, or conflicts with run evidence") from exc
        return completed

    def evidence_anchor_status(self, completed: Mapping[str, Any]) -> dict[str, Any]:
        """Report coverage; independent protection requires an operator-controlled trust boundary."""
        status = {"classification": "internally_validated", "independent_protection": "not_verified",
                  "matched_units": [], "unanchored_units": sorted(completed)}
        directory = os.environ.get("RQ1_EVIDENCE_ANCHOR_DIR")
        if directory and self.manifest.get("schema") == "rq1-feasibility-execution-manifest-v1":
            from rq1.university_gpu.evidence_anchor import verify_anchor_dir
            try:
                check = verify_anchor_dir(self.root, self.manifest_path, Path(directory))
            except (OSError, ValueError, KeyError) as exc:
                raise SchedulerError("external evidence anchor is missing, unsafe, or conflicts with run evidence") from exc
            missing = check["unanchored_units"]
            status.update({"classification": "compared_to_supplied_anchors", "unanchored_units": missing,
                           "matched_units": sorted(set(completed) - set(missing))})
        return status

    def validate_evidence(self, result: Mapping[str, Any], attempt_id: str) -> list[str]:
        from rq1.university_gpu.evidence import validate_attempt_evidence
        if result.get("attempt_id") != attempt_id or not attempt_id.isalnum():
            return ["result attempt identity mismatch"]
        unit_key = str(result.get("unit_key", ""))
        if not _valid_unit_key(unit_key):
            return ["result unit key is unsafe"]
        try:
            parent = self.path(Path("attempts") / unit_key)
        except SchedulerError:
            return ["attempt path escapes or traverses the run root"]
        matches = list(parent.glob("*-" + attempt_id))
        if len(matches) != 1 or matches[0].is_symlink() or self.attempts_dir.resolve() not in matches[0].resolve().parents:
            return ["attempt directory missing, ambiguous or unsafe"]
        task = next(t for t in self.manifest["tasks"] if t["task_id"] == result["task_id"])
        return validate_attempt_evidence(matches[0], result, task, self.run_id)

    def record_result(self, unit: Mapping[str, Any], result: Mapping[str, Any], *, attempt_id: str, worker_id: str) -> Path:
        unit_key = str(unit["unit_key"])
        if not _valid_unit_key(unit_key):
            raise SchedulerError("assigned unit key is unsafe")
        expected_units = {str(item["unit_key"]): item for item in self.manifest["units"]}
        expected = expected_units.get(unit_key)
        if expected is None:
            raise SchedulerError(f"assigned unit is outside the frozen manifest: {unit_key}")
        path = self.path(Path("results/units") / f"{unit_key}.json")
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
        if self.manifest.get("schema") == "rq1-feasibility-execution-manifest-v1":
            result_problems = _validate_feasibility_result(body, expected, self.manifest)
            result_problems.extend(self.validate_evidence(body, attempt_id))
            if result_problems:
                raise SchedulerError("feasibility worker result differs from its non-scientific profile: "
                                      + "; ".join(result_problems))
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
        self.stop_file = store.path("STOP")

    def request_stop(self, reason: str = "operator requested safe stop") -> None:
        if self.stop_file.is_symlink():
            raise SchedulerError("STOP marker path is a symlink")
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
        with (nullcontext() if lock_held else RunLock(self.store.path("RUN.lock"), resume=resume)):
            if self.stop_file.exists() and not resume:
                raise SchedulerError("STOP file exists; resume only after reviewing the run state")
            rows = self.store.event_rows()
            attempts: dict[str, int] = {}
            latest: dict[str, str] = {}
            for row in rows:
                if row.get("event") == "attempt_started":
                    key = str(row["unit_key"])
                    attempts[key] = attempts.get(key, 0) + 1
                    latest[key] = "running"
                elif row.get("event") in {"attempt_failed", "unit_completed", "recovered_authoritative_result"}:
                    latest[str(row["unit_key"])] = str(row["event"])
            completed = self.store.completed()
            completion_events = {str(row["unit_key"]) for row in rows
                                 if row.get("event") in {"unit_completed", "recovered_authoritative_result"}}
            if completion_events - set(completed):
                raise SchedulerError("completion event exists without its authoritative result; manual inspection required")
            failures: list[dict[str, Any]] = []
            backup_failed = False
            backup_succeeded = False
            full_snapshot_failed = False
            full_snapshot_succeeded = False
            # A coordinator interruption may occur after atomic result publication but
            # before checkpoint publication. Repair only missing checkpoints; an
            # existing invalid checkpoint is evidence of conflict and is never replaced.
            from rq1.university_gpu.snapshot import backup_completed_unit, verify_unit_checkpoint
            try:
                unit_checkpoints = self.store.path("backups/units")
                if unit_checkpoints.exists():
                    for entry in unit_checkpoints.iterdir():
                        if entry.name.startswith(".partial-"):
                            continue
                        if (not _valid_unit_key(entry.name) or entry.is_symlink() or not entry.is_dir()
                                or entry.name not in completed):
                            raise SchedulerError(
                                f"checkpoint exists without matching authoritative unit result: {entry.name}")
            except Exception as exc:
                backup_failed = True
                failures.append({"failure_class": "backup", "error_type": type(exc).__name__, "error": str(exc)})
            for key in sorted(completed):
                if backup_failed:
                    break
                try:
                    backup_completed_unit(self.store, key)
                    backup_succeeded = True
                except Exception as exc:
                    backup_failed = True
                    failures.append({"failure_class": "backup", "unit_key": key,
                                     "error_type": type(exc).__name__, "error": str(exc)})
                    break
            if resume and not backup_failed:
                self.stop_file.unlink(missing_ok=True)
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
                        attempt_dir = self.store.path(Path("attempts") / key / f"{attempt_number:03d}-{attempt_id}")
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
                        result_published = False
                        try:
                            result = dict(future.result())
                            if result.get("unit_key") != key:
                                raise SchedulerError("worker returned a different unit identity")
                            output_path = self.store.record_result(unit, result, attempt_id=attempt_id, worker_id=worker.worker_id)
                            result_published = True
                            self.store.append_event({"event": "unit_completed", "unit_key": key, "attempt_id": attempt_id,
                                                     "worker_id": worker.worker_id, "result_sha256": sha256_file(output_path)})
                            free_workers.append(worker)
                            try:
                                from rq1.university_gpu.snapshot import backup_completed_unit
                                backup_completed_unit(self.store, key)
                                backup_succeeded = True
                            except Exception as exc:
                                backup_failed = True
                                failures.append({"failure_class": "backup", "unit_key": key,
                                                 "error_type": type(exc).__name__, "error": str(exc)})
                                try:
                                    self.request_stop(f"per-unit checkpoint failed for {key}: {type(exc).__name__}: {exc}")
                                except OSError:
                                    pass
                            done_count = len(self.store.completed())
                            planned_count = len(self.store.manifest["units"])
                            # Retain the historical periodic full-run bundle for portability;
                            # per-unit checkpoints above are the immediate recovery authority.
                            if not backup_failed and (done_count % 10 == 0 or done_count == planned_count):
                                try:
                                    from rq1.university_gpu.snapshot import backup_checkpoint
                                    backup_checkpoint(self.store)
                                    full_snapshot_succeeded = True
                                except Exception as exc:
                                    backup_failed = True
                                    full_snapshot_failed = True
                                    failures.append({"failure_class": "backup", "backup_type": "full_snapshot",
                                                     "error_type": type(exc).__name__, "error": str(exc)})
                                    try:
                                        self.request_stop(f"automatic backup failed: {type(exc).__name__}: {exc}")
                                    except OSError:
                                        pass  # A full or failed disk must still stop new assignments.
                        except Exception as exc:
                            try:
                                result_published = result_published or self.store.path(Path("results/units") / f"{key}.json").exists()
                            except (OSError, ValueError, SchedulerError):
                                result_published = True
                            if result_published:
                                backup_failed = True
                                failures.append({"failure_class": "durability", "unit_key": key,
                                                 "attempt_id": attempt_id, "error_type": type(exc).__name__, "error": str(exc)})
                                try:
                                    self.request_stop(f"post-publication failure for {key}: {type(exc).__name__}: {exc}")
                                except OSError:
                                    pass
                                continue
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
            execution_status = ("completed" if len(completed_now) == len(self.store.manifest["units"])
                                else "incomplete" if failures else "paused")
            backup_status = "failed" if backup_failed else "succeeded" if backup_succeeded else "not_attempted"
            checkpointed = []
            checkpoint_missing = []
            for key in completed_now:
                try:
                    from rq1.university_gpu.snapshot import verify_unit_checkpoint
                    verify_unit_checkpoint(self.store, key)
                    checkpointed.append(key)
                except Exception as exc:
                    checkpoint_missing.append(key)
                    backup_failed = True
                    failures.append({"failure_class": "checkpoint_verification", "unit_key": key,
                                     "error_type": type(exc).__name__, "error": str(exc)})
            # A resume that already holds every authoritative result may still
            # lack the final full snapshot if its publication failed on the
            # completion pass. Retry before reporting success; backup_checkpoint
            # publishes a new create-only bundle and never replaces a snapshot.
            if resume and not backup_failed and completed_now \
                    and len(completed_now) == len(self.store.manifest["units"]):
                try:
                    from rq1.university_gpu.snapshot import backup_checkpoint
                    backup_checkpoint(self.store)
                    full_snapshot_succeeded = True
                except Exception as exc:
                    backup_failed = True
                    full_snapshot_failed = True
                    failures.append({"failure_class": "backup", "backup_type": "full_snapshot",
                                     "error_type": type(exc).__name__, "error": str(exc)})
                    try:
                        self.request_stop(f"resume full snapshot failed: {type(exc).__name__}: {exc}")
                    except OSError:
                        pass
            if backup_failed:
                backup_status = "failed"
            # Episode authority survives backup failure; callers cannot mistake
            # successful execution for verified durability.
            return {"run_id": self.store.run_id,
                    "status": "durability_failed" if backup_failed else execution_status,
                    "execution_status": execution_status, "backup_status": backup_status,
                    "unit_checkpoint_status": ("not_attempted" if not completed_now else
                                                "failed" if checkpoint_missing else "verified"),
                    "checkpointed_units": len(checkpointed), "uncheckpointed_units": checkpoint_missing,
                    "full_snapshot_status": ("failed" if full_snapshot_failed else
                                             "succeeded" if full_snapshot_succeeded else "not_attempted"),
                    "export_status": "not_performed", "independent_copy_status": "not_verified",
                    "planned_units": len(self.store.manifest["units"]), "completed_units": len(completed_now), "attempts_started": attempted,
                    "failures": failures, "cutoff_reached": interrupted_by_cutoff, "active_workers_at_return": len(active),
                    "worker_ids": [worker.worker_id for worker in self.workers]}

    def validate(self) -> dict[str, Any]:
        problems = _validate_run_manifest(self.store.manifest)
        rows = []
        try:
            rows = self.store.event_rows()
        except (OSError, ValueError, TypeError, SchedulerError) as exc:
            problems.append(f"invalid attempt event log: {type(exc).__name__}: {exc}")
        completed = self.store.completed()
        completed_events = {str(row["unit_key"]) for row in rows
                            if row.get("event") in {"unit_completed", "recovered_authoritative_result"}}
        if completed_events - set(completed):
            problems.append("completion event exists without its authoritative result")
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
            if _validate_feasibility_result(result, expected[key], self.store.manifest):
                bad.append(key)
        if bad:
            problems.append(f"invalid result identity/status for {len(set(bad))} units")
        result = {"passed": not problems and len(completed) == len(expected), "problems": problems,
                  "expected": len(expected), "completed": len(completed), "missing": sorted(set(expected) - set(completed))}
        if self.store.manifest.get("schema") == "rq1-feasibility-execution-manifest-v1":
            result["evidence_anchor_status"] = self.store.evidence_anchor_status(completed)
        return result

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
