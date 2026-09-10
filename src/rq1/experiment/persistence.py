from __future__ import annotations

import json
import hashlib
import os
import platform
import socket
import subprocess
from contextlib import AbstractContextManager
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable, Mapping

from rq1.experiment.models import ExperimentUnit, canonical_hash, canonical_json
from rq1.utils.time import utc_now


SCHEMA_VERSION = 1
CRITICAL_FILES = (
    "checkpoint.json",
    "checkpoint.backup.json",
    "results.jsonl",
    "errors.jsonl",
    "run_manifest.json",
)


class ExperimentStateError(RuntimeError):
    pass


class CompatibilityError(ExperimentStateError):
    pass


class DuplicateResultError(ExperimentStateError):
    pass


class BackupError(ExperimentStateError):
    pass


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_bytes(path: Path, payload: bytes, *, backup: Path | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    _write_bytes(temporary, payload)
    if backup is not None and path.is_file():
        backup_temporary = backup.with_suffix(".tmp")
        _write_bytes(backup_temporary, path.read_bytes())
        os.replace(backup_temporary, backup)
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def atomic_write_json(path: Path, payload: Mapping[str, Any], *, backup: Path | None = None) -> None:
    data = (json.dumps(dict(payload), indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
    atomic_write_bytes(path, data, backup=backup)


def durable_append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (canonical_json(dict(payload)) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _git_commit(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ("git", "rev-parse", "HEAD"), cwd=root, text=True, capture_output=True,
            timeout=5, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = result.stdout.strip()
    return value if result.returncode == 0 and value else None


def runtime_manifest(root: Path) -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for name in (
        "hermes-agent", "alfworld", "torch", "transformers", "accelerate",
        "numpy", "matplotlib", "hermes-alfworld-rq1",
    ):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "packages": packages,
        "git_commit": _git_commit(root),
    }


def bind_repository_configuration(
    root: Path, configuration: Mapping[str, Any]
) -> dict[str, Any]:
    """Bind the supplied runtime settings to versioned local scientific inputs."""
    hashes: dict[str, str] = {}
    for relative_root in (Path("configs"), Path("hermes") / "prompts"):
        directory = root / relative_root
        if not directory.is_dir():
            continue
        for path in sorted(item for item in directory.rglob("*") if item.is_file()):
            relative = str(path.relative_to(root)).replace("\\", "/")
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            hashes[relative] = digest.hexdigest()
    return {**dict(configuration), "repository_input_hashes": hashes}


class ExperimentLock(AbstractContextManager["ExperimentLock"]):
    def __init__(self, path: Path) -> None:
        self.path = path
        self.acquired = False

    def __enter__(self) -> "ExperimentLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists() and self._stale():
            self.path.unlink(missing_ok=True)
        payload = canonical_json({"pid": os.getpid(), "hostname": socket.gethostname(), "created_at": utc_now()})
        try:
            descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as exc:
            raise ExperimentStateError(f"Experiment is already locked: {self.path}") from exc
        try:
            os.write(descriptor, payload.encode("utf-8"))
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        self.acquired = True
        return self

    def _stale(self) -> bool:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if payload.get("hostname") != socket.gethostname():
                return True
            pid = int(payload["pid"])
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        except (PermissionError, OSError):
            return False
        return False

    def __exit__(self, *_: object) -> None:
        if self.acquired:
            self.path.unlink(missing_ok=True)


class ExperimentStore:
    def __init__(self, root: Path, experiment_id: str, *, base: Path | None = None) -> None:
        if not experiment_id or any(value in experiment_id for value in ("/", "\\", "..")):
            raise ValueError("experiment_id must be a non-empty path-safe identifier")
        self.root = root.resolve()
        self.experiment_id = experiment_id
        self.directory = (base or self.root / "results" / "final") / experiment_id
        self.checkpoint_path = self.directory / "checkpoint.json"
        self.backup_checkpoint_path = self.directory / "checkpoint.backup.json"
        self.results_path = self.directory / "results.jsonl"
        self.errors_path = self.directory / "errors.jsonl"
        self.manifest_path = self.directory / "run_manifest.json"
        self.manifests = self.directory / "manifests"
        self.logs = self.directory / "logs"
        self.lock = ExperimentLock(self.directory / ".experiment.lock")

    def initialize(self) -> None:
        self.logs.mkdir(parents=True, exist_ok=True)
        self.manifests.mkdir(parents=True, exist_ok=True)
        for path in (self.results_path, self.errors_path):
            if not path.exists():
                _write_bytes(path, b"")

    def register_phase(
        self,
        phase: str,
        units: Iterable[ExperimentUnit],
        configuration: Mapping[str, Any],
        *,
        resume: bool,
    ) -> dict[str, Any]:
        units_list = [unit.to_dict() for unit in units]
        definition = {
            "schema_version": SCHEMA_VERSION,
            "phase": phase,
            "configuration": dict(configuration),
            "configuration_hash": canonical_hash(dict(configuration)),
            "planned_runs": units_list,
            "planned_run_count": len(units_list),
            "task_ids": [unit["task_id"] for unit in units_list],
            "conditions": sorted({unit["condition"] for unit in units_list}),
            "seeds": sorted({unit["seed"] for unit in units_list if unit["seed"] is not None}),
            "library_hashes": sorted({unit["library_hash"] for unit in units_list if unit["library_hash"]}),
        }
        phase_hash = canonical_hash(definition)
        phase_path = self.manifests / f"{phase}.json"
        if phase_path.exists():
            existing = self._read_json(phase_path, "phase manifest")
            if existing.get("content_hash") != phase_hash:
                raise CompatibilityError(
                    f"Cannot resume {phase}: task/configuration/condition/seed/library definition changed"
                )
            if not resume:
                raise ExperimentStateError(f"Phase already exists; use resume: {phase}")
        else:
            if resume:
                raise ExperimentStateError(f"Cannot resume unknown phase: {phase}")
            phase_payload = {**definition, "created_at": utc_now(), "content_hash": phase_hash}
            atomic_write_json(phase_path, phase_payload)

        if self.manifest_path.exists():
            manifest = self._read_json(self.manifest_path, "run manifest")
            if manifest.get("experiment_id") != self.experiment_id:
                raise CompatibilityError("run manifest experiment ID mismatch")
            self._validate_runtime_compatibility(manifest.get("runtime", {}))
        else:
            manifest = {
                "schema_version": SCHEMA_VERSION,
                "experiment_id": self.experiment_id,
                "experiment_start_time": utc_now(),
                "runtime": runtime_manifest(self.root),
                "phases": {},
            }
        phases = dict(manifest.get("phases", {}))
        expected_ref = {"path": str(phase_path.relative_to(self.directory)), "sha256": phase_hash}
        if phase in phases and phases[phase] != expected_ref:
            raise CompatibilityError(f"run manifest phase reference changed: {phase}")
        if phase not in phases:
            phases[phase] = expected_ref
            manifest["phases"] = phases
            manifest["updated_at"] = utc_now()
            atomic_write_json(self.manifest_path, manifest)
        return self._read_json(phase_path, "phase manifest")

    def _validate_runtime_compatibility(self, original: object) -> None:
        if not isinstance(original, dict):
            raise CompatibilityError("run manifest runtime metadata is invalid")
        current = runtime_manifest(self.root)
        mismatches: list[str] = []
        for field in ("python_version", "python_implementation"):
            if original.get(field) != current.get(field):
                mismatches.append(field)
        original_commit = original.get("git_commit")
        if original_commit is not None and original_commit != current.get("git_commit"):
            mismatches.append("git_commit")
        old_packages = original.get("packages")
        new_packages = current.get("packages")
        if isinstance(old_packages, dict) and isinstance(new_packages, dict):
            for name, version in old_packages.items():
                if version is not None and new_packages.get(name) != version:
                    mismatches.append(f"package:{name}")
        if mismatches:
            raise CompatibilityError(
                "Cannot resume with scientifically relevant runtime drift: "
                + ", ".join(mismatches)
            )

    def manifest_runtime(self) -> dict[str, Any]:
        if not self.manifest_path.is_file():
            return {}
        value = self._read_json(self.manifest_path, "run manifest").get("runtime", {})
        return dict(value) if isinstance(value, dict) else {}

    def _read_json(self, path: Path, label: str) -> dict[str, Any]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise ExperimentStateError(f"Invalid {label}: {path}") from exc
        if not isinstance(value, dict):
            raise ExperimentStateError(f"Invalid {label}: {path}")
        return value

    def load_checkpoint(self) -> tuple[dict[str, Any] | None, str | None]:
        for path, source in ((self.checkpoint_path, "primary"), (self.backup_checkpoint_path, "backup")):
            if not path.exists():
                continue
            try:
                value = self._read_json(path, "checkpoint")
            except ExperimentStateError:
                continue
            if value.get("experiment_id") == self.experiment_id:
                return value, source
        return None, None

    def write_checkpoint(self, payload: Mapping[str, Any]) -> None:
        value = dict(payload)
        value["schema_version"] = SCHEMA_VERSION
        value["experiment_id"] = self.experiment_id
        value["timestamp"] = utc_now()
        rotate_to_backup = None
        if self.checkpoint_path.is_file():
            try:
                current = self._read_json(self.checkpoint_path, "checkpoint")
                if current.get("experiment_id") == self.experiment_id:
                    rotate_to_backup = self.backup_checkpoint_path
            except ExperimentStateError:
                # Preserve an existing known-good backup until the replacement
                # primary has been atomically installed.
                rotate_to_backup = None
        atomic_write_json(self.checkpoint_path, value, backup=rotate_to_backup)

    def append_result(self, payload: Mapping[str, Any]) -> None:
        durable_append_jsonl(self.results_path, payload)

    def append_error(self, payload: Mapping[str, Any]) -> None:
        durable_append_jsonl(self.errors_path, payload)

    def read_results(self, *, repair_tail: bool = True) -> list[dict[str, Any]]:
        return self._read_jsonl(self.results_path, "results", repair_tail=repair_tail)

    def read_errors(self, *, repair_tail: bool = True) -> list[dict[str, Any]]:
        return self._read_jsonl(self.errors_path, "errors", repair_tail=repair_tail)

    def _read_jsonl(self, path: Path, label: str, *, repair_tail: bool) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        raw = path.read_bytes()
        lines = raw.splitlines(keepends=True)
        values: list[dict[str, Any]] = []
        valid_end = 0
        for index, line in enumerate(lines):
            final = index == len(lines) - 1
            content = line.rstrip(b"\r\n")
            if not content:
                valid_end += len(line)
                continue
            try:
                value = json.loads(content.decode("utf-8"))
            except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
                if not final or not repair_tail:
                    raise ExperimentStateError(f"Invalid interior {label} JSONL record at line {index + 1}") from exc
                self._recover_tail(path, raw[:valid_end], raw[valid_end:], label)
                break
            if not isinstance(value, dict):
                raise ExperimentStateError(f"Non-object {label} JSONL record at line {index + 1}")
            values.append(value)
            valid_end += len(line)
        if raw and lines and not lines[-1].endswith((b"\n", b"\r")) and valid_end == len(raw):
            if repair_tail:
                durable_append_bytes(path, b"\n")
        return values

    def _recover_tail(self, path: Path, valid: bytes, tail: bytes, label: str) -> None:
        recovery = self.logs / "recovery"
        recovery.mkdir(parents=True, exist_ok=True)
        suffix = hashlib.sha256(tail).hexdigest()[:12]
        name = f"{label}-partial-{utc_now().replace(':', '-')}-{suffix}.bin"
        _write_bytes(recovery / name, tail)
        atomic_write_bytes(path, valid)

    def terminal_results(
        self, *, phase: str | None = None, repair_tail: bool = True
    ) -> dict[str, dict[str, Any]]:
        latest: dict[str, dict[str, Any]] = {}
        for record in self.read_results(repair_tail=repair_tail):
            if phase is not None and record.get("phase") != phase:
                continue
            key = record.get("run_key")
            attempt = record.get("attempt_id")
            if not isinstance(key, str) or not isinstance(attempt, str):
                raise ExperimentStateError("Result record lacks run_key or attempt_id")
            previous = latest.get(key)
            if previous is not None:
                if record.get("supersedes_attempt_id") != previous.get("attempt_id"):
                    raise DuplicateResultError(f"Unexplained duplicate result for run_key={key}")
                if record.get("retry_reason") != "retry_failed" or previous.get("status") != "failed":
                    raise DuplicateResultError(f"Invalid retry lineage for run_key={key}")
            latest[key] = record
        return latest

    def mirror(self, backup_dir: Path, *, required: bool = False) -> Path | None:
        destination = backup_dir.resolve() / self.experiment_id
        try:
            if destination == self.directory.resolve():
                raise BackupError("backup destination must differ from the experiment directory")
            destination.mkdir(parents=True, exist_ok=True)
            paths = [self.directory / name for name in CRITICAL_FILES]
            paths.extend(sorted(self.manifests.glob("*.json")))
            for source in paths:
                if not source.is_file():
                    continue
                relative = source.relative_to(self.directory)
                target = destination / relative
                atomic_write_bytes(target, source.read_bytes())
            _fsync_directory(destination)
            return destination
        except Exception as exc:
            if required:
                if isinstance(exc, BackupError):
                    raise
                raise BackupError(f"critical-file backup failed: {type(exc).__name__}: {exc}") from exc
            return None


def durable_append_bytes(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_APPEND)
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def resolve_experiment_directory(root: Path, experiment_id: str) -> Path:
    candidates = [
        root / "results" / "final" / experiment_id,
        root / "results" / "checkpoint-tests" / experiment_id,
    ]
    existing = [path for path in candidates if path.is_dir()]
    if len(existing) != 1:
        if not existing:
            raise FileNotFoundError(f"Unknown experiment: {experiment_id}")
        raise ExperimentStateError(f"Ambiguous experiment ID exists in multiple output roots: {experiment_id}")
    return existing[0]


def copy_critical_directory(source: Path, backup_dir: Path, *, required: bool = True) -> Path | None:
    store = ExperimentStore(source.parents[2], source.name, base=source.parent)
    return store.mirror(backup_dir, required=required)
