from __future__ import annotations

import signal
import time
import traceback
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from types import FrameType
from typing import Any, Callable, Iterable, Mapping
from uuid import uuid4

from rq1.experiment.models import (
    ExperimentUnit,
    RunExecutionContext,
    RunExecutor,
    RunFailure,
    RunOutcome,
)
from rq1.experiment.persistence import ExperimentStateError, ExperimentStore
from rq1.utils.time import utc_now


FaultHook = Callable[[str, ExperimentUnit, Mapping[str, Any] | None], None]


@dataclass(frozen=True)
class RunnerOptions:
    resume: bool = False
    retry_failed: bool = False
    max_runs: int | None = None
    fail_fast: bool = False
    backup_dir: Path | None = None
    require_backup: bool = False

    def __post_init__(self) -> None:
        if self.max_runs is not None and self.max_runs < 1:
            raise ValueError("max_runs must be at least 1")
        if self.retry_failed and not self.resume:
            raise ValueError("retry_failed requires resume")
        if self.require_backup and self.backup_dir is None:
            raise ValueError("require_backup requires backup_dir")


class _SignalBoundary(AbstractContextManager["_SignalBoundary"]):
    def __init__(self, callback: Callable[[int], None]) -> None:
        self.callback = callback
        self.previous: dict[int, Any] = {}

    def __enter__(self) -> "_SignalBoundary":
        try:
            for value in (signal.SIGINT, signal.SIGTERM):
                self.previous[value] = signal.getsignal(value)
                signal.signal(value, self._handle)
        except ValueError:
            self.previous.clear()  # Signal registration is main-thread only.
        return self

    def _handle(self, number: int, _frame: FrameType | None) -> None:
        self.callback(number)

    def __exit__(self, *_: object) -> None:
        for number, previous in self.previous.items():
            signal.signal(number, previous)


class DurableExperimentRunner:
    """Serial crash-safe runner whose JSONL journal is completion authority."""

    def __init__(
        self,
        store: ExperimentStore,
        *,
        fault_hook: FaultHook | None = None,
        progress: Callable[[str], None] | None = print,
    ) -> None:
        self.store = store
        self.fault_hook = fault_hook
        self.progress = progress
        self._stop_requested = False
        self._signal_number: int | None = None

    def request_stop(self, signal_number: int | None = None) -> None:
        self._stop_requested = True
        self._signal_number = signal_number

    def run(
        self,
        phase: str,
        units: Iterable[ExperimentUnit],
        configuration: Mapping[str, Any],
        executor: RunExecutor,
        options: RunnerOptions | None = None,
    ) -> dict[str, Any]:
        options = options or RunnerOptions()
        unit_list = list(units)
        self._validate_units(phase, unit_list)
        self._validate_configuration(configuration)
        started_monotonic = time.monotonic()
        if options.resume and not self.store.manifest_path.is_file():
            raise ExperimentStateError(
                f"Cannot resume without run_manifest.json: {self.store.directory}"
            )
        self.store.initialize()
        with self.store.lock, _SignalBoundary(self.request_stop):
            phase_manifest = self.store.register_phase(
                phase, unit_list, configuration, resume=options.resume
            )
            checkpoint, checkpoint_source = self.store.load_checkpoint()
            if options.resume and checkpoint is not None and checkpoint.get("blocking_error"):
                raise ExperimentStateError(
                    "Resume blocked by uncertain prior mutation: "
                    + str(checkpoint["blocking_error"].get("message", "unknown error"))
                )
            self.store.read_errors()
            # Validate retry lineage across the whole shared journal before
            # selecting the current phase's terminal identities.
            self.store.terminal_results()
            latest = self.store.terminal_results(phase=phase)
            self._validate_result_keys(unit_list, latest)
            self._record_stale_attempt(checkpoint, latest, phase)
            candidates = self._candidates(phase, unit_list, latest, options)
            if options.max_runs is not None:
                candidates = candidates[: options.max_runs]
            self._startup_message(phase, unit_list, latest, candidates, checkpoint_source, options)
            attempted = 0
            blocking_error: dict[str, Any] | None = None
            halted_on_failure = False
            for unit in candidates:
                if self._stop_requested:
                    break
                previous = latest.get(unit.run_key)
                attempt_index = int(previous.get("attempt_index", 0)) + 1 if previous else 1
                attempt_id = str(uuid4())
                output = self.store.logs / phase / unit.run_key / attempt_id
                output.mkdir(parents=True, exist_ok=False)
                self.store.write_checkpoint(
                    self._checkpoint(
                        phase, unit_list, latest, "running", phase_manifest,
                        current=unit, attempt_id=attempt_id,
                    )
                )
                self._backup(options)
                self._fault("before_execute", unit, None)
                context = RunExecutionContext(
                    self.store.experiment_id, attempt_id, attempt_index, output,
                    lambda: self._stop_requested,
                )
                before = time.monotonic()
                try:
                    outcome = executor(unit, context)
                    if not isinstance(outcome, RunOutcome):
                        raise TypeError("run executor must return RunOutcome")
                    self._validate_outcome(phase, outcome)
                    elapsed = time.monotonic() - before
                    record = self._result_record(
                        phase, unit, attempt_id, attempt_index, outcome, elapsed, previous
                    )
                except KeyboardInterrupt:
                    self.request_stop(int(signal.SIGINT))
                    self._record_interruption(phase, unit, attempt_id, attempt_index)
                    interruption_block = None
                    interruption_status = "interrupted"
                    if phase == "acquisition":
                        interruption_status = "blocked"
                        interruption_block = {
                            "run_key": unit.run_key,
                            "attempt_id": attempt_id,
                            "message": "Acquisition interruption has an unknown skill-write outcome",
                            "mutation_state_known": False,
                        }
                        blocking_error = interruption_block
                    self.store.write_checkpoint(
                        self._checkpoint(
                            phase, unit_list, latest, interruption_status, phase_manifest,
                            current=unit, attempt_id=attempt_id,
                            blocking_error=interruption_block,
                        )
                    )
                    self._backup(options)
                    break
                except BaseException as exc:
                    if isinstance(exc, (SystemExit, GeneratorExit)):
                        raise
                    attempted += 1
                    elapsed = time.monotonic() - before
                    record, safe = self._record_failure(
                        phase, unit, attempt_id, attempt_index, exc, elapsed, previous
                    )
                    self._backup(options)
                    latest[unit.run_key] = record
                    blocking = None if safe else {
                        "run_key": unit.run_key,
                        "attempt_id": attempt_id,
                        "message": str(exc),
                        "mutation_state_known": isinstance(exc, RunFailure) and exc.mutation_state_known,
                    }
                    self.store.write_checkpoint(
                        self._checkpoint(
                            phase, unit_list, latest, "failed" if safe else "blocked",
                            phase_manifest, current=unit, attempt_id=attempt_id,
                            blocking_error=blocking,
                        )
                    )
                    self._backup(options)
                    self._print_progress(unit, unit_list, latest, started_monotonic)
                    if options.fail_fast or not safe:
                        blocking_error = blocking
                        halted_on_failure = options.fail_fast and safe
                        break
                    continue

                attempted += 1
                self._fault("before_result_append", unit, record)
                self.store.append_result(record)
                # Results are the recovery authority, so mirror the committed
                # row before the checkpoint. A stale mirrored checkpoint is
                # reconciled from this journal on resume.
                self._backup(options)
                self._fault("after_result_fsync", unit, record)
                latest[unit.run_key] = record
                status = "stopping" if self._stop_requested else "running"
                self.store.write_checkpoint(
                    self._checkpoint(
                        phase, unit_list, latest, status, phase_manifest,
                        current=None, attempt_id=None,
                    )
                )
                self._fault("after_checkpoint", unit, record)
                self._backup(options)
                self._print_progress(unit, unit_list, latest, started_monotonic)

            remaining_options = RunnerOptions(
                resume=True, retry_failed=options.retry_failed
            )
            final_candidates = self._candidates(
                phase, unit_list, latest, remaining_options,
            )
            if blocking_error is not None:
                status = "blocked"
            elif halted_on_failure:
                status = "failed"
            elif self._stop_requested:
                status = "interrupted"
            elif final_candidates:
                status = "paused" if options.max_runs is not None else "incomplete"
            else:
                status = "completed"
            checkpoint = self._checkpoint(
                phase, unit_list, latest, status, phase_manifest,
                current=final_candidates[0] if final_candidates else None,
                attempt_id=None,
                blocking_error=blocking_error,
                next_override=final_candidates[0] if final_candidates else None,
            )
            self.store.write_checkpoint(checkpoint)
            self._backup(options)
            return {
                "experiment_id": self.store.experiment_id,
                "phase": phase,
                "status": status,
                "attempted_this_invocation": attempted,
                "completed": checkpoint["completed_run_count"],
                "failed": checkpoint["failed_run_count"],
                "total": len(unit_list),
                "next_run": checkpoint["next_run"],
                "output_directory": str(self.store.directory),
            }

    @staticmethod
    def _validate_units(phase: str, units: list[ExperimentUnit]) -> None:
        keys: set[str] = set()
        for expected_index, unit in enumerate(units, 1):
            if unit.phase != phase:
                raise ValueError(f"unit phase mismatch: {unit.phase} != {phase}")
            if unit.task_index != expected_index:
                raise ValueError("task/run order indices must be contiguous and unchanged")
            if unit.run_key in keys:
                raise ValueError(f"duplicate planned run identity: {unit.run_key}")
            keys.add(unit.run_key)

    @staticmethod
    def _validate_configuration(configuration: Mapping[str, Any]) -> None:
        missing = {"version", "model_name", "runtime_settings"} - set(configuration)
        if missing:
            raise ValueError(
                "experiment configuration lacks required reproducibility fields: "
                + ", ".join(sorted(missing))
            )
        if not isinstance(configuration.get("runtime_settings"), Mapping):
            raise ValueError("runtime_settings must be a mapping")

    @staticmethod
    def _validate_outcome(phase: str, outcome: RunOutcome) -> None:
        if not isinstance(outcome.success, bool):
            raise TypeError("RunOutcome.success must be boolean")
        if outcome.library_size_after is not None and outcome.library_size_after < 0:
            raise ValueError("library_size_after cannot be negative")
        if phase == "acquisition" and (
            outcome.skill_library_hash_after is None
            or outcome.library_size_after is None
        ):
            raise RunFailure(
                "Acquisition executor did not prove the post-run skill-library hash and size",
                safe_to_continue=False,
                mutation_state_known=False,
            )

    @staticmethod
    def _validate_result_keys(
        units: list[ExperimentUnit], latest: Mapping[str, Mapping[str, Any]]
    ) -> None:
        planned = {unit.run_key for unit in units}
        unexpected = set(latest) - planned
        if unexpected:
            raise ExperimentStateError(
                "results contain run identities outside the registered phase plan: "
                + ", ".join(sorted(unexpected))
            )

    def _candidates(
        self,
        phase: str,
        units: list[ExperimentUnit],
        latest: Mapping[str, Mapping[str, Any]],
        options: RunnerOptions,
    ) -> list[ExperimentUnit]:
        if options.retry_failed:
            candidates = [unit for unit in units if latest.get(unit.run_key, {}).get("status") == "failed"]
            if phase == "acquisition" and candidates:
                first = units.index(candidates[0])
                later_terminal = any(unit.run_key in latest for unit in units[first + 1 :])
                if later_terminal:
                    raise ExperimentStateError(
                        "Cannot retry an acquisition failure after later chronological results exist"
                    )
            return candidates
        return [unit for unit in units if unit.run_key not in latest]

    def _result_record(
        self,
        phase: str,
        unit: ExperimentUnit,
        attempt_id: str,
        attempt_index: int,
        outcome: RunOutcome,
        elapsed: float,
        previous: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        measurements = dict(outcome.measurements)
        record = {
            **measurements,
            "schema_version": 1,
            "experiment_id": self.store.experiment_id,
            "run_id": self.store.experiment_id,
            "run_key": unit.run_key,
            "attempt_id": attempt_id,
            "attempt_index": attempt_index,
            "phase": phase,
            "status": "completed",
            "task_id": unit.task_id,
            "task_index": unit.task_index,
            "condition": unit.condition,
            "seed": unit.seed,
            "library_name": unit.library_name,
            "library_size": unit.library_size,
            "skill_library_hash": unit.library_hash,
            "skill_library_hash_after": outcome.skill_library_hash_after,
            "library_size_after": outcome.library_size_after,
            "success": outcome.success,
            "recovery_success": outcome.recovery_success,
            "retrieved_skill_ids": list(outcome.retrieved_skill_ids),
            "retrieval_similarity_scores": list(outcome.retrieval_similarity_scores),
            "actions": outcome.actions,
            "steps": outcome.steps,
            "invalid_actions": outcome.invalid_actions,
            "latency_ms": outcome.latency_ms,
            "runtime_seconds": outcome.runtime_seconds if outcome.runtime_seconds is not None else elapsed,
            "errors": None,
            "timestamp": utc_now(),
            "log_paths": self._portable_log_paths(outcome.log_paths),
        }
        if previous is not None:
            record["supersedes_attempt_id"] = previous["attempt_id"]
            record["retry_reason"] = "retry_failed"
        return record

    def _portable_log_paths(self, paths: Iterable[str]) -> list[str]:
        values: list[str] = []
        for raw in paths:
            path = Path(raw)
            if path.is_absolute():
                try:
                    path = path.resolve().relative_to(self.store.directory.resolve())
                except ValueError as exc:
                    raise ValueError(
                        "run log paths must remain inside the portable experiment directory"
                    ) from exc
            if ".." in path.parts:
                raise ValueError("run log paths cannot escape the experiment directory")
            values.append(str(path))
        return values

    def _record_failure(
        self,
        phase: str,
        unit: ExperimentUnit,
        attempt_id: str,
        attempt_index: int,
        exc: BaseException,
        elapsed: float,
        previous: Mapping[str, Any] | None,
    ) -> tuple[dict[str, Any], bool]:
        error_id = str(uuid4())
        safe = isinstance(exc, RunFailure) and exc.safe_to_continue and exc.mutation_state_known
        detail = dict(exc.details) if isinstance(exc, RunFailure) else {}
        error = {
            "schema_version": 1,
            "error_id": error_id,
            "experiment_id": self.store.experiment_id,
            "run_key": unit.run_key,
            "attempt_id": attempt_id,
            "phase": phase,
            "task_id": unit.task_id,
            "condition": unit.condition,
            "seed": unit.seed,
            "error_type": type(exc).__name__,
            "message": str(exc),
            "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
            "safe_to_continue": safe,
            "mutation_state_known": isinstance(exc, RunFailure) and exc.mutation_state_known,
            "details": detail,
            "timestamp": utc_now(),
        }
        self.store.append_error(error)
        record = {
            "schema_version": 1,
            "experiment_id": self.store.experiment_id,
            "run_id": self.store.experiment_id,
            "run_key": unit.run_key,
            "attempt_id": attempt_id,
            "attempt_index": attempt_index,
            "phase": phase,
            "status": "failed",
            "task_id": unit.task_id,
            "task_index": unit.task_index,
            "condition": unit.condition,
            "seed": unit.seed,
            "library_name": unit.library_name,
            "library_size": unit.library_size,
            "skill_library_hash": unit.library_hash,
            "success": None,
            "recovery_success": None,
            "retrieved_skill_ids": [],
            "retrieval_similarity_scores": [],
            "actions": None,
            "steps": None,
            "invalid_actions": None,
            "latency_ms": None,
            "runtime_seconds": elapsed,
            "errors": [error_id],
            "timestamp": utc_now(),
            "log_paths": [],
        }
        if previous is not None:
            record["supersedes_attempt_id"] = previous["attempt_id"]
            record["retry_reason"] = "retry_failed"
        self.store.append_result(record)
        return record, safe

    def _record_interruption(
        self, phase: str, unit: ExperimentUnit, attempt_id: str, attempt_index: int
    ) -> None:
        self.store.append_error({
            "schema_version": 1,
            "error_id": str(uuid4()),
            "experiment_id": self.store.experiment_id,
            "run_key": unit.run_key,
            "attempt_id": attempt_id,
            "attempt_index": attempt_index,
            "phase": phase,
            "task_id": unit.task_id,
            "status": "interrupted",
            "signal": self._signal_number,
            "message": "Run interrupted before a terminal result was committed; it will restart from the beginning.",
            "timestamp": utc_now(),
        })

    def _record_stale_attempt(
        self,
        checkpoint: Mapping[str, Any] | None,
        latest: Mapping[str, Mapping[str, Any]],
        phase: str,
    ) -> None:
        if not checkpoint or checkpoint.get("current_phase") != phase:
            return
        current = checkpoint.get("current_run")
        if not isinstance(current, Mapping):
            return
        run_key = current.get("run_key")
        attempt_id = current.get("attempt_id")
        if not isinstance(run_key, str) or not isinstance(attempt_id, str):
            return
        if run_key in latest:
            return
        for error in self.store.read_errors():
            if error.get("attempt_id") == attempt_id and error.get("status") in {
                "interrupted", "crashed_or_interrupted"
            }:
                return
        self.store.append_error({
            "schema_version": 1,
            "error_id": str(uuid4()),
            "experiment_id": self.store.experiment_id,
            "run_key": run_key,
            "attempt_id": attempt_id,
            "phase": phase,
            "task_id": current.get("task_id"),
            "status": "crashed_or_interrupted",
            "message": "A prior in-progress attempt had no terminal result and will restart from the beginning.",
            "timestamp": utc_now(),
        })

    def _checkpoint(
        self,
        phase: str,
        units: list[ExperimentUnit],
        latest: Mapping[str, Mapping[str, Any]],
        status: str,
        phase_manifest: Mapping[str, Any],
        *,
        current: ExperimentUnit | None,
        attempt_id: str | None,
        blocking_error: Mapping[str, Any] | None = None,
        next_override: ExperimentUnit | None = None,
    ) -> dict[str, Any]:
        pending = [unit for unit in units if unit.run_key not in latest]
        completed = sum(value.get("status") == "completed" for value in latest.values())
        failed = sum(value.get("status") == "failed" for value in latest.values())
        next_unit = next_override or (pending[0] if pending else None)
        config = dict(phase_manifest.get("configuration", {}))
        ordered_latest = [latest[unit.run_key] for unit in units if unit.run_key in latest]
        last_library = next(
            (
                item for item in reversed(ordered_latest)
                if item.get("skill_library_hash_after") is not None
            ),
            None,
        )
        library_name = current.library_name if current else (next_unit.library_name if next_unit else None)
        library_size = current.library_size if current else (next_unit.library_size if next_unit else None)
        library_hash = current.library_hash if current else (next_unit.library_hash if next_unit else None)
        if last_library is not None:
            library_size = last_library.get("library_size_after", library_size)
            library_hash = last_library.get("skill_library_hash_after", library_hash)
        return {
            "status": status,
            "current_phase": phase,
            "completed_run_count": completed,
            "failed_run_count": failed,
            "terminal_run_count": completed + failed,
            "total_planned_run_count": len(units),
            "current_run": self._unit_pointer(current, attempt_id),
            "next_run": self._unit_pointer(next_unit, None),
            "phase_manifest_hash": phase_manifest.get("content_hash"),
            "experiment_configuration_version": config.get("version"),
            "config_hash": phase_manifest.get("configuration_hash"),
            "model_name": config.get("model_name"),
            "model_runtime_settings": config.get("runtime_settings", {}),
            "library_name": library_name,
            "library_size": library_size,
            "skill_library_hash": library_hash,
            "git_commit": config.get("git_commit") or self.store.manifest_runtime().get("git_commit"),
            "blocking_error": dict(blocking_error) if blocking_error else None,
        }

    @staticmethod
    def _unit_pointer(unit: ExperimentUnit | None, attempt_id: str | None) -> dict[str, Any] | None:
        if unit is None:
            return None
        return {
            "run_key": unit.run_key,
            "attempt_id": attempt_id,
            "task_id": unit.task_id,
            "task_index": unit.task_index,
            "condition": unit.condition,
            "seed": unit.seed,
            "phase": unit.phase,
        }

    def _backup(self, options: RunnerOptions) -> None:
        if options.backup_dir is None:
            return
        destination = self.store.mirror(options.backup_dir, required=options.require_backup)
        if destination is None and self.progress is not None:
            self.progress("WARNING: critical-file backup failed; local durable state is intact")

    def _fault(
        self, point: str, unit: ExperimentUnit, record: Mapping[str, Any] | None
    ) -> None:
        if self.fault_hook is not None:
            self.fault_hook(point, unit, record)

    def _startup_message(
        self,
        phase: str,
        units: list[ExperimentUnit],
        latest: Mapping[str, Mapping[str, Any]],
        candidates: list[ExperimentUnit],
        checkpoint_source: str | None,
        options: RunnerOptions,
    ) -> None:
        if self.progress is None:
            return
        if options.resume:
            self.progress(f"Checkpoint found ({checkpoint_source or 'reconstructed from results'}).")
            self.progress(f"{len(latest)} / {len(units)} runs already terminal.")
            if candidates:
                self.progress(f"Resuming from run {candidates[0].task_index}. No completed runs will be repeated.")
            else:
                self.progress(f"No unfinished {phase} runs remain.")
        else:
            self.progress(f"Starting {phase}: 0 / {len(units)} runs completed.")

    def _print_progress(
        self,
        unit: ExperimentUnit,
        units: list[ExperimentUnit],
        latest: Mapping[str, Mapping[str, Any]],
        started: float,
    ) -> None:
        if self.progress is None:
            return
        elapsed = max(time.monotonic() - started, 0.001)
        terminal = len(latest)
        completed = sum(value.get("status") == "completed" for value in latest.values())
        failed = sum(value.get("status") == "failed" for value in latest.values())
        remaining = len(units) - terminal
        eta = elapsed / terminal * remaining if terminal else 0.0
        self.progress(
            f"[{terminal}/{len(units)}] condition={unit.condition} task={unit.task_id} "
            f"seed={unit.seed} | completed={completed} failed={failed} | "
            f"elapsed={elapsed:.1f}s ETA={eta:.1f}s"
        )
