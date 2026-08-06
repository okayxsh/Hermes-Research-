from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from rq1.orchestration.locks import StageLock
from rq1.setup.models import (
    SETUP_STAGES,
    SETUP_STAGE_MAP,
    SetupOptions,
    SetupStageResult,
    SetupState,
    path_for_report,
)
from rq1.setup.probes import executable_in_venv, python_executable
from rq1.setup.registry import SetupRegistry
from rq1.setup.runner import CommandRunner, SubprocessRunner
from rq1.setup.stages import (
    PRIMARY_MODEL,
    STAGE_HANDLERS,
    StageContext,
    StageFailure,
    StageOutcome,
    _model_info,
    _ollama_available,
    _resolve_hermes,
    _alfworld_package_available,
    _hermes_help_capabilities,
    _hermes_profiles_available,
    _python_environment_available,
    _smoke_model,
    _system_packages_installed,
    _valid_alfworld_data,
    commands_since,
    preflight_available,
)
from rq1.utils.hashing import sha256_text
from rq1.utils.time import utc_now


class SetupError(RuntimeError):
    pass


class SetupOrchestrator:
    def __init__(
        self,
        root: Path,
        options: SetupOptions,
        *,
        runner: CommandRunner | None = None,
        handlers: dict[str, Callable[[StageContext], StageOutcome]] | None = None,
        context_factory: Callable[[Path, SetupOptions, CommandRunner], StageContext] | None = None,
    ) -> None:
        self.root = root.resolve()
        self.options = options
        self.runner = runner or SubprocessRunner(dry_run=options.dry_run, verbose=options.verbose)
        self.handlers = handlers or STAGE_HANDLERS
        self.registry = SetupRegistry(self.root / "state" / "setup_status.json")
        self.run_id = str(uuid4())
        self._context_factory = context_factory or (lambda root, opts, command_runner: StageContext(root, opts, command_runner))
        self.context = self._context_factory(self.root, options, self.runner)

    def status(self) -> dict[str, dict[str, Any]]:
        return {name: asdict(state) for name, state in self.registry.load().items()}

    def _fingerprint(self, stage: str) -> str:
        payload = {
            "schema_version": 1,
            "stage": stage,
            "material_options": {
                "skip_system_packages": self.options.skip_system_packages,
                "skip_model": self.options.skip_model,
                "skip_alfworld_data": self.options.skip_alfworld_data,
                "install_fallback_model": self.options.install_fallback_model,
            },
            "python_version": self.root.joinpath(".python-version").read_text(encoding="utf-8").strip()
            if self.root.joinpath(".python-version").exists()
            else None,
            "lock_sha256": sha256_text(self.root.joinpath("uv.lock").read_text(encoding="utf-8"))
            if self.root.joinpath("uv.lock").exists()
            else None,
        }
        return sha256_text(json.dumps(payload, sort_keys=True))

    def _current(self, stage: str) -> bool:
        if stage == "preflight":
            return preflight_available(self.context)
        if stage == "system-packages":
            return _system_packages_installed(self.context)[0]
        if stage == "python-environment":
            return _python_environment_available(self.context)[0] and (self.root / "uv.lock").exists()
        if stage == "ollama":
            return _ollama_available(self.context)
        if stage == "hermes":
            hermes = _resolve_hermes(self.context)
            return bool(hermes and _hermes_help_capabilities(self.context, hermes)[0].get("supported"))
        if stage == "alfworld-package":
            return _alfworld_package_available(self.context)[0]
        if stage == "alfworld-data":
            return _valid_alfworld_data(self.context.data_dir)
        if stage == "candidate-models":
            return _smoke_model(self.context, PRIMARY_MODEL)[0]
        if stage == "base-profiles":
            return _hermes_profiles_available(self.context)[0]
        if stage == "installation-verification":
            # Always rerun the final smoke lifecycle; a historical report is evidence, not current state.
            return False
        return False

    def _prepare_force(self) -> list[str]:
        if not self.options.force_stage:
            return []
        if self.options.force_stage not in SETUP_STAGE_MAP:
            raise SetupError(f"Unknown setup stage: {self.options.force_stage}")
        if not self.options.yes and not self.options.dry_run:
            raise SetupError("--force-stage requires --yes because it invalidates setup state")
        if self.options.dry_run:
            names = [item.name for item in SETUP_STAGES]
            return names[names.index(self.options.force_stage) :]
        return self.registry.invalidate_from(self.options.force_stage)

    def run(self, only_stage: str | None = None, stop_after: str | None = None) -> dict[str, Any]:
        if only_stage and only_stage not in SETUP_STAGE_MAP:
            raise SetupError(f"Unknown setup stage: {only_stage}")
        if stop_after and stop_after not in SETUP_STAGE_MAP:
            raise SetupError(f"Unknown setup stage: {stop_after}")
        invalidated = self._prepare_force()
        results: list[SetupStageResult] = []
        stages = [SETUP_STAGE_MAP[only_stage]] if only_stage else list(SETUP_STAGES)
        states = self.registry.load()
        for stage in stages:
            state = states[stage.name]
            if state.status == "passed":
                fingerprint_matches = state.input_fingerprint == self._fingerprint(stage.name)
                if self.options.resume and fingerprint_matches and self._current(stage.name):
                    continue
                if not self.options.resume:
                    raise SetupError(f"{stage.name} already passed; use --resume or --force-stage {stage.name} --yes")
                if not self.options.dry_run:
                    self.registry.invalidate_from(stage.name)
                    states = self.registry.load()
            missing = [name for name in stage.prerequisites if states[name].status not in {"passed", "skipped"}]
            if missing and not self.options.dry_run:
                raise SetupError(f"Cannot start {stage.name}; prerequisites incomplete: {', '.join(missing)}")
            result = self._run_stage(stage.name)
            results.append(result)
            states = self.registry.load() if not self.options.dry_run else states
            if result.status in {"failed", "blocked"}:
                break
            if stop_after == stage.name:
                break
        aggregate = self._write_aggregate(results, invalidated)
        return aggregate

    def _run_stage(self, stage: str) -> SetupStageResult:
        started = utc_now()
        attempt_id = str(uuid4())
        fingerprint = self._fingerprint(stage)
        report_path = path_for_report(self.root, stage, attempt_id)
        command_offset = len(self.runner.commands)
        lock = self.root / "state" / "locks" / f"setup-{stage}.lock"
        if not self.options.dry_run:
            states = self.registry.load()
            states[stage] = SetupState(status="running", attempt_id=attempt_id, input_fingerprint=fingerprint)
            self.registry.save(states)
        try:
            with StageLock(lock):
                outcome = self.handlers[stage](self.context)
            status = "skipped" if self.options.dry_run else outcome.status
            skip_reason = "dry-run; no external mutation was executed" if self.options.dry_run else outcome.skip_reason
            result = SetupStageResult(
                stage=stage,
                status=status,
                started_at=started,
                completed_at=utc_now(),
                run_id=self.run_id,
                attempt_id=attempt_id,
                dry_run=self.options.dry_run,
                input_fingerprint=fingerprint,
                commands=commands_since(self.context, self.runner, command_offset),
                probes=[self.context.sanitize_payload(probe.to_dict()) for probe in outcome.probes],
                artifacts=[self.context.portable(path) for path in outcome.artifacts],
                warnings=[self.context.sanitize_text(item) for item in outcome.warnings],
                remediation=self.context.sanitize_text(outcome.remediation) if outcome.remediation else None,
                skip_reason=self.context.sanitize_text(skip_reason) if skip_reason else None,
                metadata=self.context.sanitize_payload(outcome.metadata),
            )
        except Exception as exc:
            remediation = exc.remediation if isinstance(exc, StageFailure) else "Inspect this stage report, resolve the cause, then rerun with --resume."
            result = SetupStageResult(
                stage=stage,
                status="failed",
                started_at=started,
                completed_at=utc_now(),
                run_id=self.run_id,
                attempt_id=attempt_id,
                dry_run=self.options.dry_run,
                input_fingerprint=fingerprint,
                commands=commands_since(self.context, self.runner, command_offset),
                errors=[self.context.sanitize_text(str(exc))],
                remediation=self.context.sanitize_text(remediation) if remediation else None,
            )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if not self.options.dry_run:
            states = self.registry.load()
            states[stage] = SetupState(
                status=result.status,
                attempt_id=attempt_id,
                report=self.context.portable(report_path),
                completed_at=result.completed_at,
                input_fingerprint=fingerprint,
            )
            self.registry.save(states)
        return result

    def _write_aggregate(self, results: list[SetupStageResult], invalidated: list[str]) -> dict[str, Any]:
        states = self.status() if not self.options.dry_run else {stage.name: {"status": "pending"} for stage in SETUP_STAGES}
        final = next((item for item in reversed(results) if item.stage == "installation-verification"), None)
        readiness = final.metadata if final else {}
        installation_ready = bool(readiness.get("installation_ready", False))
        payload = {
            "schema_version": 1,
            "run_id": self.run_id,
            "generated_at": utc_now(),
            "dry_run": self.options.dry_run,
            "status": "passed" if installation_ready else (results[-1].status if results else "pending"),
            "installation_ready": installation_ready,
            "pilot_ready": False,
            "real_integration_tested": False,
            "compatibility_claim": "No real Hermes-to-ALFWorld compatibility claim has been made.",
            "invalidated_stages": invalidated,
            "stages": states,
            "attempts": [result.to_dict() for result in results],
            "readiness": readiness,
        }
        path = self.root / "artifacts" / "stage_reports" / "installation.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(path)
        return payload
