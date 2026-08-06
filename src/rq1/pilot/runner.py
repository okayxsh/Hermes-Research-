"""Typed, resumable Phase 6 pilot runner."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from rq1.logging.run_registry import PilotAttemptBinding, RecoveryEvidenceBinding, RunRegistry
from rq1.orchestration.locks import StageLock
from rq1.pilot.catalog import PILOT_TESTS, PILOT_TEST_MAP, select_tests, validate_catalog
from rq1.pilot.fake import FakePilotRuntime
from rq1.pilot.gates import evidence_satisfies, unmet_prerequisites
from rq1.pilot.models import EvidenceLevel, EvidenceReference, PilotAttemptResult, PilotMode, PilotStatus, PilotTestSpec, RuntimeExecution
from rq1.pilot.real import RealPilotRuntime
from rq1.pilot.registry import PilotRegistry, PilotRegistryError
from rq1.pilot.report import generate_reports
from rq1.utils.ids import new_attempt_id
from rq1.utils.time import utc_now


PRIMARY_MODEL = "hermes3:8b"
FALLBACK_MODEL = "llama3.1:8b"


def _hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sanitize(value: Any, root: Path) -> Any:
    from rq1.hermes.models import redact
    value = redact(value)
    if isinstance(value, str): return value.replace(str(root), "$REPO")
    if isinstance(value, dict): return {str(key): _sanitize(item, root) for key, item in value.items()}
    if isinstance(value, list): return [_sanitize(item, root) for item in value]
    if isinstance(value, tuple): return [_sanitize(item, root) for item in value]
    return value


def input_fingerprint(root: Path, mode: PilotMode, candidate_model: str) -> str:
    digest = hashlib.sha256()
    digest.update(f"phase6:{mode.value}:{candidate_model}".encode())
    for relative in ("pyproject.toml", "uv.lock", "AGENTS.md"):
        path = root / relative
        if path.exists(): digest.update(relative.encode()); digest.update(path.read_bytes())
    configs = root / "configs"
    if configs.exists():
        for path in sorted(item for item in configs.rglob("*") if item.is_file()):
            digest.update(str(path.relative_to(root)).encode()); digest.update(path.read_bytes())
    return digest.hexdigest()


def catalog_payload() -> dict[str, Any]:
    return {"schema_version": 1, "errors": validate_catalog(), "tests": [item.to_dict() for item in PILOT_TESTS]}


class PilotRunner:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.registry = PilotRegistry(self.root)
        self.run_registry = RunRegistry(self.root / "state" / "run_registry.sqlite")

    def create_and_run(
        self, mode: PilotMode, *, test_id: str | None = None, group: str | None = None,
        start: str | None = None, end: str | None = None, include_prerequisites: bool = False,
        candidate_model: str = PRIMARY_MODEL,
    ) -> dict[str, Any]:
        self._validate_model(candidate_model)
        selected = select_tests(test_id=test_id, group=group, start=start, end=end, include_prerequisites=include_prerequisites)
        run_id = f"phase6-{mode.value}-{uuid4()}"
        fingerprint = input_fingerprint(self.root, mode, candidate_model)
        self.registry.create(run_id, mode.value, fingerprint, [item.test_id for item in selected], candidate_model)
        return self._run(run_id, selected, retry_failed=False)

    def resume(self, run_id: str, *, retry_failed: bool = False) -> dict[str, Any]:
        state = self.registry.load(run_id)
        mode = PilotMode(state["mode"])
        current = input_fingerprint(self.root, mode, state["candidate_model"])
        if current != state["input_fingerprint"]:
            self.registry.invalidate_all_passed(run_id)
            state = self.registry.load(run_id)
            state["input_fingerprint"] = current
            self.registry.save(state)
        self.registry.mark_stale_running_interrupted(run_id)
        selected = tuple(PILOT_TEST_MAP[test_id] for test_id in state["selected_tests"])
        return self._run(run_id, selected, retry_failed=retry_failed)

    def _run(self, run_id: str, selected: tuple[PilotTestSpec, ...], *, retry_failed: bool) -> dict[str, Any]:
        state = self.registry.load(run_id)
        mode = PilotMode(state["mode"])
        runtime = FakePilotRuntime(self.root) if mode == PilotMode.FAKE else RealPilotRuntime(self.root)
        lock_path = self.root / "state" / "locks" / f"pilot-{run_id}.lock"
        with StageLock(lock_path):
            for spec in selected:
                state = self.registry.load(run_id)
                current = PilotStatus(state["tests"][spec.test_id]["status"])
                if current == PilotStatus.PASSED:
                    continue
                if current == PilotStatus.FAILED and not retry_failed:
                    continue
                self._attempt(run_id, spec, runtime, state)
            state = self.registry.load(run_id)
            aggregate = generate_reports(self.root, state)
            state["status"] = "passed" if aggregate["mock_orchestration_ready"] or aggregate["experimental_ready"] else "blocked"
            self.registry.save(state)
            return aggregate

    def _attempt(self, run_id: str, spec: PilotTestSpec, runtime: Any, state: dict[str, Any]) -> None:
        attempt_id = new_attempt_id()
        started = utc_now()
        output = self.root / "artifacts" / "pilot_reports" / run_id / "tests" / spec.test_id / attempt_id
        output.mkdir(parents=True, exist_ok=False)
        self.registry.transition(run_id, spec.test_id, PilotStatus.RUNNING, attempt_id=attempt_id)
        missing = unmet_prerequisites(spec, state["tests"])
        if missing and spec.test_id != "pilot_36":
            execution = RuntimeExecution(
                PilotStatus.BLOCKED, EvidenceLevel.STATIC,
                {"unmet_prerequisites": list(missing), "operation_executed": False},
                f"Prerequisites not passed: {', '.join(missing)}", "Pass or revalidate prerequisites, then resume this pilot run.",
            )
        else:
            try:
                before = time.monotonic()
                execution = runtime.execute(spec, run_id=run_id, attempt_id=attempt_id, output_dir=output)
                elapsed_ms = int((time.monotonic() - before) * 1000)
                execution.details["elapsed_ms"] = elapsed_ms
                execution.details["timeout_seconds"] = spec.timeout_seconds
                if elapsed_ms > spec.timeout_seconds * 1000:
                    execution = RuntimeExecution(PilotStatus.INTERRUPTED, execution.evidence_level, execution.details, "Pilot test exceeded its catalog timeout.", "Start a new attempt from this test boundary; do not merge the timed-out attempt.")
            except Exception as exc:
                execution = RuntimeExecution(PilotStatus.FAILED, EvidenceLevel.STATIC, {}, f"{type(exc).__name__}: {exc}", "Inspect immutable attempt evidence and retry as a new attempt.")
        execution = RuntimeExecution(
            execution.status, execution.evidence_level, _sanitize(execution.details, self.root),
            _sanitize(execution.error, self.root) if execution.error else None,
            _sanitize(execution.remediation, self.root) if execution.remediation else None,
        )
        raw_path = output / "evidence.json"
        raw_payload = {"schema_version": 1, "pilot_run_id": run_id, "pilot_test_id": spec.test_id, "attempt_id": attempt_id, "mode": state["mode"], "details": execution.details}
        raw_path.write_text(json.dumps(raw_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        json.loads(raw_path.read_text(encoding="utf-8"))
        evidence = EvidenceReference(
            str(raw_path.relative_to(self.root)), _hash_bytes(raw_path.read_bytes()), execution.evidence_level,
            "phase6-json-evidence-v1", utc_now(), state["mode"] == "fake",
        )
        status = execution.status
        if status == PilotStatus.PASSED and not evidence_satisfies(PilotMode(state["mode"]), spec, execution.evidence_level):
            status = PilotStatus.BLOCKED
            execution = RuntimeExecution(status, execution.evidence_level, execution.details, "Evidence level is below this test's required level.", "Collect validated evidence at the required level.")
        result = PilotAttemptResult(
            1, run_id, spec.test_id, attempt_id, PilotMode(state["mode"]), status, execution.evidence_level,
            started, utc_now(), state["input_fingerprint"], [evidence], execution.details,
            execution.error, execution.remediation, self._next_allowed(spec.test_id, state["tests"], status),
        )
        report_path = output / "attempt-report.json"
        report_path.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        relative = str(report_path.relative_to(self.root))
        self.registry.transition(run_id, spec.test_id, status, attempt_id=attempt_id, attempt_path=relative, evidence_level=execution.evidence_level.value)
        self.run_registry.bind_pilot_attempt(PilotAttemptBinding(run_id, spec.test_id, attempt_id, state["mode"], status.value, relative))
        if spec.test_id in {"pilot_16", "pilot_17", "pilot_18", "pilot_19", "pilot_20", "pilot_21", "pilot_22", "pilot_23", "pilot_24"}:
            self.run_registry.bind_recovery_evidence(RecoveryEvidenceBinding(
                run_id, attempt_id, execution.details.get("checkpoint_id", "phase6-cp"),
                execution.details.get("perturbation_id", "phase6-pert"), "rq1-pilot", None,
                str(raw_path.relative_to(self.root)), relative,
            ))

    @staticmethod
    def _next_allowed(test_id: str, states: dict[str, Any], completed_status: PilotStatus) -> str | None:
        ids = [item.test_id for item in PILOT_TESTS]
        for candidate in ids[ids.index(test_id) + 1:]:
            spec = PILOT_TEST_MAP[candidate]
            missing = set(unmet_prerequisites(spec, states))
            if completed_status == PilotStatus.PASSED:
                missing.discard(test_id)
            if not missing: return candidate
        return None

    @staticmethod
    def _validate_model(candidate_model: str) -> None:
        if candidate_model not in {PRIMARY_MODEL, FALLBACK_MODEL}:
            raise ValueError("candidate model must be hermes3:8b or llama3.1:8b; DeepSeek is out of scope")


def add_manual_evidence(root: Path, run_id: str, test_id: str, source: Path, level: EvidenceLevel) -> dict[str, Any]:
    from rq1.hermes.models import redact
    if test_id not in PILOT_TEST_MAP: raise ValueError(f"Unknown pilot test: {test_id}")
    if not source.is_file(): raise FileNotFoundError(source)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict): raise ValueError("Manual evidence must be a JSON object")
    payload = redact(payload)
    registry = PilotRegistry(root)
    evidence_id = str(uuid4())
    destination = root / "artifacts" / "pilot_reports" / run_id / "raw" / "manual" / test_id / f"{evidence_id}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    evidence = EvidenceReference(str(destination.relative_to(root)), _hash_bytes(destination.read_bytes()), level, "manual-json-object-v1", utc_now(), False).to_dict()
    registry.add_manual_evidence(run_id, test_id, evidence)
    return evidence
