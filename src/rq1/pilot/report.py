"""Phase 6 capability, runtime, protocol, and go/no-go reports."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from rq1.pilot.catalog import PILOT_TESTS
from rq1.pilot.models import GoNoGoDecision, PilotMode
from rq1.utils.time import utc_now


def _read_attempt(root: Path, state: dict[str, Any], test_id: str) -> dict[str, Any]:
    attempts = state["tests"][test_id].get("attempts", [])
    if not attempts: return {}
    path = root / attempts[-1]
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _preserve(path: Path) -> None:
    if not path.exists(): return
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    history = path.parent / "report-history" / f"{path.stem}-{digest}{path.suffix}"
    history.parent.mkdir(parents=True, exist_ok=True)
    if not history.exists(): shutil.copy2(path, history)


def _write_latest(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _preserve(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if isinstance(payload, str): temporary.write_text(payload, encoding="utf-8")
    else: temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def build_decision(state: dict[str, Any]) -> GoNoGoDecision:
    mode = PilotMode(state["mode"])
    blocking = [item.test_id for item in PILOT_TESTS if item.test_id != "pilot_36"]
    incomplete = [test_id for test_id in blocking if state["tests"][test_id]["status"] != "passed"]
    if mode == PilotMode.FAKE:
        reasons = ("Fake execution validates runner orchestration only.", "Real Hermes, ALFWorld, and recovery evidence are required in Phase 7.")
        return GoNoGoDecision("no_go", False, reasons, mode)
    if incomplete:
        return GoNoGoDecision("no_go", False, (f"Blocking tests not passed: {', '.join(incomplete)}",), mode)
    return GoNoGoDecision("go", True, ("All blocking real pilot gates passed with validated evidence.",), mode)


def generate_reports(root: Path, state: dict[str, Any]) -> dict[str, Any]:
    run_id = str(state["run_id"])
    output = root / "artifacts" / "pilot_reports" / run_id
    statuses = {test_id: value["status"] for test_id, value in state["tests"].items()}
    counts = {status: list(statuses.values()).count(status) for status in sorted(set(statuses.values()))}
    decision = build_decision(state)
    matrix = {
        "schema_version": 1, "pilot_run_id": run_id, "mode": state["mode"],
        "tests": {test_id: {"status": value["status"], "evidence_level": value.get("evidence_level")} for test_id, value in state["tests"].items()},
        "real_evidence_promoted_from_mock": False,
    }
    runtime_attempt = _read_attempt(root, state, "pilot_32")
    runtime = {"schema_version": 1, "pilot_run_id": run_id, "mode": state["mode"], "measurements": runtime_attempt.get("details", {}), "simulated": state["mode"] == "fake"}
    protocol_attempt = _read_attempt(root, state, "pilot_35")
    protocol = {"schema_version": 1, "pilot_run_id": run_id, **protocol_attempt.get("details", {"approval_state": "unapproved", "availability": "unavailable"})}
    aggregate = {
        "schema_version": 1,
        "phase": 6,
        "pilot_run_id": run_id,
        "generated_at": utc_now(),
        "mode": state["mode"],
        "candidate_model": state["candidate_model"],
        "status_counts": counts,
        "tests": matrix["tests"],
        "mock_orchestration_ready": state["mode"] == "fake" and all(statuses[item.test_id] == "passed" for item in PILOT_TESTS),
        "pilot_ready": decision.experimental_ready,
        "real_integrated": decision.experimental_ready,
        "experimental_ready": decision.experimental_ready,
        "go_no_go": decision.to_dict(),
        "protocol_frozen": False,
        "final_acquisition_run": False,
        "final_snapshots_created": False,
        "final_recovery_evaluation_run": False,
        "phase7_required": True,
        "artifacts": {
            "capability_matrix": "capability-matrix.json",
            "runtime_benchmark": "runtime-benchmark.json",
            "proposed_protocol": "proposed-recovery-protocol.json",
            "decision": "go-no-go.json",
        },
    }
    markdown = "\n".join((
        "# Phase 6 pilot report", "", f"- Run: `{run_id}`", f"- Mode: `{state['mode']}`",
        f"- Decision: **{decision.decision}**", f"- Experimental ready: `{str(decision.experimental_ready).lower()}`",
        f"- Status counts: `{json.dumps(counts, sort_keys=True)}`", "",
        "Fake success is runner-contract evidence only. Phase 7 is the real university pilot and environment/recovery-protocol freeze.", "",
    ))
    _write_latest(output / "capability-matrix.json", matrix)
    _write_latest(output / "runtime-benchmark.json", runtime)
    _write_latest(output / "proposed-recovery-protocol.json", protocol)
    _write_latest(output / "go-no-go.json", decision.to_dict())
    _write_latest(output / "pilot-report.json", aggregate)
    _write_latest(output / "pilot-report.md", markdown)
    stage = root / "artifacts" / "stage_reports" / f"phase6-pilot-{run_id}.json"
    if not stage.exists():
        stage.parent.mkdir(parents=True, exist_ok=True)
        stage.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return aggregate
