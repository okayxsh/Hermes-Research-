"""Proposed exposure-risk acceptance for scientific roster review (Phase 2B).  Never upgrades a task's status.

The retrospective investigation is closed (no further broad audits are authorised).  What remains is a documented,
residual risk.  This module turns the Phase 1 exposure ledger plus a coverage record into a hash-bound STATEMENT:

* every roster task keeps its ledger class.  UNKNOWN / unresolved stays UNKNOWN; the strongest available description of a
  task without located outcomes is ``NO_LOCATED_MODEL_OUTCOME`` -- never ``VERIFIED_UNEXPOSED`` (no code path emits it);
* tasks with KNOWN model-outcome exposure (``EXPOSED`` or ``POSSIBLY_EXPOSED``) can never enter an independent
  evaluation roster: building the statement fails;
* the inspected coverage and the remaining gaps are part of the hashed statement;
* the statement binds the exact roster hash and the frozen protocol hash (prospective freezing).

An ACCEPTANCE is a future human approval record whose ``artifact_sha256`` equals ``statement_sha256``; this module only
validates it (``readiness`` consumes it).  Without it, scientific execution stays blocked.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

from rq1.five_arm import readiness

SCHEMA = "rq1-five-arm-exposure-risk-statement-v1"
KNOWN_EXPOSURE_CLASSES = ("EXPOSED", "POSSIBLY_EXPOSED")
RESIDUAL_CLASSES = {"NO_MODEL_OUTCOME_FOUND": "NO_LOCATED_MODEL_OUTCOME", "UNKNOWN": "UNKNOWN"}
FORBIDDEN_STATUSES = ("VERIFIED_UNEXPOSED", "UNEXPOSED", "VERIFIED")
REQUIRED_COVERAGE = ("files_examined", "located_model_outcomes_for_roster", "located_rating_calibration_for_roster",
                     "oracle_preparation_outcome_blind", "known_gaps", "investigation_closed")

# The Phase 2B coverage record as reported by the university investigation (input data, not a conclusion of this code).
UNIVERSITY_INVESTIGATION_COVERAGE = {
    "files_examined": 904,
    "located_model_outcomes_for_roster": 0,
    "located_rating_calibration_for_roster": 0,
    "oracle_preparation_outcome_blind": True,
    "known_gaps": [
        "primary execution records for 2026-09-15 to 2026-10-29 are incomplete",
        "additional undocumented ad hoc activity remains possible",
    ],
    "incomplete_record_period": "15 September to 29 October 2026",
    "investigation_closed": True,
    "source": "university targeted retrospective investigation (reported to the authors; Phase 2B brief)",
}

MANUSCRIPT_LIMITATION_TEXT = (
    "Exposure limitation. Before the evaluation roster was frozen, we screened all local experiment records available "
    "to us ({files_examined} files) for model outcomes, human rating calibration and other prior use of the candidate "
    "evaluation tasks. We located no model outcomes and no rating calibration involving the {roster_size} roster tasks, "
    "and the oracle preparation used for the controlled disruption was verified to be outcome-blind. However, primary "
    "execution records for part of the development period ({period}) are incomplete, and undocumented ad hoc use of these "
    "tasks cannot be ruled out. The absence of located outcomes therefore does not prove that the tasks were never "
    "previously exposed to the model or to the investigators; we treat their exposure status as unresolved rather than "
    "verified. Tasks with any known model-outcome exposure were excluded. The roster, analysis plan and protocol were "
    "frozen before any evaluation outcome was generated, and the residual risk was accepted explicitly by the authors "
    "against this documented coverage."
)


class ExposureAcceptanceError(ValueError):
    pass


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def build_statement(*, roster: Sequence[str], ledger_rows: Sequence[Mapping[str, Any]], coverage: Mapping[str, Any],
                    roster_sha256: str, protocol_sha256: str, purpose: str) -> dict[str, Any]:
    """``ledger_rows`` are Phase 1 exposure-ledger rows (``task_id``, ``exposure_class``, ``exposure_kinds``,
    ``evidence_complete``, ``provisional_gaps``).  Raises if any roster task is known-exposed or unknown to the ledger."""
    if purpose not in ("pilot", "confirmatory"):
        raise ExposureAcceptanceError("purpose must be pilot or confirmatory")
    missing = [k for k in REQUIRED_COVERAGE if k not in coverage]
    if missing:
        raise ExposureAcceptanceError(f"coverage record lacks {missing}")
    for name, value in (("roster_sha256", roster_sha256), ("protocol_sha256", protocol_sha256)):
        if len(str(value)) != 64:
            raise ExposureAcceptanceError(f"{name} must be a SHA-256 digest (prospective freezing)")
    if len(set(roster)) != len(roster) or not roster:
        raise ExposureAcceptanceError("roster must be a non-empty list of unique task IDs")
    by_task = {str(r["task_id"]): r for r in ledger_rows}
    rows, excluded_known = [], []
    for task_id in sorted(roster):
        row = by_task.get(task_id)
        if row is None:
            raise ExposureAcceptanceError(f"{task_id} is not in the exposure ledger; its status cannot be described")
        klass = str(row.get("exposure_class"))
        if klass in FORBIDDEN_STATUSES:
            raise ExposureAcceptanceError(f"{task_id}: ledger claims {klass}; verified non-exposure is not an admissible status")
        if klass in KNOWN_EXPOSURE_CLASSES or "model_outcome_exposure" in (row.get("exposure_kinds") or []):
            excluded_known.append(task_id)
            continue
        if klass not in RESIDUAL_CLASSES:
            raise ExposureAcceptanceError(f"{task_id}: unrecognised exposure class {klass!r}")
        rows.append({"task_id": task_id, "ledger_class": klass, "residual_status": RESIDUAL_CLASSES[klass],
                     "exposure_kinds": sorted(row.get("exposure_kinds") or []),
                     "ledger_evidence_complete": bool(row.get("evidence_complete")),
                     "provisional_gaps": list(row.get("provisional_gaps") or [])})
    if excluded_known:
        raise ExposureAcceptanceError(f"{len(excluded_known)} roster task(s) have known model-outcome exposure and cannot "
                                      f"enter an independent evaluation roster: {excluded_known[:5]}")
    body = {"schema": SCHEMA, "purpose": purpose, "roster_sha256": roster_sha256, "protocol_sha256": protocol_sha256,
            "roster_size": len(rows), "tasks": rows, "coverage": dict(coverage),
            "interpretation": "absence of located model outcomes is NOT proof of non-exposure; statuses are not upgraded",
            "status_counts": {s: sum(r["residual_status"] == s for r in rows) for s in sorted(set(RESIDUAL_CLASSES.values()))}}
    body["statement_sha256"] = _sha(body)
    body["acceptance_status"] = "AWAITING_AUTHOR_ACCEPTANCE"
    return body


def verify_statement(statement: Mapping[str, Any]) -> list[str]:
    body = {k: v for k, v in statement.items() if k not in ("statement_sha256", "acceptance_status", "acceptance")}
    problems = [] if statement.get("statement_sha256") == _sha(body) else ["statement hash does not match its contents"]
    for row in statement.get("tasks", []):
        if row.get("residual_status") not in RESIDUAL_CLASSES.values():
            problems.append(f"{row.get('task_id')}: inadmissible residual status {row.get('residual_status')!r}")
    return problems


def acceptance_valid(statement: Mapping[str, Any], acceptance: Any, *, roster_sha256: str, protocol_sha256: str,
                     real: bool = True) -> list[str]:
    problems = verify_statement(statement)
    if statement.get("roster_sha256") != roster_sha256:
        problems.append("statement is bound to a different roster")
    if statement.get("protocol_sha256") != protocol_sha256:
        problems.append("statement is bound to a different frozen protocol")
    if not readiness.valid_approval(acceptance, statement.get("statement_sha256"), real=real):
        problems.append("no human acceptance bound to this exact exposure-risk statement")
    return problems


def manuscript_text(statement: Mapping[str, Any]) -> str:
    coverage = statement["coverage"]
    period = coverage.get("incomplete_record_period") or "period not recorded"
    return MANUSCRIPT_LIMITATION_TEXT.format(files_examined=coverage["files_examined"], roster_size=statement["roster_size"],
                                            period=period)
