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


def _independent_problems(statement: Mapping[str, Any], acceptance: Any, *, purpose: str, roster_sha256: str,
                          protocol_sha256: str, real: bool) -> list[str]:
    if not isinstance(statement, Mapping) or statement.get("schema") != SCHEMA:
        return [f"independent eligibility needs a {SCHEMA} statement"]
    problems = acceptance_valid(statement, acceptance, roster_sha256=roster_sha256, protocol_sha256=protocol_sha256, real=real)
    if statement.get("purpose") != purpose:
        problems.append(f"independent exposure statement purpose is {statement.get('purpose')!r}, not {purpose!r}")
    return problems


# --- Phase 3O: developmental-pilot eligibility (alongside the independent-roster validator above) --------------------------
# The revised 12-task valid_seen pilot was used in the completed D0 diagnostic, so it can never pass the independent-roster
# validator.  The authors approved (Phase 3N) a replacement SD-19 choice and an exposure statement that ACKNOWLEDGE that
# exposure.  This validator accepts only those exact approved bytes, bound to the diagnostic evidence they describe, and only
# for the seven-condition PILOT.  It never upgrades a task's status and never authorises confirmatory execution.
DEVELOPMENTAL_SCHEMA = "rq1-proposed-pilot-exposure-update-v1"
DEVELOPMENTAL_PURPOSE = "seven-condition developmental pilot"
DEVELOPMENTAL_EXPOSURE_CLASS = "MODEL_OUTCOME_EXPOSED_IN_D0_DEVELOPMENT_DIAGNOSTIC"
DEVELOPMENTAL_STAGE_SCOPE = "developmental_pilot_only"
SD19_KEY = "pilot_roster_composition"
# Digests of the author-approved artifacts (Phase 3N reconciled approvals; recorded inputs, not produced by this code).
APPROVED_DEVELOPMENTAL = {
    "exposure_statement_sha256": "1438f055fb82156fa3b6defedd513a425cde447273914fcb45db0287adfeeb05",   # file bytes
    "sd19_decision_sha256": "b9da3b4ca333ce82f856141bf33fe653fccac7c841f34bd0f345b6874968f4be",          # readiness.decision_sha256
    "cpu_amendment_sha256": "f0744805f83cb80be832f0e7115802d2b9b5ebf45e8316cfddbc6ba070f5b70c",          # file bytes
    "roster_sha256": "e9eb2f49495bc7cf65bdbeb5d3a2f5a4fc9c3ca80aa07810c0fd58598aea1fb8",                 # file bytes
    "diagnostic_manifest_sha256": "35ad22dfc9cf7e14d6faefd6da79668c59015c8c13510b8b208ae3f61f31ea5c",    # manifest_hash
    "diagnostic_run_id": "rq1-phase2g-d0-diagnostic",
    "diagnostic_units": 72,
    "task_count": 12,
}
DEVELOPMENTAL_EVIDENCE_FILES = ("statement", "roster", "cpu_amendment", "diagnostic_records", "diagnostic_receipt",
                                "diagnostic_manifest", "diagnostic_journal")


def _bytes_sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_lines(data: bytes) -> list[dict[str, Any]]:
    return [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]


def developmental_eligibility(evidence: Mapping[str, Any], *, readiness_state: Mapping[str, Any],
                              approved: Mapping[str, Any] | None = None, real: bool = True) -> dict[str, Any]:
    """Validate developmental-pilot eligibility from the EXACT approved bytes.

    ``evidence`` holds the raw bytes of ``DEVELOPMENTAL_EVIDENCE_FILES`` plus the two author approval records
    ``exposure_acceptance`` and ``cpu_amendment_approval``.  The SD-19 approval is read from the readiness state.  Returns
    ``{"problems": [...], "acknowledgement": {...}, "task_ids": [...], "statement_sha256": ...}``; eligible only if
    ``problems`` is empty.  The acknowledgement states both the authorised diagnostic exposure and the unresolved
    historical exposure explicitly; nothing is upgraded."""
    from rq1.five_arm.manifest import DIAGNOSTIC_SCHEMA, manifest_hash
    from rq1.five_arm.measurements import DIAGNOSTIC_CONDITIONS, DIAGNOSTIC_PROTOCOL_VERSION

    pins = dict(APPROVED_DEVELOPMENTAL if approved is None else approved)
    problems: list[str] = []
    missing = [k for k in DEVELOPMENTAL_EVIDENCE_FILES if not isinstance(evidence.get(k), (bytes, bytearray))]
    if missing:
        return {"problems": [f"developmental evidence lacks the exact bytes of {missing}"], "acknowledgement": None,
                "task_ids": [], "statement_sha256": None}
    raw = {k: bytes(evidence[k]) for k in DEVELOPMENTAL_EVIDENCE_FILES}
    statement_sha = _bytes_sha(raw["statement"])
    # 1. the exact approved statement and its author approval
    if statement_sha != pins["exposure_statement_sha256"]:
        problems.append("exposure statement bytes differ from the approved developmental statement")
    acceptance = evidence.get("exposure_acceptance")
    if not readiness.valid_approval(acceptance, pins["exposure_statement_sha256"], real=real):
        problems.append("no author approval bound to the approved developmental exposure statement")
    elif (acceptance.get("stage_scope") != DEVELOPMENTAL_STAGE_SCOPE
          or acceptance.get("roster_sha256") != pins["roster_sha256"]
          or acceptance.get("protocol_amendment_sha256") != pins["cpu_amendment_sha256"]):
        problems.append("exposure approval is not scoped to the developmental pilot, roster and CPU amendment")
    try:
        statement = json.loads(raw["statement"].decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return {"problems": problems + ["exposure statement is not valid JSON"], "acknowledgement": None, "task_ids": [],
                "statement_sha256": statement_sha}
    tasks = statement.get("tasks") or []
    task_ids = sorted(str(t.get("task_id")) for t in tasks)
    exposure = statement.get("known_pilot_roster_model_outcome_exposure") or {}
    coverage = statement.get("retrospective_coverage_as_of_original_audit") or {}
    if (statement.get("schema"), statement.get("purpose")) != (DEVELOPMENTAL_SCHEMA, DEVELOPMENTAL_PURPOSE):
        problems.append("statement is not the developmental-pilot exposure statement")
    if statement.get("roster_sha256") != pins["roster_sha256"] or statement.get("protocol_amendment_sha256") != pins["cpu_amendment_sha256"]:
        problems.append("statement is bound to a different roster or protocol amendment")
    if (len(task_ids) != pins["task_count"] or len(set(task_ids)) != len(task_ids)
            or statement.get("roster_size") != pins["task_count"] or not all(t.startswith("valid_seen:") for t in task_ids)):
        problems.append(f"statement must list exactly {pins['task_count']} unique valid_seen tasks")
    per_task = int(exposure.get("per_task") or 0)
    if (exposure.get("run_id") != pins["diagnostic_run_id"] or exposure.get("valid_d0_units") != pins["diagnostic_units"]
            or exposure.get("task_count") != pins["task_count"] or per_task * pins["task_count"] != pins["diagnostic_units"]):
        problems.append("statement does not bind the approved 72-unit diagnostic exposure")
    for row in tasks:
        ledger = row.get("historical_ledger") or {}
        if row.get("current_known_exposure") != DEVELOPMENTAL_EXPOSURE_CLASS or row.get("valid_diagnostic_units") != per_task:
            problems.append(f"{row.get('task_id')}: diagnostic exposure not acknowledged")
        if ledger.get("ledger_class") not in RESIDUAL_CLASSES or ledger.get("ledger_class") in FORBIDDEN_STATUSES:
            problems.append(f"{row.get('task_id')}: historical status upgraded or unrecognised ({ledger.get('ledger_class')!r})")
    if any(s in (statement.get("historical_status_counts") or {}) for s in FORBIDDEN_STATUSES):
        problems.append("statement claims verified non-exposure")
    if not coverage.get("known_gaps") or coverage.get("investigation_closed") is not True:
        problems.append("unresolved historical exposure (known retrospective gaps) is not acknowledged")
    # 2. roster, CPU amendment and SD-19
    if _bytes_sha(raw["roster"]) != pins["roster_sha256"]:
        problems.append("roster bytes differ from the approved developmental roster")
    else:
        roster = json.loads(raw["roster"].decode("utf-8"))
        if sorted(str(t.get("task_id")) for t in roster.get("tasks") or []) != task_ids:
            problems.append("roster task identities differ from the exposure statement")
    if _bytes_sha(raw["cpu_amendment"]) != pins["cpu_amendment_sha256"]:
        problems.append("CPU protocol/SAP amendment bytes differ from the approved amendment")
    if not readiness.valid_approval(evidence.get("cpu_amendment_approval"), pins["cpu_amendment_sha256"], real=real):
        problems.append("no author approval bound to the CPU protocol/SAP amendment")
    sd19 = ((readiness_state or {}).get("scientific_decisions") or {}).get(SD19_KEY)
    if not (isinstance(sd19, Mapping) and readiness.decision_sha256(SD19_KEY, sd19.get("choice")) == pins["sd19_decision_sha256"]
            and readiness.valid_approval(sd19, pins["sd19_decision_sha256"], real=real)):
        problems.append("SD-19 is not approved with the replacement developmental-pilot choice")
    # 3. the bound 72-unit diagnostic evidence
    if _bytes_sha(raw["diagnostic_records"]) != exposure.get("records_sha256"):
        problems.append("diagnostic records differ from those bound by the exposure statement")
    if _bytes_sha(raw["diagnostic_receipt"]) != exposure.get("completion_receipt_sha256"):
        problems.append("diagnostic completion receipt differs from the one bound by the exposure statement")
    try:
        records = _json_lines(raw["diagnostic_records"])
        journal = _json_lines(raw["diagnostic_journal"])
        receipt = json.loads(raw["diagnostic_receipt"].decode("utf-8"))
        dmanifest = json.loads(raw["diagnostic_manifest"].decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return {"problems": problems + ["diagnostic evidence is not valid JSON"], "acknowledgement": None, "task_ids": task_ids,
                "statement_sha256": statement_sha}
    if not (manifest_hash(dmanifest) == dmanifest.get("manifest_sha256") == pins["diagnostic_manifest_sha256"]
            and dmanifest.get("schema") == DIAGNOSTIC_SCHEMA and dmanifest.get("stage") == "pilot_diagnostic"
            and dmanifest.get("status") == "frozen" and dmanifest.get("split") == "valid_seen"
            and dmanifest.get("run_id") == pins["diagnostic_run_id"] and dmanifest.get("synthetic") is False
            and len(dmanifest.get("units") or []) == pins["diagnostic_units"]
            and sorted(str(t["task_id"]) for t in dmanifest.get("tasks") or []) == task_ids):
        problems.append("diagnostic manifest is not the approved frozen 72-unit diagnostic manifest for these tasks")
    unit_keys = {u.get("unit_key") for u in dmanifest.get("units") or []}
    per = {}
    for r in records:
        per[str(r.get("task_id"))] = per.get(str(r.get("task_id")), 0) + 1
    if (len(records) != pins["diagnostic_units"] or len({r.get("unit_key") for r in records}) != len(records)
            or any(r.get("schema") != "rq1-five-arm-diagnostic-v1" or r.get("protocol_version") != DIAGNOSTIC_PROTOCOL_VERSION
                   or r.get("outcome_status") != "VALID" or r.get("synthetic") is not False or r.get("unit_key") not in unit_keys
                   or r.get("condition") not in DIAGNOSTIC_CONDITIONS for r in records)
            or sorted(per) != task_ids or set(per.values()) != {per_task}
            or sorted({int(r.get("seed")) for r in records}) != sorted(int(s) for s in exposure.get("seeds") or [])
            or sorted({str(r.get("condition")) for r in records}) != sorted(exposure.get("conditions") or [])):
        problems.append("diagnostic records are not the 72 valid units, 6 per approved task, bound to the diagnostic manifest")
    finished = {e.get("unit_key") for e in journal if e.get("event") == "attempt_finished" and e.get("outcome_status") == "VALID"}
    started = {e.get("unit_key") for e in journal if e.get("event") == "attempt_started"}
    if not {r.get("unit_key") for r in records} <= (finished & started):
        problems.append("diagnostic journal does not record the start and valid finish of every bound unit")
    summary, validation = receipt.get("summary") or {}, receipt.get("validation") or {}
    if (summary.get("run_id") != pins["diagnostic_run_id"] or summary.get("units") != pins["diagnostic_units"]
            or (summary.get("final_status_counts") or {}).get("VALID") != pins["diagnostic_units"]
            or summary.get("pending_units") != 0 or summary.get("synthetic") is not False
            or validation.get("ok") is not True or validation.get("complete") is not True):
        problems.append("diagnostic completion receipt does not record a complete, valid 72-unit run")
    acknowledgement = {
        "authorized_diagnostic_exposure": {"run_id": exposure.get("run_id"), "valid_units": exposure.get("valid_d0_units"),
                                           "tasks": len(task_ids), "seeds": exposure.get("seeds"),
                                           "status": DEVELOPMENTAL_EXPOSURE_CLASS},
        "unresolved_historical_exposure": {"known_gaps": list(coverage.get("known_gaps") or []),
                                           "status_upgraded": False},
        "scope": DEVELOPMENTAL_STAGE_SCOPE, "confirmatory_eligibility": False}
    return {"problems": problems, "acknowledgement": acknowledgement if not problems else None, "task_ids": task_ids,
            "statement_sha256": statement_sha}


def exposure_eligibility_problems(manifest: Mapping[str, Any], eligibility: Mapping[str, Any] | None, *,
                                  readiness_state: Mapping[str, Any], artifacts: Mapping[str, str], real: bool = True,
                                  approved: Mapping[str, Any] | None = None) -> list[str]:
    """Stage-specific exposure gate shared by manifest freeze and execution preflight (seven-condition manifests only).

    * pilot: ``kind`` ``developmental`` (the Phase 3N approval) or ``independent`` (the existing statement + acceptance);
    * confirmatory: ``independent`` only; developmental evidence is always refused, and every task of a supplied
      ``known_exposed_statements`` entry (the approved developmental statement bytes, required) is excluded.
    The manifest's ``exposure_statement_sha256`` must equal the digest the author approval binds."""
    from rq1.five_arm.manifest import SCHEMA as SEVEN_CONDITION_SCHEMA
    from rq1.five_arm.protocol import PROTOCOL_VERSION

    if manifest.get("schema") != SEVEN_CONDITION_SCHEMA:
        return []                                               # pilot-diagnostic manifests: unchanged requirements
    pins = dict(APPROVED_DEVELOPMENTAL if approved is None else approved)
    stage = manifest.get("stage")
    if not isinstance(eligibility, Mapping) or eligibility.get("kind") not in ("developmental", "independent"):
        return ["no exposure eligibility evidence (developmental or independent) was supplied"]
    kind = eligibility["kind"]
    hashes = manifest.get("input_hashes") or {}
    manifest_ids = sorted(str(t.get("task_id")) for t in manifest.get("tasks") or [])
    problems: list[str] = []
    if manifest.get("protocol_version") != PROTOCOL_VERSION:
        problems.append("manifest is not a seven-condition protocol manifest")
    if stage == "confirmatory":
        if kind == "developmental":
            return problems + ["developmental-pilot exposure eligibility never authorises confirmatory execution"]
        statement = eligibility.get("statement")
        if _sha_or_none(statement) == pins["exposure_statement_sha256"] or (
                isinstance(statement, Mapping) and statement.get("schema") == DEVELOPMENTAL_SCHEMA):
            return problems + ["developmental exposure artifacts cannot serve as confirmatory exposure evidence"]
        if artifacts.get("confirmatory_exposure_statement") == pins["exposure_statement_sha256"]:
            problems.append("the confirmatory exposure artifact is the developmental-pilot statement")
        problems += _independent_problems(statement, eligibility.get("acceptance"), purpose="confirmatory",
                                          roster_sha256=hashes.get("roster_sha256"),
                                          protocol_sha256=str(eligibility.get("protocol_sha256")), real=real)
        exposed: set[str] = set()
        known = eligibility.get("known_exposed_statements") or []
        if not known:
            problems.append("confirmatory eligibility must include the approved developmental statement as an exclusion list")
        for data in known:
            if not isinstance(data, (bytes, bytearray)) or _bytes_sha(bytes(data)) != pins["exposure_statement_sha256"]:
                problems.append("known-exposure exclusion list is not the approved developmental statement")
                continue
            exposed |= {str(t.get("task_id")) for t in json.loads(bytes(data).decode("utf-8")).get("tasks") or []}
        listed = {str(r.get("task_id")) for r in (statement or {}).get("tasks", [])} if isinstance(statement, Mapping) else set()
        entering = sorted((set(manifest_ids) | listed) & exposed)
        if entering:
            problems.append(f"{len(entering)} task(s) with known model-outcome exposure cannot enter the confirmatory roster")
        if isinstance(statement, Mapping) and statement.get("statement_sha256") != hashes.get("exposure_statement_sha256"):
            problems.append("manifest exposure digest differs from the accepted confirmatory statement")
        return problems
    if stage != "pilot":
        return problems + [f"exposure eligibility is undefined for stage {stage!r}"]
    if kind == "independent":
        statement = eligibility.get("statement")
        problems += _independent_problems(statement, eligibility.get("acceptance"), purpose="pilot",
                                          roster_sha256=hashes.get("roster_sha256"),
                                          protocol_sha256=str(eligibility.get("protocol_sha256")), real=real)
        if isinstance(statement, Mapping) and statement.get("statement_sha256") != hashes.get("exposure_statement_sha256"):
            problems.append("manifest exposure digest differs from the accepted pilot statement")
        return problems
    result = developmental_eligibility(eligibility, readiness_state=readiness_state, approved=pins, real=real)
    problems += result["problems"]
    if manifest.get("split") != "valid_seen":
        problems.append("the developmental pilot must use valid_seen")
    if manifest_ids != result["task_ids"]:
        problems.append("manifest task identities differ from the 12 approved developmental-pilot tasks")
    if hashes.get("roster_sha256") != pins["roster_sha256"] or artifacts.get("pilot_roster") != pins["roster_sha256"]:
        problems.append("manifest or readiness roster digest differs from the approved developmental roster")
    approved_digest = (eligibility.get("exposure_acceptance") or {}).get("artifact_sha256")
    if not (hashes.get("exposure_statement_sha256") == result["statement_sha256"] == approved_digest
            == artifacts.get("pilot_exposure_statement") == pins["exposure_statement_sha256"]):
        problems.append("exposure digest must match the approved statement, its author approval, the manifest and readiness")
    return problems


def _sha_or_none(value: Any) -> str | None:
    return _bytes_sha(bytes(value)) if isinstance(value, (bytes, bytearray)) else None


def manuscript_text(statement: Mapping[str, Any]) -> str:
    coverage = statement["coverage"]
    period = coverage.get("incomplete_record_period") or "period not recorded"
    return MANUSCRIPT_LIMITATION_TEXT.format(files_examined=coverage["files_examined"], roster_size=statement["roster_size"],
                                            period=period)
