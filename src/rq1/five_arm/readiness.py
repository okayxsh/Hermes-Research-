"""Six separate readiness levels; scientific execution fails closed.  Passing engineering tests never approves science.

Levels (each includes every prerequisite of the levels it builds on, and nothing irrelevant to it):

1. ``local_synthetic_development``   -- local synthetic test suite passed.  No real data, no model.
2. ``university_cpu_integration``    -- reviewed, pushed code commit.  Authorises CPU-only work on the workstation:
   tests, runtime METADATA observation (no inference), frozen-ranking export, rating export, exact-cap oracle and
   cap-invariance replays, real-tokenizer filler audits.  Needs no labels, filler, roster or runtime approval.
3. ``provider_compatibility_probe``  -- a separately authorised NON-TASK probe (no ALFWorld content) of the installed
   model: needs the observed runtime record and an authorisation bound to the exact probe plan.
4. ``d0_failure_manipulation_pilot`` -- D0-only undisturbed/detoured diagnostic on the pilot roster.  Needs runtime
   verification, probe results, pilot roster, exposure-risk acceptance, the diagnostic protocol (thresholds, allowance,
   budget rule) and exact-cap oracle replay -- NOT filler or relevance labels (never used).
5. ``five_arm_scientific_pilot``     -- adds the diagnostic review, the budget decision, approved filler, pilot labels.
6. ``confirmatory``                  -- adds pilot GO, the final protocol, SAP, confirmatory roster, labels, exposure
   acceptance and sample-size justification.

Approval entries are human records bound to the artifact they approve::

    {"approved": true, "approved_by": "<name>", "approval_reference": "<ref>", "artifact_sha256": "<64 hex>"}

``artifact_sha256`` must equal the hash of the artifact actually used (supplied by the caller from the loaded files, or
computed here for decisions).  For REAL execution, any entry marked synthetic / fixture is refused.  This module only
READS approvals; nothing here creates one.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Mapping

from rq1.five_arm.protocol import PROTOCOL_VERSION, SCIENTIFIC_DECISIONS, decisions_blocking

LEVELS = ("local_synthetic_development", "university_cpu_integration", "provider_compatibility_probe",
          "d0_failure_manipulation_pilot", "five_arm_scientific_pilot", "confirmatory")
MODEL_EXECUTION_LEVELS = ("provider_compatibility_probe", "d0_failure_manipulation_pilot",
                          "five_arm_scientific_pilot", "confirmatory")
TASK_EXECUTION_LEVELS = ("d0_failure_manipulation_pilot", "five_arm_scientific_pilot", "confirmatory")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_SYNTHETIC_MARKERS = ("synthetic", "fixture", "test-only", "example")

# level -> list of (state key, artifact key bound by artifact_sha256, human description)
_RULES: dict[str, list[tuple[str, str, str]]] = {
    "provider_compatibility_probe": [
        ("provider_probe_authorization", "provider_probe_plan", "no separate authorisation for the non-task provider probe"),
    ],
    "d0_failure_manipulation_pilot": [
        ("provider_probe_results", "provider_probe_results", "provider compatibility probe results not reviewed"),
        ("runtime_verification", "runtime_config", "unverified model/runtime/tokenizer identity"),
        ("pilot_roster", "pilot_roster", "unapproved pilot roster"),
        ("pilot_exposure_acceptance", "pilot_exposure_statement", "no exposure-risk acceptance for the pilot roster"),
        ("diagnostic_protocol", "diagnostic_protocol", "unapproved diagnostic protocol (thresholds, allowance, budget rule)"),
        ("pilot_oracle_replay_diagnostic_cap", "pilot_oracle_replay_diagnostic_cap",
         "no exact-cap oracle replay of the pilot roster at the diagnostic cap"),
    ],
    "five_arm_scientific_pilot": [
        ("diagnostic_review", "diagnostic_report", "failure-manipulation diagnostic not reviewed (proceed / revise)"),
        ("budget_decision", "budget_decision", "no approved action-budget decision"),
        ("filler_corpus", "filler_corpus", "unapproved neutral filler corpus"),
        ("pilot_relevance_labels", "pilot_relevance_labels", "missing frozen pilot relevance judgments"),
        ("pilot_oracle_replay_chosen_cap", "pilot_oracle_replay_chosen_cap",
         "no exact-cap oracle replay of the pilot roster at the chosen cap"),
    ],
    "confirmatory": [
        ("pilot_go_decision", "pilot_report", "no pilot GO decision"),
        ("final_protocol", "final_protocol", "final protocol not approved and frozen"),
        ("statistical_analysis_plan", "statistical_analysis_plan", "unapproved statistical analysis plan"),
        ("sample_size_justification", "sample_size_justification", "insufficient sample-size justification"),
        ("confirmatory_roster", "confirmatory_roster", "unapproved confirmatory roster"),
        ("confirmatory_exposure_acceptance", "confirmatory_exposure_statement",
         "no exposure-risk acceptance for the confirmatory roster"),
        ("confirmatory_relevance_labels", "confirmatory_relevance_labels", "missing confirmatory relevance judgments"),
        ("confirmatory_oracle_replay_chosen_cap", "confirmatory_oracle_replay_chosen_cap",
         "no exact-cap oracle replay of the confirmatory roster at the chosen cap"),
    ],
}
_DECISION_STAGE = {"d0_failure_manipulation_pilot": "d0_failure_manipulation_pilot",
                   "five_arm_scientific_pilot": "five_arm_scientific_pilot", "confirmatory": "confirmatory"}


class ExecutionBlocked(RuntimeError):
    pass


def is_synthetic(entry: Any) -> bool:
    if not isinstance(entry, Mapping):
        return False
    if entry.get("synthetic") is True:
        return True
    text = " ".join(str(entry.get(k, "")) for k in ("approved_by", "approval_reference")).lower()
    return any(marker in text for marker in _SYNTHETIC_MARKERS)


def valid_approval(entry: Any, expected_sha256: str | None = None, *, real: bool = True) -> bool:
    """A human approval record; when ``expected_sha256`` is given it must be bound to exactly that artifact."""
    if not (isinstance(entry, Mapping) and entry.get("approved") is True and bool(str(entry.get("approved_by", "")).strip())
            and bool(str(entry.get("approval_reference", "")).strip()) and bool(_HEX64.match(str(entry.get("artifact_sha256", ""))))):
        return False
    if real and is_synthetic(entry):
        return False
    return expected_sha256 is None or entry["artifact_sha256"] == expected_sha256


def decision_sha256(name: str, choice: Any) -> str:
    """The artifact an approval of one scientific decision is bound to: the exact register entry plus the choice."""
    entry = SCIENTIFIC_DECISIONS[name]
    payload = {"protocol_version": PROTOCOL_VERSION, "decision": name, "entry": entry, "choice": choice}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _decision_blockers(stage: str, decisions: Any, real: bool) -> list[str]:
    blockers = []
    for name in decisions_blocking(stage):
        entry = decisions.get(name) if isinstance(decisions, Mapping) else None
        ok = (isinstance(entry, Mapping) and entry.get("choice") is not None
              and valid_approval(entry, decision_sha256(name, entry.get("choice")), real=real))
        if not ok:
            blockers.append(f"scientific decision {SCIENTIFIC_DECISIONS[name]['id']} {name} not approved")
    return blockers


def evaluate(state: Mapping[str, Any], artifacts: Mapping[str, str] | None = None, *, real: bool = True) -> dict[str, Any]:
    """``artifacts`` maps artifact keys to the SHA-256 of the artifacts actually loaded.  An approval whose artifact
    hash is unknown or different is a blocker (never trusted on its own)."""
    artifacts = dict(artifacts or {})
    levels: dict[str, list[str]] = {}
    base = []
    if real and state.get("synthetic") is True:
        base.append("readiness state is marked synthetic; it can never authorise real execution")
    levels["local_synthetic_development"] = base + ([] if state.get("local_tests_passed") is True
                                                    else ["local synthetic test suite has not passed"])
    levels["university_cpu_integration"] = levels["local_synthetic_development"] + (
        [] if _HEX40.match(str(state.get("code_commit", ""))) else ["no reviewed, pushed code commit recorded"])
    levels["provider_compatibility_probe"] = list(levels["university_cpu_integration"])
    if not (isinstance(state.get("observed_runtime"), Mapping) and state["observed_runtime"]
            and not (real and is_synthetic(state["observed_runtime"]))):
        levels["provider_compatibility_probe"].append("no observed runtime record from the workstation")
    previous = "provider_compatibility_probe"
    for level in LEVELS[2:]:
        blockers = list(levels[previous]) if level != "provider_compatibility_probe" else levels[level]
        for key, artifact_key, text in _RULES[level]:
            expected = artifacts.get(artifact_key)
            if expected is None or not _HEX64.match(str(expected)):
                blockers.append(f"{text} (artifact {artifact_key!r} not loaded/hashed)")
            elif not valid_approval(state.get(key), expected, real=real):
                blockers.append(text)
        if level in _DECISION_STAGE:
            blockers += _decision_blockers(_DECISION_STAGE[level], state.get("scientific_decisions"), real)
        levels[level] = blockers
        previous = level
    return {"levels": {name: {"status": "READY" if not levels[name] else "BLOCKED", "blockers": levels[name],
                              "authorises_model_inference": name in MODEL_EXECUTION_LEVELS}
                       for name in LEVELS},
            "real_execution_semantics": real,
            "note": "READY for local_synthetic_development or university_cpu_integration never authorises model inference"}


def require(level: str, state: Mapping[str, Any], artifacts: Mapping[str, str] | None = None, *, real: bool = True) -> None:
    if level not in MODEL_EXECUTION_LEVELS:
        raise ExecutionBlocked(f"{level!r} does not authorise model execution")
    report = evaluate(state, artifacts, real=real)["levels"][level]
    if report["status"] != "READY":
        raise ExecutionBlocked(f"{level} execution blocked: " + "; ".join(report["blockers"]))
