"""Separate readiness states; scientific execution fails closed.  Passing engineering tests never approves science.

State is a mapping of evidence entries.  Approval entries must be human records::

    {"approved": true, "approved_by": "<name>", "approval_reference": "<ref>", "artifact_sha256": "<64 hex>"}

This module only READS approvals; nothing here creates one.
"""
from __future__ import annotations

import re
from typing import Any, Mapping

from rq1.five_arm.protocol import SCIENTIFIC_DECISIONS

LEVELS = ("local_development", "university_integration", "cpu_oracle", "scientific_pilot", "confirmatory")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")

# blocker name -> (state key, kind, human description)
_PILOT = {
    "pilot_roster_approved": ("pilot_roster", "approval", "unapproved pilot roster"),
    "runtime_pin_verified": ("runtime_verification", "approval", "unverified model/runtime pin"),
    "tokenizer_provider_linked": ("tokenizer_provider_linkage", "approval", "unverified tokenizer/provider linkage"),
    "filler_approved": ("filler_corpus", "approval", "unapproved neutral filler"),
    "pilot_labels_frozen": ("pilot_relevance_labels", "approval", "missing pilot relevance judgments"),
    "pilot_protocol_approved": ("pilot_protocol", "approval", "unapproved pilot protocol (budget-decision rule included)"),
    "exact_cap_oracle_replay": ("exact_cap_oracle_replay", "approval", "no exact-cap oracle replay at the run's action cap"),
    "decisions_approved": ("scientific_decisions", "decisions", "open scientific decisions are not approved"),
}
_CONFIRMATORY = {
    "exposure_provenance_complete": ("exposure_provenance", "approval", "incomplete exposure provenance"),
    "confirmatory_roster_approved": ("confirmatory_roster", "approval", "unapproved confirmatory roster"),
    "confirmatory_labels_frozen": ("confirmatory_relevance_labels", "approval", "missing confirmatory relevance judgments"),
    "checkpoint_budget_policy_approved": ("checkpoint_and_budget_policy", "approval", "unapproved checkpoint and action-budget policy"),
    "failure_manipulation_validated": ("failure_manipulation_validation", "approval", "unvalidated failure manipulation"),
    "statistical_plan_approved": ("statistical_analysis_plan", "approval", "unapproved statistical plan"),
    "sample_justification_approved": ("sample_size_justification", "approval", "insufficient sample justification"),
    "pilot_go": ("pilot_go_decision", "approval", "no pilot GO decision"),
}


def valid_approval(entry: Any) -> bool:
    return (isinstance(entry, Mapping) and entry.get("approved") is True and bool(str(entry.get("approved_by", "")).strip())
            and bool(str(entry.get("approval_reference", "")).strip()) and bool(_HEX64.match(str(entry.get("artifact_sha256", "")))))


def _decisions_ok(entry: Any) -> bool:
    return isinstance(entry, Mapping) and all(valid_approval(entry.get(name)) and entry[name].get("choice") is not None
                                              for name in SCIENTIFIC_DECISIONS)


def _check(rules: Mapping[str, tuple[str, str, str]], state: Mapping[str, Any]) -> list[str]:
    blockers = []
    for _name, (key, kind, text) in rules.items():
        entry = state.get(key)
        ok = _decisions_ok(entry) if kind == "decisions" else valid_approval(entry)
        if not ok:
            blockers.append(text)
    return blockers


def evaluate(state: Mapping[str, Any]) -> dict[str, Any]:
    levels: dict[str, Any] = {}
    local = [] if state.get("local_tests_passed") is True else ["local test suite has not passed"]
    levels["local_development"] = local
    levels["university_integration"] = local + ([] if isinstance(state.get("observed_runtime"), Mapping)
                                                 else ["no observed runtime record from the workstation"])
    levels["cpu_oracle"] = levels["university_integration"] + (
        [] if state.get("oracle_validation_evidence_sha256") and _HEX64.match(str(state["oracle_validation_evidence_sha256"]))
        else ["no environment-validated oracle evidence"])
    levels["scientific_pilot"] = levels["cpu_oracle"] + _check(_PILOT, state)
    levels["confirmatory"] = levels["scientific_pilot"] + _check(_CONFIRMATORY, state)
    return {"levels": {name: {"status": "READY" if not b else "BLOCKED", "blockers": b} for name, b in levels.items()},
            "note": "READY for local_development / university_integration / cpu_oracle never authorises model execution"}


class ExecutionBlocked(RuntimeError):
    pass


def require(level: str, state: Mapping[str, Any]) -> None:
    if level not in ("scientific_pilot", "confirmatory"):
        raise ExecutionBlocked("model execution is only possible at scientific_pilot or confirmatory readiness")
    report = evaluate(state)["levels"][level]
    if report["status"] != "READY":
        raise ExecutionBlocked(f"{level} execution blocked: " + "; ".join(report["blockers"]))
