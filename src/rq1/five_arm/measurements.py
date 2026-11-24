"""Uniform per-episode measurement schema for all five arms, plus retrieved-vs-injected relevance metrics.

Natural retrieval (computed for every task, model-visible only in D18) and injected memory (what the model actually
saw) are SEPARATE fields.  Retrieved Precision@3 uses the natural top-3; injected relevance uses the injected IDs --
never the unchanged natural retrieval IDs.  D0 carries every field, with explicit empty / not-applicable values.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from rq1.five_arm.protocol import ACTIVE_SKILL_COUNT, CONDITIONS, PROTOCOL_VERSION

MEASUREMENT_SCHEMA = "rq1-five-arm-measurement-v1"
OUTCOME_STATUSES = ("VALID", "MISSING_INFRASTRUCTURE", "MISSING_CELL", "UNAPPROVED_INTERVENTION", "INVALID_MEASUREMENT")
TERMINATION_REASONS = ("success", "budget_exhausted", "invalid_action_limit", "environment_terminal_failure",
                       "provider_error", "environment_error", "infrastructure_interruption", "missing_cell",
                       "unapproved_intervention", "invalid_measurement")
OUTCOME_TERMINATIONS = {"success", "budget_exhausted", "invalid_action_limit", "environment_terminal_failure"}
INFRASTRUCTURE_TERMINATIONS = {"provider_error", "environment_error", "infrastructure_interruption"}
REQUIRED_FIELDS = (
    "schema", "protocol_version", "synthetic", "unit_key", "attempt", "task_id", "goal_template", "scene", "family",
    "exposure_policy", "oracle_id", "oracle_sha256", "checkpoint_digest", "post_detour_digest", "condition", "seed",
    "runtime_identity_sha256", "natural_top3_skill_ids", "natural_top3_scores", "injected_skill_ids",
    "injected_skill_count", "analysis_only_injected_labels", "filler_corpus_sha256", "filler_sentence_indices",
    "memory_block_sha256", "first_recovery_prompt_sha256", "tokens", "total_action_cap", "prefix_actions",
    "detour_actions", "recovery_actions", "success", "termination_reason", "invalid_actions", "wall_clock_seconds",
    "inference_latency_seconds", "outcome_status", "not_applicable",
)
TOKEN_FIELDS = ("raw_text_tokens", "memory_block_tokens", "chat_template_tokens", "provider_prompt_tokens")


class MeasurementError(ValueError):
    pass


def validate_record(record: Mapping[str, Any]) -> list[str]:
    problems = [f"missing field {name}" for name in REQUIRED_FIELDS if name not in record]
    if problems:
        return problems
    if record["schema"] != MEASUREMENT_SCHEMA or record["protocol_version"] != PROTOCOL_VERSION:
        problems.append("schema/protocol mismatch")
    if record["condition"] not in CONDITIONS:
        problems.append(f"condition {record['condition']!r} is not one of the five arms")
        return problems
    arm = record["condition"]
    if record["outcome_status"] not in OUTCOME_STATUSES:
        problems.append("invalid outcome_status")
    if record["termination_reason"] not in TERMINATION_REASONS:
        problems.append("invalid termination_reason")
    if not isinstance(record["tokens"], Mapping) or set(record["tokens"]) != set(TOKEN_FIELDS):
        problems.append(f"tokens must have exactly {TOKEN_FIELDS}")
    if record["injected_skill_count"] != len(record["injected_skill_ids"]):
        problems.append("injected_skill_count differs from injected_skill_ids")
    if record["injected_skill_count"] != ACTIVE_SKILL_COUNT[arm]:
        problems.append(f"{arm} must inject exactly {ACTIVE_SKILL_COUNT[arm]} procedural skill(s)")
    if arm == "D18" and list(record["injected_skill_ids"]) != list(record["natural_top3_skill_ids"]):
        problems.append("D18 must inject exactly the natural top-3 in rank order")
    if arm == "D0":
        if record["filler_corpus_sha256"] is not None or record["filler_sentence_indices"]:
            problems.append("D0 must not carry neutral filler")
        if "memory" not in record["not_applicable"]:
            problems.append("D0 must declare memory fields not applicable")
    if arm in ("Rel", "Irr", "Fill") and not record["filler_corpus_sha256"]:
        problems.append(f"{arm} must identify its neutral filler corpus")
    status, reason = record["outcome_status"], record["termination_reason"]
    if status == "VALID":
        if not isinstance(record["success"], bool):
            problems.append("a VALID outcome needs a boolean success")
        elif record["success"] != (reason == "success"):
            problems.append("success and termination_reason disagree")
        if reason not in OUTCOME_TERMINATIONS:
            problems.append(f"termination {reason!r} is not an outcome; the record cannot be VALID")
        used = record["prefix_actions"] + record["detour_actions"] + record["recovery_actions"]
        if not all(isinstance(record[k], int) for k in ("prefix_actions", "detour_actions", "recovery_actions", "total_action_cap")):
            problems.append("action counts must be integers")
        elif used > record["total_action_cap"]:
            problems.append("actions exceed the total action cap")
    else:
        if record["success"] is not None:
            problems.append("a non-VALID record must have success None (never coerced to failure)")
        if status == "MISSING_INFRASTRUCTURE" and reason not in INFRASTRUCTURE_TERMINATIONS:
            problems.append("MISSING_INFRASTRUCTURE requires an infrastructure termination reason")
    return problems


def make_record(**fields: Any) -> dict[str, Any]:
    record = {"schema": MEASUREMENT_SCHEMA, "protocol_version": PROTOCOL_VERSION, **fields}
    problems = validate_record(record)
    if problems:
        raise MeasurementError("; ".join(problems))
    return record


# --- relevance metrics -----------------------------------------------------------------------------------------------
def _is_relevant(label: str, mode: str) -> bool:
    if label == "UNCERTAIN":
        raise MeasurementError("UNCERTAIN labels must be adjudicated before analysis")
    if mode == "strict":
        return label == "RELEVANT"
    if mode == "lenient":
        return label in ("RELEVANT", "PARTIAL")
    raise MeasurementError(f"unknown relevance mode {mode!r}")


def _labels_for(task_id: str, skill_ids: Sequence[str], labels: Mapping[tuple[str, str], str]) -> list[str]:
    out = []
    for skill_id in skill_ids:
        label = labels.get((task_id, skill_id))
        if label is None:
            raise MeasurementError(f"missing frozen label for {task_id} / {skill_id}")
        out.append(label)
    return out


def retrieved_precision_at_3(record: Mapping[str, Any], labels: Mapping[tuple[str, str], str], mode: str = "strict") -> float:
    ids = list(record["natural_top3_skill_ids"])
    if len(ids) != 3:
        raise MeasurementError("retrieved Precision@3 needs exactly three natural retrieved skills")
    return sum(_is_relevant(l, mode) for l in _labels_for(record["task_id"], ids, labels)) / 3


def injected_relevance(record: Mapping[str, Any], labels: Mapping[tuple[str, str], str], mode: str = "strict") -> dict[str, Any]:
    ids = list(record["injected_skill_ids"])
    if not ids:
        return {"injected_skill_count": 0, "injected_relevant_fraction": None, "injected_irrelevant_count": 0,
                "status": "not_applicable_zero_injected"}
    found = _labels_for(record["task_id"], ids, labels)
    relevant = [_is_relevant(l, mode) for l in found]
    return {"injected_skill_count": len(ids), "injected_relevant_fraction": sum(relevant) / len(ids),
            "injected_irrelevant_count": sum(l == "IRRELEVANT" for l in found), "status": "computed"}
