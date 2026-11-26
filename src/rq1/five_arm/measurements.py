"""Uniform per-episode measurement schemas (five-arm v2 and the ISOLATED pilot diagnostic), plus relevance metrics.

Natural retrieval (computed for every five-arm episode, model-visible only in D18) and injected memory (what the model
actually saw) are SEPARATE fields.  Retrieved Precision@3 uses the natural top-3; injected relevance uses the injected
IDs.  D0 carries every field, with explicit empty / not-applicable values.

v2 adds what the executed path must persist: a per-decision ``recovery_trace`` (needed for exact truncation to a smaller
action cap), every provider call (provider prompt_eval_count next to local token counts, latency, errors), the prompt
sequence identity, the requested provider options, and a ``failure_detail`` for non-VALID records.

Pilot-diagnostic records (UNDISTURBED / DETOURED) use their own schema and protocol tag; the five-arm analysis refuses
them, so diagnostic episodes can never be merged into the confirmatory dataset.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from rq1.five_arm.protocol import ACTIVE_SKILL_COUNT, CONDITIONS, PROTOCOL_VERSION, UNDISTURBED_CONDITIONS

MEASUREMENT_SCHEMA = "rq1-seven-condition-measurement-v3"
DIAGNOSTIC_SCHEMA = "rq1-five-arm-diagnostic-v1"
# Pinned literal: the completed D0 diagnostic (Phase 3A) produced records with this tag; it must stay valid.
DIAGNOSTIC_PROTOCOL_VERSION = "rq1-five-arm-v1-pilot-diagnostic"
DIAGNOSTIC_CONDITIONS = ("UNDISTURBED", "DETOURED")
OUTCOME_STATUSES = ("VALID", "MISSING_INFRASTRUCTURE", "MISSING_CELL", "UNAPPROVED_INTERVENTION", "INVALID_MEASUREMENT")
TERMINATION_REASONS = ("success", "budget_exhausted", "invalid_action_limit", "environment_terminal_failure",
                       "provider_error", "environment_error", "infrastructure_interruption", "missing_cell",
                       "unapproved_intervention", "invalid_measurement")
OUTCOME_TERMINATIONS = {"success", "budget_exhausted", "invalid_action_limit", "environment_terminal_failure"}
INFRASTRUCTURE_TERMINATIONS = {"provider_error", "environment_error", "infrastructure_interruption"}
EXECUTOR_MODES = ("arm", "diagnostic_undisturbed", "diagnostic_detoured")
REQUIRED_FIELDS = (
    "schema", "protocol_version", "synthetic", "executor_mode", "unit_key", "attempt", "task_id", "goal_template", "scene",
    "family", "exposure_policy", "oracle_id", "oracle_sha256", "checkpoint_digest", "post_detour_digest", "condition",
    "seed", "runtime_identity_sha256", "natural_top3_skill_ids", "natural_top3_scores", "injected_skill_ids",
    "injected_skill_count", "analysis_only_injected_labels", "filler_corpus_sha256", "filler_sentence_indices",
    "memory_block_sha256", "first_recovery_prompt_sha256", "prompt_sequence_sha256", "tokens", "provider_calls",
    "requested_provider_options", "total_action_cap", "prefix_actions", "detour_actions", "recovery_action_allowance",
    "recovery_actions", "recovery_trace", "success", "termination_reason", "invalid_actions", "wall_clock_seconds",
    "inference_latency_seconds", "outcome_status", "failure_detail", "not_applicable",
)
# Seven-condition records also carry execution provenance (endpoint, GPU lane, worker, host, request/response identity);
# the diagnostic schema is unchanged so the completed diagnostic records remain valid.
ARM_REQUIRED_FIELDS = REQUIRED_FIELDS + ("execution_provenance", "memory_source")
MEMORY_SOURCES = ("none", "natural_retrieval_at_post_detour_state", "frozen_block_from_detoured_natural_retrieval",
                  "controlled_injection_from_frozen_selection")
TOKEN_FIELDS = ("raw_text_tokens", "memory_block_tokens", "chat_template_tokens", "provider_prompt_tokens")
TRACE_KEYS = ("decision", "action", "rejections", "env_action_valid", "done", "success", "selection_exhausted")


class MeasurementError(ValueError):
    pass


def _validate_trace(record: Mapping[str, Any]) -> list[str]:
    trace = record["recovery_trace"]
    if not isinstance(trace, list):
        return ["recovery_trace must be a list"]
    problems = []
    for index, step in enumerate(trace, 1):
        if not isinstance(step, Mapping) or tuple(sorted(step)) != tuple(sorted(TRACE_KEYS)):
            return [f"recovery_trace entry {index} must have exactly {TRACE_KEYS}"]
        if step["decision"] != index:
            problems.append("recovery_trace decisions must be numbered 1..n")
    if record["outcome_status"] == "VALID":
        actions = sum(1 for s in trace if s["action"] is not None)
        if actions != record["recovery_actions"]:
            problems.append("recovery_trace action count differs from recovery_actions")
        traced_success = any(s["success"] is True for s in trace)
        if traced_success != record["success"]:
            problems.append("recovery_trace success differs from success")
        if sum(len(s["rejections"]) for s in trace) != record["invalid_actions"]:
            problems.append("recovery_trace rejections differ from invalid_actions")
    return problems


def _validate_common(record: Mapping[str, Any]) -> list[str]:
    problems = []
    if record["executor_mode"] not in EXECUTOR_MODES:
        problems.append("invalid executor_mode")
    if record["outcome_status"] not in OUTCOME_STATUSES:
        problems.append("invalid outcome_status")
    if record["termination_reason"] not in TERMINATION_REASONS:
        problems.append("invalid termination_reason")
    if not isinstance(record["tokens"], Mapping) or set(record["tokens"]) != set(TOKEN_FIELDS):
        problems.append(f"tokens must have exactly {TOKEN_FIELDS}")
    if not isinstance(record["provider_calls"], list):
        problems.append("provider_calls must be a list")
    if record["injected_skill_count"] != len(record["injected_skill_ids"]):
        problems.append("injected_skill_count differs from injected_skill_ids")
    status, reason = record["outcome_status"], record["termination_reason"]
    if status == "VALID":
        if not isinstance(record["success"], bool):
            problems.append("a VALID outcome needs a boolean success")
        elif record["success"] != (reason == "success"):
            problems.append("success and termination_reason disagree")
        if reason not in OUTCOME_TERMINATIONS:
            problems.append(f"termination {reason!r} is not an outcome; the record cannot be VALID")
        ints = ("prefix_actions", "detour_actions", "recovery_actions", "total_action_cap", "recovery_action_allowance")
        if not all(isinstance(record[k], int) for k in ints):
            problems.append("action counts must be integers")
        else:
            if record["recovery_actions"] > record["recovery_action_allowance"]:
                problems.append("recovery actions exceed the recovery action allowance")
            if record["prefix_actions"] + record["detour_actions"] + record["recovery_actions"] > record["total_action_cap"]:
                problems.append("actions exceed the total action cap")
        if record["failure_detail"] is not None:
            problems.append("a VALID outcome carries no failure_detail")
    else:
        if record["success"] is not None:
            problems.append("a non-VALID record must have success None (never coerced to failure)")
        if status == "MISSING_INFRASTRUCTURE" and reason not in INFRASTRUCTURE_TERMINATIONS:
            problems.append("MISSING_INFRASTRUCTURE requires an infrastructure termination reason")
        if not record["failure_detail"]:
            problems.append("a non-VALID record needs a failure_detail")
    return problems + _validate_trace(record)


def validate_record(record: Mapping[str, Any]) -> list[str]:
    problems = [f"missing field {name}" for name in ARM_REQUIRED_FIELDS if name not in record]
    if problems:
        return problems
    if record["schema"] != MEASUREMENT_SCHEMA or record["protocol_version"] != PROTOCOL_VERSION:
        return ["schema/protocol mismatch (not a seven-condition measurement)"]
    if record["executor_mode"] != "arm":
        return ["seven-condition measurements must come from the arm executor mode"]
    if record["condition"] not in CONDITIONS:
        return [f"condition {record['condition']!r} is not one of the seven conditions"]
    problems = _validate_common(record)
    arm = record["condition"]
    if not isinstance(record["execution_provenance"], Mapping) or not record["execution_provenance"].get("mode"):
        problems.append("execution_provenance must be a mapping with an execution mode")
    if record["memory_source"] not in MEMORY_SOURCES:
        problems.append(f"memory_source must be one of {MEMORY_SOURCES}")
    expected_detour = 0 if arm in UNDISTURBED_CONDITIONS else 1
    if record["detour_actions"] != expected_detour:
        problems.append(f"{arm} episodes contain exactly {expected_detour} controlled detour action(s)")
    if record["outcome_status"] != "VALID" and "memory_not_constructed_before_failure" in record["not_applicable"]:
        if record["injected_skill_ids"] or record["filler_corpus_sha256"] or record["memory_block_sha256"]:
            problems.append("a record whose memory was never constructed cannot report injected memory")
        return problems
    if record["injected_skill_count"] != ACTIVE_SKILL_COUNT[arm]:
        problems.append(f"{arm} must inject exactly {ACTIVE_SKILL_COUNT[arm]} procedural skill(s)")
    if arm in ("D18", "D18U") and list(record["injected_skill_ids"]) != list(record["natural_top3_skill_ids"]):
        problems.append(f"{arm} must inject exactly the (frozen) natural top-3 in rank order")
    expected_source = {"D0": "none", "D0U": "none", "D18": "natural_retrieval_at_post_detour_state",
                       "D18U": "frozen_block_from_detoured_natural_retrieval"}.get(arm, "controlled_injection_from_frozen_selection")
    if record["memory_source"] != expected_source:
        problems.append(f"{arm} memory_source must be {expected_source!r}")
    if arm in ("D0", "D0U"):
        if record["filler_corpus_sha256"] is not None or record["filler_sentence_indices"]:
            problems.append("D0 must not carry neutral filler")
        if "memory" not in record["not_applicable"]:
            problems.append(f"{arm} must declare memory fields not applicable")
    if arm in ("Rel", "Irr", "Fill") and not record["filler_corpus_sha256"]:
        problems.append(f"{arm} must identify its neutral filler corpus")
    return problems


def validate_diagnostic_record(record: Mapping[str, Any]) -> list[str]:
    problems = [f"missing field {name}" for name in REQUIRED_FIELDS if name not in record]
    if problems:
        return problems
    if record["schema"] != DIAGNOSTIC_SCHEMA or record["protocol_version"] != DIAGNOSTIC_PROTOCOL_VERSION:
        return ["schema/protocol mismatch (not a pilot-diagnostic record)"]
    if record["condition"] not in DIAGNOSTIC_CONDITIONS:
        return [f"diagnostic condition must be one of {DIAGNOSTIC_CONDITIONS}"]
    expected_mode = {"UNDISTURBED": "diagnostic_undisturbed", "DETOURED": "diagnostic_detoured"}[record["condition"]]
    problems = _validate_common(record)
    if record["executor_mode"] != expected_mode:
        problems.append("executor_mode does not match the diagnostic condition")
    if record["injected_skill_ids"] or record["natural_top3_skill_ids"] or record["filler_corpus_sha256"]:
        problems.append("diagnostic episodes use no retrieval, no skills and no filler")
    if record["detour_actions"] != (0 if record["condition"] == "UNDISTURBED" else 1):
        problems.append("UNDISTURBED has no detour action; DETOURED has exactly one")
    return problems


def make_record(**fields: Any) -> dict[str, Any]:
    record = {"schema": MEASUREMENT_SCHEMA, "protocol_version": PROTOCOL_VERSION, **fields}
    problems = validate_record(record)
    if problems:
        raise MeasurementError("; ".join(problems))
    return record


def make_diagnostic_record(**fields: Any) -> dict[str, Any]:
    record = {"schema": DIAGNOSTIC_SCHEMA, "protocol_version": DIAGNOSTIC_PROTOCOL_VERSION, **fields}
    problems = validate_diagnostic_record(record)
    if problems:
        raise MeasurementError("; ".join(problems))
    return record


def truncate_to_cap(record: Mapping[str, Any], cap: int) -> dict[str, Any]:
    """Outcome of a VALID record had it been run at the smaller total cap ``cap`` (requires the cap-invariance
    assumptions documented in the protocol: model inputs and environment transitions before the cap do not depend on
    the cap).  Uses the per-decision recovery trace; never extrapolates beyond the executed trajectory."""
    if record.get("outcome_status") != "VALID":
        raise MeasurementError("only VALID records can be truncated")
    run_cap = int(record["total_action_cap"])
    if cap > run_cap:
        raise MeasurementError("cannot extrapolate to a larger cap than the one executed")
    allowance = cap - int(record["prefix_actions"]) - int(record["detour_actions"])
    if allowance < 1:
        return {"cap": cap, "feasible": False, "success": None, "termination_reason": None}
    actions = invalid = 0
    for step in record["recovery_trace"]:
        invalid += len(step["rejections"])
        if step["selection_exhausted"]:
            return {"cap": cap, "feasible": True, "success": False, "termination_reason": "invalid_action_limit",
                    "recovery_actions": actions, "invalid_actions": invalid}
        actions += 1
        if step["success"] is True:
            return {"cap": cap, "feasible": True, "success": True, "termination_reason": "success",
                    "recovery_actions": actions, "invalid_actions": invalid}
        if step["done"]:
            return {"cap": cap, "feasible": True, "success": False, "termination_reason": "environment_terminal_failure",
                    "recovery_actions": actions, "invalid_actions": invalid}
        if actions >= allowance:
            return {"cap": cap, "feasible": True, "success": False, "termination_reason": "budget_exhausted",
                    "recovery_actions": actions, "invalid_actions": invalid}
    return {"cap": cap, "feasible": True, "success": bool(record["success"]), "termination_reason": record["termination_reason"],
            "recovery_actions": actions, "invalid_actions": invalid}


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
