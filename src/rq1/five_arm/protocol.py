"""Versioned five-arm protocol definition (PROPOSED / DRAFT) and the register of open scientific decisions.

Code never marks a decision approved.  Approvals are external human records validated by ``readiness``.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping

PROTOCOL_VERSION = "rq1-five-arm-v1"
PROTOCOL_STATUS = "DRAFT"
CONDITIONS = ("D0", "D18", "Rel", "Irr", "Fill")
MEMORY_BEARING = ("D18", "Rel", "Irr", "Fill")
ALWAYS_SCHEDULED = ("D0", "D18", "Fill")            # natural analysis arms: every eligible task
MECHANISM_ARMS = ("Rel", "Irr")                     # scheduled only where the frozen selection rule yields a skill
DEFERRED_ARMS = {"Gate": "deferred: the historical Gate arm could become an exact D18 no-op; not part of v1"}
PRIMARY_CONTRASTS = (("D18", "D0"), ("D18", "Fill"), ("Irr", "Fill"))
SECONDARY_CONTRASTS = (("Rel", "Irr"), ("Rel", "Fill"))
HISTORICAL_SIX_ARM_PROTOCOL = "rq1-selective-memory-six-arm-v1"
HISTORICAL_SIX_ARM_CONDITIONS = ("D0", "D18", "Rel", "Irr", "Fill", "Gate")
ACTIVE_SKILL_COUNT = {"D0": 0, "D18": 3, "Rel": 1, "Irr": 1, "Fill": 0}


class ProtocolError(ValueError):
    pass


def contrast_name(treatment: str, reference: str) -> str:
    return f"{treatment}-{reference}"


def assert_five_arm_records(records: Iterable[Mapping[str, Any]]) -> None:
    """Refuse historical six-arm (or any other) records: they are never silently reinterpreted as five-arm."""
    for index, record in enumerate(records, 1):
        if record.get("protocol_version") != PROTOCOL_VERSION:
            raise ProtocolError(f"record {index} has protocol {record.get('protocol_version')!r}; six-arm or other "
                                f"records cannot be analysed as {PROTOCOL_VERSION}")
        if record.get("condition") not in CONDITIONS:
            raise ProtocolError(f"record {index} has condition {record.get('condition')!r} outside the five-arm design")


# --- open scientific decisions -------------------------------------------------------------------------------------
# Each entry: the question, the admissible options, the engineering PROPOSAL (never a default used for execution),
# and why it matters.  Execution requires an approved decision record for every entry (see readiness).
SCIENTIFIC_DECISIONS: dict[str, dict[str, Any]] = {
    "d0_representation": {
        "question": "How is the no-memory baseline presented to the model?",
        "options": ["absent_memory_section", "historical_nolib_marker"],
        "proposal": "absent_memory_section",
        "rationale": "No RECOVERY MEMORY section at all is the natural no-memory baseline and carries no condition flag; "
                     "the historical marker ('no_retrieved_skills_available': true) tells the model memory is missing. "
                     "Either way D0 differs structurally from the memory-bearing arms; this is documented, not matched.",
    },
    "rel_candidate_scope": {
        "question": "Where may the Rel skill come from?",
        "options": ["natural_top3", "full_ranking"],
        "proposal": "natural_top3",
        "rationale": "natural_top3 keeps Rel a single skill the natural retriever actually surfaced and makes complete "
                     "candidate-set annotation possible; full_ranking increases Rel coverage but requires labels for all 18 skills.",
    },
    "irr_candidate_scope": {
        "question": "Where may the Irr skill come from?",
        "options": ["natural_top3", "full_ranking"],
        "proposal": "natural_top3",
        "rationale": "the highest-ranked adjudicated-irrelevant skill inside the natural top 3 is a plausible retrieval false "
                     "positive; full_ranking falls back to lower-ranked (less plausible) irrelevant skills.",
    },
    "partial_label_policy": {
        "question": "How do PARTIAL / UNCERTAIN adjudicated labels enter injection and Precision@3?",
        "options": ["injection_binary_only__p3_strict_primary_lenient_sensitivity"],
        "proposal": "injection_binary_only__p3_strict_primary_lenient_sensitivity",
        "rationale": "only adjudicated RELEVANT / IRRELEVANT skills may be injected; for Precision@3 PARTIAL counts as "
                     "not relevant (strict, primary) and as relevant (lenient, sensitivity); UNCERTAIN must be adjudicated away.",
    },
    "active_slot_position": {
        "question": "Which slot carries the single active skill in Rel/Irr?",
        "options": ["slot_1"],
        "proposal": "slot_1",
        "rationale": "mirrors the D18 top-ranked position; a fixed position avoids position randomisation as an extra factor.",
    },
    "token_tolerance": {
        "question": "Maximum memory-block token residual versus the task's D18 anchor?",
        "options": ["max(8 tokens, 2% of anchor)", "other value set by reviewers"],
        "proposal": {"absolute_tokens": 8, "relative": 0.02},
        "rationale": "filler is assembled from whole reviewed sentences, never truncated, so exact equality is not always "
                     "achievable; cells outside tolerance are blocked, never silently run.",
    },
    "action_cap": {
        "question": "Total action cap (prefix + detour + recovery)?",
        "options": [50, 100],
        "proposal": None,
        "rationale": "decided by the pre-specified pilot budget rule before any confirmatory outcome exists.",
    },
    "inference_seeds": {
        "question": "Inference seeds per task and arm?",
        "options": [[11], [11, 29, 47]],
        "proposal": [11, 29, 47],
        "rationale": "seeds are provider inference seeds only; ALFWorld state is fixed by the frozen checkpoint, so seeds are "
                     "repeated measurements within a task, never independent environment randomisations.",
    },
    "annotation_design": {
        "question": "Full-library or candidate-set relevance annotation?",
        "options": ["full_library", "candidate_set_natural_top3"],
        "proposal": "candidate_set_natural_top3 only if both Rel and Irr scopes are natural_top3; otherwise full_library",
        "rationale": "the candidate set must contain every skill that can be retrieved or injected under the frozen rules.",
    },
    "primary_inference": {
        "question": "Primary test for each primary contrast?",
        "options": ["task_level_sign_flip_permutation_holm"],
        "proposal": "task_level_sign_flip_permutation_holm",
        "rationale": "task is the unit; seed outcomes are averaged within task; two-sided exact/Monte-Carlo sign-flip test "
                     "on task-level paired differences; Holm across the three primary contrasts.",
    },
    "missing_data_policy": {
        "question": "Retry and missing-cell handling?",
        "options": ["retry_infrastructure_up_to_2__pairwise_complete_case__bounds_sensitivity"],
        "proposal": "retry_infrastructure_up_to_2__pairwise_complete_case__bounds_sensitivity",
        "rationale": "infrastructure failures (provider/environment/interruption) are retried with the same unit identity; "
                     "model failures, budget exhaustion and invalid actions are outcomes and never retried.",
    },
    "filler_corpus": {
        "question": "Which neutral filler corpus is approved?",
        "options": ["a reviewed corpus file identified by SHA-256"],
        "proposal": None,
        "rationale": "filler requires separate scientific review; token length alone never approves it.",
    },
    "runtime_model_digest": {
        "question": "Which Gemma 4 12B digest may scientific execution use?",
        "options": ["4eb23ef1… (historical pin)", "6114515d… (installed engineering digest)"],
        "proposal": "6114515d… only after runtime and tokenizer verification and an explicit amendment",
        "rationale": "same tag and quantisation do not establish equivalence; historical records keep their pin.",
    },
}


def decision_register() -> dict[str, Any]:
    return {"protocol_version": PROTOCOL_VERSION, "status": PROTOCOL_STATUS,
            "decisions": {name: {**value, "approved": False} for name, value in SCIENTIFIC_DECISIONS.items()},
            "note": "proposals are engineering suggestions; none is approved by this code"}
