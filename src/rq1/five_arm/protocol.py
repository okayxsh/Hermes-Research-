"""Versioned seven-condition protocol definition (PROPOSED / DRAFT), hypothesis hierarchy and the decision register.

Code never marks a decision approved.  Approvals are external human records validated by ``readiness``.

Phase 3C design (author-authorised implementation; the protocol itself remains DRAFT until approved):

* Seven conditions.  Detoured (after the controlled one-step navigation perturbation, total cap 50): D0, D18, Rel, Irr,
  Fill.  Undisturbed (no perturbation, total cap 49, identical model-action allowance): D0U, D18U.
* Research question (narrowed): task continuation -- completion of a partly executed task after a controlled navigation
  perturbation -- with one retrieval from a frozen 18-skill library; no recovery-mechanism claim.
* D18U shows the SAME frozen memory block that natural retrieval produced at the post-detour state of the task.  No
  retrieval happens in the undisturbed conditions; D18U is NOT independent natural retrieval at the undisturbed state.
  It holds memory content fixed so the memory x perturbation interaction is not confounded by different retrieved text.

Hypothesis hierarchy:

* ONE primary comparison: D18 versus D0 (both detoured).  Primary outcome: environment-reported task success within the approved total
  action horizon after the controlled disruption.  Primary estimand: the task-weighted mean (over the screened task
  population) of the within-task difference in recovery success between natural retrieval from the frozen Accum-18
  library and the no-memory baseline, averaging inference seeds within each task.  It is the TOTAL effect of the
  specified natural-retrieval intervention (procedural content + memory presence + prompt length); it does not isolate
  procedural relevance.
* Secondary, exploratory comparisons (separate multiplicity family, never confirmatory discoveries): D18-Fill,
  Irr-Fill, Rel-Irr, Rel-Fill; retrieval-relevance associations; action efficiency; invalid actions; latency;
  acquisition-template-overlap sensitivity.
* Secondary factorial ESTIMATES (intervals only, no tests): D18U-D0U (continuation benefit without perturbation),
  D0-D0U and D18-D18U (perturbation effect without / with memory) and the memory x perturbation interaction
  (D18-D0) - (D18U-D0U).
* Claims boundary: a fixed library cannot establish effects of progressive library growth, optimal library size,
  efficacy on unrelated benchmarks or other model families, or autonomous improvement over time.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping

PROTOCOL_VERSION = "rq1-seven-condition-v2"
PROTOCOL_STATUS = "DRAFT"
DETOURED_CONDITIONS = ("D0", "D18", "Rel", "Irr", "Fill")
UNDISTURBED_CONDITIONS = ("D0U", "D18U")
CONDITIONS = DETOURED_CONDITIONS + UNDISTURBED_CONDITIONS
BASE_CONDITION = {"D0U": "D0", "D18U": "D18"}       # undisturbed condition -> condition whose frozen memory it shows
MEMORY_BEARING = ("D18", "Rel", "Irr", "Fill", "D18U")
NATURAL_ARMS = ("D0", "D18", "D0U", "D18U")         # never depend on filler or labels: every eligible task
FILLER_ARMS = ("Rel", "Irr", "Fill")                # depend on the approved filler corpus (cells may be blocked)
ALWAYS_SCHEDULED = ("D0", "D18", "Fill")            # Fill is scheduled for every task unless its cell is blocked
MECHANISM_ARMS = ("Rel", "Irr")                     # scheduled only where the frozen selection rule yields a skill
DEFERRED_ARMS = {"Gate": "deferred: the historical Gate arm could become an exact D18 no-op; not part of v1"}
PRIMARY_CONTRAST = ("D18", "D0")
PRIMARY_CONTRASTS = (PRIMARY_CONTRAST,)
SECONDARY_CONTRASTS = (("D18", "Fill"), ("Irr", "Fill"), ("Rel", "Irr"), ("Rel", "Fill"))
FACTORIAL_ESTIMATES = (("D18U", "D0U"), ("D0", "D0U"), ("D18", "D18U"))   # plus the interaction; estimation only
HISTORICAL_SIX_ARM_PROTOCOL = "rq1-selective-memory-six-arm-v1"
HISTORICAL_SIX_ARM_CONDITIONS = ("D0", "D18", "Rel", "Irr", "Fill", "Gate")
ACTIVE_SKILL_COUNT = {"D0": 0, "D18": 3, "Rel": 1, "Irr": 1, "Fill": 0, "D0U": 0, "D18U": 3}

CLAIMS_BOUNDARY = (
    "The seven-condition experiment uses one fixed, frozen library (Accum-18).  It cannot establish: the effect of progressive "
    "library growth; an optimal library size; efficacy on unrelated benchmarks; efficacy across model families; or "
    "autonomous learning improvements over time.  The primary comparison estimates the total effect of natural "
    "retrieval from this library versus no memory for one model, one environment and the screened task population.  "
    "The perturbation is a controlled one-step navigation detour whose disruptiveness the pilot diagnostic left "
    "INCONCLUSIVE; no recovery-from-failure mechanism is claimed."
)

# Execution stages used by readiness and by the decision register's ``blocks`` field.
STAGES = ("university_cpu_integration", "provider_compatibility_probe", "d0_failure_manipulation_pilot",
          "five_arm_scientific_pilot", "confirmatory")


class ProtocolError(ValueError):
    pass


def contrast_name(treatment: str, reference: str) -> str:
    return f"{treatment}-{reference}"


def assert_five_arm_records(records: Iterable[Mapping[str, Any]]) -> None:
    """Refuse historical six-arm, pilot-diagnostic or any other records: never reinterpreted as five-arm."""
    for index, record in enumerate(records, 1):
        if record.get("protocol_version") != PROTOCOL_VERSION:
            raise ProtocolError(f"record {index} has protocol {record.get('protocol_version')!r}; six-arm or other "
                                f"records cannot be analysed as {PROTOCOL_VERSION}")
        if record.get("condition") not in CONDITIONS:
            raise ProtocolError(f"record {index} has condition {record.get('condition')!r} outside the seven-condition design")


# --- scientific decision register --------------------------------------------------------------------------------------
# Every entry: id, question, scientific consequence, engineering RECOMMENDATION (never a default used for execution),
# alternatives, evidence required before approval, and the execution stages it blocks.  Every entry awaits author
# approval; implementation directions in a task brief are not approvals.
def _d(id_: str, question: str, consequence: str, recommendation: Any, alternatives: list[Any], evidence: str,
       blocks: tuple[str, ...]) -> dict[str, Any]:
    unknown = set(blocks) - set(STAGES)
    if unknown:
        raise ProtocolError(f"{id_}: unknown stages {sorted(unknown)}")
    return {"id": id_, "question": question, "consequence": consequence, "recommendation": recommendation,
            "options": [recommendation, *alternatives] if recommendation is not None else list(alternatives),
            "alternatives": alternatives, "required_evidence": evidence, "blocks": list(blocks)}


_PILOTS = ("d0_failure_manipulation_pilot", "five_arm_scientific_pilot", "confirmatory")
_FIVE = ("five_arm_scientific_pilot", "confirmatory")

SCIENTIFIC_DECISIONS: dict[str, dict[str, Any]] = {
    "d0_representation": _d(
        "SD-01", "How is the no-memory baseline presented to the model?",
        "Defines the reference arm of the primary comparison; the historical marker tells the model memory is missing, "
        "the absent section carries no flag.  Either way D0 differs structurally from memory-bearing arms.",
        "absent_memory_section", ["historical_nolib_marker"],
        "Prompt-structure audit (compare_arm_prompts) on frozen pilot checkpoints.", _PILOTS),
    "rel_candidate_scope": _d(
        "SD-02", "Where may the Rel skill come from?",
        "natural_top3 keeps Rel a skill the retriever actually surfaced and makes candidate-set annotation sufficient; "
        "full_ranking raises Rel coverage but needs labels for all 18 skills per task.",
        "natural_top3", ["full_ranking"], "Frozen rankings; candidate-set label coverage.", _FIVE),
    "irr_candidate_scope": _d(
        "SD-03", "Where may the Irr skill come from?",
        "natural_top3 makes Irr a plausible retrieval false positive; full_ranking admits less plausible irrelevant skills.",
        "natural_top3", ["full_ranking"], "Frozen rankings; candidate-set label coverage.", _FIVE),
    "partial_label_policy": _d(
        "SD-04", "How do PARTIAL / UNCERTAIN adjudicated labels enter injection and Precision@3?",
        "Determines Rel/Irr eligibility and the relevance metrics.",
        "injection_binary_only__p3_strict_primary_lenient_sensitivity", [],
        "Pilot label distribution after adjudication.", _FIVE),
    "active_slot_position": _d(
        "SD-05", "Which slot carries the single active skill in Rel/Irr?",
        "A fixed slot mirrors D18's top-ranked position; randomising position adds a factor the sample cannot support.",
        "slot_1", ["randomised_slot_per_task"], "None beyond the protocol rationale.", _FIVE),
    "token_tolerance": _d(
        "SD-06", "Maximum memory-block token residual versus the task's D18 anchor?",
        "Cells outside tolerance are blocked (never run); a tighter tolerance blocks more Fill/Rel/Irr cells.",
        {"absolute_tokens": 8, "relative": 0.02}, ["reviewer-set value"],
        "Matching audit with the REAL tokenizer on frozen top-3 texts (university CPU step).", _FIVE),
    "action_cap": _d(
        "SD-07", "Total action cap (prefix + detour + recovery)?",
        "Defines the primary outcome horizon and checkpoint eligibility.",
        "chosen_by_budget_selection_rule_SD-17", [50, 100],
        "Pre-specified budget rule (SD-17) applied to the D0 diagnostic pilot; cap-invariance replay.",
        _FIVE),
    "inference_seed_policy": _d(
        "SD-08", "Inference seeds per task and arm?",
        "Seeds are repeated provider measurements within a task, never independent tasks; extra seeds add precision only "
        "if trajectories vary at temperature 0.",
        "one_primary_seed_plus_prospective_reproducibility_subset", ["three_seeds_all_tasks", "one_seed_only"],
        "Pilot seed-variability measurement (identical-trajectory rate across seeds).", _PILOTS),
    "annotation_design": _d(
        "SD-09", "Full-library or candidate-set relevance annotation?",
        "Candidate set (natural top 3) is valid only if every skill that can enter any prompt is in the frozen top 3.",
        "candidate_set_natural_top3", ["full_library"],
        "Retrieval-freezing verification (SD-14); frozen rankings.", _FIVE),
    "hypothesis_hierarchy_and_primary_test": _d(
        "SD-10", "Primary comparison and its test?",
        "One primary comparison avoids an underpowered three-way family; the sign-flip test is exact for the sharp null "
        "of no effect in any task.",
        "single_primary_D18_vs_D0__task_level_sign_flip_sharp_null__template_level_sensitivity",
        ["three_primary_contrasts_holm", "estimation_only"],
        "Prospective power simulation (SAP section 4).", ("confirmatory",)),
    "missing_data_and_retry_policy": _d(
        "SD-11", "Retry and missing-cell handling?",
        "Infrastructure failures are retried with the same unit identity; model failures are outcomes.",
        "retry_infrastructure_up_to_2__pairwise_complete_case__bounds_sensitivity", ["retry_up_to_1", "no_retry"],
        "Pilot infrastructure-failure rate.", _PILOTS),
    "filler_corpus": _d(
        "SD-12", "Which neutral filler corpus is approved?",
        "Filler defines Fill and the neutral slots of Rel/Irr; lint and token length never approve it.",
        "candidate corpus rq1-five-arm-neutral-filler-candidate-v2 after human review (v1 superseded)", ["further revised corpus", "out-of-domain procedural filler"],
        "Human content review; embedding-similarity screen; real-tokenizer matching audit.", _FIVE),
    "runtime_model_digest": _d(
        "SD-13", "Which Gemma 4 12B digest may scientific execution use?",
        "Same tag and quantisation do not establish equivalence; historical records keep the historical pin.",
        "installed 6114515d... after runtime/tokenizer verification and an explicit amendment",
        ["historical pin 4eb23ef1... if still installable"],
        "Observed runtime record (/api/show, blobs, tokenizer, template) and the provider compatibility probe.", _PILOTS),
    # ---- new decisions found in Phase 2B --------------------------------------------------------------------------
    "frozen_retrieval_intervention": _d(
        "SD-14", "Is retrieval performed once at the recovery checkpoint and held fixed for the whole recovery?",
        "The executor retrieves exactly once (post-detour failure context) and repeats the same memory block in every "
        "recovery prompt; no re-retrieval or memory reload.  This departs from any per-step memory behaviour of normal "
        "Hermes use and must be described as such.  Live top-3 must equal the frozen ranking or the unit is invalid.",
        "single_checkpoint_retrieval_static_memory__live_top3_must_equal_frozen__tie_fragile_tasks_flagged",
        ["per-step re-retrieval (requires full-library labels; incompatible with the candidate-set design)"],
        "Static audit (done) plus CPU ranking reproducibility check on the workstation.", _FIVE),
    "diagnostic_action_allowance": _d(
        "SD-15", "How are recovery allowances matched between undisturbed and detoured diagnostic continuations?",
        "Equal post-checkpoint model-action allowance R = cap - prefix - 1 for both; the undisturbed run's total limit is "
        "cap - 1 (the detour is an environment action, not a model action).",
        "equal_model_action_allowance", ["equal_total_cap (undisturbed gets one extra model action)"],
        "None beyond the rationale.", ("d0_failure_manipulation_pilot",)),
    "diagnostic_decision_thresholds": _d(
        "SD-16", "Numerical thresholds for the failure-manipulation diagnostic?",
        "Decide whether the detour is a meaningful disruption before any pilot outcome exists.",
        "see rq1.five_arm.pilot.PROPOSED_DIAGNOSTIC_THRESHOLDS", ["reviewer-set thresholds"],
        "Rationale in the pilot section of the protocol draft; must be approved before the diagnostic pilot.",
        ("d0_failure_manipulation_pilot",)),
    "budget_selection_rule": _d(
        "SD-17", "How is the action cap chosen?",
        "Uses only baseline (D0, detoured) feasibility and ceiling/floor behaviour from the diagnostic pilot run at cap "
        "100; never any treatment difference.",
        "rq1.five_arm.pilot.budget_decision on D0-detoured diagnostic records", ["fix cap 100 without a pilot rule"],
        "Cap-invariance replay (CPU) confirming truncation validity.", ("d0_failure_manipulation_pilot",) + _FIVE),
    "exposure_risk_acceptance": _d(
        "SD-18", "Is the residual (unresolved, not disproven) exposure risk accepted for the exact roster?",
        "Absence of located outcomes is not proof of non-exposure; the limitation is reported in the manuscript.",
        "accept with documented limitation, bound to roster + coverage statement + frozen protocol",
        ["reject and re-screen (not authorised now)"], "Exposure-risk statement (exposure_acceptance module).", _PILOTS),
    "pilot_roster_composition": _d(
        "SD-19", "Which tasks form the pilot roster?",
        "Pilot tasks must not be confirmatory tasks or known-exposed; 12 tasks is a diagnostic, not a hypothesis test.",
        "12 valid_seen tasks, family-stratified, no known model-outcome exposure, oracle-validated at cap 100",
        ["different size"], "Phase 1 inventory and exposure ledger; exact-cap oracle replay.", _PILOTS),
    "secondary_multiplicity": _d(
        "SD-20", "How are secondary hypothesis tests adjusted?",
        "Secondary contrasts are exploratory; Holm within the secondary family only, never pooled with the primary.",
        "holm_within_secondary_family_labelled_exploratory", ["unadjusted_estimates_only"], "None.", ("confirmatory",)),
    "primary_uncertainty_interval": _d(
        "SD-21", "Which interval accompanies the primary estimate?",
        "Tasks sharing a goal template are dependent; a task bootstrap ignores that dependence.",
        "template_cluster_bootstrap_percentile_95", ["task_bootstrap_95", "both_reported_task_primary"],
        "Power/precision simulation under template dependence.", ("confirmatory",)),
    "provider_token_discrepancy_tolerance": _d(
        "SD-22", "Allowed difference between provider prompt_eval_count and the local chat-template count?",
        "Guards against silent provider truncation and template mismatch; a measurement outside tolerance is INVALID.",
        {"tokens": 0}, ["small positive tolerance set from the compatibility probe"],
        "Provider compatibility probe results.", _PILOTS),
    "filler_blocked_cell_policy": _d(
        "SD-23", "What happens to tasks whose filler-bearing cell cannot be matched?",
        "Blocked Fill/Rel/Irr cells are excluded prospectively from the contrasts that need them; the D0/D18 pair of the "
        "task is never discarded for that reason.",
        "exclude_blocked_filler_cells_prospectively_keep_natural_pair", ["drop the whole task"],
        "Real-tokenizer matching audit before freezing.", _FIVE),
}


def decisions_blocking(stage: str) -> list[str]:
    if stage not in STAGES:
        raise ProtocolError(f"unknown stage {stage!r}")
    return [name for name, d in SCIENTIFIC_DECISIONS.items() if stage in d["blocks"]]


def decision_register() -> dict[str, Any]:
    return {"protocol_version": PROTOCOL_VERSION, "status": PROTOCOL_STATUS,
            "hypothesis_hierarchy": {"primary": contrast_name(*PRIMARY_CONTRAST),
                                     "secondary_exploratory": [contrast_name(*c) for c in SECONDARY_CONTRASTS]},
            "claims_boundary": CLAIMS_BOUNDARY,
            "decisions": {name: {**value, "approved": False, "awaiting_author_approval": True}
                          for name, value in SCIENTIFIC_DECISIONS.items()},
            "note": "recommendations are engineering proposals; none is approved by this code or by a task brief"}
