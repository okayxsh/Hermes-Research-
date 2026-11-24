# RQ1 Phase 2B: scientific decision register (DRAFT)

**Status: DRAFT. Every decision below is NOT APPROVED and awaits author approval.**
- Implementation directions in a task brief are not approvals.
- A decision becomes approved only through an external human record bound to `readiness.decision_sha256(name, choice)`.
- This file is generated from `src/rq1/five_arm/protocol.py` (`SCIENTIFIC_DECISIONS`). Print the live register with `python -m rq1.five_arm decisions`.

**Hypothesis hierarchy (proposed):** primary D18-D0; secondary exploratory D18-Fill, Irr-Fill, Rel-Irr, Rel-Fill.

**Claims boundary:** The five-arm experiment uses one fixed, frozen library (Accum-18).  It cannot establish: the effect of progressive library growth; an optimal library size; efficacy on unrelated benchmarks; efficacy across model families; or autonomous learning improvements over time.  The primary comparison estimates the total effect of natural retrieval from this library versus no memory for one model, one environment and the screened task population.

**Stages blocked** use these short names: CPU = `university_cpu_integration`; probe = `provider_compatibility_probe`; diagnostic pilot = `d0_failure_manipulation_pilot`; five-arm pilot = `five_arm_scientific_pilot`; confirmation = `confirmatory`.

## Summary

| ID | Decision | Recommended option | Blocks | Status |
|---|---|---|---|---|
| SD-01 | `d0_representation` | absent_memory_section | diagnostic pilot, five-arm pilot, confirmation | awaiting author approval |
| SD-02 | `rel_candidate_scope` | natural_top3 | five-arm pilot, confirmation | awaiting author approval |
| SD-03 | `irr_candidate_scope` | natural_top3 | five-arm pilot, confirmation | awaiting author approval |
| SD-04 | `partial_label_policy` | injection_binary_only__p3_strict_primary_lenient_sensitivity | five-arm pilot, confirmation | awaiting author approval |
| SD-05 | `active_slot_position` | slot_1 | five-arm pilot, confirmation | awaiting author approval |
| SD-06 | `token_tolerance` | `{"absolute_tokens": 8, "relative": 0.02}` | five-arm pilot, confirmation | awaiting author approval |
| SD-07 | `action_cap` | chosen_by_budget_selection_rule_SD-17 | five-arm pilot, confirmation | awaiting author approval |
| SD-08 | `inference_seed_policy` | one_primary_seed_plus_prospective_reproducibility_subset | diagnostic pilot, five-arm pilot, confirmation | awaiting author approval |
| SD-09 | `annotation_design` | candidate_set_natural_top3 | five-arm pilot, confirmation | awaiting author approval |
| SD-10 | `hypothesis_hierarchy_and_primary_test` | single_primary_D18_vs_D0__task_level_sign_flip_sharp_null__template_level_sensitivity | confirmation | awaiting author approval |
| SD-11 | `missing_data_and_retry_policy` | retry_infrastructure_up_to_2__pairwise_complete_case__bounds_sensitivity | diagnostic pilot, five-arm pilot, confirmation | awaiting author approval |
| SD-12 | `filler_corpus` | candidate corpus rq1-five-arm-neutral-filler-candidate-v2 after human review (v1 superseded) | five-arm pilot, confirmation | awaiting author approval |
| SD-13 | `runtime_model_digest` | installed 6114515d... after runtime/tokenizer verification and an explicit amendment | diagnostic pilot, five-arm pilot, confirmation | awaiting author approval |
| SD-14 (new in 2B) | `frozen_retrieval_intervention` | single_checkpoint_retrieval_static_memory__live_top3_must_equal_frozen__tie_fragile_tasks_flagged | five-arm pilot, confirmation | awaiting author approval |
| SD-15 (new in 2B) | `diagnostic_action_allowance` | equal_model_action_allowance | diagnostic pilot | awaiting author approval |
| SD-16 (new in 2B) | `diagnostic_decision_thresholds` | see rq1.five_arm.pilot.PROPOSED_DIAGNOSTIC_THRESHOLDS | diagnostic pilot | awaiting author approval |
| SD-17 (new in 2B) | `budget_selection_rule` | rq1.five_arm.pilot.budget_decision on D0-detoured diagnostic records | diagnostic pilot, five-arm pilot, confirmation | awaiting author approval |
| SD-18 (new in 2B) | `exposure_risk_acceptance` | accept with documented limitation, bound to roster + coverage statement + frozen protocol | diagnostic pilot, five-arm pilot, confirmation | awaiting author approval |
| SD-19 (new in 2B) | `pilot_roster_composition` | 12 valid_seen tasks, family-stratified, no known model-outcome exposure, oracle-validated at cap 100 | diagnostic pilot, five-arm pilot, confirmation | awaiting author approval |
| SD-20 (new in 2B) | `secondary_multiplicity` | holm_within_secondary_family_labelled_exploratory | confirmation | awaiting author approval |
| SD-21 (new in 2B) | `primary_uncertainty_interval` | template_cluster_bootstrap_percentile_95 | confirmation | awaiting author approval |
| SD-22 (new in 2B) | `provider_token_discrepancy_tolerance` | `{"tokens": 0}` | diagnostic pilot, five-arm pilot, confirmation | awaiting author approval |
| SD-23 (new in 2B) | `filler_blocked_cell_policy` | exclude_blocked_filler_cells_prospectively_keep_natural_pair | five-arm pilot, confirmation | awaiting author approval |

## Details

### SD-01 `d0_representation`

- **Question:** How is the no-memory baseline presented to the model?
- **Scientific consequence:** Defines the reference arm of the primary comparison; the historical marker tells the model memory is missing, the absent section carries no flag.  Either way D0 differs structurally from memory-bearing arms.
- **Recommended option:** absent_memory_section
- **Alternatives:** historical_nolib_marker
- **Evidence required before approval:** Prompt-structure audit (compare_arm_prompts) on frozen pilot checkpoints.
- **Blocks:** diagnostic pilot, five-arm pilot, confirmation
- **Status:** NOT APPROVED (awaiting author approval)

### SD-02 `rel_candidate_scope`

- **Question:** Where may the Rel skill come from?
- **Scientific consequence:** natural_top3 keeps Rel a skill the retriever actually surfaced and makes candidate-set annotation sufficient; full_ranking raises Rel coverage but needs labels for all 18 skills per task.
- **Recommended option:** natural_top3
- **Alternatives:** full_ranking
- **Evidence required before approval:** Frozen rankings; candidate-set label coverage.
- **Blocks:** five-arm pilot, confirmation
- **Status:** NOT APPROVED (awaiting author approval)

### SD-03 `irr_candidate_scope`

- **Question:** Where may the Irr skill come from?
- **Scientific consequence:** natural_top3 makes Irr a plausible retrieval false positive; full_ranking admits less plausible irrelevant skills.
- **Recommended option:** natural_top3
- **Alternatives:** full_ranking
- **Evidence required before approval:** Frozen rankings; candidate-set label coverage.
- **Blocks:** five-arm pilot, confirmation
- **Status:** NOT APPROVED (awaiting author approval)

### SD-04 `partial_label_policy`

- **Question:** How do PARTIAL / UNCERTAIN adjudicated labels enter injection and Precision@3?
- **Scientific consequence:** Determines Rel/Irr eligibility and the relevance metrics.
- **Recommended option:** injection_binary_only__p3_strict_primary_lenient_sensitivity
- **Alternatives:** none listed
- **Evidence required before approval:** Pilot label distribution after adjudication.
- **Blocks:** five-arm pilot, confirmation
- **Status:** NOT APPROVED (awaiting author approval)

### SD-05 `active_slot_position`

- **Question:** Which slot carries the single active skill in Rel/Irr?
- **Scientific consequence:** A fixed slot mirrors D18's top-ranked position; randomising position adds a factor the sample cannot support.
- **Recommended option:** slot_1
- **Alternatives:** randomised_slot_per_task
- **Evidence required before approval:** None beyond the protocol rationale.
- **Blocks:** five-arm pilot, confirmation
- **Status:** NOT APPROVED (awaiting author approval)

### SD-06 `token_tolerance`

- **Question:** Maximum memory-block token residual versus the task's D18 anchor?
- **Scientific consequence:** Cells outside tolerance are blocked (never run); a tighter tolerance blocks more Fill/Rel/Irr cells.
- **Recommended option:** `{"absolute_tokens": 8, "relative": 0.02}`
- **Alternatives:** reviewer-set value
- **Evidence required before approval:** Matching audit with the REAL tokenizer on frozen top-3 texts (university CPU step).
- **Blocks:** five-arm pilot, confirmation
- **Status:** NOT APPROVED (awaiting author approval)

### SD-07 `action_cap`

- **Question:** Total action cap (prefix + detour + recovery)?
- **Scientific consequence:** Defines the primary outcome horizon and checkpoint eligibility.
- **Recommended option:** chosen_by_budget_selection_rule_SD-17
- **Alternatives:** 50, 100
- **Evidence required before approval:** Pre-specified budget rule (SD-17) applied to the D0 diagnostic pilot; cap-invariance replay.
- **Blocks:** five-arm pilot, confirmation
- **Status:** NOT APPROVED (awaiting author approval)

### SD-08 `inference_seed_policy`

- **Question:** Inference seeds per task and arm?
- **Scientific consequence:** Seeds are repeated provider measurements within a task, never independent tasks; extra seeds add precision only if trajectories vary at temperature 0.
- **Recommended option:** one_primary_seed_plus_prospective_reproducibility_subset
- **Alternatives:** three_seeds_all_tasks, one_seed_only
- **Evidence required before approval:** Pilot seed-variability measurement (identical-trajectory rate across seeds).
- **Blocks:** diagnostic pilot, five-arm pilot, confirmation
- **Status:** NOT APPROVED (awaiting author approval)

### SD-09 `annotation_design`

- **Question:** Full-library or candidate-set relevance annotation?
- **Scientific consequence:** Candidate set (natural top 3) is valid only if every skill that can enter any prompt is in the frozen top 3.
- **Recommended option:** candidate_set_natural_top3
- **Alternatives:** full_library
- **Evidence required before approval:** Retrieval-freezing verification (SD-14); frozen rankings.
- **Blocks:** five-arm pilot, confirmation
- **Status:** NOT APPROVED (awaiting author approval)

### SD-10 `hypothesis_hierarchy_and_primary_test`

- **Question:** Primary comparison and its test?
- **Scientific consequence:** One primary comparison avoids an underpowered three-way family; the sign-flip test is exact for the sharp null of no effect in any task.
- **Recommended option:** single_primary_D18_vs_D0__task_level_sign_flip_sharp_null__template_level_sensitivity
- **Alternatives:** three_primary_contrasts_holm, estimation_only
- **Evidence required before approval:** Prospective power simulation (SAP section 4).
- **Blocks:** confirmation
- **Status:** NOT APPROVED (awaiting author approval)

### SD-11 `missing_data_and_retry_policy`

- **Question:** Retry and missing-cell handling?
- **Scientific consequence:** Infrastructure failures are retried with the same unit identity; model failures are outcomes.
- **Recommended option:** retry_infrastructure_up_to_2__pairwise_complete_case__bounds_sensitivity
- **Alternatives:** retry_up_to_1, no_retry
- **Evidence required before approval:** Pilot infrastructure-failure rate.
- **Blocks:** diagnostic pilot, five-arm pilot, confirmation
- **Status:** NOT APPROVED (awaiting author approval)

### SD-12 `filler_corpus`

- **Question:** Which neutral filler corpus is approved?
- **Scientific consequence:** Filler defines Fill and the neutral slots of Rel/Irr; lint and token length never approve it.
- **Recommended option:** candidate corpus rq1-five-arm-neutral-filler-candidate-v2 after human review (v1 superseded)
- **Alternatives:** further revised corpus, out-of-domain procedural filler
- **Evidence required before approval:** Human content review; embedding-similarity screen; real-tokenizer matching audit.
- **Blocks:** five-arm pilot, confirmation
- **Status:** NOT APPROVED (awaiting author approval)

### SD-13 `runtime_model_digest`

- **Question:** Which Gemma 4 12B digest may scientific execution use?
- **Scientific consequence:** Same tag and quantisation do not establish equivalence; historical records keep the historical pin.
- **Recommended option:** installed 6114515d... after runtime/tokenizer verification and an explicit amendment
- **Alternatives:** historical pin 4eb23ef1... if still installable
- **Evidence required before approval:** Observed runtime record (/api/show, blobs, tokenizer, template) and the provider compatibility probe.
- **Blocks:** diagnostic pilot, five-arm pilot, confirmation
- **Status:** NOT APPROVED (awaiting author approval)

### SD-14 `frozen_retrieval_intervention`

- **Question:** Is retrieval performed once at the recovery checkpoint and held fixed for the whole recovery?
- **Scientific consequence:** The executor retrieves exactly once (post-detour failure context) and repeats the same memory block in every recovery prompt; no re-retrieval or memory reload.  This departs from any per-step memory behaviour of normal Hermes use and must be described as such.  Live top-3 must equal the frozen ranking or the unit is invalid.
- **Recommended option:** single_checkpoint_retrieval_static_memory__live_top3_must_equal_frozen__tie_fragile_tasks_flagged
- **Alternatives:** per-step re-retrieval (requires full-library labels; incompatible with the candidate-set design)
- **Evidence required before approval:** Static audit (done) plus CPU ranking reproducibility check on the workstation.
- **Blocks:** five-arm pilot, confirmation
- **Status:** NOT APPROVED (awaiting author approval)

### SD-15 `diagnostic_action_allowance`

- **Question:** How are recovery allowances matched between undisturbed and detoured diagnostic continuations?
- **Scientific consequence:** Equal post-checkpoint model-action allowance R = cap - prefix - 1 for both; the undisturbed run's total limit is cap - 1 (the detour is an environment action, not a model action).
- **Recommended option:** equal_model_action_allowance
- **Alternatives:** equal_total_cap (undisturbed gets one extra model action)
- **Evidence required before approval:** None beyond the rationale.
- **Blocks:** diagnostic pilot
- **Status:** NOT APPROVED (awaiting author approval)

### SD-16 `diagnostic_decision_thresholds`

- **Question:** Numerical thresholds for the failure-manipulation diagnostic?
- **Scientific consequence:** Decide whether the detour is a meaningful disruption before any pilot outcome exists.
- **Recommended option:** see rq1.five_arm.pilot.PROPOSED_DIAGNOSTIC_THRESHOLDS
- **Alternatives:** reviewer-set thresholds
- **Evidence required before approval:** Rationale in the pilot section of the protocol draft; must be approved before the diagnostic pilot.
- **Blocks:** diagnostic pilot
- **Status:** NOT APPROVED (awaiting author approval)

### SD-17 `budget_selection_rule`

- **Question:** How is the action cap chosen?
- **Scientific consequence:** Uses only baseline (D0, detoured) feasibility and ceiling/floor behaviour from the diagnostic pilot run at cap 100; never any treatment difference.
- **Recommended option:** rq1.five_arm.pilot.budget_decision on D0-detoured diagnostic records
- **Alternatives:** fix cap 100 without a pilot rule
- **Evidence required before approval:** Cap-invariance replay (CPU) confirming truncation validity.
- **Blocks:** diagnostic pilot, five-arm pilot, confirmation
- **Status:** NOT APPROVED (awaiting author approval)

### SD-18 `exposure_risk_acceptance`

- **Question:** Is the residual (unresolved, not disproven) exposure risk accepted for the exact roster?
- **Scientific consequence:** Absence of located outcomes is not proof of non-exposure; the limitation is reported in the manuscript.
- **Recommended option:** accept with documented limitation, bound to roster + coverage statement + frozen protocol
- **Alternatives:** reject and re-screen (not authorised now)
- **Evidence required before approval:** Exposure-risk statement (exposure_acceptance module).
- **Blocks:** diagnostic pilot, five-arm pilot, confirmation
- **Status:** NOT APPROVED (awaiting author approval)

### SD-19 `pilot_roster_composition`

- **Question:** Which tasks form the pilot roster?
- **Scientific consequence:** Pilot tasks must not be confirmatory tasks or known-exposed; 12 tasks is a diagnostic, not a hypothesis test.
- **Recommended option:** 12 valid_seen tasks, family-stratified, no known model-outcome exposure, oracle-validated at cap 100
- **Alternatives:** different size
- **Evidence required before approval:** Phase 1 inventory and exposure ledger; exact-cap oracle replay.
- **Blocks:** diagnostic pilot, five-arm pilot, confirmation
- **Status:** NOT APPROVED (awaiting author approval)

### SD-20 `secondary_multiplicity`

- **Question:** How are secondary hypothesis tests adjusted?
- **Scientific consequence:** Secondary contrasts are exploratory; Holm within the secondary family only, never pooled with the primary.
- **Recommended option:** holm_within_secondary_family_labelled_exploratory
- **Alternatives:** unadjusted_estimates_only
- **Evidence required before approval:** None.
- **Blocks:** confirmation
- **Status:** NOT APPROVED (awaiting author approval)

### SD-21 `primary_uncertainty_interval`

- **Question:** Which interval accompanies the primary estimate?
- **Scientific consequence:** Tasks sharing a goal template are dependent; a task bootstrap ignores that dependence.
- **Recommended option:** template_cluster_bootstrap_percentile_95
- **Alternatives:** task_bootstrap_95, both_reported_task_primary
- **Evidence required before approval:** Power/precision simulation under template dependence.
- **Blocks:** confirmation
- **Status:** NOT APPROVED (awaiting author approval)

### SD-22 `provider_token_discrepancy_tolerance`

- **Question:** Allowed difference between provider prompt_eval_count and the local chat-template count?
- **Scientific consequence:** Guards against silent provider truncation and template mismatch; a measurement outside tolerance is INVALID.
- **Recommended option:** `{"tokens": 0}`
- **Alternatives:** small positive tolerance set from the compatibility probe
- **Evidence required before approval:** Provider compatibility probe results.
- **Blocks:** diagnostic pilot, five-arm pilot, confirmation
- **Status:** NOT APPROVED (awaiting author approval)

### SD-23 `filler_blocked_cell_policy`

- **Question:** What happens to tasks whose filler-bearing cell cannot be matched?
- **Scientific consequence:** Blocked Fill/Rel/Irr cells are excluded prospectively from the contrasts that need them; the D0/D18 pair of the task is never discarded for that reason.
- **Recommended option:** exclude_blocked_filler_cells_prospectively_keep_natural_pair
- **Alternatives:** drop the whole task
- **Evidence required before approval:** Real-tokenizer matching audit before freezing.
- **Blocks:** five-arm pilot, confirmation
- **Status:** NOT APPROVED (awaiting author approval)
