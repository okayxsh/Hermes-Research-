# RQ1 workflow hierarchy: which protocol is current?

This page reconciles the historical experiment designs with the proposed Phase 2 scientific workflow. **Historical protocols are preserved unchanged**; nothing below edits or reinterprets their evidence.

| Layer | Status | Code | Documents | May it run? |
|---|---|---|---|---|
| 1. Original four-condition design (NoLib / Clean-24 / Accum-60 / Accum-96) | **Historical, superseded** | `rq1.evaluation`, `rq1.experiment` | `docs/EVALUATION_PROTOCOL.md`, decisions 001–011, the original `AGENTS.md` protocol section | No new scientific runs |
| 2. Amended evaluation (Decision 012: pre-evaluation library amendment, Accum-18 core) | **Historical; components reused** | `rq1.evaluation.amended_*`, `rq1.pilot.real_runtime.harnesses`, `rq1.hermes.episode_driver` | `docs/EVALUATION_AMENDED_PROTOCOL.md`, `docs/decisions/012-…` | Its frozen components (harness, single-retrieval boundary, prompt builder, bridge) are reused **unchanged** by Phase 2 |
| 3. University GPU six-arm workflow (D0/D18/Rel/Irr/Fill/Gate) and engineering scale-up | **Closed engineering workflow** | `rq1.university_gpu` | `university-gpu/README.md`, `docs/UNIVERSITY_RUNBOOK.md` | No: engineering evidence only; its 11 audit findings are confirmed (Phase 2 protocol draft §1) |
| 4. Phase 1 scientific preparation (inventory, exposure ledger, oracle validation) | **Completed preparation tooling** | `rq1.scientific_prep` | `docs/SCIENTIFIC_PHASE1_TASK_ORACLE_PREP.md` | CPU-only preparation commands |
| 5. Phase 2/2B five-arm scientific workflow (D0/D18/Rel/Irr/Fill; single primary D18 − D0) | **Current proposal: DRAFT, unapproved** | `rq1.five_arm` | `docs/PHASE2_FIVE_ARM_PROTOCOL_DRAFT.md`, `PHASE2_STATISTICAL_ANALYSIS_PLAN_DRAFT.md`, `PHASE2_DECISION_REGISTER_DRAFT.md`, `PHASE2_RATER_INSTRUCTIONS_DRAFT.md` | Only as far as its readiness levels allow (below) |

## Current readiness requirements (layer 5)

| Level | What it authorises | Main prerequisites |
|---|---|---|
| `local_synthetic_development` | synthetic tests | local suite passes |
| `university_cpu_integration` | CPU tools on the workstation; **no inference** | reviewed, pushed commit |
| `provider_compatibility_probe` | one non-task probe | observed runtime record; authorisation bound to the probe plan |
| `d0_failure_manipulation_pilot` | D0 undisturbed/detoured diagnostic | probe results, runtime verification, pilot roster, exposure acceptance, diagnostic protocol and thresholds, oracle replay, decisions |
| `five_arm_scientific_pilot` | five-arm pilot (valid_seen) | diagnostic review, budget decision, approved filler, pilot labels, chosen-cap oracle replay, decisions |
| `confirmatory` | confirmatory run (valid_unseen) | pilot GO, final protocol, SAP, sample-size justification, confirmatory roster, exposure acceptance and labels, chosen-cap oracle replay, decisions |

Run `python -m rq1.five_arm readiness --state <state.json>` to see which levels are BLOCKED and why. **As of this branch, every level above `university_cpu_integration` is BLOCKED**: no approvals exist.

## Rules that hold across layers

- Valid_unseen is touched only for final evaluation (and authorised CPU task preparation).
- Historical records keep their protocol tags and the historical model pin `4eb23ef…`.
- Five-arm analysis refuses six-arm, four-condition and pilot-diagnostic records.
- Agents and developers must not approve scientific decisions, fabricate labels or approvals, or run models without the matching readiness level and an explicit authorisation.
