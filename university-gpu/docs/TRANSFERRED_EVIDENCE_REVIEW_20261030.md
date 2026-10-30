# Transferred evidence review — unapproved prospective amendment

Archive checksum verified: `11c8ca453ef4ceaef5dce189c976ffc09636fdc218e40e22a0803deac9e429e2`. The original archive is preserved. Candidate route/state definitions and the byte-identical original proposal are outside the repository in `rq1-evidence-audit-20261030-v2`. Each oracle member hash is retained in candidate-failures.json. No candidate has oracle.validated=true. Earlier absence-of-evidence reports describe the pre-transfer state; this review supersedes that limitation.

| Family | Route | Prefix | Remaining route | Recovery budget | Oracle actions | Verdict |
|---|---:|---:|---:|---:|---:|---|
| clean_and_place | 95 | 48 | 47 | 1 | 96 | BLOCKED |
| cool_and_place | 13 | 6 | 7 | 43 | 14 | REQUIRES HUMAN REVIEW |
| heat_and_place | 14 | 6 | 8 | 43 | 15 | REQUIRES HUMAN REVIEW |
| look_at_object | 11 | 5 | 6 | 44 | 12 | REQUIRES HUMAN REVIEW |
| pick_and_place | 6 | 4 | 2 | 45 | 7 | REQUIRES HUMAN REVIEW |
| pick_two_and_place | 146 | 74 | 72 | -25 | 147 | BLOCKED |

All six original oracle logs report success with a 200-action cap. Main replay and independent oracle checkpoint/post-detour digests match. Four successful complete traces use fewer than 50 actions; this is observed bounded-length success, while replay under cap 50 remains untested. The two long routes cannot pass the frozen unchanged-script oracle. No shorter-plan impossibility claim is made. Raw logs contradict the earlier 92/109 lengths: actual reference arrays are 95/146. Full task IDs, detours and evidence member paths are in the JSON report.

## Selection and exposure decision proposal
The actual pilot proposal explicitly uses task-selection-v1, seed 11, round_robin_families and status proposed with null approvals. The historical longest-route policy belongs to amended valid-unseen evaluation; it does not automatically override the new pilot proposal. Authors must resolve whether the prospective pilot inherits this seeded proposal or needs a new prospective rule. The pick-and-place task has exposed Gemma recovery outcomes in the engineering demonstration. Preserve exposure records; do not use successes to select or retain candidates.
Proposed rule for author review: retain seeded family ordering, prespecify a deterministic expert-route eligibility screen under total cap 50 and its failure/replacement rule before rerunning selection; report how the screen changes the target task population. Use the next candidate in the original seeded within-family ordering, not the easiest task or one selected by Gemma success. Reconstruct the ordering from original discovery inventory and seed; do not infer it from excluded-task prose. If full independent selection is required after outcome exposure, use a newly prespecified seed and disjoint tasks and document that amendment. None of these changes is approved.

## Model, tokenizer, ratings and control decisions
Historical digest remains 4eb23ef187e2c5462566d6a1d3bbbc2f1346d0b4327cbb66d58fffbcc9b2b05c. Prospective installed digest is 6114515d63c17436a7c0417d82820ac65ad643e2806c5a3c89cb62846436ed0b. Same tag/Q4_K_M does not establish equivalence. Authors must approve a separate prospective environment/protocol identity and contemporary D0/D18 cells; historical results cannot fill primary new cells. No new identity is activated. Tokenizer and Accum-18 library files are absent from this transferred archive. Do not fabricate tokenizer source/revision/hash or emit incomplete 108-pair forms. Recover the exact frozen library and approved task contexts first. Pilot requires 216 independent judgments plus adjudication of 108 pairs; confirmation requires 1080 independent judgments plus adjudication of 540 pairs unless a documented compatibility review authorizes reuse.
Candidate filler for human semantic review only: “This is a neutral reference separator.” No approval or claim of behavioral neutrality is made. Token matching must use the existing exact sorted ensure_ascii=False JSON memory-block serialization and real pinned tokenizer; full-prompt equality is not claimed. Gate remains uncomputed pending human labels. Existing tests cover memory accounting, unchanged D18, relevance/leakage gates and outcome-blind calibration.

## Pre-confirmatory statistical review checklist
- Prespecify primary binary recovery success within eligible-task budget, target population and primary paired arm contrasts before confirmation.
- Treat task as the uncertainty cluster; repeated seeds/arms are paired within task. Thirty tasks, not 540 independent observations, govern generalization.
- Prespecify multiplicity correction or a primary contrast hierarchy; retain descriptive family summaries and exploratory cluster intervals.
- Simulate paired task/seed outcomes using justified pilot variance/correlation assumptions and assess precision and power across sensitivity scenarios. Do not declare 540 sufficient from episode count.
- Specify eligibility exclusions before outcomes; distinguish infrastructure failure/retry, invalid perturbation, timeout/censoring and genuine budget-exhaustion failures. Publish missing cells and denominator rules.
- Keep original rater columns, adjudication and label-conflict sensitivity for Precision@3/noise and kappa; never invent agreement.
- Document historical holdout and engineering outcome exposure, stopping criteria, pilot-to-confirmatory decisions and contemporaneous anchors.

## Remaining work by owner
Codex-resolvable: source-level prospective identity mechanism with separate schema and full hash-bound approval tests; evidence-schema completeness checks; provisional rating generator after actual library/context recovery; deeper tokenizer/filler review once exact assets exist. Human-only: selection/budget/model amendment, relevance judgments and adjudication, filler review, Gate/freeze/launch authorization, publication permissions. University-only: cap-50 replay, original task-data rehash, tokenizer/model manifest/blob export, real four-worker placement/coordinator smoke and independent archive backup readback.
Adapter and discovery data digests differ in the records. They are computed by separate code paths, so inequality alone is not proof of data corruption. Original task files are absent from the archive; establish identity by rehashing the source/game files against the proposal and documenting both digest definitions.

## Definition of done
Engineering: partial; existing resolver and resilience tests pass, but current-digest scientific support and live coordinator validation are pending. Pilot inputs: not ready. 36 episodes: not authorized. 540 confirmation: not ready. The next visit still requires verification and scientific input work before experiments.

Verification: 38 existing readiness/GPU-package tests and 2 new archive-parser tests passed; compilation and Git whitespace checks passed. No inference was launched. Existing broader-suite result remains 349 tests run, 4 skipped; it was not repeated for this archive-only change.
