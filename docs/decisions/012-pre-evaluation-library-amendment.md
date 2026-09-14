# 012 — Pre-evaluation library amendment and amended evaluation protocol

- Status: **proposed, UNAPPROVED.** It takes effect only when the researcher explicitly approves
  the evaluation amendment, environment, and protocol freeze requests and the task freeze.
- Date: 2026-09-14
- Timing: after the complete 240-episode acquisition closed at its pre-approved hard cap and
  **before any evaluation**. No evaluation episode has run and no evaluation outcome exists.
- Amends: `configs/libraries.yaml` (Clean-24 / Accum-60 / Accum-96) and the four conditions of
  `configs/tasks/evaluation.yaml`, for the RQ1 evaluation only.

## Why

1. Acquisition ended at the pre-approved hard cap of 240 TRAIN episodes (Decisions 007 and 011).
   The raw pool has 50 skills: pick_and_place 11, pick_two_and_place 8, look_at_object 4,
   clean_and_place 9, heat_and_place 15, cool_and_place 3.
2. The original sizes cannot be built. Clean-24 needs 4 per family (cool_and_place has 3),
   Accum-60 needs 10 per family, and Accum-96 needs 16 per family.
3. The amendment is driven by acquisition feasibility only. No skill is fabricated, no
   acquisition episode is discarded, no quality rule is loosened, and no semantic deduplication
   is introduced.

## Amended conditions (Rule A)

4. Conditions and sizes are **0 / 6 / 12 / 18**:
   - `NoLib`: 0 skills.
   - `Core-6`: one human-validated core skill per family.
   - `Accum-12`: the same core plus one chronological extra per family.
   - `Accum-18`: the same core plus two chronological extras per family.

   The original labels (Clean-24, Accum-60, Accum-96) are not reused for these sizes.
5. **Rule A**, which keeps the approved library logic:
   - The core of each family is the chronologically earliest skill marked PASS under the human
     rubric `docs/SKILL_QUALITY_RUBRIC.md` (skill-quality-rubric-v1).
   - Extras are the chronologically earliest acquired skills not in the core. They are not
     quality-filtered and not deduplicated, because accumulated extras were never meant to be
     cleaned.
   - Libraries are balanced per family and strictly nested (Core-6 ⊂ Accum-12 ⊂ Accum-18), with
     the same core in every non-empty library.
6. The raw pool supports 3 skills per family in every family. The only remaining feasibility
   condition is at least one PASS per family. If any family has none, evaluation does not start.
7. Only the core needs human review, so the reviewer judges each family in chronological order
   and stops at its first PASS (`core-validation-fast.csv`).
8. **Structure retained:** four conditions, 30 valid_unseen tasks (5 per family), seeds 11, 29,
   and 47, and 360 evaluation episodes.

## Protocol made explicit before any result (`docs/EVALUATION_AMENDED_PROTOCOL.md`)

9. **Tasks.** Take the five longest ALFWorld hand-coded expert routes per family, with ties
   broken by task ID. A task whose controlled failure cannot be validated before the freeze is
   replaced by the next-longest task of the same family.
10. **Controlled failure.** This is a controlled reversible action perturbation, never object
    relocation.
    - **Checkpoint.** The first midpoint-near checkpoint whose next reference action is
      navigation.
    - **Detour.** The lexicographically first other admissible `go to` action.
    - **Validity.** The detour must be a valid, non-terminal transition that changes the
      observation.
    - **Oracle.** In the real bridge, the sequence "prefix → detour → remaining reference route"
      must complete successfully within the 50-action episode budget.
    - **Runtime.** Every episode must reproduce the frozen checkpoint and post-detour digests.
    - **Visibility.** Oracle and reference actions are never shown to the model.
11. **Budget.** Each episode has 50 environment actions in total, the standard ALFWorld limit
    already used for acquisition. The recovery budget is 50 − prefix length − 1, fixed per task
    and identical across conditions and seeds.
12. **Retrieval.**
    - SBERT `all-mpnet-base-v2` at revision `e8c3b32…` (snapshot `bbfecb04…`, 768 dimensions).
    - Top-3 by cosine, exactly once, immediately after the controlled failure.
    - Query `query-v2`: task goal, current observation, inventory field, canonical failure
      message; no action history.
    - Scores are logged but never shown to the model.
    - One structured recovery-memory block (rank, skill ID, text) enters the post-failure
      prompts; NoLib receives the same block marked `no_retrieved_skills_available`.
13. **Model and controller.**
    - Identical to acquisition: `gemma4:12b` (Q4_K_M, digest `4eb23ef1…`), `num_predict` 2048,
      `num_ctx` 32768, temperature 0, `think: false`, `action-index-history-v3`, three
      selection attempts, and the same output-failure classification.
    - **Change:** the inference seed of each episode is its evaluation replicate seed (11, 29, or
      47), identical across conditions. Acquisition used 42.
    - **Limitation:** with temperature 0 decoding, replicates may be identical or nearly so; the
      seeds are reported as replicates, not as independent samples.
14. **Metrics.**
    - **Primary:** conditional recovery rate. Recovery successes are divided by eligible units,
      which reproduced the frozen checkpoint and detour and finished without an infrastructure
      failure.
    - **Secondary:** task completion, recovery latency (actions and seconds from recovery-memory
      injection to success), invalid action selections, retries, and infrastructure failure rate.
    - **Retrieval quality:** Precision@3 and Retrieval Noise = 1 − Precision@3, from two
      independent binary raters (`docs/RETRIEVAL_RELEVANCE_RUBRIC.md`), with Cohen's kappa on the
      original labels and adjudication afterwards.
    - **Uncertainty and association:** percentile bootstrap CIs over task-seed cells, plus a
      descriptive, non-causal Spearman correlation between noise and recovery.
15. **Schedule.** The 90 task-seed cells are ordered by family, task, and seed. Conditions are
    rotated within each cell. Six shards of 15 cells keep all four conditions of a cell together,
    so each shard is balanced across conditions and can run on its own worker. Nothing is
    reordered based on outcomes.

## Process

16. **valid_unseen access before the freeze.** Task selection and oracle validation need
    valid_unseen metadata and hand-coded expert routes.
    - They run only under `RQ1_VALID_UNSEEN_ACCESS=evaluation-task-preparation`, which logs every
      task access.
    - They start only after the selection and perturbation policy are committed.
    - No model is called and no agent outcome is produced.
    - The scientific evaluation requires `RQ1_VALID_UNSEEN_ACCESS=scientific-evaluation`.
17. **Approval path.** The legacy L0–L100 snapshot/activation path (Phase-7 pilot report, Hermes
    snapshot profiles) is replaced for RQ1 by these approval-gated freezes:
    - the amendment
    - the evaluation task freeze
    - the evaluation environment freeze
    - the evaluation protocol freeze
    - the deterministic library freeze built from the completed core review

    Evidence comes from a passed non-scientific evaluation check on valid_seen at the frozen
    commit. Evaluation never writes skills.
