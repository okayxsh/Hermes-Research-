# RQ1 Phase 2: five-arm protocol (DRAFT, PROPOSED)

**Status: DRAFT. Nothing here is approved.**
- No roster, label set, filler corpus, runtime digest, action cap or statistical plan has been approved.
- No pilot or confirmatory model execution has taken place.
- All tests use synthetic fixtures. They are not scientific evidence.

| | |
|---|---|
| Implementation | `src/rq1/five_arm/` |
| Tests | `tests/test_five_arm_*.py` |
| Branch | `sci/rq1-phase2-five-arm-runner`, based on `f71b4da` |

The historical six-arm implementation (`rq1.university_gpu`) is unchanged. Its records are refused by the five-arm analysis (`protocol.assert_five_arm_records`) and are never reinterpreted as five-arm records.

## 1. Audit of the existing six-arm runner (code at `f71b4da`)

| # | University audit finding | Verified in code | Where |
|---|---|---|---|
| 1 | Six arms: D0, D18, Rel, Irr, Fill, Gate | **Confirmed** | `university_gpu/manifest.py` `CONDITIONS` |
| 2 | Rel/Irr/Fill/filtered Gate add a model-visible `neutral_filler` key absent from D0 and D18 | **Confirmed** | `interventions._with_filler` |
| 3 | Memory-availability flags differ by condition | **Confirmed.** `no_retrieved_skills_available` is true in D0 and Fill, false in D18/Rel/Irr. Fill tells the model there is no memory *and* shows filler | `interventions.py:165`, `retrieval/injection.py` |
| 4 | Token matching covers only the serialised memory block | **Confirmed.** Only the inner JSON is counted; the rest of the prompt and the chat template are not | `interventions.token_count` |
| 5 | Rel/Irr select from full-library label pools | **Confirmed** (seeded random choice among all 18 skills) | `manifest._select_skill_ids` |
| 6 | Rel/Irr can inject skills outside the natural top 3 | **Confirmed** | as above |
| 7 | Noise analysis uses retrieved, not injected, IDs | **Confirmed.** Rel/Irr/Fill noise is computed from the natural top 3 the model never saw | `analysis.analyze` |
| 8 | D0 metadata incomplete | **Confirmed.** No budget, memory hash or filler fields | `interventions.intervene` (D0 branch) |
| 9 | Gate can be an exact D18 no-op | **Confirmed.** If no item falls below the threshold, the D18 block is returned unchanged | `interventions.py:170` |
| 10 | No confirmatory hypothesis tests or missingness policy | **Confirmed.** Exploratory bootstrap only; any missing unit aborts the analysis | `analysis.analyze` |
| 11 | Runner expects a digest not installed at the university | **Confirmed.** `MODEL_DIGEST = 4eb23ef…`; `probe_endpoint` fails on any other digest | `manifest.py`, `ollama.probe_endpoint` |

**Additional findings:**
- **IDs and ranks are model-visible.** Skill IDs and retrieval ranks appear in the memory JSON in every memory-bearing arm. In Rel/Irr the rank is always 1.
- **The old manifest cannot represent the 79-task population.** It hard-requires exactly 6 or 30 tasks, with all six families including pick-two. The 79-task population has 0 pick-two tasks.
- **Provider token counts are lost.** The provider's `prompt_eval_count` is parsed but never logged, so provider-side token accounting is unavailable.
- **`AGENTS.md` is stale.** It still describes a superseded four-condition design (NoLib/Clean-24/Accum-60/Accum-96). It was left unchanged; reviewers should reconcile it.

## 2. Conditions (protocol `rq1-five-arm-v1`)

| Arm | What the model sees | Active procedural skills |
|---|---|---|
| D0 | No memory (see §3; presentation is an open decision) | 0 |
| D18 | The natural Sentence-BERT top 3 from the frozen Accum-18 library, in rank order (pinned `all-mpnet-base-v2`, cosine, ties broken by `(-score, skill_id)`) | 3 |
| Rel | One adjudicated-RELEVANT skill in slot 1; reviewed neutral content in slots 2–3 | 1 |
| Irr | One adjudicated-IRRELEVANT skill in slot 1; reviewed neutral content in slots 2–3 | 1 |
| Fill | Reviewed neutral content in all three slots | 0 |

Gate is **deferred** (finding 9).

**Natural retrieval runs for every arm,** including D0, and is checked against the frozen ranking. Only the memory policy decides what the model sees. Retrieval scores and human labels never change the ranking.

**Rel and Irr are controlled single-skill substitutions.** Rel vs D18 also changes the number of active skills (1 vs 3), so it does **not** isolate relevance. The relevance contrast is Rel vs Irr, with matched count, slot, format and token anchor.

## 3. Prompt construction

The prompt is built by the frozen builder (`render_action_prompt`, unchanged) and sent as one user message with no system message. The memory-bearing block, placed in the existing `RECOVERY MEMORY:` section and serialised `json.dumps(..., ensure_ascii=False, sort_keys=True)`, is:

```json
{"memory_entries": [{"text": "…"}, {"text": "…"}, {"text": "…"}]}
```

**The block is identical in form across D18, Rel, Irr and Fill:**
- the same key and the same three slots;
- no IDs, ranks, scores, labels, arm names or availability flags;
- no condition-specific instructions.

**D0 (decision `d0_representation`):**
- **Proposed:** `absent_memory_section`. The section is omitted entirely, which works through the existing driver because `{"recovery_memory": None}` renders no section.
- **Alternative:** the historical NoLib marker.

Either way, D0 differs structurally from the memory-bearing arms. This difference is documented, and D0 is not claimed to be length-matched.

**Token anchor:** the task's D18 memory-block token count.
- Neutral content is assembled deterministically from **whole reviewed sentences**, approaching the anchor from below.
- Procedural text is never truncated. If a single active skill exceeds the anchor, the cell is flagged instead.
- Cells outside the approved tolerance (proposed: max(8 tokens, 2% of anchor)) are **blocked** in the manifest, and a manifest with blocked cells cannot be frozen.

**Accounting levels** (`prompting.account_prompt`):

| Level | Source | Notes |
|---|---|---|
| Raw prompt tokens | local count | |
| Memory-block tokens | local count | |
| Chat-template tokens | local count | only with a renderer verified against the provider |
| Provider `prompt_eval_count` | real requests | recorded by `ProviderTokenRecorder` |

- Exact provider equality is never claimed.
- A local prompt plus maximum output that exceeds `num_ctx` raises an error before any call.
- A provider count at or above `num_ctx` turns the episode into `INVALID_MEASUREMENT`.

**Automated paired checks** (`compare_arm_prompts`, `scan_leakage`) verify:
- every non-memory section is byte-identical across arms;
- the memory schema is exact, with no unexpected filler keys or condition flags;
- no arm-name, label or meta-term leakage;
- no skill IDs and no score-like numbers in the memory;
- no oracle or expert action inside the memory;
- no oracle continuation as executed history;
- no future observations.

Natural skill texts are excluded from the term scan (they are legitimately visible).

## 4. Rel/Irr selection (`selection.py`)

**Rule:**
- **Rel** = the highest-ranked skill in the Rel scope whose frozen adjudicated label is RELEVANT.
- **Irr** = the highest-ranked skill in the Irr scope labelled IRRELEVANT. That makes it a plausible retrieval false positive.
- PARTIAL and UNCERTAIN labels are never injected. Low similarity is never treated as irrelevance.
- Every skill in a scope must carry a frozen label, or selection fails. There are no fallback labels.

**Scopes (decisions `rel_candidate_scope`, `irr_candidate_scope`):**
- **Proposed:** `natural_top3` for both.
- **Alternative:** `full_ranking` for more coverage. It requires full-library annotation.

**Missing categories (pre-specified):**
- A task with no qualifying skill gets no Rel (or Irr) unit and is excluded only from that mechanism contrast.
- It remains in D0/D18/Fill and in D18-vs-D0. No substitute intervention is defined.
- The mechanism subsets are fixed by `mechanism_subsets()` from frozen labels and rankings. They are hash-bound before any model outcome exists.

## 5. Neutral filler

- Filler comes from a corpus file. Its approval is a human review bound to the corpus SHA-256 (`FillerCorpus.approved`).
- `lint_filler_sentence` screens for:
  - action verbs;
  - ALFWorld objects and receptacles;
  - directive or meta terms (task, goal, memory, recovery, relevant, you, must, …);
  - digits and markup.
- Passing the lint **never** approves filler, and token length never approves it.
- No real corpus exists yet; the test corpus is a fixture.
- **Status: UNAPPROVED. A blocker.**

## 6. Measurements (`measurements.py`)

**One schema for all arms.** It records:
- task, template, scene, family, exposure policy and oracle hashes;
- checkpoint and post-detour digests;
- condition, seed and runtime identity;
- **natural** top-3 IDs and scores, and the **injected** IDs and count;
- analysis-only injected labels;
- filler identity, memory hash and first-prompt hash;
- the four token levels;
- cap, prefix, detour and recovery actions;
- success, termination reason and invalid actions;
- wall-clock and inference latency;
- outcome status.

D0 carries every field, with explicit empty or not-applicable values.

**Relevance metrics:**
- **Retrieved Precision@3** uses the natural top 3 (strict primary; lenient sensitivity).
- **Injected relevance** uses the injected IDs only. It is `not_applicable_zero_injected` for D0 and Fill.

**Statuses** separate:
- outcomes: success, budget exhaustion, invalid-action limit, environment terminal failure;
- infrastructure: provider, environment, interruption;
- missing cell, unapproved intervention and invalid measurement.

A non-valid record has `success = None` and is never coerced into a failure.

## 7. Runtime identity (`runtime.py`)

**Digests:**
- The historical pin `4eb23ef…` is preserved, and historical records keep it.
- The installed `6114515d…` is **PROPOSED_UNVERIFIED**.

**What must be verified on DESKTOP-AIPC-02** (all are null in the proposed config until observed):
- manifest digest, every referenced blob digest, and quantisation (`ollama show`);
- tokenizer hash and revision, linked to the GGUF tokenizer (vocabulary, special tokens, template);
- chat-template hash;
- Ollama, Hermes and ALFWorld versions;
- Sentence-BERT revision and snapshot;
- context length, temperature, top_p, top_k, `num_predict` and `think` (sent and reported);
- inference seeds;
- that a real request's `prompt_eval_count` equals the local chat-template count, within tolerance.

**Seeds** are inference seeds only. ALFWorld state is fixed by the checkpoint, so seeds are not independent environment randomisations.

**Fail-closed checks:** `verify_runtime` blocks on:
- missing or unbound approval;
- any unset required field;
- digest, tokenizer, template or Sentence-BERT mismatch;
- `think` not false, or temperature not 0.

## 8. Readiness (`readiness.py`)

**Levels:**
- local development;
- university integration;
- CPU oracle;
- scientific pilot;
- confirmatory.

Only the last two permit model execution. Both require valid human approval records, each `{approved, approved_by, approval_reference, artifact_sha256}`.

**Blockers enforced:**
- incomplete exposure provenance;
- unapproved pilot or confirmatory roster;
- unverified model/runtime pin;
- unverified tokenizer/provider linkage;
- unapproved filler;
- missing pilot or confirmatory relevance judgments;
- unapproved checkpoint and action-budget policy;
- no exact-cap oracle replay;
- unvalidated failure manipulation;
- unapproved statistical plan;
- insufficient sample justification;
- no pilot GO;
- unapproved scientific decisions.

**The worker runs its gate first.** `FiveArmWorker` checks readiness, then the runtime, then that the manifest is frozen and bound to the approved runtime. Only then does it import or contact any provider, environment or model component.

## 9. Human relevance rating (`rating.py`)

**What raters see:** an anonymous item ID (salted hash), the task goal, the permitted recovery context, the skill text, and the rubric, with a rating and an optional rationale to fill in.

**What raters never see:** rank, score, arm, outcome, other raters' answers, or the expert continuation. Forbidden strings are checked, and item order is shuffled per rater.

**Rubric** (`rq1-five-arm-relevance-rubric-v1-draft`): RELEVANT / PARTIAL / IRRELEVANT / UNCERTAIN. Topical similarity is explicitly not relevance.

**Binary mapping for injection:**
- RELEVANT → relevant; IRRELEVANT → irrelevant;
- PARTIAL → never injected;
- UNCERTAIN → must be adjudicated.

**Workflow:**
1. Each rater submits separately, with a declaration that the labels are human and that no other rater's answers were seen. AI labels are refused.
2. Agreement is measured: percent agreement, 4-category and binary κ, and a confusion table.
3. Every conflict, and every agreed UNCERTAIN, is adjudicated by a third person.
4. The labels are frozen with provenance hashes as `frozen_pending_human_approval`.

**Workload, before adjudication:**

| Design | Pairs (79 tasks) | Judgments (2 raters) | Pilot (12 tasks) |
|---|---:|---:|---:|
| A: full library (18 skills) | 1,422 | 2,844 | 216 pairs / 432 judgments |
| B: candidate set (natural top 3), valid only if both scopes are `natural_top3` | 237 | 474 | 36 pairs / 72 judgments |

Design B covers every skill that can be retrieved (for Precision@3) or injected under the frozen `natural_top3` rules (`required_annotation_candidates`). Under any `full_ranking` scope, Design A is required. B is not chosen merely because it is cheaper.

## 10. Pilot (valid_seen; not executed)

**Objectives:**
- provider compatibility;
- memory injection and prompt accounting (comparing provider and local counts);
- retrieval and intervention behaviour;
- ceiling and floor effects;
- budget suitability;
- infrastructure reliability;
- compute estimates.

**Failure-manipulation validation** (`pilot.disruption_report`):
- For the same task, checkpoint and seed, without memory, the model continues either **undisturbed** (budget = cap − prefix) or **after the detour** (budget = cap − prefix − 1).
- Evidence of disruption is a lower completion rate, more actions to success, or more invalid or repeated actions.
- The thresholds (`min_completion_drop`, `min_extra_actions`, `ceiling`, `floor`) are decisions to be approved before the pilot.
- A trivially solved detour is reported as a methodological issue.
- The pilot needs a no-detour runner mode. This is a **remaining implementation item** for the university integration step; it is not yet wired into `FiveArmWorker`.

**Action budget** (`pilot.budget_decision`):
- Candidates are 50 (historical; every observed unseen witness ≤ 25 actions) and 100.
- The pilot runs at 100. Because the prompt never shows the remaining budget, cap-50 outcomes follow by truncation.
- **Choose 50** unless more than `late_success_threshold` of the pooled successes need more than the cap-50 recovery budget, or pooled cap-50 success is at the floor while cap-100 success is higher.
- The rule is pooled over arms and never sees an arm difference.
- **Before any confirmatory run:**
  - the chosen cap is frozen;
  - every roster task's oracle is replayed at exactly that cap (Phase 1 `oracle-validate --selection-cap <cap>`; readiness blocker `exact_cap_oracle_replay`).

The confirmatory population is never changed after its outcomes are seen.

## 11. Scientific decisions still open (`python -m rq1.five_arm decisions`)

- `d0_representation`
- `rel_candidate_scope`, `irr_candidate_scope`
- `partial_label_policy`
- `active_slot_position`
- `token_tolerance`
- `action_cap`
- `inference_seeds`
- `annotation_design`
- `primary_inference`
- `missing_data_policy`
- `filler_corpus`
- `runtime_model_digest`

Each has an engineering *proposal*. None is a default used for execution.

## 12. University deployment (after the branch is pushed with authorisation)

```sh
git fetch origin sci/rq1-phase2-five-arm-runner && git worktree add ~/rq1-phase2 origin/sci/rq1-phase2-five-arm-runner
cd ~/rq1-phase2 && export PYTHONPATH=src:tests PYTHONDONTWRITEBYTECODE=1
python3.11 -B -m unittest test_five_arm_interventions test_five_arm_prompts test_five_arm_runtime_readiness test_five_arm_rating test_five_arm_analysis test_five_arm_cli
python3.11 -B -m rq1.five_arm decisions
python3.11 -B -m rq1.five_arm workload --tasks 79 --skills 18 --candidates-per-task 3 --pilot-tasks 12
python3.11 -B -m rq1.five_arm readiness --state <state.json>        # expected: pilot/confirmatory BLOCKED
```

**Integration steps that remain** (no model calls until readiness passes):
1. Record the observed runtime, including `ollama show` and the tokenizer/template linkage.
2. Compute the frozen natural rankings for the pilot and roster tasks on CPU.
3. Export the rating items.
4. Wire the undisturbed-continuation pilot mode.
5. Build the manifest audits with the real tokenizer.
