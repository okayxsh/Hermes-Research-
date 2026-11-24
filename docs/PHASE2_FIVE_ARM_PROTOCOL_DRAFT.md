# RQ1 Phase 2B: five-arm protocol (DRAFT, PROPOSED)

**Status: DRAFT. Nothing here is approved.**
- No roster, label set, filler corpus, runtime digest, action cap, threshold or statistical plan has been approved.
- Every scientific decision is listed in `PHASE2_DECISION_REGISTER_DRAFT.md`; all 23 await author approval.
- No pilot or confirmatory model execution has taken place. No provider, Ollama or GPU was used.
- All tests use synthetic fixtures (scripted world, scripted provider). They are not scientific evidence.

| | |
|---|---|
| Implementation | `src/rq1/five_arm/` |
| Tests | `tests/test_five_arm_*.py`, fixtures `tests/five_arm_fixtures.py`, `tests/five_arm_e2e_fixtures.py` |
| Branch | `sci/rq1-phase2-five-arm-runner` (base `f71b4da`) |
| Related drafts | `PHASE2_STATISTICAL_ANALYSIS_PLAN_DRAFT.md`, `PHASE2_DECISION_REGISTER_DRAFT.md`, `PHASE2_RATER_INSTRUCTIONS_DRAFT.md`, `PHASE2_WORKFLOW_HIERARCHY.md` |

The historical six-arm implementation (`rq1.university_gpu`) is unchanged. Its records and manifests are refused by the five-arm code and are never reinterpreted.

## 1. Audits

### 1.1 Six-arm runner (code at `f71b4da`), from Phase 2

| # | University audit finding | Verified | Where |
|---|---|---|---|
| 1 | Six arms: D0, D18, Rel, Irr, Fill, Gate | **Confirmed** | `university_gpu/manifest.py` `CONDITIONS` |
| 2 | Rel/Irr/Fill/filtered Gate add a model-visible `neutral_filler` key absent from D0 and D18 | **Confirmed** | `interventions._with_filler` |
| 3 | Memory-availability flags differ by condition (Fill says "no memory" *and* shows filler) | **Confirmed** | `interventions.py:165`, `retrieval/injection.py` |
| 4 | Token matching covers only the serialised memory block | **Confirmed** | `interventions.token_count` |
| 5 | Rel/Irr select from full-library label pools | **Confirmed** | `manifest._select_skill_ids` |
| 6 | Rel/Irr can inject skills outside the natural top 3 | **Confirmed** | as above |
| 7 | Noise analysis uses retrieved, not injected, IDs | **Confirmed** | `analysis.analyze` |
| 8 | D0 metadata incomplete | **Confirmed** | `interventions.intervene` |
| 9 | Gate can be an exact D18 no-op | **Confirmed** | `interventions.py:170` |
| 10 | No confirmatory hypothesis tests or missingness policy | **Confirmed** | `analysis.analyze` |
| 11 | Runner expects a digest not installed at the university | **Confirmed** | `manifest.py`, `ollama.probe_endpoint` |

Additional Phase 2 findings: skill IDs and ranks were model-visible; the old manifest cannot represent the 79-task population (no pick-two tasks); provider `prompt_eval_count` was parsed but never logged.

### 1.2 Phase 2 five-arm worker (commit `3de474a`), audited in Phase 2B

| Component | State at `3de474a` | State now |
|---|---|---|
| Executable scientific worker | `FiveArmWorker`: gate + single-unit wrapper around `AmendedEvaluationExecutor` | `FiveArmEpisodeExecutor` (`executor.py`), composed from the frozen harness, episode runner, retrieval boundary and prompt builder |
| Provider token capture | **Broken**: `ProviderTokenRecorder` wrapped `driver._model_response`, which exists on `RealEpisodeSession`, not on `RealEpisodeDriver`; construction would fail and no count would be captured | `ProviderCallCapture` wraps every **session's** `_model_response`; every call is persisted (`provider_calls`) |
| Manifest builder | present | v2: per-arm eligibility, exact cap and allowance per unit, exact-cap oracle requirement, diagnostic manifest |
| Work-queue orchestration | **absent** | `queue.RunQueue`: ordered, single process, lock, journal, fsync'd records |
| Five-arm intervention dispatch | memory policy only | memory policy + executor modes (`arm`, `diagnostic_undisturbed`, `diagnostic_detoured`) |
| Environment reconstruction | via the frozen harness | unchanged frozen harness; checkpoint and post-detour digests verified every episode |
| Termination handling | via executor | success / budget / invalid-action limit / environment terminal; provider, environment and interruption failures separated |
| Measurement persistence | **absent** (returned a dict) | `records.jsonl` + `journal.jsonl` + per-attempt episode logs |
| Resume / retry | **absent** | stale attempts closed as `infrastructure_interruption` on resume; only infrastructure failures retried, at most 2 times |
| Post-run validation | **absent** | `RunQueue.validate()` + `analysis.resolve_units` |
| Undisturbed pilot mode | **absent** | `UndisturbedContinuationHarness` |
| University entry points | **absent** | `university_cli` (CPU tools and the gated `run-queue`) |

## 2. Hypotheses and claims boundary

**Single primary comparison: D18 versus D0.**
- **Primary outcome:** environment-reported task success within the approved total action horizon after the controlled disruption.
- **Primary estimand:** the task-weighted mean, over the screened task population, of the within-task difference in recovery success between natural retrieval from the frozen Accum-18 library and no memory, averaging inference seeds within each task. It is reported in percentage points, with an interval (SAP §1).
- **Interpretation:** this is the **total** effect of the specified natural-retrieval intervention (procedural content, memory presence and prompt length together). It does not isolate procedural relevance.

**Secondary, exploratory comparisons** (separate multiplicity family, never presented as confirmatory discoveries):
- D18 − Fill, Irr − Fill, Rel − Irr, Rel − Fill;
- retrieval-relevance associations;
- action efficiency (restricted-mean recovery actions), invalid actions, recovery latency;
- acquisition-template-overlap sensitivity.

**Claims boundary.** One fixed, frozen library is used. The experiment cannot establish:
- the effect of progressive library growth;
- an optimal library size;
- efficacy on unrelated benchmarks or other model families;
- autonomous improvement over time.

The broader research motivation (whether accumulated procedural memory helps recovery) is retained as motivation only.

## 3. Conditions (`rq1-five-arm-v1`)

| Arm | Model-visible memory | Active skills | Scheduled for |
|---|---|---|---|
| D0 | none (SD-01 proposes no memory section) | 0 | every roster task |
| D18 | natural top 3 from frozen Accum-18, rank order | 3 | every roster task |
| Rel | adjudicated-RELEVANT skill (slot 1) + neutral slots 2–3 | 1 | frozen Rel subset |
| Irr | adjudicated-IRRELEVANT skill (slot 1) + neutral slots 2–3 | 1 | frozen Irr subset |
| Fill | neutral content in all 3 slots | 0 | every task whose Fill cell is not blocked |

Gate is deferred. Natural retrieval runs for **every** arm, including D0, and its live top 3 must equal the frozen ranking (otherwise a protocol violation stops the queue).

## 4. Retrieval-freezing verification (SD-14)

The actual path was audited statically and is now asserted at run time:

| Question | Finding | Evidence |
|---|---|---|
| Retrieves once at the recovery checkpoint? | **Yes.** One crossing of `SingleRetrievalBoundary`, on the post-detour failure context, before the first recovery action | `recovery_executor.run_recovery_episode` step 6; the boundary raises on a second crossing |
| Retrieves repeatedly during recovery? | **No.** `run_model_loop` never calls retrieval | `episode_driver.run_model_loop`, `choose_action` |
| Changes the query as observations evolve? | **No.** The query is built once from the failure context | as above |
| Reloads skill memory in later calls? | **No.** The same `recovery_memory` dict is passed to every decision; the model is called directly through Ollama `/api/chat`, and Hermes is used only as the ALFWorld tool registry (no Hermes skill loading) | `harnesses.run_recovery`; `_RegistryWorker` config enables only the ALFWorld plugin |
| Run-time assertion | `ProviderCallCapture` refuses any recovery prompt whose memory section differs from the first one | `executor.py`; e2e test |

So every skill that can enter a prompt is in the task's frozen natural top 3, and the candidate-set annotation (Rel/Irr scope `natural_top3`) covers every injectable or retrievable skill.

**Departure from normal Hermes use.** The intervention is "retrieve once at the disruption checkpoint, then hold memory fixed". It is not Hermes's own skill loading. Manuscripts must say so.

**Tie fragility.** The CPU ranking export records the rank-3/rank-4 score margin. Tasks with a margin below 1e-6 are flagged; if a live ranking differs from the frozen one, the unit is a protocol violation, never silently relabelled.

## 5. Prompt construction and leakage

- The unchanged builder `render_action_prompt` is used. There is one user message and no system message.
- Shared block for D18/Rel/Irr/Fill: `{"memory_entries": [{"text": …} ×3]}`. It has no IDs, ranks, scores, labels, arm names or availability flags.
- **Runtime guards** in the executor:
  - the full leakage scan of the first recovery prompt runs **before** it is sent: arm names, label and meta terms, skill IDs, score-like decimals, oracle or expert actions in non-library memory content, oracle continuation as history, future observations;
  - memory is checked to stay static;
  - a local context-overflow check runs before each call.
- Frozen library texts are legitimately visible. Acquired skills can quote concrete actions, so an oracle action counts as leakage only in memory content that is not a frozen library text.
- **Remaining documented differences:**
  - D0 has no memory section;
  - Rel/Irr carry 1 skill versus 3 in D18;
  - neutral prose reads differently from procedural text;
  - token matching is approximate (§7).

## 6. Rel/Irr selection and annotation (SD-02/03/04/09)

- **Rel** = the highest-ranked RELEVANT skill within the natural top 3.
- **Irr** = the highest-ranked IRRELEVANT skill within the natural top 3.
- PARTIAL and UNCERTAIN are never injected, and there are no fallback labels.
- Rel and Irr share slot layout, slot count, active-skill count and metadata keys.
- Mechanism subsets (`arm_subsets`) are frozen from labels and rankings and hashed into the manifest before any outcome.
- Tasks without a qualifying Rel or Irr skill stay in D0/D18/Fill.
- **Annotation workload, natural top-3 candidate set** (planning estimates; recompute with `rating.workload_from_rankings` on the frozen rankings):

  | Population | Distinct pairs | Judgments, 2 raters, before adjudication |
  |---|---:|---:|
  | 79 unseen contexts | 237 | 474 |
  | 12 pilot contexts | 36 | 72 |

- **Rater protocol:** see `PHASE2_RATER_INSTRUCTIONS_DRAFT.md`.
  - Exports hide scores, ranks, arms, outcomes and oracle continuations (checked by `_export_leaks`).
  - Order is per-rater random; item IDs are anonymous and stable.
  - Declarations use pseudonymous codes and are paired with a coordinator receipt. A declaration is an attestation, not proof; the independence screen flags suspicious patterns for review.

## 7. Neutral filler (SD-06/12/23)

**Candidate corpus:** `configs/five_arm/neutral_filler_candidate_v1.json`.
- 139 hand-written sentences about weather, landscape, nature, music, astronomy and history.
- 30 of them are short (≤ 4 words), for fine length control.
- Canonical SHA-256: `cd9f3e07f848c0f289310a431bf1adb483b4a710cbf50a521d9c04ba60c2ca31`.
- **Status: PROPOSED_CANDIDATE_NOT_APPROVED** (empty review record).

**Lint** (`lint_filler_sentence`) screens for:
- action verbs;
- directive and meta terms;
- digits and markup;
- ALFWorld objects and receptacles;
- spatial and navigation words;
- task-family concepts (heating, cooling, cleaning, light, opening);
- household and object words;
- advice and evaluation language;
- leakage label terms.

All 139 sentences pass. **Passing the lint does not show the prose is behaviourally inert**, and it never approves the corpus.

**Matching** (synthetic-tokenizer demonstration in `test_five_arm_governance`):
- whole sentences are added deterministically, approaching the D18 anchor from below;
- procedural skills are never truncated;
- total memory-block tokens are matched within max(8, 2% of anchor);
- Rel and Irr have identical slot structure and metadata keys;
- matching is deterministic per task.

**What is matched:** only the total memory-block length. Slot-level lengths are **not** matched: in Rel/Irr the neutral slots absorb the difference between the active skill and the anchor.

**Remaining confounds:**
- *Position:* the active skill is always in slot 1, while D18 skills occupy ranks 1–3.
- *Semantic:* neutral prose is a different register and may itself act as a cue ("memory present but unhelpful").
- *Length:* Rel and Irr skills differ in length, so their neutral portions differ.

**Real-tokenizer audit:** a university CPU step. Cells outside tolerance are **BLOCKED** and excluded prospectively from the contrasts that need them. They never remove a task's D18/D0 pair (SD-23).

## 8. Measurements (`measurements.py`, schema v2)

The v1 fields are kept, plus:
- `executor_mode`;
- a per-decision `recovery_trace` (action, rejected attempts, environment validity, done, success, selection exhaustion);
- `provider_calls` (provider `prompt_eval_count` next to local raw and chat-template counts, latency, errors);
- `prompt_sequence_sha256`;
- `requested_provider_options`;
- `recovery_action_allowance`;
- `failure_detail`.

Pilot-diagnostic records use a separate schema (`rq1-five-arm-diagnostic-v1`). The five-arm analysis refuses them.

## 9. Execution path

`university_cli run-queue` → `runner.build_university_queue`:

1. **Artifact hashes and gates.** Artifact hashes are computed from the configured files. Then the gates run:
   - readiness for the manifest's stage;
   - observed versus approved runtime;
   - frozen manifest with a bound human approval;
   - manifest built from the reviewed commit.
2. **Real components** (only after the gates pass):
   - provider metadata preflight;
   - tokenizer and the hash-bound chat-template counter;
   - `RealEpisodeDriver`;
   - Sentence-BERT (arm mode only);
   - the approved filler (arm mode only);
   - the memory policy and the executor.
3. **`RunQueue.run()`.** Units run in global order. Each attempt gets a fresh directory and a durable journal entry, and the record is validated before it is appended.
   - A protocol violation halts the queue.
   - Resume closes stale attempts and never re-runs a unit with a final non-infrastructure outcome.
4. **`RunQueue.validate()`.**

**Real versus synthetic separation:**
- A synthetic configuration can never drive the real driver.
- A non-real driver can never produce non-synthetic records.
- A real queue refuses synthetic manifests.
- Fixture or synthetic approvals are refused under real semantics.

## 10. Pilot failure-manipulation diagnostic (pilot only; SD-15/16)

**Conditions.** Both are run without memory (the D0 presentation from SD-01) on the pilot roster (valid_seen), with matched seeds and provider settings:
- **UNDISTURBED:** frozen checkpoint, no detour.
- **DETOURED:** frozen checkpoint plus the frozen detour.

**Construction:**
- Both start from the same task reset and the same scripted prefix. The checkpoint digest is verified in both, and the undisturbed run also verifies that the detour rule *would* select the frozen detour.
- **Allowance (SD-15):** an identical post-checkpoint model-action allowance R = cap − prefix − 1. The undisturbed bridge limit is therefore cap − 1.
- **History rule:** every executed step is shown, oldest first. The detoured history contains the detour step and its observation; the undisturbed history does not.
- **Documented prompt difference:** the detour changes the history and the current observation. The prompts are **not** identical, by design.
- **Isolation:** separate manifest schema, separate queue run and separate record schema. Diagnostic episodes are never merged into five-arm data.

**Proposed thresholds** (`pilot.PROPOSED_DIAGNOSTIC_THRESHOLDS`, SD-16; not approved; each with a stated reason):

| Criterion | Proposed rule | Why |
|---|---|---|
| Meaningful disruption (any one) | completion drop ≥ 0.15 (interval lower bound); median extra actions when both succeed ≥ 2; extra rejected selections ≥ 0.5 per episode (interval lower bound) | 0.15 ≈ 1 task-level SE with 12 tasks; the oracle needs exactly +1 action, so ≥ 2 means re-orientation |
| Weak disruption | drop ≥ 0.05 but no criterion met | inert below 5 points |
| Insufficient disruption | drop < 0.05 and no criterion met | — |
| Indeterminate | any criterion's interval straddles its threshold | 12 tasks cannot settle it; design review, never automatic GO |
| Ceiling | detoured completion ≥ 0.90 | ≤ 10 points of headroom |
| Floor | detoured completion ≤ 0.10 | memory effects floor-limited |
| Task-ability floor | undisturbed completion ≤ 0.20 | disruption not measurable |
| Interface problem | > 1.0 rejected selections per episode, or > 10% selection exhaustion | action interface, not disruption |
| Infrastructure unstable | > 5% units missing after retries | complete-case validity |

**Recommendation logic:**
- MEANINGFUL with no findings → proceed to five-arm pilot review.
- WEAK or INSUFFICIENT → narrow the claim to "recovery after a controlled navigation perturbation", or strengthen the manipulation.
- Anything else → design review.

A one-action detour is never described as a validated long-horizon failure.

**Sensitivity:** with 12 tasks the completion-drop interval is about ±0.25 wide, so INDETERMINATE is the expected result unless effects are large. Seed variability (`pilot.seed_variability`) is measured in the same diagnostic. The proposal is 2–3 seeds in the diagnostic only.

## 11. Action budget and cap equivalence (SD-07/17)

**Budget-dependent behaviour audit:**

| Behaviour | Depends on the cap? |
|---|---|
| Prompt construction | **No.** No remaining-budget or step-limit text; `action_count`/`action_limit` returned by the bridge are logged, not rendered |
| Model calls | Each is stateless `/api/chat` with identical options |
| Environment | The bridge sets `done` only when `action_count ≥ action_limit` (observation "Action limit reached."). ALFWorld receives `max_nb_steps_per_episode = action_limit`; TextWorld also terminates there only. Before the cap, transitions are identical (verified by `cap_invariance_replay`) |
| Tool availability | Identical |
| Checkpoint eligibility | **Yes:** the oracle route must fit (prefix + detour + continuation ≤ cap); the manifest requires an oracle replay at exactly the cap |
| Worker stopping | `run_model_loop` stops at the allowance or on `done`; the allowance is cap-derived |
| Retry logic | Independent of the cap |
| Scheduling | Independent of the cap. No episode wall-clock limit (only the per-call provider timeout), so longer cap-100 episodes cannot hit a time limit a cap-50 episode would not |

**Conclusion.** Truncating a cap-100 trajectory to cap 50 (`measurements.truncate_to_cap`, using the per-decision trace) is valid under these assumptions:
1. the prompt and model inputs at decision *k* do not depend on the cap (verified statically);
2. environment transitions before the cap are identical (verified per task by the CPU cap-invariance replay);
3. per-call provider behaviour depends only on the prompt and options (stateless calls; same nondeterminism in both).

If (2) fails for any task, `budget_decision` refuses truncation and separate cap-specific diagnostic runs are required.

**Rule (SD-17).** It uses **only DETOURED no-memory diagnostic records** at cap 100, so it uses baseline feasibility and ceiling/floor behaviour, never a treatment difference.
- Choose 50 unless:
  - more than 20% of baseline successes need more than the cap-50 allowance; or
  - baseline success at 50 is ≤ 0.10 while higher at 100; or
  - any task is infeasible at 50.

**Exact-cap oracle replay:**
- Diagnostic roster: at cap 100 before the diagnostic.
- Five-arm pilot and confirmatory rosters: at the chosen cap.

Command: `university_cli oracle-replay --cap <cap>`. The manifest refuses a task without `oracle_replay_total_action_limit == cap`. These replays are CPU-only and are **not** run locally.

## 12. Runtime identity and provider measurements

**Model digests:**
- The historical pin `4eb23ef…` is preserved, and historical records keep it.
- The installed `6114515d…` is the proposed Phase 2 runtime (PROPOSED_UNVERIFIED).

**Static, CPU-only observation** (`university_cli observe-runtime`, metadata endpoints only):
- tag and manifest digest;
- blob digests from the modelfile;
- quantisation;
- template hash;
- default `top_p`/`top_k`;
- Ollama version;
- locally supplied tokenizer, Hermes, ALFWorld and Sentence-BERT identities.

**What needs real inference** (the separately authorised non-task probe, `provider_probe.py`):
- whether `prompt_eval_count` is returned;
- exact equality of the chat-template counter with provider counts (the counter spec is hash-bound as `chat_template_counter_sha256`);
- repeat-call determinism at temperature 0.

The *effective* sampling parameters cannot be read back from Ollama. The requested options are logged per record (`requested_provider_options`).

**Per-episode persistence and guards:**
- provider prompt count, local counts, prompt hashes, errors and done reasons are persisted per episode;
- a provider count ≥ `num_ctx`, or a discrepancy beyond SD-22's tolerance, turns the episode INVALID_MEASUREMENT;
- a missing provider count in real execution also turns it INVALID_MEASUREMENT.

## 13. Readiness levels (`readiness.py`)

| Level | Needs (in addition to the previous level) | Authorises |
|---|---|---|
| 1 `local_synthetic_development` | local synthetic suite passed | synthetic tests |
| 2 `university_cpu_integration` | reviewed, pushed code commit | CPU tools, runtime metadata, ranking and rating exports, oracle and cap replays |
| 3 `provider_compatibility_probe` | observed runtime record; authorisation bound to the probe-plan hash | non-task probe only |
| 4 `d0_failure_manipulation_pilot` | probe results; runtime verification; pilot roster; pilot exposure acceptance; diagnostic protocol; diagnostic-cap oracle replay; decisions blocking this stage | D0 diagnostic only (no filler, no labels needed) |
| 5 `five_arm_scientific_pilot` | diagnostic review; budget decision; filler; pilot labels; chosen-cap oracle replay; decisions | five-arm pilot |
| 6 `confirmatory` | pilot GO; final protocol; SAP; sample-size justification; confirmatory roster, exposure acceptance and labels; chosen-cap oracle replay; decisions | confirmatory run |

- Every approval must be bound (`artifact_sha256`) to the hash of the artifact actually loaded.
- Decision approvals are bound to the exact register entry plus the chosen option.
- Synthetic or fixture records never authorise real execution.
- The levels have no circular dependencies: CPU work needs no labels, filler or runtime approval, and the D0 diagnostic needs no filler or labels.

## 14. Exposure-risk acceptance (SD-18)

**Inputs** (`exposure_acceptance.build_statement`):
- the Phase 1 exposure ledger;
- the closed university investigation coverage:
  - 904 files examined;
  - no located outcomes for the candidate tasks;
  - no located rating calibration;
  - outcome-blind oracle preparation;
  - incomplete primary execution records for 15 September – 29 October 2026;
  - undocumented ad hoc activity possible.

**Output:** a hash-bound statement per roster. It:
- preserves UNKNOWN;
- describes the remaining tasks as `NO_LOCATED_MODEL_OUTCOME`, never `VERIFIED_UNEXPOSED`;
- refuses any task with known model-outcome exposure (`EXPOSED`/`POSSIBLY_EXPOSED`);
- binds the roster hash and the frozen protocol hash.

**Acceptance:** a future author approval bound to `statement_sha256`. None exists, so execution stays blocked. No further broad audit is implied.

**Proposed manuscript wording** (`exposure_acceptance.MANUSCRIPT_LIMITATION_TEXT`):

> Exposure limitation. Before the evaluation roster was frozen, we screened all local experiment records available to us (904 files) for model outcomes, human rating calibration and other prior use of the candidate evaluation tasks. We located no model outcomes and no rating calibration involving the N roster tasks, and the oracle preparation used for the controlled disruption was verified to be outcome-blind. However, primary execution records for part of the development period (15 September to 29 October 2026) are incomplete, and undocumented ad hoc use of these tasks cannot be ruled out. The absence of located outcomes therefore does not prove that the tasks were never previously exposed to the model or to the investigators; we treat their exposure status as unresolved rather than verified. Tasks with any known model-outcome exposure were excluded. The roster, analysis plan and protocol were frozen before any evaluation outcome was generated, and the residual risk was accepted explicitly by the authors against this documented coverage.

## 15. University integration and deployment

**Prerequisite:** the branch is pushed with explicit authorisation. Until then, nothing below can run.

```sh
git fetch origin sci/rq1-phase2-five-arm-runner
git worktree add ~/rq1-phase2 origin/sci/rq1-phase2-five-arm-runner
cd ~/rq1-phase2 && export PYTHONPATH=src:tests PYTHONDONTWRITEBYTECODE=1
python3.11 -B -m unittest test_five_arm_interventions test_five_arm_prompts test_five_arm_runtime_readiness \
    test_five_arm_rating test_five_arm_analysis test_five_arm_cli test_five_arm_governance test_five_arm_e2e
python3.11 -B -m rq1.five_arm decisions
python3.11 -B -m rq1.five_arm readiness --state state.json     # expect: CPU READY, all model levels BLOCKED
python3.11 -B -m rq1.five_arm filler-audit --corpus configs/five_arm/neutral_filler_candidate_v1.json
```

`state.json` for CPU integration: `{"local_tests_passed": true, "code_commit": "<pushed sha>"}`.

**CPU-only integration steps** (no model calls):
1. **Observe the runtime:**
   `python3.11 -B -m rq1.five_arm.university_cli observe-runtime --state state.json --endpoint http://127.0.0.1:11434 --local-json local-identities.json --output-root <out> --label rt1`
2. **Frozen rankings** for the pilot and roster tasks (valid_unseen preparation needs the task-preparation unseen-access authorisation):
   `university_cli cpu-rankings --state state.json --config cpu-unseen.json --cap 100 --work-dir <new dir> --output-root <out> --label rank1`
3. **Oracle replays:** `university_cli oracle-replay --cap 100 …` for the diagnostic roster, and `cap-invariance --caps 50 100 …`.
4. **Annotation:**
   - compute the candidate-set workload with `rating.workload_from_rankings`;
   - build rating exports (`python -m rq1.five_arm rating-export`);
   - distribute them per the rater instructions.
5. **Filler:** a real-tokenizer matching audit of the candidate filler on the frozen top-3 texts, for review.
6. **Exposure:** build the exposure-risk statements for the pilot and confirmatory rosters, for author review.

Model-using steps (probe, diagnostic, pilot, confirmatory) require the readiness levels in §13 and separate authorisation.
