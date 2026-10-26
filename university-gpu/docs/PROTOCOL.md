# Prospective six-arm protocol freeze

## Scope and source protocol

This package uses the amended RQ1 controlled-recovery executor in the verified scientific checkout, with the Gemma 4 12B digest and Sentence-BERT snapshot already recorded there. It does not modify the historical Oct 20 evaluation. The new intervention is a separate prospective follow-up. Its results must not be merged with the historical evaluation or relabeled as a replication of an untouched holdout.

The existing scientific execution path remains: replay the frozen ALFWorld checkpoint, apply the approved reversible navigation detour, verify checkpoint and post-detour digests, form the frozen failure query, perform the same one-time deterministic top-3 retrieval from the Accum-18 candidate pool, inject one recovery-memory block, and evaluate recovery with the existing action budget. No oracle route or condition name is shown to the model, and the evaluator does not write skills.

Frozen identity checked by the package:

| Setting | Required value |
| --- | --- |
| Ollama model tag | `gemma4:12b` |
| Ollama model digest | `4eb23ef187e2c5462566d6a1d3bbbc2f1346d0b4327cbb66d58fffbcc9b2b05c` |
| Quantization | `Q4_K_M` |
| Inference | temperature 0, `think=false`, `num_predict=2048`, `num_ctx=32768` |
| Replicate seeds | 11, 29, 47, paired across conditions |
| Retrieval | `sentence-transformers/all-mpnet-base-v2`, revision `e8c3b32edf5434bc2275fc9bab85f82640a19130`, top 3 |
| Retrieval snapshot SHA-256 | `bbfecb04a834241144af31e9200a3dd0df83ff0fbb2453d56a4bb383a6bfbe11` |
| Task eligibility | frozen valid-seen pilot tasks and frozen valid-unseen confirmation tasks with validated controlled-failure oracles |

The operator check also verifies the repository's historical model and protocol files against these values. Older instructions that name Llama, Hermes-native memory, or a different retrieval path are not accepted as execution defaults.

## Arms and estimands

All nonempty-memory arms use the same frozen Accum-18 skill library and the same retrieval event. The intervention changes only the recovery-memory contents, subject to the exact token accounting below.

| Arm | Model-facing memory | Comparison purpose |
| --- | --- | --- |
| `D0` | The executor's unchanged no-retrieval block | No-memory reference |
| `D18` | The executor's unchanged top-3 Accum-18 retrieval block | Current retrieval reference |
| `Rel` | One adjudicated relevant skill, padded with approved neutral filler to the D18 token count | Applicability signal |
| `Irr` | One adjudicated irrelevant skill, padded to the D18 token count | Irrelevant-memory control |
| `Fill` | No skill text, only approved neutral filler, padded to the D18 token count | Memory/token-load control |
| `Gate` | D18's top-3 items whose cosine similarity is at least the outcome-blind valid-seen threshold; pad remaining tokens with approved filler | Minimal adaptive selective-memory extension |

`D18` is deliberately byte-for-byte unchanged at the recovery-memory mapping boundary. Its memory-block token count uses the exact `json.dumps(..., ensure_ascii=False, sort_keys=True)` serialization emitted by the frozen prompt builder. Every `Rel`, `Irr`, `Fill`, and `Gate` block must match that serialized block count under the hash-pinned local Gemma tokenizer. This is a memory-block dosage match; the package does not claim identical whole-prompt token counts, which can also depend on context and tokenizer boundaries. A test tokenizer is permitted in unit tests only. Tokenizer file, revision, hash, serialization, and filler text must be frozen before the manifest; no production tokenizer or neutral-filler artifact is supplied by this code change.

`Gate` keeps only already-retrieved top-3 items above threshold `tau`; it does not trigger a second retrieval, rerank candidates, or write persistent memory. Its threshold is selected by maximizing balanced accuracy over valid-seen task/skill cosine scores against adjudicated relevance labels, with ties resolved to the higher threshold. The calibration file records the complete candidate scores, label-file hashes, threshold, procedure, and human approval. Recovery outcomes and valid-unseen data are prohibited from threshold selection.

The random-skill arm is excluded because the blinded irrelevant-skill arm directly tests the relevance contrast while `Fill` separately controls token load. An undisturbed continuation is excluded because every arm here shares the same controlled disturbance; estimating disturbance cost is a separate question and would need its own justified control and workload.

The outcome is the existing binary successful recovery within the frozen recovery action budget, conditional on a valid controlled-failure episode. The analysis reports episode counts, task-family descriptive rates, paired task-seed differences, and task-cluster bootstrap intervals. It reports retrieval noise (`1 - Precision@3`) under adjudicated labels and both original-rater label columns as a sensitivity analysis. Task-cluster intervals and family summaries are exploratory; the frozen Oct 20 analysis remains the primary historical result.

## Planned matrix and limits

| Stage | Tasks | Seeds per task | Arms | Planned episodes |
| --- | ---: | ---: | ---: | ---: |
| Valid-seen pilot | 6 (one per task family) | 1 (seed 11) | 6 | 36 |
| Valid-unseen confirmatory follow-up | 30 (five per family) | 3 (11, 29, 47) | 6 | 540 |
| Total planned floor | 36 stage-specific task slots | — | — | 576 |

The floor is not a sample-size sufficiency claim. Before confirmatory approval, simulate the planned paired contrast and task-cluster estimator using prespecified minimum effect, task heterogeneity, within-task seed correlation, eligibility loss, and multiplicity assumptions. Show operating characteristics over plausible ranges and a Monte Carlo error estimate. The current code intentionally does not invent those inputs. Lack of that simulation blocks any claim that 540 confirmatory episodes are adequate.

The 30 confirmatory task identities have historical RQ1 result records. The task split is valid-unseen relative to acquisition, but the follow-up is not independent of those prior evaluation outcomes. The manifest and approval preserve this history. Authors must decide whether to present it as a follow-up on previously observed tasks or select a genuinely untouched task set; operator code cannot erase the exposure.

## Historical anchor compatibility

Historical NoLib and Accum-18 episodes remain separate references. They are not substituted for `D0` or `D18` in a primary six-arm analysis. The available compatibility record does not establish equivalence across the intervention protocol, exact recovery-memory block/token handling, relevance controls, task/checkpoint set, seed/order, run environment, eligibility accounting, and planned contrasts. Any sensitivity reuse requires an explicit, evidence-backed compatibility table, preserved historical results, and author approval. Counts such as 450 or 360 episodes are hypothetical accounting only, not an accepted reduced design.

## Eligibility, labels, and leakage controls

Before a proposal can be frozen:

- Every task must be in its named split, include a validated controlled-failure oracle, and have a frozen checkpoint, post-detour digest, prefix, detour, reference route, and recovery budget.
- The pilot has exactly six tasks, one from each frozen family; confirmation has exactly thirty tasks, five from each family.
- Each task's 18 Accum-18 skills have two blinded rater labels and an adjudicated label. The parser rejects missing coverage, duplicate task-skill rows, nonlibrary skill IDs, and nonbinary labels. Original rater columns are retained in the source CSV and used only for the labeled sensitivity analysis.
- The Gate threshold proposal uses only valid-seen similarity/relevance data, the predeclared calibration rule, no recovery outcomes, and no valid-unseen observations.
- Confirmatory freeze and launch approval explicitly acknowledge the previously observed task outcomes.

The code cannot verify that raters were truly blinded or independent, that adjudication followed the documented rubric, that filler is semantically neutral, or that the task-oracle review was conducted correctly. Those remain documented human verification gates.
