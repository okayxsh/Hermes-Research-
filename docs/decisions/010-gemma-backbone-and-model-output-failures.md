# 010 — Gemma 4 12B backbone and model-output failure classification (pre-run amendment)

- Status: approved methodology decision by the study owner
- Date: 2026-09-13
- Timing: frozen before the scientific acquisition run; no scientific acquisition data existed when it was made
- Supersedes: Decision 001 (model selection); the Llama 3.1 8B fallback is withdrawn
- Amends: Decision 007 (the experimental agent is `gemma4:12b`), the Decision 008/009 controller (`action-index-history-v2` becomes `action-index-history-v3`), and the acquisition policy (`acquisition-execution-v1` becomes `acquisition-execution-v2`)
- Applies identically to: acquisition and every recovery condition (NoLib, Clean-24, Accum-60, Accum-96)

## Model selection

1. `hermes3:8b` was technically compatible with the execution stack but achieved
   0/18 successes in the final non-scientific capability checks.
2. `gemma4:12b` was then tested on the same six TRAIN tasks in the same order:
   2/6 before the Decision 009 interface corrections (gemma-a, commit 72b8f77) and
   3/6 after them (gemma-b, commit 578016d). This meets the pre-stated prelaunch
   capability criterion of at least 3/6.
3. The official gemma-b score is 3/6 under the chronological retry rule that
   already existed in the repository before the gate ran; the scoring rule was not
   changed after the result. Recorded separately: the first heat-and-place attempt
   failed when the model produced an effectively endless response (more than 7,100
   generated tokens) that reached the 180-second provider timeout; the harness as
   it then existed classified that event as an infrastructure failure; the
   pre-existing retry then ran a fresh attempt, which succeeded.
4. Model selection happened before scientific acquisition. No scientific outcome
   data influenced it, and all prelaunch evidence remains non-scientific.
5. Frozen identity: Ollama tag `gemma4:12b`, digest
   `4eb23ef187e2c5462566d6a1d3bbbc2f1346d0b4327cbb66d58fffbcc9b2b05c`, quantization
   Q4_K_M (11.9B parameters), temperature 0, inference seed 42, `think: false`, and
   context window `num_ctx` 32768. The installed digest, quantization, and provider
   settings are recorded and enforced by the acquisition environment freeze.

## Model-output failures versus infrastructure failures

6. The previous behaviour was scientifically wrong: an unbounded model response
   ran until the provider timeout, was labelled an infrastructure failure, and gave
   the model a fresh episode.
7. **Output cap.** Every model call sends Ollama `options.num_predict = 2048`
   (maximum output tokens). The longest normal Gemma response observed in prelaunch
   checks was about 900 tokens. The cap is a controller safety bound, not task
   knowledge, and is identical for every task and condition.
8. **Model/controller attempt failures** consume one of the three action-selection
   attempts for that single ALFWorld decision: no `ACTION_INDEX`; a malformed
   `ACTION_INDEX`; more than one mention; an out-of-range index; or any response
   that stops at the output cap (`done_reason = "length"`), which is incomplete by
   definition and is never parsed. These failures never restart the episode, reset
   ALFWorld, create a fresh episode-level sample, count as an infrastructure
   failure, or advance the environment action counter. After three failed attempts
   the episode ends as `action_selection_exhausted`, a scientific failure. A
   post-success skill response that stops at the cap is rejected as
   `output_token_cap_reached`; skill generation remains a single attempt.
9. **Infrastructure failures** are reserved for genuine execution failures: Ollama
   unavailable, provider connection failure or timeout, process crash, CUDA/device
   failure, bridge/plugin failure, ALFWorld infrastructure failure, and
   OS/network/storage failure. They keep the existing crash/resume policy: a failed
   result without a skill, a halt, and chronological `retry-failed`. With the cap a
   normal response cannot approach the 180-second provider timeout (2048 tokens take
   about one minute), so a timeout indicates an execution fault.
10. Three action-selection attempts per ALFWorld decision remain.

## Determinism

11. Seed 42 and temperature 0 stay frozen, but provider/model inference is not
    claimed to be deterministic: byte-identical prompts produced different Gemma
    responses in prelaunch checks. The task queue, configuration, and controller are
    deterministic. Completed scientific units are never rerun, and a failed
    scientific episode is never re-sampled to obtain a preferred result.

## Record keeping

12. Prelaunch evidence and earlier decision records are not rewritten; Decisions
    001 and 007 keep their historical text with amendment notes.
