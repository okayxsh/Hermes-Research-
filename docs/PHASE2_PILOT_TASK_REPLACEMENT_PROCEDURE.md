# RQ1 Phase 2E: outcome-blind replacement of an infeasible pilot task (DRAFT)

**Status: DRAFT procedure, not approved.**
- It applies to the pilot task whose goal places a Box on a CoffeeTable. Its validated oracle route needs **139 environment actions**, which exceeds both candidate scientific horizons (50 and 100).
- It applies equally to any later pilot task found infeasible for the same reason.
- No diagnostic or other model episode has been run on the pilot roster. The replacement must be completed **before** any D0 diagnostic outcome exists.

## 1. Why the task is excluded

| Check | Result |
|---|---|
| Exclusion basis | The oracle route length (prefix + detour + continuation = 139 > 100) |
| Evidence used | Environment and oracle evidence only: no model outcome, rating or diagnostic result |
| Consequence | Cap-invariance cannot reach a comparable post-detour transition, so the task is `UNTESTED`, never PASS. Exact-cap oracle replay fails at 50 and 100 |

The exclusion is recorded with its reason and the oracle-length evidence. The task is not silently dropped.

## 2. Candidate pool (fixed before any candidate is evaluated)

A candidate must satisfy **all** of the following:
1. It is in the `valid_seen` split and installed (Phase 1 inventory).
2. It is the **same task family** as the excluded task, so the pilot's family stratification (SD-19) is preserved.
3. It is not already in the pilot roster, and never in the confirmatory roster (different split; checked anyway).
4. **Exposure screen:**
   - its Phase 1 ledger class is `NO_MODEL_OUTCOME_FOUND`;
   - its `exposure_kinds` contain neither `model_outcome_exposure` nor `exact_task_reuse`;
   - `UNKNOWN` tasks are not eligible for the pilot;
   - known prelaunch and Kaggle check tasks are excluded by this rule (for example, `look_at_obj_in_light-AlarmClock-None-DeskLamp-323`).
5. Where possible, its goal template does not coincide with a confirmatory roster template. If no candidate satisfies this, the coincidence is recorded rather than used to block.

## 3. Deterministic order

- **Salt.** Before evaluating any candidate, record the salt: the SHA-256 of the current pilot roster file, plus the label `pilot-replacement-<n>`.
- **Order.** Sort candidates by `sha256(salt + "|" + task_id)`.
- **No discretion.** No other ordering or preference may be applied, and no candidate may be skipped except by a failed check.

## 4. Checks per candidate, in order (all CPU-only, no model)

| Step | Tool | Pass condition |
|---|---|---|
| a | Phase 1 checkpoint and detour rule | Mid-route navigation checkpoint exists; the detour rule selects an admissible `go to`; oracle-validated (Phase 1 `oracle-validate`) |
| b | `university_cli oracle-replay --cap 50` and `--cap 100` | `validated: true` at **both** exact caps. The budget decision is not yet made, so the replacement must be feasible at either |
| c | `university_cli cap-invariance --caps 50 100` | `status: PASS` (UNTESTED or FAIL rejects the candidate) |
| d | `university_cli cpu-rankings --cap 100` | Frozen natural ranking produced; `tie_fragile` recorded (a flag, not a rejection) |

**The first candidate in the §3 order that passes a–d becomes the replacement.**
- Every candidate examined is appended to a replacement log, with each check's result and output hash.
- The log is append-only and is reviewed with the decision.

## 5. Regenerate every dependent artifact

1. **Pilot roster file** (new hash), with the exclusion and the replacement recorded in it.
2. **Exposure-risk statement** for the new roster (`exposure_acceptance.build_statement`). The roster hash changes, so any earlier acceptance is void.
3. **Frozen rankings** for the pilot roster, including the replacement.
4. **Rating preparation:**
   - candidate sets (natural top 3) and the workload recomputed with `rating.workload_from_rankings`;
   - rating exports regenerated under a new label, because item IDs and ordering change;
   - any export already distributed is withdrawn, and this is recorded.
5. **Diagnostic manifest** rebuilt (new `manifest_sha256`).
6. **Approvals:** every approval bound to a superseded hash (roster, exposure statement, oracle replay, diagnostic manifest) is void. New approvals are required.

## 6. Prohibitions

- No model output, diagnostic result, rating or Study 1 outcome may influence the candidate pool, the order or the choice.
- A replacement is never chosen after any pilot model episode has run. If one has, the pilot is restarted under a new roster and run label.
- The excluded task is not reinstated at a different cap.
