# 011 — Balanced acquisition extension to 240 (activation of the pre-existing hard-cap contingency)

- Status: approved methodology decision by the study owner; the extension freezes remain separate human approvals
- Date: 2026-09-14
- Timing: after the initial 180-task scientific acquisition completed normally and before any final RQ1 evaluation; no evaluation outcomes exist
- Activates: the `later_hard_cap` of the frozen acquisition protocol (Decision 007 execution policy, `configs/acquisition/protocol.yaml`)
- Changes no scientific setting of Decisions 003, 007, 008, 009, or 010

## The pre-existing contingency

1. The acquisition protocol froze an initial queue of 180 TRAIN tasks (30 per family) and
   a later hard cap of 240 tasks (40 per family) with `automatic_extension: false`, so the
   cap is reached only by an explicit decision. The cap entered the repository in commits
   48ead8c and 36b3030 (2026-09-12), was recorded in the approved acquisition-protocol freeze
   (input fingerprint `ff693298…`) before the 180-task run launched on 2026-09-13, and the
   approved task-freeze request stated that the initial queue is not automatically extended
   to 240.
2. The study owner approved this contingency before any evaluation result existed: when the
   initial queue leaves the planned library quotas unmet, acquisition is extended in a
   balanced way up to the hard cap. This record is that explicit activation, not a new
   experiment.

## Why it is activated

3. The parent run `rq1-acquisition-gemma4-12b` (commit 8bd452e) completed normally: 180/180
   units, 86 successful episodes, 94 scientific failures, 0 infrastructure failures, 0
   acquisition retrieval events, and a final append-only pool of 34 accepted skills (hash
   `11579cfe…`). Its closeout manifest (sha256 `98c54bb3…`) certifies these values.
4. Accepted skills per family: pick_and_place 9, pick_two_and_place 6, look_at_object 3,
   clean_and_place 6, heat_and_place 8, cool_and_place 2. Clean-24 needs at least four
   eligible skills per family (look_at_object and cool_and_place are short even before the
   human quality rubric), and Accum-60 (10 per family) and Accum-96 (16 per family) are
   unmet in every family. The planned accumulated-library quotas are therefore not met.
5. The extension adds balanced TRAIN experience only. It cannot guarantee the quotas: an
   arithmetic projection from the parent yield leaves cool_and_place below four. Whatever the
   outcome, acquisition stops at 240 and library construction keeps failing closed under the
   frozen rules. No quota, quality rubric, or skill rule is relaxed by this decision.

## What is frozen

6. **Nothing is discarded or rerun.** The parent results, checkpoint, skill pool, freezes,
   and approvals are read-only evidence. No failed episode is replaced and the parent queue is
   not regenerated.
7. **Queue.** 60 new TRAIN tasks, 10 per family, logical acquisition positions 181–240. The
   frozen task-selection-v1 policy (seed 1, round-robin family balancing, TRAIN metadata only)
   is recomputed at 240; positions 1–180 must reproduce the parent frozen queue exactly and
   positions 181–240 are the extension queue. No task is chosen by apparent difficulty or
   outcome, and no valid_seen, valid_unseen, or evaluation information is used. The queue has
   no overlap with the parent 180 and no duplicates.
8. **Starting pool.** The extension starts from the exact final 34-skill parent pool, rebuilt
   from the parent `results.jsonl` and verified against its `skill_pool.json`, the closeout
   manifest, and the recorded hash; an immutable copy is recorded under
   `artifacts/acquisition-extension/`. The pool is append-only: new skills continue at pool
   index 35; parent skills are never modified, deleted, re-ranked, patched, or semantically
   deduplicated; near-duplicates are preserved; exact normalized duplicates are rejected against
   every parent and extension skill; at most one create-only candidate per successful episode.
9. **Settings.** Identical to the parent run: `gemma4:12b` (Q4_K_M, digest `4eb23ef1…`),
   `num_predict` 2048, `num_ctx` 32768, temperature 0, seed 42, `think: false`, a 50-action
   cap, `action-index-history-v3` with the full observable history, verbatim initial
   observation, and inventory only through the inventory action, three action-selection
   attempts, the same post-success skill writer, and zero acquisition retrieval. The inherited
   protocol sha256 is unchanged.
10. **Separate run.** The extension runs as `rq1-acquisition-gemma4-12b-ext-181-240` under
    `results/final/`, never appending to the parent result file. Every unit and appended skill
    records the parent run ID and its logical position. The combined chronological order for
    later library construction is parent positions 1–180 followed by extension positions 181–240.
11. **Failures and resume.** Invalid or capped model output remains a scientific outcome;
    only genuine execution failures are infrastructure failures, which halt for chronological
    `retry-failed`. Resume never reruns a completed unit.

## Checkpoint backup of the parent

12. The parent `checkpoint.backup.json` has status `running` while listing all 180 completed
    units. `ExperimentStore.write_checkpoint` copies the current primary checkpoint to
    `checkpoint.backup.json` before installing each new primary, so after the final write the
    backup is the previous generation, identical except `status`. It is read only if
    `checkpoint.json` is unreadable, and resume reconciles from `results.jsonl`, the recovery
    authority. It is a non-authoritative stale generation, preserved unchanged; this is not a
    bug, and a regression test pins that such a backup never causes a completed unit to rerun.
