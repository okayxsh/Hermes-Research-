# Amended controlled-recovery evaluation (Decision 012)

- **Authority.** The machine-readable authority is `configs/evaluation/amended.yaml`, which equals
  `rq1.evaluation.amended_protocol.evaluation_protocol_definition()`. The rationale is in
  [Decision 012](decisions/012-pre-evaluation-library-amendment.md).
- **Scope.** Nothing here may run scientific units before the amendment, task, environment, and
  protocol freezes are human-approved and the core review is complete.

## Design summary

| Item | Frozen value |
|---|---|
| Conditions | NoLib 0 · Core-6 6 · Accum-12 12 · Accum-18 18 (Rule A, nested, 1/2/3 per family) |
| Tasks | 30 valid_unseen, 5 per family, the longest hand-coded expert routes (ties by task ID) |
| Seeds | 11, 29, 47 (also the model inference seed of each unit) |
| Units | 30 × 4 × 3 = 360, in 6 balanced shards of 60 |
| Controlled failure | midpoint-near navigation checkpoint + deterministic reversible `go to` detour, oracle-validated before freeze |
| Episode budget | 50 environment actions; recovery budget = 50 − prefix − 1 |
| Retrieval | SBERT all-mpnet-base-v2 @ e8c3b32…, top-3, once after the failure, none for NoLib |
| Model | gemma4:12b Q4_K_M @ 4eb23ef1…, num_predict 2048, num_ctx 32768, temperature 0, think false |
| Primary metric | conditional recovery rate |

## Commands

Run on the Pod from the repository root (`PY=/opt/rq1-venv/bin/python`).

1. **Human core review.** `$PY -m rq1.cli evaluation-amended core-package` writes
   `artifacts/skill-validation/rq1-acquisition-240/core-validation-fast.csv`. Review each family
   until its first PASS.
2. **Task preparation.** No model is called; valid_unseen access is logged.
   ```bash
   RQ1_VALID_UNSEEN_ACCESS=evaluation-task-preparation $PY -m rq1.cli evaluation-amended prepare-tasks --yes
   ```
3. **Non-scientific check.** Run on valid_seen with the real model:
   ```bash
   $PY -m rq1.cli evaluation-amended check --run-id prelaunch-evaluation-check-<label> --max-runs 2
   $PY -m rq1.cli evaluation-amended check --run-id prelaunch-evaluation-check-<label> --resume
   $PY -m rq1.cli evaluation-amended check-report --run-id prelaunch-evaluation-check-<label>
   ```
4. **Approval requests.** `$PY -m rq1.cli evaluation-amended prepare-approvals --preparation <dir> --evidence-report <report>`
   writes UNAPPROVED requests to `artifacts/approvals/evaluation-amended/<commit12>/`.
5. **Human approval.** Set the approval metadata in each request and run its recorded command:
   `evaluation-amended freeze-tasks`, then `freeze evaluation-amendment`,
   `freeze evaluation-environment`, and `freeze evaluation-protocol`.
6. **Libraries.** After the core review, `$PY -m rq1.cli evaluation-amended build-libraries --yes`
   builds the deterministic Rule A libraries from the completed review.
7. **Gate.**
   - `$PY -m rq1.cli evaluation-amended preflight --backup-dir /workspace/persistent/backups`
   - `$PY -m rq1.cli evaluation-amended plan` must report `launch_permitted: true`.
8. **Run.** Each shard is independent (k = 1–6); run them one after another on a single worker,
   or one per worker in parallel:
   ```bash
   RQ1_VALID_UNSEEN_ACCESS=scientific-evaluation $PY -m rq1.cli evaluation-amended run --shard k --yes --backup-dir /workspace/persistent/backups --require-backup
   ```
   `resume` and `retry-failed` take the same arguments. Completed units never rerun.
9. **After all shards.**
   - `$PY -m rq1.cli evaluation-amended merge --yes` verifies 360/360 and writes the merged report.
   - `$PY -m rq1.cli evaluation-amended rater-export --yes` writes the two blinded relevance files.
   - `$PY -m rq1.cli evaluation-amended analyze` computes the metrics, adding Precision@3 once
     labels exist.

## Integrity rules

- Units run in fresh sessions; the condition label, cosine scores, and oracle or reference actions
  are never in a prompt.
- Every unit must reproduce its frozen checkpoint and post-detour digests. A mismatch is a failed
  unit (infrastructure) and is never scored.
- Evaluation never writes skills. Libraries, tasks, the matrix, and shards are immutable and
  hash-bound into each shard's run configuration; resume fails closed on drift.
- Parallel shards share no mutable state. Each has its own result directory, checkpoint, lock,
  and backup mirror, and the merge rejects gaps, overlaps, and units run in the wrong shard.
