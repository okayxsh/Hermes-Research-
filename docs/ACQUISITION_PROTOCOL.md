# Acquisition protocol

Use only frozen `train` tasks. Start a fresh Hermes session for each task. The
execution policy is frozen in [Decision 007](decisions/007-acquisition-execution-policy.md)
and `configs/acquisition/protocol.yaml`: at most 50 environment actions per
episode, no retrieval, and after a successful episode only the same `hermes3:8b`
agent may write at most one create-only candidate skill. Candidates pass the
deterministic Decision 003 validation; only exact normalized duplicates are
rejected and near-duplicates are preserved. Action selection follows
[Decision 008](decisions/008-action-selection-episode-history.md) and
[Decision 009](decisions/009-observation-interface-corrections.md): the full
episode history, the verbatim initial observation, a fixed not-observed
inventory marker, and exactly one `ACTION_INDEX` line per response.

The frozen queue is generated only from deterministic installed-data discovery
(180 TRAIN tasks, 30 per family, `task-selection-v1`, seed 1). It is rejected if
it overlaps a pilot/evaluation manifest or includes a task-specific source/game
identity already assigned elsewhere.

Every terminal task attempt is journalled and checkpointed immediately.
`results.jsonl` is the authority for the append-only skill pool: each accepted
skill lives inside its source episode's completed result row, and
`skill_pool.json` is a derived snapshot. Resume uses the canonical
task/order/profile identity, preserves one-worker chronology, and fails closed on
commit, runtime, configuration, queue, or skill-pool drift. An infrastructure
failure records a failed row without a skill and halts so it can be retried in
chronological order. See `EXPERIMENT_RECOVERY.md`.

## Operational sequence

1. Propose the queue: `python -m rq1.cli tasks propose --kind acquisition --count 180 --seed 1`.
2. NON-SCIENTIFIC check with TRAIN tasks outside that queue:
   `python -m rq1.cli acquisition check --run-id prelaunch-acquisition-check-<id> --task-id <train task> ... --max-runs <n>`,
   then `--resume`, then `python -m rq1.cli acquisition check-report --run-id <id>`.
3. `python -m rq1.cli acquisition prepare-approvals --proposal <proposal> --evidence-report <report>`
   writes three UNAPPROVED approval requests under `artifacts/approvals/acquisition/<commit>/`.
4. A human reviewer approves each request and runs its recorded command
   (`tasks freeze`, `freeze acquisition-environment`, `freeze acquisition-protocol`).
5. `python -m rq1.cli acquisition plan` must report `launch_permitted: true`.
6. Scientific run: `python -m rq1.cli acquisition run --run-id <id> --yes --backup-dir /workspace/persistent/backups --require-backup`;
   continue with `resume`, or `retry-failed` after an infrastructure failure; check with `validate`.
