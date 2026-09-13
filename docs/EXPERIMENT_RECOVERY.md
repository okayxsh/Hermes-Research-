# Crash-safe acquisition and evaluation

Final acquisition and final evaluation use a shared durable boundary beneath
`results/final/<experiment-id>/`. The real Hermes and ALFWorld adapters remain
capability-gated; checkpoint support does not enable them, read `valid_unseen`,
or relax activation and profile-isolation requirements.

## Durable files

The directory contains `run_manifest.json`, immutable phase manifests,
`results.jsonl`, `errors.jsonl`, `checkpoint.json`,
`checkpoint.backup.json`, and per-attempt `logs/`. Copy the complete directory,
not an individual checkpoint file.

`results.jsonl` is the completion authority. Each terminal attempt is flushed
and synced before the checkpoint is replaced through `checkpoint.tmp`. If a
machine stops after the result write, resume finds the run key in the journal
and does not execute it again. Only an incomplete final JSONL tail is repaired;
the original bytes are retained in `logs/recovery/`. Interior corruption and
unexplained duplicate identities are fatal.

Normal task termination with `success=false` is completed evidence. An
exception is a failed attempt and is also written with its traceback to
`errors.jsonl`. Normal resume skips completed and failed attempts. Use the
explicit `retry-failed` command to create a linked replacement attempt. A
chronological acquisition failure cannot be retried after later acquisition
results exist. Every normally terminated acquisition attempt must report the
observed post-run library hash and size; missing or uncertain library evidence
blocks the phase.

SIGINT and SIGTERM request a cooperative stop. A completed current unit is
committed; a non-terminal unit gets no result row and restarts from the
beginning. Exact mid-episode ALFWorld restoration is never claimed.

## Safe checkpoint smoke test

This uses synthetic units and produces no scientific evidence:

```bash
python3 -m rq1.cli experiment checkpoint-test \
  --run-id checkpoint-smoke --max-runs 3 --total-runs 6 --delay-ms 500

python3 -m rq1.cli experiment checkpoint-test \
  --run-id checkpoint-smoke --resume --max-runs 3 --total-runs 6 --delay-ms 500
```

Use a new run ID to repeat the test. Add `--backup-dir /workspace/persistent`
to mirror every durable commit. Add `--require-backup` when loss of that mirror
must stop the runner at the next safe boundary.

## Final commands after real adapters are approved

Fresh chronological acquisition:

```bash
python3 -m rq1.cli acquisition run --run-id <EXPERIMENT_ID> --yes \
  --backup-dir /workspace/persistent --require-backup
```

Resume acquisition:

```bash
python3 -m rq1.cli acquisition resume --run-id <EXPERIMENT_ID> --yes \
  --backup-dir /workspace/persistent --require-backup
```

Fresh activated evaluation:

```bash
RQ1_RUN_FINAL_EVALUATION=1 python3 -m rq1.cli evaluation run \
  --run-id <EXPERIMENT_ID> --activation-manifest <ACTIVATION.json> --yes \
  --backup-dir /workspace/persistent --require-backup
```

Resume evaluation with the identical activation and frozen inputs:

```bash
RQ1_RUN_FINAL_EVALUATION=1 python3 -m rq1.cli evaluation resume \
  --run-id <EXPERIMENT_ID> --activation-manifest <ACTIVATION.json> --yes \
  --backup-dir /workspace/persistent --require-backup
```

Acquisition commands run the real executor but refuse until the task, environment,
and protocol freezes are human-approved (see ACQUISITION_RUNBOOK.md). Evaluation
commands remain capability-gated until acquisition outputs, snapshots, and
activation evidence exist. Neither is ever replaced by a fake fallback.

## Status and backup

```bash
python3 -m rq1.cli experiment status --run-id <EXPERIMENT_ID>
python3 -m rq1.cli experiment backup --run-id <EXPERIMENT_ID> \
  --backup-dir /workspace/persistent
```

On a replacement VM, install the identical frozen environment and checkout,
copy `results/final/<EXPERIMENT_ID>/` to the same relative location, then run
the matching resume command. Task/order, condition, seed, model/config,
activation, library, Python, package, and available Git hashes must match.
