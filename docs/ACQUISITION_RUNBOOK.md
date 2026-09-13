# RQ1 acquisition runbook

Scientific acquisition runs the frozen 180-task TRAIN queue with `gemma4:12b`
([Decision 010](decisions/010-gemma-backbone-and-model-output-failures.md)). Nothing
here may start it before a human approves the three acquisition freezes.

Run every command on the Pod from the repository root:

```bash
source /workspace/persistent/rq1-env.sh
cd /workspace/persistent/rq1-protocol-migration
PY=/opt/rq1-venv/bin/python
RUN=rq1-acquisition-gemma4-12b
BACKUP=/workspace/persistent/backups
```

## 1. Approve and freeze (human)

The UNAPPROVED requests are in `artifacts/approvals/acquisition/<commit12>/`. For
each request the reviewer checks the recorded inputs, sets `status` to `APPROVED`,
fills `approved_by`, `approved_at` (UTC ISO-8601), and `reference`, and runs the
request's recorded `command` (`tasks freeze`, `freeze acquisition-environment`,
`freeze acquisition-protocol`). Afterwards both must pass:

```bash
$PY -m rq1.cli acquisition preflight --run-id $RUN --backup-dir $BACKUP
$PY -m rq1.cli acquisition plan          # launch_permitted: true
```

## 2. Launch (single owner)

```bash
tmux new-session -d -s rq1-acquisition \
  "$PY -m rq1.cli acquisition run --run-id $RUN --yes --backup-dir $BACKUP --require-backup >> /workspace/persistent/logs/$RUN.log 2>&1"
```

One process owns a run: another `run`, `resume`, or `retry-failed` for the same run
ID refuses while the lock is held.

## 3. Monitor (read-only)

```bash
$PY -m rq1.cli experiment status --run-id $RUN     # checkpoint and terminal result count
tail -n 3 /workspace/persistent/logs/$RUN.log       # [n/180] progress and ETA
$PY -m rq1.cli acquisition validate --run-id $RUN   # skill pool, retrieval, and retry-lineage invariants
curl -s localhost:11434/api/ps
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv
```

Never edit files under `results/final/$RUN/`.

## 4. Stop safely

```bash
kill -TERM "$(pgrep -f "rq1.cli acquisition (run|resume|retry-failed) --run-id $RUN")"
```

The runner finishes the current unit, commits its result and checkpoint, and exits
with status `interrupted`. Do not press Ctrl-C in the tmux pane (it also signals the
Hermes worker and turns the in-flight unit into an infrastructure failure), and use
`kill -9` only for a hung process.

## 5. Resume

```bash
$PY -m rq1.cli acquisition resume --run-id $RUN --yes --backup-dir $BACKUP --require-backup
```

Completed units are never rerun. An attempt that was in progress during a crash or
`kill -9` restarts from the beginning and is recorded as `crashed_or_interrupted`.

## 6. Infrastructure failures

A genuine execution failure (Ollama unavailable, provider timeout or connection
error, bridge/plugin or ALFWorld error, device, OS, network, or storage failure)
writes a failed row and halts the run. Model output never does: invalid or capped
responses consume action-selection attempts inside the episode. After fixing the
cause:

```bash
$PY -m rq1.cli acquisition retry-failed --run-id $RUN --yes --backup-dir $BACKUP --require-backup
$PY -m rq1.cli acquisition resume --run-id $RUN --yes --backup-dir $BACKUP --require-backup
```

The retry is the failed unit's authorized successor; `acquisition validate` reports
any duplicate completed unit or unauthorized retry.

## 7. Move to another Pod

1. Stop with SIGTERM (section 4) and wait for the process to exit.
2. `$PY -m rq1.cli experiment backup --run-id $RUN --backup-dir $BACKUP`
3. On the new Pod install the identical environment: the frozen commit, the
   `uv.lock` environment, ALFWorld 0.4.2 data, the Hermes commit, the Ollama
   version, and `gemma4:12b` with the frozen digest.
4. Restore `results/final/$RUN/`, `artifacts/task_manifests/frozen/`, and
   `artifacts/freezes/` to the same relative paths.
5. `acquisition plan` must report `launch_permitted: true`; then resume (section 5).

Launch and resume enforce the commit, Python version, dependency lock, ALFWorld
version, Hermes version and commit, Ollama version, model tag, digest, quantization,
provider settings, inference seed, prompt hashes, and config hashes. Host name and GPU
are recorded only, and a lock left by another host is treated as stale.

## 8. Balanced extension 181–240 (Decision 011)

The completed 180-task run is read-only. The extension is the separate run
`rq1-acquisition-gemma4-12b-ext-181-240`: 60 TRAIN tasks (10 per family) that continue
the frozen selection, starting from the parent's exact final 34-skill pool, with identical
settings ([Decision 011](decisions/011-acquisition-extension-181-240.md)).

```bash
EXT=rq1-acquisition-gemma4-12b-ext-181-240
$PY -m rq1.cli acquisition-extension propose        # queue proposal + immutable starting pool
$PY -m rq1.cli acquisition-extension check --run-id prelaunch-acquisition-extension-check-<label> --task-id <non-queue TRAIN task> --max-runs 1
$PY -m rq1.cli acquisition-extension check --run-id prelaunch-acquisition-extension-check-<label> --resume
$PY -m rq1.cli acquisition-extension check-report --run-id prelaunch-acquisition-extension-check-<label>
$PY -m rq1.cli acquisition-extension prepare-approvals --proposal <proposal> --evidence-report <report>
$PY -m rq1.cli acquisition-extension preflight --backup-dir $BACKUP   # only human approval may remain
```

A human reviewer approves the three requests in
`artifacts/approvals/acquisition-extension/<commit12>/` and runs their recorded commands
(`acquisition-extension freeze-tasks`, `freeze acquisition-extension-environment`,
`freeze acquisition-extension-protocol`). Then `acquisition-extension plan` must report
`launch_permitted: true`, and the extension is launched, monitored, stopped, resumed, and
retried exactly as in sections 2–6 with `acquisition-extension` in place of `acquisition`
and `--run-id $EXT`. `acquisition-extension validate --run-id $EXT` reports the starting pool,
the appended skills, and the combined per-family pool.
