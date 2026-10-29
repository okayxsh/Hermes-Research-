# Friday command sheet (ordered)

Every command below is copied from `OPERATIONS.md`, `README.md`, or
`FRIDAY_FINAL_READINESS.md`; no flags were invented. Run from the repository root on the university
Linux host. Tags:

- `[OFFLINE-SAFE]` — reads/writes local files only; no model call, no GPU, no network.
- `[GPU/CONSUMES-TIME]` — starts model services and/or runs episodes; consumes GPU and clock time.
- `[HUMAN-APPROVAL-REQUIRED]` — needs a signed approval bound to an exact hash, or a human decision.

Several commands are both offline and approval-gated; both tags are shown.

## 0. Set the environment (shell only)

`[OFFLINE-SAFE]`

```bash
export RQ1_CONFIG=/approved/path/outside/checkout/config.json
export RQ1_EVIDENCE=/approved/path/outside/checkout/evidence
export RQ1_RUN_ROOT="$RQ1_RUN_OUTPUT_DIR/rq1-university-pilot-001"
export RQ1_FROZEN_MANIFEST="$RQ1_EVIDENCE/pilot-frozen.json"
export RQ1_RELEVANCE_CSV="$RQ1_TASK_SKILL_RELEVANCE_CSV"
export RQ1_ACCESS_END_EPOCH=<actual-access-deadline-unix-seconds>
```

Sources: `FRIDAY_FINAL_READINESS.md:59,67-75`. Keep evidence and outputs outside the checkout
(`OPERATIONS.md:16`). The deadline is the actual access end, not a guess; assignment stops 30 minutes
early (`OPERATIONS.md:114`).

## 1. Read-only check (always first)

`[OFFLINE-SAFE]`

```bash
PYTHONPATH=src python3.11 -m rq1.cli university-gpu check --json
```

`README.md:20-23`, `OPERATIONS.md:8`. A blocked result is expected until host hardware and frozen
inputs exist; it creates no files and performs no model calls.

## 2. Display the setup plan (never applies)

`[OFFLINE-SAFE]`

```bash
PYTHONPATH=src python3.11 -m rq1.cli university-gpu setup
```

`OPERATIONS.md:9`, `README.md:29`. **Do not** add `--apply` here. Applying project-local dependencies
is a separate approval-gated action (`setup --apply --yes --approval-file PATH`) that requires
`project_python_environment`, `project_dependencies`, an approver, and an approval reference
(`OPERATIONS.md:12`). No system package install or model pull is supported.

## 3. Gate scores (CPU embeddings only)

`[OFFLINE-SAFE]`

```bash
PYTHONPATH=src python3.11 -m rq1.cli university-gpu gate-scores \
  --tasks "$RQ1_PILOT_TASKS" --failures "$RQ1_PILOT_FAILURES" --contexts "$RQ1_PILOT_CONTEXTS" \
  --library "$RQ1_ACCUM18_LIBRARY" --relevance "$RQ1_PILOT_RELEVANCE" \
  --sbert-cache "$RQ1_HUGGINGFACE_CACHE" --output-dir "$RQ1_EVIDENCE/gate-scores"
export RQ1_GATE_SCORES="$RQ1_EVIDENCE/gate-scores/scores.csv"
export RQ1_GATE_LABELS="$RQ1_EVIDENCE/gate-scores/labels.csv"
export RQ1_VALID_SEEN_GATE_SCORES_CSV="$RQ1_GATE_SCORES"
export RQ1_VALID_SEEN_GATE_LABELS_CSV="$RQ1_GATE_LABELS"
export RQ1_TASK_SKILL_RELEVANCE_CSV="$RQ1_PILOT_RELEVANCE"
```

`OPERATIONS.md:34-43`. Uses the frozen Sentence-BERT snapshot; does not call Gemma or the GPUs.

## 4. Gate calibration: propose and validate

`[OFFLINE-SAFE]`

```bash
PYTHONPATH=src python3.11 -m rq1.cli university-gpu gate-calibration propose \
  --scores "$RQ1_GATE_SCORES" --labels "$RQ1_GATE_LABELS" \
  --output "$RQ1_EVIDENCE/gate-proposal.json"
PYTHONPATH=src python3.11 -m rq1.cli university-gpu gate-calibration validate \
  --calibration "$RQ1_EVIDENCE/gate-proposal.json"
```

`OPERATIONS.md:48-52`.

## 5. Gate calibration: freeze

`[OFFLINE-SAFE] [HUMAN-APPROVAL-REQUIRED]`

```bash
PYTHONPATH=src python3.11 -m rq1.cli university-gpu gate-calibration freeze \
  --calibration "$RQ1_EVIDENCE/gate-proposal.json" \
  --approval-file "$RQ1_EVIDENCE/gate-approval.json" \
  --output "$RQ1_EVIDENCE/gate-frozen.json"
export RQ1_GATE_CALIBRATION_FROZEN="$RQ1_EVIDENCE/gate-frozen.json"
```

`OPERATIONS.md:53-57`. Approval must bind the exact `calibration_sha256` (`OPERATIONS.md:60`).

## 6. Manifest: propose and validate

`[OFFLINE-SAFE]`

```bash
PYTHONPATH=src python3.11 -m rq1.cli university-gpu manifest propose \
  --stage pilot --run-id rq1-university-pilot-001 \
  --config "$RQ1_CONFIG" --output "$RQ1_EVIDENCE/pilot-proposal.json"
PYTHONPATH=src python3.11 -m rq1.cli university-gpu manifest validate \
  --manifest "$RQ1_EVIDENCE/pilot-proposal.json"
```

`OPERATIONS.md:67-71`. Requires a committed, clean repository and hashes every input
(`OPERATIONS.md:64,27`).

## 7. Manifest: freeze

`[OFFLINE-SAFE] [HUMAN-APPROVAL-REQUIRED]`

```bash
PYTHONPATH=src python3.11 -m rq1.cli university-gpu manifest freeze \
  --manifest "$RQ1_EVIDENCE/pilot-proposal.json" \
  --approval-file "$RQ1_EVIDENCE/pilot-approval.json" \
  --output "$RQ1_EVIDENCE/pilot-frozen.json"
```

`OPERATIONS.md:72-75`. Review the proposed manifest and hash before signing; freeze refuses to
overwrite an existing file (`OPERATIONS.md:78`).

## 8. Static preflight

`[OFFLINE-SAFE]`

```bash
PYTHONPATH=src python3.11 -m rq1.cli university-gpu preflight \
  --config "$RQ1_CONFIG" --manifest "$RQ1_EVIDENCE/pilot-frozen.json"
```

`OPERATIONS.md:86-89`. Without `--check-endpoints` it can only return `STATIC_PASS`; endpoint checks
require four live endpoints (`OPERATIONS.md:84`). Do not start services for this step.

## 9. Pilot benchmark launch (four attempts)

`[GPU/CONSUMES-TIME] [HUMAN-APPROVAL-REQUIRED]`

```bash
PYTHONPATH=src python3.11 -m rq1.cli university-gpu launch \
  --config "$RQ1_CONFIG" --manifest "$RQ1_EVIDENCE/pilot-frozen.json" \
  --approval-file "$RQ1_EVIDENCE/pilot-launch-approval.json" --yes \
  --cutoff-epoch "$RQ1_ACCESS_END_EPOCH" --max-attempts 4
```

`FRIDAY_FINAL_READINESS.md:75-80`. Starts four private Ollama services, checks model identity, runs
four approved smoke requests, verifies GPU placement, then starts at most four real attempts. A
paused result after four attempts is expected and exits nonzero.

## 10. Benchmark report

`[OFFLINE-SAFE]`

```bash
PYTHONPATH=src python3.11 -m rq1.cli university-gpu benchmark-report \
  --run-root "$RQ1_RUN_ROOT" --manifest "$RQ1_EVIDENCE/pilot-frozen.json"
```

`FRIDAY_FINAL_READINESS.md:79-82`. File-only; needs four completed attempts on distinct workers.
Four attempts are a rough benchmark, not a representative sample (`FRIDAY_FINAL_READINESS.md:37`).

## 11. Resume the pilot without the attempt cap

`[GPU/CONSUMES-TIME] [HUMAN-APPROVAL-REQUIRED]`

```bash
PYTHONPATH=src python3.11 -m rq1.cli university-gpu resume \
  --config "$RQ1_CONFIG" --manifest "$RQ1_EVIDENCE/pilot-frozen.json" \
  --approval-file "$RQ1_EVIDENCE/pilot-launch-approval.json" --yes \
  --cutoff-epoch "$RQ1_ACCESS_END_EPOCH"
```

`FRIDAY_FINAL_READINESS.md:82-85`. Only after the recorded pilot go decision; same inputs and
approval.

## 12. Validate the completed run

`[OFFLINE-SAFE]`

```bash
( set -C; PYTHONPATH=src python3.11 -m rq1.cli university-gpu validate \
  --run-root "$RQ1_RUN_ROOT" --manifest "$RQ1_EVIDENCE/pilot-frozen.json" \
  > "$RQ1_EVIDENCE/pilot-validation.json" )
```

`FRIDAY_FINAL_READINESS.md:86`. File-only; the `set -C` guard refuses to overwrite an existing
validation file.

## 13. Monitor or stop (second terminal)

`[OFFLINE-SAFE]`

```bash
PYTHONPATH=src python3.11 -m rq1.cli university-gpu status \
  --run-root "$RQ1_RUN_ROOT" --manifest "$RQ1_FROZEN_MANIFEST"

PYTHONPATH=src python3.11 -m rq1.cli university-gpu stop \
  --run-root "$RQ1_RUN_ROOT" --reason "access deadline"
```

`FRIDAY_FINAL_READINESS.md:91-96`, `OPERATIONS.md:106-107`. `stop` assigns no new units but lets
active episodes finish; **wait for them to finish** (`OPERATIONS.md:114`).

## 14. Crash recovery, only after review

`[GPU/CONSUMES-TIME] [HUMAN-APPROVAL-REQUIRED]`

```bash
PYTHONPATH=src python3.11 -m rq1.cli university-gpu status \
  --run-root "$RQ1_RUN_ROOT" --manifest "$RQ1_FROZEN_MANIFEST"

PYTHONPATH=src python3.11 -m rq1.cli university-gpu resume \
  --config "$RQ1_CONFIG" --manifest "$RQ1_FROZEN_MANIFEST" \
  --approval-file "$RQ1_LAUNCH_APPROVAL" --yes --retry-failed \
  --cutoff-epoch "$RQ1_ACCESS_END_EPOCH"
```

`OPERATIONS.md:106-108`, `FRIDAY_FINAL_READINESS.md:98`. Inspect failures first; `--retry-failed`
creates new attempts only after operator review. A stale lock needs manual inspection. Resume works
at episode-unit boundaries and is designed for the original run location/worker identities.

## 15. Partial export of a stopped run

`[OFFLINE-SAFE]`

```bash
PYTHONPATH=src python3.11 -m rq1.cli university-gpu progress-report \
  --run-root "$RQ1_RUN_ROOT" --manifest "$RQ1_FROZEN_MANIFEST"
PYTHONPATH=src python3.11 -m rq1.cli university-gpu export-progress \
  --run-root "$RQ1_RUN_ROOT" --manifest "$RQ1_FROZEN_MANIFEST" \
  --relevance "$RQ1_RELEVANCE_CSV" --destination "$RQ1_EXPORT_DIR"
PYTHONPATH=src python3.11 -m rq1.cli university-gpu verify-export \
  --destination "$RQ1_EXPORT_DIR"
```

`FRIDAY_FINAL_READINESS.md:100-106`. Use a new destination for each stopped run; the exporter refuses
an existing one (`FRIDAY_FINAL_READINESS.md:108`).

## 16. Complete-run merge and export (only after a passing `validate`)

`[OFFLINE-SAFE]`

```bash
PYTHONPATH=src python3.11 -m rq1.cli university-gpu merge \
  --run-root "$RQ1_RUN_ROOT" --manifest "$RQ1_FROZEN_MANIFEST"
PYTHONPATH=src python3.11 -m rq1.cli university-gpu export \
  --run-root "$RQ1_RUN_ROOT" --manifest "$RQ1_FROZEN_MANIFEST" \
  --destination "$RQ1_EXPORT_DIR"
```

`OPERATIONS.md:110-111`. Both require a complete, hash-valid run (`README.md:58`); do not run them
on an incomplete or stopped run.

## 17. Copy out and verify the copied export

`[OFFLINE-SAFE]`

```powershell
$env:PYTHONPATH = "."
py -3.11 -B -m rq1.cli university-gpu verify-export --destination ..
py -3.11 -B -m rq1.cli university-gpu progress-report --run-root ..\run --manifest ..\run\manifest.json
```

`FRIDAY_FINAL_READINESS.md:110-116`. Run from the **copied** export's `analysis-src` directory, after
copying the directory to an independent drive; `-B` prevents bytecode files inside the export.

## 18. Offline exploratory analysis (complete run only)

`[OFFLINE-SAFE]`

```bash
PYTHONPATH=src python3.11 -m rq1.cli university-gpu analyze \
  --results "$RQ1_RUN_ROOT/merged-results.jsonl" \
  --manifest "$RQ1_FROZEN_MANIFEST" --relevance "$RQ1_RELEVANCE_CSV" \
  --output "$RQ1_EVIDENCE/exploratory-analysis.json"
```

`OPERATIONS.md:121-124`. Exploratory only; it does not establish sample-size adequacy or replace the
frozen Oct 20 analysis (`OPERATIONS.md:116-118`).

## Do-not-run list for Friday

- `university-gpu setup --apply ...` — installs project-local dependencies; needs separate
  authorization (`OPERATIONS.md:12`).
- Any `sudo`, system package install, driver change, or `ollama pull` — refused by the package
  (`OPERATIONS.md:12`).
- `launch`/`resume` without a matching signed approval and `--yes`.
- Deleting a stale lock or a run directory by hand.
- `export`/`merge` on an incomplete run.
