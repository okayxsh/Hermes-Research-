# Friday 30 October: final readiness

**BLOCKED.** The original 240 acquisition and 360 evaluation episodes are complete. The proposed six-arm study is not ready for scientific inference. The four RTX 5000 Ada GPUs and their placement, model, throughput and permissions remain unverified on the university machine.

## What is available and what is missing

- The historical 30-task valid-unseen manifest, 30 oracle-validated failure definitions and frozen Accum-18 library were recovered from the hash-verified evaluation archive. Copies and their hashes are in the separate local `experiment/university-friday-preparation-20261029/historical-inputs/` folder. Do not alter the originals.
- No frozen six-task valid-seen pilot manifest or corresponding oracle-validated failures were found in the examined local checkouts and two final experiment archives. Six eligible tasks, one per family, and their validated failures are required.
- The original 810 rater items cover 154 distinct confirmatory task–skill pairs among the 540 required pairs; 386 pairs have no historical rating in the join. Repeated ratings differ for 28 task–skill pairs. The existing ratings were collected for a different retrieval audit and require a documented compatibility and blinding decision before reuse. The new pilot needs 108 pairs. Thus at least **494 pairs and 988 individual ratings** are missing if all 154 historical pairs are accepted; otherwise up to **648 pairs and 1,296 ratings** require fresh independent review. Adjudicate disagreements and fill final labels for every pair. The 540-row confirmatory blank forms and outcome-free worksheet are in `experiment/university-friday-preparation-20261029/confirmatory-rating-templates/`. They contain no ratings. Two independent raters and an adjudicator must complete the work. Create the 108 pilot rows after its six tasks are frozen.
- The valid-seen Gate scores, adjudicated labels, approved frozen threshold and evidence of outcome-blind calibration were not located. Six pilot tasks imply 108 task–skill scores and 108 adjudicated labels if they are the calibration set. `gate-scores` can prepare them using the frozen Sentence-BERT snapshot on CPU once reviewed pilot contexts and ratings exist. That real model path is untested here because the snapshot and dependency are missing; context task instructions and inventory must be reviewed and compared with observed pilot contexts before confirmation.
- A hash-pinned Gemma tokenizer, reviewed neutral filler, ALFWorld 0.4.2 data tree and the frozen Sentence-BERT snapshot were not found in the checked local project and standard caches. The local Ollama installation has no `gemma4:12b` manifest in its default model directory. Python 3.11 is present, but the checked local Python environments lack ALFWorld, sentence-transformers, tokenizers, torch and pytest. University-host availability is unknown.
- Pilot and confirmatory proposal/freeze approvals, launch approvals, independent eligibility and leakage review, a pilot go decision, and an author-reviewed task-cluster power/precision report are still needed. Confirmatory approval must acknowledge previous exposure to these task outcomes. A local human must supply and sign these records; no approval is inferred from this document.
- Historical NoLib and Accum-18 are references only: full prompt/token/runtime compatibility and contemporaneous randomization are not established. Neither may replace a primary new cell without a separate compatibility review and author approval.

## What the code can and cannot establish

The package implements immutable manifests, four isolated Ollama services, process-to-GPU UUID checks, one coordinator lock, unique attempts, per-episode authoritative results, heartbeats, safe stop, a 30-minute assignment cutoff, retry/resume, complete-run analysis, and stopped-run partial exports. It stops new assignments if free storage falls below 1 GB or the storage check fails; active work still needs space to finish. Every ten completed units it copies immutable manifest, event and result records into a checksummed local checkpoint. An interrupted checkpoint remains a `.partial-*` directory and is not a completed backup. Attempt logs are included in the stopped-run export, not in those compact periodic checkpoints. A failed real worker is quarantined; remaining healthy workers continue. Review the failure before resuming or retrying. A freeze has a one-hour worker response timeout; an active episode is not automatically killed at the access cutoff.

The latest repository-wide offline run completed **325 tests with four skipped and no failures**. The package suite then passed **29 tests** after the storage, approval, deadline and CPU scoring changes; the two scoring tests were rerun after final input-hash hardening. Package tests simulate four concurrent workers, worker/service failure, frozen reads, incorrect GPU mapping, stop/resume, bad inputs, low storage, duplicates, cutoff, interrupted backup/export and analysis from a copied package. These are CPU mocks, not a live university GPU, Ollama, ALFWorld, reboot or network-failure test. The exact full and package test logs are retained in the separate preparation folder.

`export-progress` refuses an active or stale run lock, copies the full stopped run and Python analysis source to a new directory, hashes each file, and marks missing units. `verify-export` checks those files and the run identities without GPU inference. This is a directory to copy or archive for transfer; it is not a signed artifact or proof of a real experiment. Keep a second copy on an independent device, then verify that copy. The complete-run analysis can produce descriptive CSV tables, a simple SVG figure and task-cluster intervals. Holm-adjusted confirmatory tests, publication-ready figures, full prompt-token equality and an author-approved prospective power decision are **not implemented or verified**; do not claim paper-ready inference from the exploratory report.

An exploratory plug-in simulation resampled the 30 historical task means for NoLib versus Accum-18 with 500 outer and 500 inner replicates. The observed paired difference was 0.111; the median simulated interval width was 0.261 and 40.2% of intervals excluded zero (Monte Carlo standard error 2.2 percentage points). This assumes the observed historical task distribution and effect represent a future run, considers one contrast and omits multiplicity correction. It is not a prospective power calculation and does not validate 576 episodes. Reproduce it with `university-gpu/historical_precision_simulation.py`; the local output is `experiment/university-friday-preparation-20261029/historical-cluster-simulation.json`.

## Timing: provisional only

All 360 historical terminal records were matched to their actual first and last instrumented episode events. Full recorded episode runtimes include unsuccessful recoveries: mean 156.4 seconds, median 139.5, 90th percentile 306.5 and maximum 596.9. Event timestamps bracket logged activity; they are not necessarily total elapsed runtime. Reproduce the 360-row timing table with `university-gpu/historical_runtime.py`; the current local table and summary are in `experiment/university-friday-preparation-20261029/historical-timing/`.

The following are **scenarios, not measured RTX 5000 Ada predictions**. They assume four independent workers, no retries, contention, loading, checkpoint costs or transfer time. Reserve one hour for setup and benchmarking and at least 30 minutes for verification and transfer, leaving 6.5 hours for scientific episodes.

| Scenario relative to historical per-episode time | Pilot 36 | Confirmation 540 | All 576 | Estimated units within 6.5 compute hours |
| --- | ---: | ---: | ---: | ---: |
| Faster: 0.5× time | 12 min | 2 h 56 min | 3 h 8 min | 576 capped at plan |
| Historical-rate illustration: 1× | 23 min | 5 h 52 min | 6 h 15 min | 576 capped at plan |
| Slower: 2× time | 47 min | 11 h 44 min | 12 h 31 min | about 299 |

There is no defensible most-likely finish time or clock time until the four actual workers are measured and the start time is known. Even the middle scenario leaves little room for overhead. Do not claim all 576 fit in eight hours without a measured rate and a margin for variable tasks and failures. Four first-pilot attempts are only a rough benchmark, not a representative sample or a power calculation.

## Tonight and tomorrow

Tonight, the authors and raters need to select and validate the pilot tasks; complete or resolve the applicable ratings and adjudications; review leakage, filler and tokenizer; prepare Gate scores and calibration; complete the precision/power decision; and prepare the hash-bound approval records. Work with university staff to confirm permitted GPU access, local storage, required dependencies and exact offline model/data availability. Resolve how files will be transferred and whether remote continuation is authorized. The local prepared input copies and forms must be transferred through an approved channel.

The files in `university-gpu/approval-templates/` are blank, unapproved examples. Copy each needed template to the external evidence folder; the responsible human fills it only after reviewing the specific decision and exact hash. They are deliberately incapable of approving a run as supplied.

Give the two independent raters the outcome-free worksheet and their separate `rater_a.csv` and `rater_b.csv`. A human adjudicator fills every row of `final-labels.csv`, resolving disagreements according to the approved rubric. After all three files are complete, this file-only command preserves both original ratings and combines the human decisions into a new file:

```bash
python3.11 university-gpu/combine_ratings.py --rater-a "$RQ1_RATER_A" --rater-b "$RQ1_RATER_B" --final-labels "$RQ1_FINAL_LABELS" --output "$RQ1_RELEVANCE_CSV"
```

The command does not decide whether historical ratings satisfy the prospective blinding and compatibility rules. `make_rating_templates.py` can create new empty forms once the pilot task and failure files exist.

On Friday, from the committed repository root, run the following **read-only check first**:

```bash
PYTHONPATH=src python3.11 -m rq1.cli university-gpu check --json
```

Then read [OPERATIONS.md](OPERATIONS.md). Set `RQ1_CONFIG`, `RQ1_EVIDENCE`, `RQ1_RUN_ROOT`, and the real access-end time from the university's instructions. Keep evidence and output outside the checkout. Run `PYTHONPATH=src python3.11 -m rq1.cli university-gpu setup` to display the setup plan. `setup --apply --yes --approval-file ...` installs project-local dependencies and requires explicit operator authorization; it may need approved package access. Model downloads or system configuration changes require separate authorization and are not performed by these commands.

After all pilot inputs and human approvals exist, use the Gate and manifest `propose`, `validate`, and `freeze` commands in OPERATIONS. They inspect files and create evidence; they do not consume GPU. Run static preflight:

```bash
PYTHONPATH=src python3.11 -m rq1.cli university-gpu preflight --config "$RQ1_CONFIG" --manifest "$RQ1_EVIDENCE/pilot-frozen.json"
```

If you used the `rq1-university-pilot-001` run ID from OPERATIONS and the example config's `RQ1_RUN_OUTPUT_DIR`, set the matching paths with these shell-only commands:

```bash
export RQ1_RUN_ROOT="$RQ1_RUN_OUTPUT_DIR/rq1-university-pilot-001"
export RQ1_FROZEN_MANIFEST="$RQ1_EVIDENCE/pilot-frozen.json"
export RQ1_RELEVANCE_CSV="$RQ1_TASK_SKILL_RELEVANCE_CSV"
```

Set `RQ1_ACCESS_END_EPOCH` to the **actual access deadline** in Unix seconds. With a separate scientific/GPU approval bound to the manifest, the next command starts four private model services, checks the model identity, runs four approved smoke requests, verifies GPU placement, and starts at most four real pilot attempts. It consumes GPU time. A paused result after four attempts is expected and exits nonzero:

```bash
PYTHONPATH=src python3.11 -m rq1.cli university-gpu launch --config "$RQ1_CONFIG" --manifest "$RQ1_EVIDENCE/pilot-frozen.json" --approval-file "$RQ1_EVIDENCE/pilot-launch-approval.json" --yes --cutoff-epoch "$RQ1_ACCESS_END_EPOCH" --max-attempts 4
PYTHONPATH=src python3.11 -m rq1.cli university-gpu benchmark-report --run-root "$RQ1_RUN_ROOT" --manifest "$RQ1_EVIDENCE/pilot-frozen.json"
```

The report is file-only and needs four completed attempts on distinct workers. Compare its observed rate with the remaining clock time; it excludes setup, recovery, transfer and later variability. If the pilot can proceed, resume with the same inputs and approval, omitting `--max-attempts`:

```bash
PYTHONPATH=src python3.11 -m rq1.cli university-gpu resume --config "$RQ1_CONFIG" --manifest "$RQ1_EVIDENCE/pilot-frozen.json" --approval-file "$RQ1_EVIDENCE/pilot-launch-approval.json" --yes --cutoff-epoch "$RQ1_ACCESS_END_EPOCH"
( set -C; PYTHONPATH=src python3.11 -m rq1.cli university-gpu validate --run-root "$RQ1_RUN_ROOT" --manifest "$RQ1_EVIDENCE/pilot-frozen.json" > "$RQ1_EVIDENCE/pilot-validation.json" )
```

Resume consumes GPU; validation is file-only. Review pilot logs, protocol deviations and token matching. Only after the recorded pilot go decision and all confirmatory gates pass should authors create the confirmatory proposal, freeze its separate approval, set its config paths and launch it with the same deadline. Never substitute old records for missing primary cells or reduce the scientific design to fit the clock.

Monitor or request a cooperative stop from a second terminal; both commands are file-only:

```bash
PYTHONPATH=src python3.11 -m rq1.cli university-gpu status --run-root "$RQ1_RUN_ROOT" --manifest "$RQ1_FROZEN_MANIFEST"
PYTHONPATH=src python3.11 -m rq1.cli university-gpu stop --run-root "$RQ1_RUN_ROOT" --reason "access deadline"
```

Wait for active attempts to finish. After the process exits, inspect failures; `resume --retry-failed` creates new attempts only after operator review. A stale lock needs manual inspection. The code cannot guarantee completion before the deadline if an active episode stalls. Resume is designed for the original run location and worker identities; restoring to a different host, path or GPU UUID can fail its worker-configuration identity check and needs a reviewed migration procedure. Continued or remote execution requires explicit university permission.

For **each stopped run**, use a new destination and the exact frozen relevance file. These commands only read/copy files and create a new export directory:

```bash
PYTHONPATH=src python3.11 -m rq1.cli university-gpu progress-report --run-root "$RQ1_RUN_ROOT" --manifest "$RQ1_FROZEN_MANIFEST"
PYTHONPATH=src python3.11 -m rq1.cli university-gpu export-progress --run-root "$RQ1_RUN_ROOT" --manifest "$RQ1_FROZEN_MANIFEST" --relevance "$RQ1_RELEVANCE_CSV" --destination "$RQ1_EXPORT_DIR"
PYTHONPATH=src python3.11 -m rq1.cli university-gpu verify-export --destination "$RQ1_EXPORT_DIR"
```

For example, after choosing an external `RQ1_EXPORT_BASE`, create a dated destination name with `export RQ1_EXPORT_DIR="$RQ1_EXPORT_BASE/rq1-pilot-$(date -u +%Y%m%dT%H%M%SZ)"`. The exporter refuses an existing destination.

Copy the directory to an independent drive and verify the **copied** directory before leaving. On Windows, from the copied export's `analysis-src` directory:

```powershell
$env:PYTHONPATH = "."
py -3.11 -B -m rq1.cli university-gpu verify-export --destination ..
py -3.11 -B -m rq1.cli university-gpu progress-report --run-root ..\run --manifest ..\run\manifest.json
```

Use `-B` for commands from the export so Python does not create unlisted bytecode files inside it. For a complete run only, analyze from the same `analysis-src` directory, writing output outside the export:

```powershell
py -3.11 -B -m rq1.cli university-gpu analyze --results ..\results.jsonl --manifest ..\run\manifest.json --relevance ..\inputs\relevance.csv --output ..\..\rq1-analysis.json
py -3.11 -B -m rq1.cli university-gpu analysis-artifacts --analysis ..\..\rq1-analysis.json --output-dir ..\..\rq1-analysis-artifacts
```

Incomplete results are descriptive only. Before leaving, confirm no active workers, check the missing-unit and failed-attempt counts, verify the independent copy, retain the original run directory and approvals, and obtain permission for any later continuation.
