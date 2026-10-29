# Operator workflow and approval records

Commands below assume the university Linux host, Python 3.11, and a checkout of the committed feature branch. `CONFIG` and all run outputs live outside the checkout. This document describes future operator steps; none of them installs or launches anything when merely read.

## Read-only checks and setup plan

```bash
PYTHONPATH=src python3.11 -m rq1.cli university-gpu check --json
PYTHONPATH=src python3.11 -m rq1.cli university-gpu setup
```

The setup plan shows the locked, project-local `uv sync --frozen --extra alfworld --extra retrieval` operation. It never offers `sudo`, driver changes, system package installation, or a model pull. Applying it requires `--apply --yes --approval-file PATH` with `approved=true`, `project_python_environment=true`, `project_dependencies=true`, an approver, and an approval reference. `uv` must already exist. No software setup was performed for this package.

## External input config

Copy `../config.example.json` to an access-controlled location and set each environment variable on the university host. Do not add data, model blobs, task routes, labels, secrets, or personal machine paths to this repository. `run_output_dir` must resolve outside the Git checkout; the launch gate rejects an in-checkout path.

Required frozen inputs:

- frozen task manifest and controlled-failure oracle definitions;
- Accum-18 library snapshot, 18 unique skill IDs and exact skill text;
- task-skill CSV with exactly `task_id,skill_id,rater_a,rater_b,adjudicated`, one row for every one of 18 skills per task;
- valid-seen similarity score CSV with exactly `task_id,skill_id,cosine_similarity` for Gate calibration, adjudicated calibration labels with exactly `task_id,skill_id,adjudicated`, and the frozen calibration JSON; all three are checked against each other and hashed into the manifest;
- local pinned Gemma `tokenizer.json`, the pinned tokenizer revision, and an author-reviewed neutral filler unit;
- ALFWorld 0.4.2 data directory, Sentence-BERT cache, existing Ollama model directory, and external run output directory.

All input files are SHA-256-bound into the manifest. The ALFWorld data directory is frozen as a sorted relative-path/file-hash tree digest; symlinks are rejected. The tokenizer revision string and tokenizer JSON hash are both bound. Frozen task paths and outcome records remain read-only. Missing files, incomplete ratings, hashes, or a mismatched dependency/model identity block rather than being reconstructed.

## Calibrate and freeze Gate

Prepare a UTF-8 JSON array of exactly six reviewed pilot contexts. Each object has `task_id`, `task_instruction`, `observation`, `inventory` (a JSON string array), and `failure_message`. Use only observed post-detour information, with the frozen canonical failure message. The `gate-scores` command below checks valid-seen task identity, the frozen post-detour observation, complete rater coverage, and the local Sentence-BERT snapshot hash. It performs **CPU embedding inference** with local files only; it does not call Gemma or the GPUs. The exact task instruction and inventory still need human review; compare each recorded query hash in `SCORE-EVIDENCE.json` with observed pilot retrieval query hashes before confirmatory approval.

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

The Gate proposal fits only the predeclared valid-seen score/label rule and includes source hashes. It is a proposal until human review and approval:

```bash
PYTHONPATH=src python3.11 -m rq1.cli university-gpu gate-calibration propose \
  --scores "$RQ1_GATE_SCORES" --labels "$RQ1_GATE_LABELS" \
  --output "$RQ1_EVIDENCE/gate-proposal.json"
PYTHONPATH=src python3.11 -m rq1.cli university-gpu gate-calibration validate \
  --calibration "$RQ1_EVIDENCE/gate-proposal.json"
PYTHONPATH=src python3.11 -m rq1.cli university-gpu gate-calibration freeze \
  --calibration "$RQ1_EVIDENCE/gate-proposal.json" \
  --approval-file "$RQ1_EVIDENCE/gate-approval.json" \
  --output "$RQ1_EVIDENCE/gate-frozen.json"
export RQ1_GATE_CALIBRATION_FROZEN="$RQ1_EVIDENCE/gate-frozen.json"
```

Gate approval JSON binds the exact `calibration_sha256` and includes `approved=true`, `approved_by`, and `approval_reference`. Calibration quality on a small validation split is itself uncertain; report its metrics and do not tune the threshold against recovery outcomes.

## Propose and freeze a manifest

First make a reviewed config file as described above. The repository must be committed and clean. Store proposal/freeze outputs outside the checkout:

```bash
PYTHONPATH=src python3.11 -m rq1.cli university-gpu manifest propose \
  --stage pilot --run-id rq1-university-pilot-001 \
  --config "$RQ1_CONFIG" --output "$RQ1_EVIDENCE/pilot-proposal.json"
PYTHONPATH=src python3.11 -m rq1.cli university-gpu manifest validate \
  --manifest "$RQ1_EVIDENCE/pilot-proposal.json"
PYTHONPATH=src python3.11 -m rq1.cli university-gpu manifest freeze \
  --manifest "$RQ1_EVIDENCE/pilot-proposal.json" \
  --approval-file "$RQ1_EVIDENCE/pilot-approval.json" \
  --output "$RQ1_EVIDENCE/pilot-frozen.json"
```

Manifest approval JSON includes `approved=true`, `approved_by`, `approval_reference`, and the exact `manifest_sha256`. Confirmatory approval must also set `previous_task_outcomes_acknowledged`, `eligibility_verified`, `blinded_relevance_verified`, `leakage_review_complete`, `pilot_go_approved`, and `precision_power_review_complete` to true, and include `power_report_sha256` and `pilot_validation_sha256`. Add `power_report_path` and `pilot_validation_path` to the external confirmatory config; preflight checks both hashes and that pilot validation reports 36/36 passed. These are human attestations, not proof of the underlying review. Freezing a manifest creates a new file and refuses to overwrite one.

Confirmatory generation/launch is gated by complete valid-unseen oracle/task coverage, both raters and adjudication for every candidate skill on every task, valid-seen-only Gate calibration, explicit historical-outcome exposure acknowledgment, and a separate prespecified task-cluster precision/power simulation approved by the authors. The current code does not synthesize or assume the power inputs.

## Setup and preflight

`preflight` without flags is static: source/manifest/input hashes, Python dependencies, ALFWorld data, SBERT snapshot, local Ollama executable/model directory, four GPU identities, memory, disk and ports. The command returns `STATIC_PASS` to state that isolated model-endpoint identity has not yet been checked. `--check-endpoints` requires four live endpoints, each with the frozen digest and quantization. The launch path starts those isolated endpoints itself, checks the exact model, runs a separately approved small smoke request on each, then verifies process-to-GPU UUID placement before assigning any episode.

```bash
PYTHONPATH=src python3.11 -m rq1.cli university-gpu preflight \
  --config "$RQ1_CONFIG" --manifest "$RQ1_EVIDENCE/pilot-frozen.json"
```

## Launch, stop, resume, validate, export

Scientific launch approval is separate from manifest freeze. It includes `approved_by`, `approval_reference`, exact `manifest_sha256`, `gpu_inference=true`, and `scientific_run=true`. For confirmatory runs it also includes `previous_task_outcomes_acknowledged=true`. This approval covers the four smoke-test inferences that must precede the experimental units.

```bash
PYTHONPATH=src python3.11 -m rq1.cli university-gpu launch \
  --config "$RQ1_CONFIG" --manifest "$RQ1_EVIDENCE/pilot-frozen.json" \
  --approval-file "$RQ1_EVIDENCE/pilot-launch-approval.json" --yes --cutoff-epoch "$RQ1_ACCESS_END_EPOCH"
```

The launch requires exactly four physical NVIDIA devices, one UUID per isolated Ollama server, four unique endpoints, sufficient free memory, the frozen local model, and exact source/input identities. Ollama documents `CUDA_VISIBLE_DEVICES` as the GPU-selection control and recommends UUIDs over numeric IDs because ordering can vary ([Ollama GPU selection](https://github.com/ollama/ollama/blob/main/docs/gpu.mdx)). The package verifies live process-to-UUID placement before assigning episodes. It never substitutes a model or pulls a missing one. Confirmatory launch is a scientific/GPU operation and is not authorized merely by having prepared the package.

The coordinator writes immutable attempt records, heartbeats, per-unit authoritative results, and append-only event files. A retry is a new physical attempt and never overwrites prior attempt logs. If the host stops unexpectedly, resume only after reviewing `status`; completed authoritative units are retained and unfinished units receive new attempts.

```bash
PYTHONPATH=src python3.11 -m rq1.cli university-gpu status --run-root "$RQ1_RUN_ROOT" --manifest "$RQ1_FROZEN_MANIFEST"
PYTHONPATH=src python3.11 -m rq1.cli university-gpu stop --run-root "$RQ1_RUN_ROOT" --reason "operator cutoff"
PYTHONPATH=src python3.11 -m rq1.cli university-gpu resume --config "$RQ1_CONFIG" --manifest "$RQ1_FROZEN_MANIFEST" --approval-file "$RQ1_LAUNCH_APPROVAL" --yes --retry-failed --cutoff-epoch "$RQ1_ACCESS_END_EPOCH"
PYTHONPATH=src python3.11 -m rq1.cli university-gpu validate --run-root "$RQ1_RUN_ROOT" --manifest "$RQ1_FROZEN_MANIFEST"
PYTHONPATH=src python3.11 -m rq1.cli university-gpu merge --run-root "$RQ1_RUN_ROOT" --manifest "$RQ1_FROZEN_MANIFEST"
PYTHONPATH=src python3.11 -m rq1.cli university-gpu export --run-root "$RQ1_RUN_ROOT" --manifest "$RQ1_FROZEN_MANIFEST" --destination "$RQ1_EXPORT_DIR"
```

`stop` lets active episodes finish and assigns no new units. `--cutoff-epoch` is the actual access deadline as a Unix UTC epoch; assignment stops 30 minutes beforehand. Active episodes are not killed by the cutoff and can exceed that margin, so watch the clock. Resumption is at the episode-unit boundary only: an interrupted episode is not restored mid-trajectory. Physical attempts are at-least-once across failure/retry; the package makes no exactly-once execution claim. Use `export-progress` after the run stops to preserve incomplete work; `export` and `merge` still require completion.

## Offline analysis

After a complete validated run has been merged, run the paired/task-cluster analysis. It keeps family results descriptive, bootstraps tasks rather than episodes, and separately reports adjudicated and original-rater retrieval-noise sensitivity. It does not replace the historical frozen analysis or establish sample-size adequacy.

```bash
PYTHONPATH=src python3.11 -m rq1.cli university-gpu analyze \
  --results "$RQ1_RUN_ROOT/merged-results.jsonl" \
  --manifest "$RQ1_FROZEN_MANIFEST" --relevance "$RQ1_RELEVANCE_CSV" \
  --output "$RQ1_EVIDENCE/exploratory-analysis.json"
```
