# Next university visit runbook

## Monday 2026-11-02: engineering feasibility path

Use this only for `rq1-feasibility-200-v1`: six existing valid-seen tasks, D0/D18, 12 engineering episodes, and a 200-total-action cap per episode. It does not activate Option B as a scientific amendment, change the frozen 50-action protocol, or authorize the 36-episode six-arm pilot. A separate candidate-bound engineering-only GPU approval is required before the four smoke requests or any episodes. Keep all run outputs, model traces, and raw logs outside Git.

### 1. Synchronize the feature branch without rewriting history

```bash
set -euo pipefail
cd ~/Hermes-Research-
test "$(git branch --show-current)" = "feat/rq1-university-4gpu"
test -z "$(git status --porcelain)"
git fetch origin refs/heads/feat/rq1-university-4gpu:refs/remotes/origin/feat/rq1-university-4gpu
LOCAL=$(git rev-parse HEAD)
REMOTE=$(git rev-parse refs/remotes/origin/feat/rq1-university-4gpu)
if git merge-base --is-ancestor "$REMOTE" "$LOCAL"; then
  echo "Local branch already contains the remote branch."
elif git merge-base --is-ancestor "$LOCAL" "$REMOTE"; then
  git merge --ff-only "$REMOTE"
else
  echo "STOP: feature histories diverged; reconcile without force-pushing."
  exit 1
fi
git status --short --branch
git log -1 --format='%H %s'
```

If the reviewed commit arrives as an approved bundle, follow the bundle verification and fast-forward instructions in `BUDGET_AMENDMENT_DECISION.md`. Never bypass SSH host verification.

### 2. Check the workstation and candidate matrix

Choose a new external evidence directory for this visit; do not reuse an old run path:

```bash
set -euo pipefail
cd ~/Hermes-Research-
export RQ1_EVIDENCE="$HOME/rq1-evidence/feasibility-2026-11-02"
mkdir -p "$(dirname "$RQ1_EVIDENCE")"
mkdir "$RQ1_EVIDENCE"
git status --short --branch
python3.11 --version
python3.11 -m pip check
nvidia-smi -L
nvidia-smi
ollama --version
ollama list
ollama ps
PYTHONPATH=src python3.11 -m rq1.cli university-gpu feasibility validate \
  --manifest university-gpu/prospective/feasibility-200-candidate.json
```

Confirm four distinct RTX 5000 Ada devices and inspect the already installed model. Do not install packages, pull models, or start inference during this check. The generic scientific `university-gpu check` may still reject the current digest because that command deliberately checks the historical model identity.

### 3. Reconcile data, model/tokenizer, and Accum-18

Point each variable to an existing asset. Do not download missing files:

```bash
set -euo pipefail
: "${RQ1_ALFWORLD_DATA_DIR:?Set the existing ALFWorld 0.4.2 data root}"
: "${RQ1_MODEL_MANIFEST:?Set the saved Ollama manifest file}"
: "${RQ1_MODEL_BLOBS:?Set its existing blobs directory}"
: "${RQ1_OLLAMA_MODELS_DIR:?Set the existing Ollama model directory}"
: "${RQ1_GEMMA_TOKENIZER_JSON:?Set the exact local tokenizer.json}"
: "${RQ1_TOKENIZER_PROVENANCE:?Set reviewed tokenizer source and reference-sample JSON}"
: "${RQ1_MODEL_TOKENIZER_LINK_EVIDENCE:?Set model-tokenizer linkage evidence}"
: "${RQ1_TOKENIZER_REVISION:?Set the verified full tokenizer revision}"
: "${RQ1_ACCUM18_LIBRARY:?Set the original frozen Accum-18 JSON}"
: "${RQ1_FROZEN_MANIFEST:?Set the authentic frozen university-GPU source manifest}"
export RQ1_ALFWORLD_DATA_DIR RQ1_MODEL_MANIFEST RQ1_MODEL_BLOBS RQ1_OLLAMA_MODELS_DIR
export RQ1_GEMMA_TOKENIZER_JSON RQ1_TOKENIZER_PROVENANCE RQ1_MODEL_TOKENIZER_LINK_EVIDENCE
export RQ1_TOKENIZER_REVISION RQ1_ACCUM18_LIBRARY RQ1_FROZEN_MANIFEST
export RQ1_FEASIBILITY_DATA_REPORT="$RQ1_EVIDENCE/data-reconciliation.json"
export RQ1_PROSPECTIVE_IDENTITY_REPORT="$RQ1_EVIDENCE/prospective-identity.json"
export RQ1_ACCUM18_REFERENCE="$RQ1_EVIDENCE/accum18-reference.json"
PYTHONPATH=src python3.11 -m rq1.cli university-gpu feasibility reconcile-data \
  --manifest university-gpu/prospective/feasibility-200-candidate.json \
  --data-root "$RQ1_ALFWORLD_DATA_DIR" --output "$RQ1_FEASIBILITY_DATA_REPORT"
PYTHONPATH=src python3.11 -m rq1.university_gpu.prospective_identity \
  --manifest "$RQ1_MODEL_MANIFEST" --blobs "$RQ1_MODEL_BLOBS" \
  --tokenizer "$RQ1_GEMMA_TOKENIZER_JSON" --provenance "$RQ1_TOKENIZER_PROVENANCE" \
  --link-evidence "$RQ1_MODEL_TOKENIZER_LINK_EVIDENCE" --output "$RQ1_PROSPECTIVE_IDENTITY_REPORT"
PYTHONPATH=src python3.11 -m rq1.cli university-gpu feasibility reference-library \
  --library "$RQ1_ACCUM18_LIBRARY" --frozen-manifest "$RQ1_FROZEN_MANIFEST" \
  --output "$RQ1_ACCUM18_REFERENCE"
```

Inspect the data report's three scoped digests, complete record differences, exclusions, and six selected-file hashes. `REVIEW_REQUIRED` means expected scoped identities reproduced and a human must review them; it does not mean they are equal or frozen. Bind the report hash in a new human data-review JSON. The prospective identity tool only emits `CANDIDATE_REVIEW_REQUIRED`; inspect all blob hashes and the tokenizer link, then bind its `content_sha256` in a new human identity-review JSON. The library reference command passes only if the exact 18-skill file hash and skill IDs match the genuine frozen source manifest. Stop if any source, reference, or expected digest is missing or mismatched.

### 4. Replay the exact expert witnesses under cap 200

This is a CPU ALFWorld oracle replay with no model or GPU inference. It preserves each exact candidate prefix, detour, checkpoint digest, post-detour digest, and unchanged route; raw logs go to a new external directory:

```bash
set -euo pipefail
export RQ1_FEASIBILITY_ORACLE_REPORT="$RQ1_EVIDENCE/oracle-replay.json"
PYTHONPATH=src python3.11 -m rq1.cli university-gpu feasibility replay-oracles \
  --manifest university-gpu/prospective/feasibility-200-candidate.json \
  --data-root "$RQ1_ALFWORLD_DATA_DIR" --evidence-dir "$RQ1_EVIDENCE/oracle-raw" \
  --output "$RQ1_FEASIBILITY_ORACLE_REPORT"
```

All six exact state/detour/action/source checks must pass. A failure blocks this batch; do not regenerate a checkpoint or replace a task. The transferred route lengths are 95, 13, 14, 11, 6, 146; detoured witnesses are 96, 14, 15, 12, 7, 147.

### 5. Complete reviews and freeze the engineering manifest

Copy the invalid-by-design templates to new evidence paths with `cp -n`, then have the named humans complete the data review, model/tokenizer review, and engineering approval. The engineering approval must bind the candidate manifest hash, set `approval_scope=engineering_feasibility_only`, `gpu_inference=true`, `scientific_run=false`, `not_for_scientific_claims=true`, and acknowledge prior pick-and-place outcome exposure. It is not a scientific approval of Option B.

```bash
set -euo pipefail
cp -n university-gpu/prospective/data-identity-review-template.json "$RQ1_EVIDENCE/data-review.json"
cp -n university-gpu/prospective/model-identity-review-template.json "$RQ1_EVIDENCE/model-review.json"
cp -n university-gpu/prospective/engineering-approval-template.json "$RQ1_EVIDENCE/engineering-approval.json"
export RQ1_FEASIBILITY_DATA_REVIEW="$RQ1_EVIDENCE/data-review.json"
export RQ1_PROSPECTIVE_IDENTITY_REVIEW="$RQ1_EVIDENCE/model-review.json"
export RQ1_HUGGINGFACE_CACHE="${RQ1_HUGGINGFACE_CACHE:?Set the existing pinned Sentence-BERT cache}"
export RQ1_FEASIBILITY_RUN_ROOT="${RQ1_FEASIBILITY_RUN_ROOT:?Set durable storage outside Git}"
cp -n university-gpu/feasibility-config.example.json "$RQ1_EVIDENCE/feasibility-config.json"
```

Edit that external config to use actual data/model/library/tokenizer paths and the exact tokenizer revision. Only after the human records and all prerequisites exist, freeze to a new external file:

```bash
export RQ1_FEASIBILITY_CONFIG="$RQ1_EVIDENCE/feasibility-config.json"
export RQ1_FEASIBILITY_MANIFEST="$RQ1_EVIDENCE/feasibility-200-frozen.json"
PYTHONPATH=src python3.11 -m rq1.cli university-gpu feasibility freeze \
  --candidate university-gpu/prospective/feasibility-200-candidate.json \
  --data-report "$RQ1_FEASIBILITY_DATA_REPORT" --data-review "$RQ1_EVIDENCE/data-review.json" \
  --oracle-report "$RQ1_FEASIBILITY_ORACLE_REPORT" \
  --identity-report "$RQ1_PROSPECTIVE_IDENTITY_REPORT" --identity-review "$RQ1_EVIDENCE/model-review.json" \
  --accum18-library "$RQ1_ACCUM18_LIBRARY" --accum18-reference "$RQ1_ACCUM18_REFERENCE" \
  --sbert-cache "$RQ1_HUGGINGFACE_CACHE" --tokenizer "$RQ1_GEMMA_TOKENIZER_JSON" \
  --engineering-approval "$RQ1_EVIDENCE/engineering-approval.json" \
  --run-id rq1-feasibility-200-v1-university-20261102 --output "$RQ1_FEASIBILITY_MANIFEST"
```

The frozen feasibility manifest must retain `option_b_author_decision_status=pending` and `scientific_evidence=false`.

### 6. Rehash all assets, then benchmark four GPUs without episodes

```bash
set -euo pipefail
PYTHONPATH=src python3.11 -m rq1.cli university-gpu feasibility preflight \
  --config "$RQ1_FEASIBILITY_CONFIG" --manifest "$RQ1_FEASIBILITY_MANIFEST"
PYTHONPATH=src python3.11 -m rq1.cli university-gpu feasibility benchmark \
  --config "$RQ1_FEASIBILITY_CONFIG" --manifest "$RQ1_FEASIBILITY_MANIFEST" \
  --approval-file "$RQ1_EVIDENCE/engineering-approval.json" --yes \
  --output "$RQ1_EVIDENCE/four-gpu-benchmark.json"
```

Preflight rehashes selected task files, both data-identity methods, model manifest/blobs, tokenizer/revision, Accum-18/reference, Sentence-BERT snapshot, review artifacts, and clean source commit without GPU use. The benchmark starts four isolated Ollama endpoints, checks exact prospective digest/Q4_K_M, sends four concurrent 8-token smoke requests, and verifies live process-to-GPU UUID placement. It starts zero ALFWorld episodes. Stop if any check fails.

### 7. Launch the 12 engineering episodes only after all gates pass

```bash
set -euo pipefail
: "${RQ1_ACCESS_END_EPOCH:?Set the real university GPU access deadline in Unix UTC seconds}"
PYTHONPATH=src python3.11 -m rq1.cli university-gpu feasibility launch \
  --config "$RQ1_FEASIBILITY_CONFIG" --manifest "$RQ1_FEASIBILITY_MANIFEST" \
  --approval-file "$RQ1_EVIDENCE/engineering-approval.json" --yes \
  --cutoff-epoch "$RQ1_ACCESS_END_EPOCH"
```

This runs exactly the 12 D0/D18 units. It is not the 36-episode scientific pilot. Retries receive distinct attempt IDs; completed authoritative results are immutable.

### 8. Monitor, stop, resume, export, and verify an independent backup

```bash
set -euo pipefail
export RQ1_RUN_ROOT="$RQ1_FEASIBILITY_RUN_ROOT/rq1-feasibility-200-v1-university-20261102"
PYTHONPATH=src python3.11 -m rq1.cli university-gpu feasibility status \
  --run-root "$RQ1_RUN_ROOT" --manifest "$RQ1_FEASIBILITY_MANIFEST"
nvidia-smi
PYTHONPATH=src python3.11 -m rq1.cli university-gpu feasibility stop \
  --run-root "$RQ1_RUN_ROOT" --reason "scheduled access cutoff"
PYTHONPATH=src python3.11 -m rq1.cli university-gpu feasibility resume \
  --config "$RQ1_FEASIBILITY_CONFIG" --manifest "$RQ1_FEASIBILITY_MANIFEST" \
  --approval-file "$RQ1_EVIDENCE/engineering-approval.json" --yes \
  --retry-failed --cutoff-epoch "$RQ1_ACCESS_END_EPOCH"
PYTHONPATH=src python3.11 -m rq1.cli university-gpu feasibility validate-run \
  --run-root "$RQ1_RUN_ROOT" --manifest "$RQ1_FEASIBILITY_MANIFEST"
PYTHONPATH=src python3.11 -m rq1.cli university-gpu feasibility merge \
  --run-root "$RQ1_RUN_ROOT" --manifest "$RQ1_FEASIBILITY_MANIFEST"
export RQ1_EXPORT="$RQ1_EVIDENCE/feasibility-export"
PYTHONPATH=src python3.11 -m rq1.cli university-gpu feasibility export-progress \
  --run-root "$RQ1_RUN_ROOT" --manifest "$RQ1_FEASIBILITY_MANIFEST" --destination "$RQ1_EXPORT"
PYTHONPATH=src python3.11 -m rq1.cli university-gpu feasibility verify-export --destination "$RQ1_EXPORT"
tar -czf "$RQ1_EXPORT.tar.gz" -C "$(dirname "$RQ1_EXPORT")" "$(basename "$RQ1_EXPORT")"
sha256sum "$RQ1_EXPORT.tar.gz" | tee "$RQ1_EXPORT.tar.gz.sha256"
```

Copy the archive and checksum to university-approved independent storage and verify the copied checksum and export manifest. A GitHub push is optional for run-data integrity; push failures must never interrupt, replace, or delete the local checkpoints and verified backup.

## Current status and remaining blockers

The 200-action pathway is engineering-only. It does not amend or replace the frozen 50-action scientific protocol, and it cannot be reported as the proposed six-arm, 36-episode pilot. The original candidate/checkpoint archive has already been verified; do not repeat that archive audit. The prospective Ollama digest `6114515d63c17436a7c0417d82820ac65ad643e2806c5a3c89cb62846436ed0b` remains user-reported until the workstation manifest, blobs, quantization, and tokenizer linkage are independently checked. It is not equivalent to historical digest `4eb23ef187e2c5462566d6a1d3bbbc2f1346d0b4327cbb66d58fffbcc9b2b05c`.

| Blocker | Owner / evidence needed |
|---|---|
| Rehash the original ALFWorld task/game files; reproduce both scoped identities and review every difference | Workstation operator + data reviewer |
| Replay all six exact prefix/detour/unchanged-continuation witnesses under cap 200 and verify terminal success | Workstation operator + task reviewer; raw logs stay outside Git |
| Verify local Ollama manifest and blobs, Q4_K_M, tokenizer revision/hash, sample tokenization, and model-tokenizer linkage | Model owner + independent reviewer |
| Recover the genuine frozen Accum-18 library and verify its hash/18 skill IDs against the frozen manifest | Workstation operator + research engineer |
| Rehash the pinned Sentence-BERT snapshot and validate Python/ALFWorld runtime without installing unapproved software | Workstation operator |
| Complete the candidate-bound engineering-only GPU approval; this is not scientific Option B approval | Authorized authors |
| Verify four isolated Ollama workers and actual process-to-GPU UUID placement; smoke test before episode scheduling | Workstation operator |
| Decide whether Option B is a scientific amendment, and separately authorize any later scientific pilot | Authors; not needed to describe the 12 episodes as engineering-only |
| Confirm WSL storage persistence, independent backup target/readback, and public artifact policy | University IT + repository owner/authors |

Human relevance labels are not a prerequisite for this D0/D18 engineering feasibility batch. No rating forms or human ratings are created by this runbook. The engineering batch must keep `scientific_evidence=false`, retain outcome exposure disclosure, and remain separate from all historical and scientific records.

The generic scientific preflight, launch, and resume paths continue to bind the historical model and 50-action protocol. Do not use those commands for the feasibility work. Use only the `university-gpu feasibility` commands above for this profile. No scientific inference is authorized here.

## Safe synchronization and offline handoff

The commands under step 1 are read/fast-forward-only and stop if local and remote histories diverge. Push only a reviewed commit using an authenticated, host-verified remote; never force push or suppress SSH host-key checks. If direct GitHub synchronization is unavailable, create a bundle only after committing the feature work, transfer the bundle and a separately verified SHA-256 through approved storage, then use the bundle fast-forward procedure in `BUDGET_AMENDMENT_DECISION.md`. Keep local run data and its independently verified archive primary; a GitHub push is not a data-integrity gate.
