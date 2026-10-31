# Next university visit runbook

## Monday 2026-11-02: engineering feasibility path

Use this only for `rq1-feasibility-200-v1`: six existing valid-seen tasks, D0/D18, 12 engineering episodes, and a 200-total-action cap per episode. It does not activate Option B as a scientific amendment, change the frozen 50-action protocol, or authorize the 36-episode six-arm pilot. A separate candidate-bound engineering-only GPU approval is required before the four smoke requests or any episodes. Keep all run outputs, model traces, and raw logs outside Git.

### 1. Synchronize the feature branch without rewriting history

```bash
set -euo pipefail
cd ~/Hermes-Research-
export RQ1_PY="$PWD/.venv/bin/python"
export PYTHONPATH="$PWD/src"
test -x "$RQ1_PY"
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
: "${RQ1_APPROVED_SOURCE_COMMIT:?Set the reviewed full commit from the handoff}"
test "$(git rev-parse HEAD)" = "$RQ1_APPROVED_SOURCE_COMMIT"
```

If the reviewed commit arrives as an approved bundle, follow the bundle verification and fast-forward instructions in `BUDGET_AMENDMENT_DECISION.md`. Never bypass SSH host verification.

### 2. Check the workstation and candidate matrix

Choose a new external evidence directory for this visit; do not reuse an old run path:

```bash
set -euo pipefail
cd ~/Hermes-Research-
export RQ1_PY="$PWD/.venv/bin/python"
export PYTHONPATH="$PWD/src"
test -x "$RQ1_PY"
export RQ1_EVIDENCE="$HOME/rq1-evidence/feasibility-2026-11-02"
mkdir -p "$(dirname "$RQ1_EVIDENCE")"
mkdir "$RQ1_EVIDENCE"
git status --short --branch
"$RQ1_PY" -c 'import sys; assert sys.version_info[:2] == (3,11); import rq1, alfworld, sentence_transformers, tokenizers; print(sys.executable); print(rq1.__file__)'
"$RQ1_PY" -m pip check
nvidia-smi -L
nvidia-smi
ollama --version
ollama list
ollama ps
PYTHONPATH=src "$RQ1_PY" -m rq1.cli university-gpu feasibility validate \
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
: "${RQ1_HISTORICAL_ARCHIVE:?Set the authentic archived evaluation tar.gz}"
: "${RQ1_HISTORICAL_ARCHIVE_MANIFEST:?Set its authentic member ledger}"
export RQ1_ALFWORLD_DATA_DIR RQ1_MODEL_MANIFEST RQ1_MODEL_BLOBS RQ1_OLLAMA_MODELS_DIR
export RQ1_GEMMA_TOKENIZER_JSON RQ1_TOKENIZER_PROVENANCE RQ1_MODEL_TOKENIZER_LINK_EVIDENCE
export RQ1_TOKENIZER_REVISION RQ1_ACCUM18_LIBRARY RQ1_HISTORICAL_ARCHIVE RQ1_HISTORICAL_ARCHIVE_MANIFEST
export RQ1_FEASIBILITY_DATA_REPORT="$RQ1_EVIDENCE/data-reconciliation.json"
export RQ1_PROSPECTIVE_IDENTITY_REPORT="$RQ1_EVIDENCE/prospective-identity.json"
export RQ1_ACCUM18_REFERENCE="$RQ1_EVIDENCE/accum18-reference.json"
PYTHONPATH=src "$RQ1_PY" -m rq1.cli university-gpu feasibility reconcile-data \
  --manifest university-gpu/prospective/feasibility-200-candidate.json \
  --data-root "$RQ1_ALFWORLD_DATA_DIR" --output "$RQ1_FEASIBILITY_DATA_REPORT"
PYTHONPATH=src "$RQ1_PY" -m rq1.university_gpu.prospective_identity \
  --manifest "$RQ1_MODEL_MANIFEST" --blobs "$RQ1_MODEL_BLOBS" \
  --tokenizer "$RQ1_GEMMA_TOKENIZER_JSON" --provenance "$RQ1_TOKENIZER_PROVENANCE" \
  --link-evidence "$RQ1_MODEL_TOKENIZER_LINK_EVIDENCE" --output "$RQ1_PROSPECTIVE_IDENTITY_REPORT"
PYTHONPATH=src "$RQ1_PY" -m rq1.cli university-gpu feasibility reference-library \
  --library "$RQ1_ACCUM18_LIBRARY" --archive "$RQ1_HISTORICAL_ARCHIVE" \
  --archive-manifest "$RQ1_HISTORICAL_ARCHIVE_MANIFEST" \
  --output "$RQ1_ACCUM18_REFERENCE"
```

Inspect the data report's three scoped digests, complete record differences, exclusions, and six selected-file hashes. `REVIEW_REQUIRED` means expected scoped identities reproduced and a human must review them; it does not mean they are equal or frozen. Bind the report hash in a new human data-review JSON. The prospective identity tool only emits `CANDIDATE_REVIEW_REQUIRED`; inspect all blob hashes and the tokenizer link, then bind its `content_sha256` in a new human identity-review JSON. The library reference command passes only if the exact library file hash, 18 skill IDs and text match the authentic historical archive/member ledger. Stop if any source, reference, or expected digest is missing or mismatched.

Transfer the original evaluation archive, its member ledger, and the exact library JSON to the workstation. The local historical archive was read and checked as SHA-256 `d52ed82e01a073f8c3382711d03f65916cb41ca89bc0b773e6845817bd37e4fd`; its ledger was `d9c379b3507855a06780f16d47ab2cd8396bc1bf4d670a0bb674d24f91acd2f8`. The archive contains one regular member `rq1-evaluation-360-final/repo/artifacts/evaluation-libraries/libraries-0030f89e76c6bc4c.json`, 22,875 bytes, SHA-256 `e01d3a569ee8373d8cc235c4343f2b52f92c9bf41751b5d3e435ced70648e1f5`; `RQ1_ACCUM18_LIBRARY` must contain those exact bytes. These are local checks; the transferred university copies still require verification there. Keep all three source files available for freeze, preflight and resume.

### 4. Replay the exact expert witnesses under cap 200

This is a CPU ALFWorld oracle replay with no model or GPU inference. It preserves each exact candidate prefix, detour, checkpoint digest, post-detour digest, and unchanged route; raw logs go to a new external directory:

```bash
set -euo pipefail
export RQ1_FEASIBILITY_ORACLE_REPORT="$RQ1_EVIDENCE/oracle-replay.json"
PYTHONPATH=src "$RQ1_PY" -m rq1.cli university-gpu feasibility replay-oracles \
  --manifest university-gpu/prospective/feasibility-200-candidate.json \
  --data-root "$RQ1_ALFWORLD_DATA_DIR" --evidence-dir "$RQ1_EVIDENCE/oracle-raw" \
  --output "$RQ1_FEASIBILITY_ORACLE_REPORT"
```

All six exact state/detour/action/source checks must pass. A failure blocks this batch; do not regenerate a checkpoint or replace a task. The transferred route lengths are 95, 13, 14, 11, 6, 146; detoured witnesses are 96, 14, 15, 12, 7, 147.

### 5. Complete reviews and freeze the engineering manifest

Copy the invalid-by-design templates to new evidence paths with `cp -n`, then have authorized humans complete and review the data, model/tokenizer and engineering records. The engineering record must bind the candidate manifest hash, exact run ID, profile, model tag/digest/quantization, 200-action cap and `["D0", "D18"]`; it must identify the approver, approval provenance and reference, acknowledge prior exposure, and have `test_only=false`, `gpu_inference=true`, `scientific_run=false` and `not_for_scientific_claims=true`. `approved_by` and `approver_id` must match. The supplied template is deliberately invalid and test-only; only the authorized human may replace its fields with genuine approval. This does not authorize Option B scientifically.

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

An authorized reviewer must independently inspect the completed record and send its canonical SHA-256 through an independently authenticated channel. The operator compares that value to the following locally computed digest and sets the environment variable to the **independently received** value, never automatically to the local output:

```bash
"$RQ1_PY" -c 'import json,sys; from rq1.university_gpu.authorization import approval_digest; print(approval_digest(json.load(open(sys.argv[1], encoding="utf-8"))))' "$RQ1_EVIDENCE/engineering-approval.json"
: "${RQ1_TRUSTED_ENGINEERING_APPROVAL_SHA256:?Set the independently received approval digest}"
```

Edit that external config to use actual data/model/library/tokenizer paths and the exact tokenizer revision. Only after the human records and all prerequisites exist, freeze to a new external file:

```bash
export RQ1_FEASIBILITY_CONFIG="$RQ1_EVIDENCE/feasibility-config.json"
export RQ1_FEASIBILITY_MANIFEST="$RQ1_EVIDENCE/feasibility-200-frozen.json"
PYTHONPATH=src "$RQ1_PY" -m rq1.cli university-gpu feasibility freeze \
  --candidate university-gpu/prospective/feasibility-200-candidate.json \
  --data-report "$RQ1_FEASIBILITY_DATA_REPORT" --data-review "$RQ1_EVIDENCE/data-review.json" \
  --oracle-report "$RQ1_FEASIBILITY_ORACLE_REPORT" \
  --identity-report "$RQ1_PROSPECTIVE_IDENTITY_REPORT" --identity-review "$RQ1_EVIDENCE/model-review.json" \
  --accum18-library "$RQ1_ACCUM18_LIBRARY" --accum18-reference "$RQ1_ACCUM18_REFERENCE" \
  --sbert-cache "$RQ1_HUGGINGFACE_CACHE" --tokenizer "$RQ1_GEMMA_TOKENIZER_JSON" \
  --engineering-approval "$RQ1_EVIDENCE/engineering-approval.json" \
  --run-id rq1-feasibility-200-v1-university-20261102 --output "$RQ1_FEASIBILITY_MANIFEST"
```

The frozen feasibility manifest must retain `option_b_author_decision_status=pending` and `scientific_evidence=false`. Have the authorized reviewer inspect the complete frozen manifest and transmit its `manifest_sha256` through an independent authenticated channel. Set `RQ1_TRUSTED_ENGINEERING_MANIFEST_SHA256` to that received value. GPU benchmark, launch and resume require both pins and `--yes`; preflight and read-only diagnostics do not. A digest entered by the same person who wrote an unreviewed record does not establish human authorization. If either independently verified value is unavailable, stop before GPU use.

```bash
"$RQ1_PY" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["manifest_sha256"])' "$RQ1_FEASIBILITY_MANIFEST"
: "${RQ1_TRUSTED_ENGINEERING_MANIFEST_SHA256:?Set the independently received frozen manifest digest}"
```

### 6. Rehash all assets, then benchmark four GPUs without episodes

```bash
set -euo pipefail
PYTHONPATH=src "$RQ1_PY" -m rq1.cli university-gpu feasibility preflight \
  --config "$RQ1_FEASIBILITY_CONFIG" --manifest "$RQ1_FEASIBILITY_MANIFEST"
PYTHONPATH=src "$RQ1_PY" -m rq1.cli university-gpu feasibility benchmark \
  --config "$RQ1_FEASIBILITY_CONFIG" --manifest "$RQ1_FEASIBILITY_MANIFEST" \
  --approval-file "$RQ1_EVIDENCE/engineering-approval.json" --yes \
  --output "$RQ1_EVIDENCE/four-gpu-benchmark.json"
```

Preflight rehashes selected task files, both data-identity methods, model manifest/blobs, tokenizer/revision, Accum-18/reference, Sentence-BERT snapshot, review artifacts, and clean source commit without GPU use. The benchmark starts four isolated Ollama endpoints, checks exact prospective digest/Q4_K_M, sends four concurrent 8-token smoke requests, and verifies live process-to-GPU UUID placement. It starts zero ALFWorld episodes. Stop if any check fails.

### 7. Launch the 12 engineering episodes only after all gates pass

```bash
set -euo pipefail
: "${RQ1_ACCESS_END_EPOCH:?Set the real university GPU access deadline in Unix UTC seconds}"
PYTHONPATH=src "$RQ1_PY" -m rq1.cli university-gpu feasibility launch \
  --config "$RQ1_FEASIBILITY_CONFIG" --manifest "$RQ1_FEASIBILITY_MANIFEST" \
  --approval-file "$RQ1_EVIDENCE/engineering-approval.json" --yes \
  --cutoff-epoch "$RQ1_ACCESS_END_EPOCH"
```

This runs exactly the 12 D0/D18 units. It is not the 36-episode scientific pilot. Retries receive distinct attempt IDs; completed authoritative results are immutable.

For each completed unit the coordinator atomically publishes `results/units/<unit-key>.json`, records the completion event, then copies that result and its closed attempt traces to the create-only `backups/units/<unit-key>/` checkpoint. It verifies the checkpoint and flushes files and directories where the platform supports this. This copies each unit's traces once; periodic full-run snapshots are published only at newly accepted 10-unit boundaries and at final completion. If the final snapshot fails after all units are complete, resume retries it by publishing a new create-only full snapshot. Before publishing, the coordinator verifies an existing complete bundle for the same run and completed count against its manifest inventory and the current source files: a valid current bundle is reused, a valid but stale bundle is left intact beside a new create-only bundle, and a partial, malformed or corrupt same-count bundle fails closed. An intermediate 10-unit bundle is not reconstructed retroactively if its boundary was crossed while resuming. A result published immediately before a coordinator interruption can exist without its checkpoint; resume verifies that authority and creates the missing checkpoint before assigning further work. A concurrent worker may have physically finished before the coordinator receives and publishes its result; inspect that nonterminal attempt before retrying, because physical work can be repeated. The checkpoint is local, is not an independent backup, and cannot survive loss of the same workstation storage.

### 8. Monitor and interrupt cooperatively

These are separate operator procedures. Do not paste monitor, stop and resume as one script.

```bash
export RQ1_RUN_ROOT="$RQ1_FEASIBILITY_RUN_ROOT/rq1-feasibility-200-v1-university-20261102"
"$RQ1_PY" -m rq1.cli university-gpu feasibility status --output-root "$RQ1_FEASIBILITY_RUN_ROOT" --run-root "$RQ1_RUN_ROOT" --manifest "$RQ1_FEASIBILITY_MANIFEST"
nvidia-smi
```

Inspect `durability_status`, `completion_events_without_results` and `local_unit_checkpoints`: `verified` lists valid local checkpoints, while `missing`, `invalid`, `orphaned`, or a completion event without its result require attention. `evidence_anchor_status` separately identifies units with supplied external anchors and unanchored units. The coordinator's launch/resume result also reports `unit_checkpoint_status`, `backup_status` and `full_snapshot_status`; `durability_failed` means do not treat local checkpointing as complete.

To interrupt, request STOP, then wait for the coordinator to return and active work to finish:

```bash
"$RQ1_PY" -m rq1.cli university-gpu feasibility stop --output-root "$RQ1_FEASIBILITY_RUN_ROOT" --run-root "$RQ1_RUN_ROOT" --reason "scheduled access cutoff"
```

The cutoff stops assignments 30 minutes early; it does not forcibly terminate active episodes. Reject nonfinite/expired deadlines. Set a measured reserve for active work and backup; a one-hour response timeout can exceed the assignment reserve. If time is insufficient, stop earlier. WSL/RDP persistence and process ownership remain university-only checks. Never kill another user's processes.

### 9. Complete run versus partial/failed run

For a complete run only:

```bash
"$RQ1_PY" -m rq1.cli university-gpu feasibility validate-run --output-root "$RQ1_FEASIBILITY_RUN_ROOT" --run-root "$RQ1_RUN_ROOT" --manifest "$RQ1_FEASIBILITY_MANIFEST"
"$RQ1_PY" -m rq1.cli university-gpu feasibility merge --output-root "$RQ1_FEASIBILITY_RUN_ROOT" --run-root "$RQ1_RUN_ROOT" --manifest "$RQ1_FEASIBILITY_MANIFEST"
```

For interrupted runs, failed workers or backup failures, skip complete-run validation/merge and proceed directly to the verified partial export below after all writers stop. Inspect execution and backup status separately. Preserve failed/partial attempts. Missing authoritative logs or corrupt accepted results must block export; partial export is not permission to suppress corruption.

If a local checkpoint is missing but its result and traces validate, resume can reconstruct it before new assignments. If a published checkpoint is corrupt, conflicts with authority, or has no matching result, do not overwrite or delete it: stop, preserve a partial export and the checkpoint directory in separate storage, and obtain a reviewed recovery procedure. A failed disk, I/O flush or directory-fsync error must be resolved before relying on another local checkpoint. Partial exports preserve authoritative result and trace files but omit local `backups/`; their `source_checkpoint_status_at_export` reports what was observed, not independent verification.

For either complete or incomplete stopped runs, create per-unit evidence records in a separate directory. These records bind the frozen manifest, run, unit, attempt, authoritative result and attempt files. Create-only publication prevents this tool overwriting an existing record; the directory is not independently protected merely because it is outside the run. Copy records to university-approved independent storage and verify that copy. Only set `RQ1_EVIDENCE_ANCHOR_DIR` to a directory whose provenance and protection the operator has independently verified. If no such directory exists, leave the variable unset and describe the result as internally validated. Keep the independently protected anchor records across resume; never replace conflicting records.

```bash
: "${RQ1_ANCHOR_STAGING:?Set a new anchor directory outside the run root}"
"$RQ1_PY" -m rq1.university_gpu.evidence_anchor create --output-root "$RQ1_FEASIBILITY_RUN_ROOT" --run-root "$RQ1_RUN_ROOT" --manifest "$RQ1_FEASIBILITY_MANIFEST" --anchor-dir "$RQ1_ANCHOR_STAGING"
"$RQ1_PY" -m rq1.university_gpu.evidence_anchor verify --run-root "$RQ1_RUN_ROOT" --manifest "$RQ1_FEASIBILITY_MANIFEST" --anchor-dir "$RQ1_ANCHOR_STAGING"
```

After a separate approved transfer, verify using the independently stored directory, inspect `unanchored_units`, and only then set `RQ1_EVIDENCE_ANCHOR_DIR` to that verified directory. Repeat for newly completed units and protect new records separately. An absent anchor remains unanchored; an unavailable configured directory or any conflict blocks validation. A local hash or a hash stored beside the run proves no independent authentication.

```bash
: "${RQ1_INDEPENDENT_ANCHORS:?Set the independently stored and protected anchor directory}"
"$RQ1_PY" -m rq1.university_gpu.evidence_anchor verify --run-root "$RQ1_RUN_ROOT" --manifest "$RQ1_FEASIBILITY_MANIFEST" --anchor-dir "$RQ1_INDEPENDENT_ANCHORS"
export RQ1_EVIDENCE_ANCHOR_DIR="$RQ1_INDEPENDENT_ANCHORS"
```

Choose a new export destination every time:

```bash
export RQ1_EXPORT="$RQ1_EVIDENCE/feasibility-export-$(date -u +%Y%m%dT%H%M%SZ)"
test ! -e "$RQ1_EXPORT"
test ! -e "$RQ1_EXPORT.tar.gz"
"$RQ1_PY" -m rq1.cli university-gpu feasibility export-progress --output-root "$RQ1_FEASIBILITY_RUN_ROOT" --run-root "$RQ1_RUN_ROOT" --manifest "$RQ1_FEASIBILITY_MANIFEST" --destination "$RQ1_EXPORT"
"$RQ1_PY" -m rq1.cli university-gpu feasibility verify-export --destination "$RQ1_EXPORT"
(set -o noclobber; tar -czf - -C "$(dirname "$RQ1_EXPORT")" "$(basename "$RQ1_EXPORT")" > "$RQ1_EXPORT.tar.gz")
(set -o noclobber; sha256sum "$RQ1_EXPORT.tar.gz" > "$RQ1_EXPORT.tar.gz.sha256")
```

Copy archive and checksum to university-approved independent storage and verify the copied checksum and export manifest there. Local checkpoint/export success does not prove independent-copy verification. GitHub is supplementary backup.
After transfer, verify the extracted copy against the independently protected anchors using `evidence_anchor verify --run-root <copied-export>/run --manifest <copied-export>/run/manifest.json --anchor-dir "$RQ1_INDEPENDENT_ANCHORS"`; also run `feasibility verify-export --destination <copied-export>`. Retain a separately protected anchor-directory snapshot for each partial export: anchors for units completed later reference results absent from an earlier export and will make that earlier verification fail. Inspect anchor coverage and the export's `independent_protection` and `unanchored_units` fields. Never treat anchor records merely copied into the export as an independent trust source.

### 10. Resume after a worker failure or interruption

After preserving a verified partial export, inspect failed attempts and ports/owned processes. If a `RUN.lock.takeover` marker remains, stop and inspect ownership and the interrupted takeover; do not delete or replace it automatically. Resolve the actual infrastructure error, recheck unchanged assets, set a new finite authorized deadline and explicitly resume.

Use `--retry-failed` after reviewing the latest event for every affected unit. It is required for two cases. For a terminal `attempt_failed` event, retry is allowed only when that event records **both** `safe_to_continue: true` and `mutation_state_known: true` (a verified pre-mutation or fully known failure); if either field is missing or false, do not use `--retry-failed`: stop and obtain a reviewed manual recovery procedure. For a nonterminal attempt (an `attempt_started` with no result or terminal event), the attempt may have physically run, so ordinary `resume` fails closed with a `SchedulerError` before starting any worker; inspect the attempt directory and the actual process state, and use `--retry-failed` only as an explicit operator decision after that review. Before using `--retry-failed`, verify that every owned process from the interrupted attempt is actually stopped. The coordinator fails closed before starting any new worker when the applicable condition is not met.

```bash
: "${RQ1_ACCESS_END_EPOCH:?Set the actual authorized future Unix UTC deadline}"
"$RQ1_PY" -m rq1.cli university-gpu feasibility resume \
  --config "$RQ1_FEASIBILITY_CONFIG" --manifest "$RQ1_FEASIBILITY_MANIFEST" \
  --approval-file "$RQ1_EVIDENCE/engineering-approval.json" --yes \
  --retry-failed --cutoff-epoch "$RQ1_ACCESS_END_EPOCH"
```

Retries have new attempts and may consume more physical work; accepted results are never overwritten. Do not discard checkpoints, edit frozen inputs, or fabricate approvals to unblock resume.

## Current status and remaining blockers

The 200-action pathway is engineering-only. It does not amend or replace the frozen 50-action scientific protocol, and it cannot be reported as the proposed six-arm, 36-episode pilot. The original candidate/checkpoint archive has already been verified; do not repeat that archive audit. The prospective Ollama digest `6114515d63c17436a7c0417d82820ac65ad643e2806c5a3c89cb62846436ed0b` remains user-reported until the workstation manifest, blobs, quantization, and tokenizer linkage are independently checked. It is not equivalent to historical digest `4eb23ef187e2c5462566d6a1d3bbbc2f1346d0b4327cbb66d58fffbcc9b2b05c`.

| Blocker | Owner / evidence needed |
|---|---|
| Rehash the original ALFWorld task/game files; reproduce both scoped identities and review every difference | Workstation operator + data reviewer |
| Replay all six exact prefix/detour/unchanged-continuation witnesses under cap 200 and verify terminal success | Workstation operator + task reviewer; raw logs stay outside Git |
| Verify local Ollama manifest and blobs, Q4_K_M, tokenizer revision/hash, sample tokenization, and model-tokenizer linkage | Model owner + independent reviewer |
| Recover the genuine frozen Accum-18 library and verify its hash/18 skill IDs against the authentic archive and member ledger | Workstation operator + research engineer |
| Rehash the pinned Sentence-BERT snapshot and validate Python/ALFWorld runtime without installing unapproved software | Workstation operator |
| Complete the candidate-bound engineering-only GPU approval; this is not scientific Option B approval | Authorized authors |
| Verify four isolated Ollama workers and actual process-to-GPU UUID placement; smoke test before episode scheduling | Workstation operator |
| Decide whether Option B is a scientific amendment, and separately authorize any later scientific pilot | Authors; not needed to describe the 12 episodes as engineering-only |
| Confirm WSL storage persistence, independent backup target/readback, and public artifact policy | University IT + repository owner/authors |

Human relevance labels are not a prerequisite for this D0/D18 engineering feasibility batch. No rating forms or human ratings are created by this runbook. The engineering batch must keep `scientific_evidence=false`, retain outcome exposure disclosure, and remain separate from all historical and scientific records.

The generic scientific preflight, launch, and resume paths continue to bind the historical model and 50-action protocol. Do not use those commands for the feasibility work. Use only the `university-gpu feasibility` commands above for this profile. No scientific inference is authorized here.

## Safe synchronization and offline handoff

The commands under step 1 are read/fast-forward-only and stop if local and remote histories diverge. Push only a reviewed commit using an authenticated, host-verified remote; never force push or suppress SSH host-key checks. If direct GitHub synchronization is unavailable, create a bundle only after committing the feature work, transfer the bundle and a separately verified SHA-256 through approved storage, then use the bundle fast-forward procedure in `BUDGET_AMENDMENT_DECISION.md`. Keep local run data and its independently verified archive primary; a GitHub push is not a data-integrity gate.
