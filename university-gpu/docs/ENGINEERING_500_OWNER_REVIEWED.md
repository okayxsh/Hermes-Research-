# Owner-reviewed 4-episode acceptance and 500-execution engineering run

Yash is the sole owner, operator and reviewer. This is owner review, **not independent review**. The two locally generated pins below detect changes; they do not authenticate the reviewer, demonstrate independent review, or grant university or backup permission. No scientific or confirmatory inference is authorized. All existing asset, source, placement, retry, storage and protected-backup gates still apply.

Use a distinct, newly authorized candidate, run ID, approval and pins for acceptance and production. Never overwrite existing evidence or launch a second time into a populated run. For an interrupted run use the reviewed resume procedure in ENGINEERING_500_RUNBOOK.md, only when its safety checks allow it.

## Deploy and prepare

In the existing university checkout, stop on a dirty worktree, unexpected commit or divergence. Replace `EXPECTED_COMMIT` with the exact full SHA communicated with this feature delivery.

```bash
cd /home/researcher/Hermes-Research-
test -z "$(git status --porcelain)" || { echo 'Dirty checkout: stop'; exit 1; }
git fetch origin refs/heads/feat/rq1-engineering-500-owner-reviewed:refs/remotes/origin/feat/rq1-engineering-500-owner-reviewed
# First use: git switch --track origin/feat/rq1-engineering-500-owner-reviewed
# Existing local branch: git switch feat/rq1-engineering-500-owner-reviewed && git merge --ff-only origin/feat/rq1-engineering-500-owner-reviewed
test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT" || { echo 'Wrong commit: stop'; exit 1; }
export PYTHONPATH="$PWD/src" RQ1_PY="$PWD/.venv/bin/python"
test -x "$RQ1_PY"
```

Use the existing external config and asset variables described in ENGINEERING_500_RUNBOOK.md and NEXT_VISIT_RUNBOOK.md. Set `RQ1_SCALE_EVIDENCE`, `RQ1_SCALE_CONFIG`, `RQ1_SCALE_RUN_ID`, `RQ1_SCALE_CANDIDATE`, `RQ1_SCALE_APPROVAL`, `RQ1_SCALE_MANIFEST`, and `RQ1_ACCESS_END_EPOCH` to actual external paths, unique run identity and authorized access end. The backup must actually exist, be approved, disjoint and independently protected. Set `acceptance_export_path` empty in the acceptance config; for production use the verified acceptance export directory. Do not fabricate a review or independence confirmation.

For the existing acceptance setup, reuse and validate an already-created **unfrozen** acceptance candidate; do not overwrite it. If none exists, create a new candidate at an unused path:

```bash
"$RQ1_PY" -m rq1.cli university-gpu engineering-scale candidate --baseline-candidate university-gpu/prospective/feasibility-200-candidate.json --acceptance --output "$RQ1_SCALE_CANDIDATE"
"$RQ1_PY" -m rq1.cli university-gpu engineering-scale validate --manifest "$RQ1_SCALE_CANDIDATE"
```

For production generate a new candidate at another unused path with the same command **without** `--acceptance`; validate it. Acceptance and production require separately completed approvals, fresh integrity pins, and run IDs. Production additionally binds the SHA-256 of the accepted four-worker `EXPORT-MANIFEST.json`.

## Step 1: review approval and confirm its integrity

Yash completes a new external copy of `university-gpu/prospective/engineering-500-approval-template.json`. Use the candidate digest shown by validation, exact source SHA, actual model and asset scope, unique run ID, access end, physical attempt count, and approved backup path. Acceptance specifies 4 units and 4 attempts unless extra attempts have explicitly been approved; production specifies 500. Set `approver_id`, `approved_by`, and the added `owner_reviewer_id` to `Yash`; add `review_mode` = `owner_reviewed`, `independent_review` = JSON `false`, and set `approval_provenance` to an honest description containing `owner-reviewed`. Set the existing non-scientific fields correctly, and set `test_only=false`, `gpu_inference=true`, and the exposure and backup confirmations only after actual review. Use an honest approval reference.

```bash
export RQ1_ENGINEERING_AUTHORITY_MODE=owner_reviewed
cat "$RQ1_SCALE_APPROVAL"
"$RQ1_PY" -c 'import json,sys; from rq1.university_gpu.authorization import approval_digest; print("OWNER APPROVAL INTEGRITY PIN:", approval_digest(json.load(open(sys.argv[1], encoding="utf-8"))))' "$RQ1_SCALE_APPROVAL"
read -r -p 'Enter the reviewed approval integrity pin: ' RQ1_OWNER_REVIEWED_ENGINEERING_APPROVAL_SHA256
export RQ1_OWNER_REVIEWED_ENGINEERING_APPROVAL_SHA256
"$RQ1_PY" -m rq1.cli university-gpu engineering-scale freeze \
  --candidate "$RQ1_SCALE_CANDIDATE" --run-id "$RQ1_SCALE_RUN_ID" \
  --data-report "$RQ1_FEASIBILITY_DATA_REPORT" --data-review "$RQ1_FEASIBILITY_DATA_REVIEW" \
  --oracle-report "$RQ1_FEASIBILITY_ORACLE_REPORT" \
  --identity-report "$RQ1_PROSPECTIVE_IDENTITY_REPORT" --identity-review "$RQ1_PROSPECTIVE_IDENTITY_REVIEW" \
  --accum18-library "$RQ1_ACCUM18_LIBRARY" --accum18-reference "$RQ1_ACCUM18_REFERENCE" \
  --sbert-cache "$RQ1_HUGGINGFACE_CACHE" --tokenizer "$RQ1_GEMMA_TOKENIZER_JSON" \
  --engineering-approval "$RQ1_SCALE_APPROVAL" --output "$RQ1_SCALE_MANIFEST"
```

Freeze is create-only. If an appropriate manifest already exists, review its exact source and approval and skip freeze; do not overwrite or reuse one bound to another commit.

## Step 2: review frozen manifest and confirm its integrity

Review the complete manifest and its candidate, approval, source, assets, deadline, counts and backup. Preserve the displayed pins separately for later comparison.

```bash
"$RQ1_PY" -m rq1.cli university-gpu engineering-scale validate --manifest "$RQ1_SCALE_MANIFEST" --require-frozen
cat "$RQ1_SCALE_MANIFEST"
"$RQ1_PY" -c 'import json,sys; print("FROZEN MANIFEST INTEGRITY PIN:", json.load(open(sys.argv[1], encoding="utf-8"))["manifest_sha256"])' "$RQ1_SCALE_MANIFEST"
read -r -p 'Enter the reviewed frozen manifest integrity pin: ' RQ1_OWNER_REVIEWED_ENGINEERING_MANIFEST_SHA256
export RQ1_OWNER_REVIEWED_ENGINEERING_MANIFEST_SHA256
"$RQ1_PY" -m rq1.cli university-gpu engineering-scale preflight --config "$RQ1_SCALE_CONFIG" --manifest "$RQ1_SCALE_MANIFEST"
```

Only after both confirmations and all real approvals and CPU preflight pass, and only if the scope explicitly includes four-worker GPU smoke and acceptance episodes, use:

```bash
"$RQ1_PY" -m rq1.cli university-gpu engineering-scale benchmark --config "$RQ1_SCALE_CONFIG" --manifest "$RQ1_SCALE_MANIFEST" --approval-file "$RQ1_SCALE_APPROVAL" --yes --output "$RQ1_SCALE_EVIDENCE/benchmark.json"
"$RQ1_PY" -m rq1.cli university-gpu engineering-scale launch --config "$RQ1_SCALE_CONFIG" --manifest "$RQ1_SCALE_MANIFEST" --approval-file "$RQ1_SCALE_APPROVAL" --yes --cutoff-epoch "$RQ1_ACCESS_END_EPOCH"
```

Benchmark includes GPU requests, and launch performs GPU requests and episodes. Do not execute either unless genuinely authorized. Follow ENGINEERING_500_RUNBOOK.md for status, controlled stop, resume, validation, merge, independent protected export and verification. Four valid acceptance records and verified backup are required before production approval and preflight. Keep the 75-minute cutoff reserve and actual institutional access constraints. Retain failed attempts; do not retry unknown mutation states.
