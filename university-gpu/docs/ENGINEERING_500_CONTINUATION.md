# Owner-reviewed continuation of the protected 500-unit run

This procedure is engineering-only and does not authorize scientific or confirmatory experiments. Yash is the owner and reviewer; this is **not independent review**. Locally calculated pins detect changes, but do not authenticate the reviewer or establish university permission or independent backup. The original manifest, approval, results, checkpoints, events and receipts remain unchanged. A new commit is necessary; the original and continuation commit are recorded separately.

## Prepare the checkout and paths

First ensure the existing run is stopped and no other operator or process is using it. Confirm that the Windows E: backup is independently protected and mounted and that its approved path agrees exactly with the original manifest. Do not remove a lock or STOP marker manually; investigate an active/stale owner through the existing runbook. Confirm actual workstation access and cutoff, including the 75-minute reserve. Obtain an approved persistent session; do not assume WSL survives disconnect or shutdown.

```bash
cd /home/researcher/Hermes-Research-
test -z "$(git status --porcelain)" || { echo 'Dirty checkout: stop'; exit 1; }
git fetch origin refs/heads/feat/rq1-engineering-500-continuation:refs/remotes/origin/feat/rq1-engineering-500-continuation
# First visit:
git switch --track origin/feat/rq1-engineering-500-continuation
# If the local branch already exists, instead switch to it and use git merge --ff-only origin/feat/rq1-engineering-500-continuation.
export RQ1_CONTINUATION_COMMIT='<full reviewed continuation commit from delivery>'
test "$(git rev-parse HEAD)" = "$RQ1_CONTINUATION_COMMIT" || { echo 'Wrong source: stop'; exit 1; }
export PYTHONPATH="$PWD/src" RQ1_PY="$PWD/.venv/bin/python"
test -x "$RQ1_PY"
```

Restore the original external configuration, assets and original owner-reviewed approval and manifest pins using the original runbook. Set the following to their **actual** existing values, not guessed paths:

```bash
export RQ1_ENGINEERING_AUTHORITY_MODE=owner_reviewed
export RQ1_SCALE_CONFIG='<existing external scale config>'
export RQ1_SCALE_MANIFEST='<original frozen manifest path>'
export RQ1_SCALE_APPROVAL='<original approval path>'
export RQ1_SCALE_OUTPUT_ROOT='<run_output_dir from config>'
export RQ1_SCALE_RUN_ID='<original run id>'
export RQ1_SCALE_RUN_ROOT="$RQ1_SCALE_OUTPUT_ROOT/$RQ1_SCALE_RUN_ID"
export RQ1_CONTINUATION_AUTH='<new unused external JSON path>'
export RQ1_ACCESS_END_EPOCH='<new actual approved Unix UTC deadline>'
read -r -p 'Original approval integrity pin: ' RQ1_OWNER_REVIEWED_ENGINEERING_APPROVAL_SHA256
read -r -p 'Original frozen manifest integrity pin: ' RQ1_OWNER_REVIEWED_ENGINEERING_MANIFEST_SHA256
export RQ1_OWNER_REVIEWED_ENGINEERING_APPROVAL_SHA256 RQ1_OWNER_REVIEWED_ENGINEERING_MANIFEST_SHA256
```

## Verify and personally authorize the checkpoint

```bash
"$RQ1_PY" -m rq1.cli university-gpu engineering-scale resume-inspect \
  --output-root "$RQ1_SCALE_OUTPUT_ROOT" --run-root "$RQ1_SCALE_RUN_ROOT" --manifest "$RQ1_SCALE_MANIFEST"
```

The command reads the existing run, every local unit checkpoint, the current protected receipt, its copied archive, inventory and anchors. For the first continuation, require `completed_count` 218 and a verified receipt at `protection/0218.json`. Do not proceed on a mismatch, failed/unknown attempt, corruption or missing independent backup. Record the entire displayed `checkpoint` object exactly. If a later continuation is interrupted after further progress, repeat the inspection and complete a **new** owner-reviewed authorization bound to the newly verified checkpoint.

Use an editor to create a new JSON file at `$RQ1_CONTINUATION_AUTH` without overwriting an existing authorization. Yash must personally review and complete the following; replace every angle-bracket value, and copy `checkpoint` exactly from the inspection output. The original `backup_destination` must match the manifest string exactly. Set the booleans only after the actual engineering GPU smoke/episode scope is authorized. The continuation code does not create an approval or signature.

```bash
( umask 077; set -o noclobber; : > "$RQ1_CONTINUATION_AUTH" ) || { echo 'Authorization path already exists or cannot be created: stop'; exit 1; }
nano "$RQ1_CONTINUATION_AUTH"
```

If `nano` is unavailable, use an already-installed editor. Paste and complete this template yourself:

```json
{
  "schema": "rq1-engineering-scale-resume-authorization-v1",
  "review_mode": "owner_reviewed",
  "independent_review": false,
  "reviewer_id": "Yash",
  "approval_reference": "<your actual owner-review reference>",
  "approval_provenance": "owner-reviewed by Yash for this engineering continuation",
  "engineering_only": true,
  "scientific_run": false,
  "gpu_inference": true,
  "test_only": false,
  "run_id": "<original run ID>",
  "manifest_sha256": "<original manifest_sha256>",
  "original_source_commit": "e27188554f354d096700287a1b3c3f62cb5d11e6",
  "continuation_source_commit": "<full reviewed continuation SHA>",
  "backup_destination": "<exact original approved backup destination>",
  "checkpoint": {
    "completed_count": 218,
    "completed_unit_keys_sha256": "<from inspection>",
    "checkpoint_manifests_sha256": "<from inspection>",
    "receipt_sha256": "<from inspection>",
    "receipt_archive_sha256": "<from inspection>"
  },
  "access_end_epoch": 0
}
```

Replace `access_end_epoch` with the actual future Unix timestamp as a JSON number. Inspect the file and explicitly confirm its canonical integrity pin:

```bash
cat "$RQ1_CONTINUATION_AUTH"
"$RQ1_PY" -c 'import json,sys; from rq1.university_gpu.models import content_hash; print("RESUME AUTHORIZATION INTEGRITY PIN:", content_hash(json.load(open(sys.argv[1], encoding="utf-8"))))' "$RQ1_CONTINUATION_AUTH"
read -r -p 'Enter the owner-reviewed resume authorization integrity pin: ' RQ1_OWNER_REVIEWED_RESUME_AUTHORIZATION_SHA256
export RQ1_OWNER_REVIEWED_RESUME_AUTHORIZATION_SHA256
"$RQ1_PY" -m rq1.cli university-gpu engineering-scale resume-preflight \
  --config "$RQ1_SCALE_CONFIG" --manifest "$RQ1_SCALE_MANIFEST" --approval-file "$RQ1_SCALE_APPROVAL" \
  --resume-authorization "$RQ1_CONTINUATION_AUTH" --cutoff-epoch "$RQ1_ACCESS_END_EPOCH"
```

Preflight verifies the original approval and manifest pins, the two source identities and allowed code changes, all inputs, protected checkpoint and storage. It does not run inference. A PASS does not prove the physical backup is independent or university access is authorized.

## Resume only after preflight passes

Start an approved installed tmux session and enter the resume command **inside** it. The command performs new GPU smoke requests before episodes and uses four isolated workers; those operations require the owner's genuine authorization. Do not pass `--retry-failed`.

```bash
tmux new -s rq1-continuation
"$RQ1_PY" -m rq1.cli university-gpu engineering-scale resume \
  --config "$RQ1_SCALE_CONFIG" --manifest "$RQ1_SCALE_MANIFEST" --approval-file "$RQ1_SCALE_APPROVAL" \
  --resume-authorization "$RQ1_CONTINUATION_AUTH" --yes --cutoff-epoch "$RQ1_ACCESS_END_EPOCH"
```

Detach with `Ctrl-b`, then `d`; reattach with `tmux attach -t rq1-continuation`. Continuation status and monitor files are under the newest `$RQ1_SCALE_RUN_ROOT/continuation-runtime/<launch-id>/`, preserving the old live-status file. Follow the original runbook for monitoring, controlled stop, validation, merge and protected export. New attempts and receipts are create-only. Inspect any failure or unknown attempt; do not retry it silently or alter old evidence. The runner stops for a new reviewed authorization after an interrupted continuation has advanced beyond its authorized checkpoint.
