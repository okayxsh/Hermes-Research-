# November 9, 2026 university continuation

Resume the same engineering run, `rq1-engineering-500-v1-university-20261104-prod-01`. The reported starting point is 218 protected results and 282 remaining. The workstation checks below establish the actual starting point. Yash personally reviews the continuation. Use the actual university access window and keep the original November 4 files and receipts in place.

## 1. Inspect the workstation before starting anything

In WSL, check for old runner, worker or Ollama processes, and compare with NVIDIA's process report. An Ollama listener or a GPU process can belong to another user or service. Resolve any conflict with the workstation operator before proceeding.

```bash
pgrep -af '[r]q1.cli|[e]ngineering_scale_worker|[o]llama' || true
nvidia-smi
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.free --format=csv
tmux list-sessions
```

Expect four RTX 5000 Ada GPUs. Check on Windows that the workstation will remain awake and university policy permits leaving the session running during an RDP disconnect. Use Disconnect, and keep Windows signed in and WSL running.

## 2. Fetch and pin the reviewed checkout

```bash
cd /home/researcher/Hermes-Research- || exit 1
RQ1_GIT_STATUS=$(git status --porcelain) || exit 1
test -z "$RQ1_GIT_STATUS" || { echo 'Dirty checkout. Stop here.'; exit 1; }
git fetch origin refs/heads/feat/rq1-engineering-500-nov09-readiness:refs/remotes/origin/feat/rq1-engineering-500-nov09-readiness || exit 1
read -r -p 'Paste the full reviewed commit from the delivery message: ' RQ1_REVIEWED_COMMIT
test "$(git rev-parse origin/feat/rq1-engineering-500-nov09-readiness)" = "$RQ1_REVIEWED_COMMIT" || { echo 'Unexpected remote SHA. Stop here.'; exit 1; }
git switch --detach "$RQ1_REVIEWED_COMMIT" || exit 1
git status --short --branch
```

## 3. Enter tmux and restore the original environment

Run the remaining setup and launch in the same pane. Use a fresh session name if one already exists, after confirming it has no running experiment. Reuse the frozen Python environment and original external assets.

```bash
tmux new -s rq1-nov09 -c /home/researcher/Hermes-Research-
```

Inside tmux:

```bash
export PYTHONPATH="$PWD/src" RQ1_PY="$PWD/.venv/bin/python"
"$RQ1_PY" --version && "$RQ1_PY" -m pip check || { echo 'Python check failed. Stop here.'; exit 1; }
export RQ1_ENGINEERING_AUTHORITY_MODE=owner_reviewed
export RQ1_ORIGINAL=/home/researcher/rq1-evidence/engineering-500-20261104/production-500
export RQ1_SCALE_CONFIG="$RQ1_ORIGINAL/config.json"
export RQ1_SCALE_MANIFEST="$RQ1_ORIGINAL/frozen.json"
export RQ1_SCALE_APPROVAL="$RQ1_ORIGINAL/approval.json"
export RQ1_SCALE_OUTPUT_ROOT=/home/researcher/rq1-engineering-500-runs
export RQ1_SCALE_RUN_ID=rq1-engineering-500-v1-university-20261104-prod-01
export RQ1_SCALE_RUN_ROOT="$RQ1_SCALE_OUTPUT_ROOT/$RQ1_SCALE_RUN_ID"
export RQ1_BACKUP=/mnt/e/rq1-engineering-500-protected/production-20261104
export RQ1_CONTINUATION_AUTH=/home/researcher/rq1-evidence/engineering-500-20261104/resume-20261109-01.json
test -r "$RQ1_SCALE_CONFIG" && test -r "$RQ1_SCALE_MANIFEST" && test -r "$RQ1_SCALE_APPROVAL" || { echo 'Missing original input. Stop here.'; exit 1; }
mountpoint /mnt/e && test -d "$RQ1_BACKUP" || { echo 'E: backup unavailable. Stop here.'; exit 1; }
df -h "$RQ1_SCALE_RUN_ROOT" "$RQ1_BACKUP"
```

Require Python 3.11 and the original frozen package versions. Stop on a missing input, failed dependency check or unavailable E: mount. In this pane, define a short wrapper for the existing CLI.

```bash
scale() { "$RQ1_PY" -m rq1.cli university-gpu engineering-scale "$@"; }
run_args=(--output-root "$RQ1_SCALE_OUTPUT_ROOT" --run-root "$RQ1_SCALE_RUN_ROOT" --manifest "$RQ1_SCALE_MANIFEST")
```

## 4. Verify the protected starting checkpoint

```bash
scale resume-inspect "${run_args[@]}"
"$RQ1_PY" -c 'import json,sys; m=json.load(open(sys.argv[1])); print(*(m[k] for k in ("run_id","source_commit","manifest_sha256","backup_destination")), sep="\n")' "$RQ1_SCALE_MANIFEST"
"$RQ1_PY" -c 'import json,sys; r=json.load(open(sys.argv[1])); print(r["status"], len(r["unit_keys"]), r["destination"], r["archive_sha256"], sep="\n")' "$RQ1_SCALE_RUN_ROOT/protection/0218.json"
git rev-parse HEAD
```

Require `VERIFIED_CHECKPOINT` and 218 completed units for this first continuation. The receipt must show `VERIFIED_AT_APPROVED_DESTINATION` under the E: destination above, and the original source must be `e27188554f354d096700287a1b3c3f62cb5d11e6`. Keep the entire inspected `checkpoint` object. Stop on a missing receipt, unexpected count, STOP marker, lock, failed attempt or unresolved attempt.

## 5. Choose the cutoff and personally review the new authorization

Enter the actual end of authorized access on November 9 in Dubai time, including `+0400`. The runner stops assigning work 75 minutes before this cutoff and drains active units. Choose an earlier deadline if the workstation must be handed back sooner.

Scaling the earlier 8–12 hour estimate for 500 units to 282 gives roughly 4 hours 31 minutes to 6 hours 46 minutes before the reserve. Allow about 5 hours 46 minutes to 8 hours 1 minute including the reserve, plus setup. This is a planning estimate. Compare it with the original run's measured elapsed time and current backup overhead before relying on a single visit.

```bash
read -r -p 'Authorized end (YYYY-MM-DD HH:MM +0400): ' RQ1_ACCESS_END_DUBAI
RQ1_ACCESS_END_EPOCH=$(date -d "$RQ1_ACCESS_END_DUBAI" +%s) || { echo 'Invalid cutoff. Stop here.'; exit 1; }
export RQ1_ACCESS_END_EPOCH
TZ=Asia/Dubai date -d "@$RQ1_ACCESS_END_EPOCH"
test "$RQ1_ACCESS_END_EPOCH" -gt "$(( $(date +%s) + 4500 ))" || { echo 'Cutoff leaves less than the required reserve. Stop here.'; exit 1; }
( umask 077; set -o noclobber; : > "$RQ1_CONTINUATION_AUTH" ) && nano "$RQ1_CONTINUATION_AUTH"
```

If any command fails, stop. In the new file, personally complete the [authorization JSON template](ENGINEERING_500_CONTINUATION.md#verify-and-personally-authorize-the-checkpoint). Use the actual review reference, the inspected checkpoint object, the original run and manifest identity, the exact checked-out continuation SHA, the E: destination and the Unix cutoff as a JSON number. Review the scope before affirming authorization for engineering GPU smoke requests and episodes.

## 6. Confirm all three integrity pins and run preflight

```bash
cat "$RQ1_CONTINUATION_AUTH"
"$RQ1_PY" -c 'import json,sys; from rq1.university_gpu.models import content_hash; print(content_hash(json.load(open(sys.argv[1], encoding="utf-8"))))' "$RQ1_CONTINUATION_AUTH"
read -r -p 'Enter the personally reviewed resume pin: ' RQ1_OWNER_REVIEWED_RESUME_AUTHORIZATION_SHA256
read -r -p 'Enter the retained November 4 approval pin: ' RQ1_OWNER_REVIEWED_ENGINEERING_APPROVAL_SHA256
read -r -p 'Enter the retained November 4 manifest pin: ' RQ1_OWNER_REVIEWED_ENGINEERING_MANIFEST_SHA256
export RQ1_OWNER_REVIEWED_RESUME_AUTHORIZATION_SHA256 RQ1_OWNER_REVIEWED_ENGINEERING_APPROVAL_SHA256 RQ1_OWNER_REVIEWED_ENGINEERING_MANIFEST_SHA256
auth_args=(--config "$RQ1_SCALE_CONFIG" --manifest "$RQ1_SCALE_MANIFEST" --approval-file "$RQ1_SCALE_APPROVAL")
auth_args+=(--resume-authorization "$RQ1_CONTINUATION_AUTH" --cutoff-epoch "$RQ1_ACCESS_END_EPOCH")
scale resume-preflight "${auth_args[@]}"
```

Use the original reviewed pins from the November 4 record. If either is missing, stop and recover the reviewed value. Require a preflight `PASS` with the same checkpoint and continuation SHA. Preflight uses CPU and file checks. Actual placement and model readiness are checked at launch.

## 7. Resume and confirm the first new attempt

Only after preflight passes and access is authorized, run inside the same tmux pane:

```bash
set -o pipefail
RQ1_SESSION_LOG=$(mktemp /home/researcher/rq1-evidence/engineering-500-20261104/resume-nov09.XXXXXX.log) || { echo 'Cannot create session log. Stop here.'; exit 1; }
scale resume "${auth_args[@]}" --yes 2>&1 | tee "$RQ1_SESSION_LOG"
```

In a second tmux pane, set the run path and compare new attempt starts with the 218 protected keys. There should be at least one new start and zero repeated protected keys once workers begin. Startup and validation can take several minutes.

```bash
export RQ1_SCALE_RUN_ROOT=/home/researcher/rq1-engineering-500-runs/rq1-engineering-500-v1-university-20261104-prod-01
cd /home/researcher/Hermes-Research-
.venv/bin/python - <<'PY'
import json, os
from pathlib import Path
root = Path(os.environ["RQ1_SCALE_RUN_ROOT"])
old = set(json.loads((root / "protection/0218.json").read_text())["unit_keys"])
events = [json.loads(p.read_text()) for p in (root / "events").glob("*.json")]
starts = sorted((r for r in events if r["event"] == "attempt_started"), key=lambda r: r["sequence"])
assert len(old) == 218 and {r["unit_key"] for r in starts[:218]} == old
new = [r["unit_key"] for r in starts[218:]]
assert not old.intersection(new) and len(set(new)) == len(new)
print("New attempts:", len(new), "First new unit:", next(iter(new), "waiting"))
PY
```

## 8. Monitor progress and confirm RDP persistence

The original tmux pane retains the variables and wrapper. To monitor in a second pane, repeat the definitions in step 3 beginning at `export PYTHONPATH`, then run:

```bash
scale status "${run_args[@]}"
find "$RQ1_SCALE_RUN_ROOT/protection" -maxdepth 1 -name '[0-9][0-9][0-9][0-9].json' | sort | tail -3
find "$RQ1_SCALE_RUN_ROOT/continuation-runtime" -name status-live.json -printf '%T@ %p\n' | sort -nr | head -1
find "$RQ1_SCALE_RUN_ROOT/continuation-runtime" \( -name ollama-server.log -o -name '*.stderr.log' \) -printf '%T@ %p\n' | sort -nr | head -8
ls -lt "$RQ1_SCALE_RUN_ROOT/launches" | head
nvidia-smi
```

Use the latest displayed paths with `cat` for status JSON and `tail -n 40` for a worker log. The status reports planned and completed counts, attempts, failures and local checkpoints. Remaining units are 500 minus completed. A zero failed-attempt count alone cannot establish a clean pause. The supervisor pauses for protection at 250, 300, 350, 400, 450 and 500, so a period without GPU activity can be expected. Read the latest numbered receipt to identify the protected copy. A receipt's presence alone cannot establish backup integrity.

After observing a new start, detach with **Ctrl-b d**, disconnect RDP, reconnect, and run `tmux attach -t rq1-nov09`. Compare new attempts, worker timestamps or protection progress with the values before disconnect. A live tmux pane by itself only establishes that tmux persisted. Resolve any lifecycle issue before relying on unattended operation.

## 9. Verify completion or a clean pause

After the resume command exits and all runner/worker processes have stopped, use the same environment and run:

```bash
scale status "${run_args[@]}"
scale resume-inspect "${run_args[@]}"
```

A clean cutoff may return a nonzero CLI exit code. Require the verified checkpoint count to equal the completed count, all accepted units to have local checkpoints, and zero failed or unresolved attempts. For 500 units also run:

```bash
scale validate-run "${run_args[@]}"
RQ1_FINAL_EXPORT=$("$RQ1_PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["destination"])' "$RQ1_SCALE_RUN_ROOT/protection/0500.json")
scale verify-export --destination "$RQ1_FINAL_EXPORT/export"
```

Expect 500 completed, 500 started, zero failed, a complete protected export and a passing readback. For a clean partial pause, keep the receipt and repeat the inspect and owner-review steps in the next approved access window, using a fresh authorization file and pin tied to the newer checkpoint. Prefer the reviewed cutoff for a planned pause. A manual STOP marker or an unresolved attempt requires review. If publication fails or the process is interrupted mid-segment, preserve the files and incomplete copies and investigate before authorizing any continuation.
