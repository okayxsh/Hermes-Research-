# Engineering scale runner: 4-episode acceptance and 500 executions

For owner-reviewed acceptance and production commands, see [ENGINEERING_500_OWNER_REVIEWED.md](ENGINEERING_500_OWNER_REVIEWED.md). Owner review is not independent review; locally generated pins are integrity checks only.

Status: offline preparation. Real four-GPU acceptance, independent reviewer decisions, university access and backup destination approval must be obtained on the workstation. Nothing here authorizes scientific inference. Original 12-episode profile and scientific freezes are preserved.

## Identity and workload

Profile `rq1-engineering-500-v1`, schema `rq1-engineering-scale-manifest-v1`, seed 11 for environment and model; exact six existing valid-seen tasks and 200 TOTAL actions. The default manifest has 500 logical executions, 250 paired repetitions, D0=250/D18=250. Tasks in original candidate order receive 42,42,42,42,41,41 pairs. Temperature 0 does not promise deterministic GPU/provider output. These are repeated observations on SIX contexts, not 500 independent scientific samples.

Candidate generation supports a separate `--acceptance` manifest: four executions (first two task pairs), four unique GPU workers, distinct run ID/approval, excluded from production 500. Acceptance requires valid infrastructure/evidence, not universal task recovery success.

Default total physical attempt allowance is the matrix size. If prospectively approved, `--additional-attempts N` permits N additional attempts (0..20) while logical count remains fixed. Never automatically retry failures. `resume --retry-failed` additionally requires latest failure to record BOTH safety flags true; unknown mutation state is blocked before workers start. An unknown attempt or exhausted allowance requires a reviewed recovery procedure/new authorization, never edited events.

## Laptop implementation and deployment

The branch begins at `6af25622464142beef25d2c248655946389e4f8b`. Run offline checks on the laptop before reviewing/staging changes:

```powershell
$env:PYTHONPATH='src'
.venv/Scripts/python.exe -m unittest discover -s tests -v
.venv/Scripts/python.exe -m compileall -q src/rq1 tests
git diff --check
```

The persisted 500-unit case can take 40+ minutes on native Windows. It uses synthetic providers and temporary files, not GPUs. On university access time, retain the reviewed laptop results; run focused CPU portability checks and the actual preflight/acceptance rather than repeating the entire suite unless source changes.

After offline validation and review:

```powershell
git status --short --branch
git diff --check
git add -- src/rq1/university_gpu tests/test_engineering_scale.py tests/test_engineering_scale_recovery.py university-gpu/engineering-500-config.example.json university-gpu/prospective/engineering-500-approval-template.json university-gpu/docs/ENGINEERING_500_RUNBOOK.md university-gpu/docs/ENGINEERING_500_IMPLEMENTATION.md
git diff --cached --check
git commit -m "Add isolated 500-execution engineering profile"
git push origin HEAD:refs/heads/feat/rq1-engineering-500
```

Do not force push. Stage review must exclude evidence, models, credentials, personal paths and unrelated files. The implementation report records actual validation/commit results.

University, existing checkout and environment only:

```bash
set -euo pipefail
cd /home/researcher/Hermes-Research-
test -z "$(git status --porcelain)"
git fetch origin refs/heads/feat/rq1-engineering-500:refs/remotes/origin/feat/rq1-engineering-500
# First visit to this branch:
git switch --track origin/feat/rq1-engineering-500
# If already present, instead use: git switch feat/rq1-engineering-500; git merge --ff-only origin/feat/rq1-engineering-500
: "${RQ1_APPROVED_SOURCE_COMMIT:?Independently supplied reviewed full source SHA}"
test "$(git rev-parse HEAD)" = "$RQ1_APPROVED_SOURCE_COMMIT"
export PYTHONPATH="$PWD/src"
export RQ1_PY="$PWD/.venv/bin/python"
test -x "$RQ1_PY"
"$RQ1_PY" --version
"$RQ1_PY" -m pip check
nvidia-smi -L
ollama --version
ollama list
```

Stop on dirty source, unexpected SHA, divergence, model differences or missing dependencies. No install/pull is performed by this workflow. Obtain separate approval for necessary environmental changes.

## Candidate, evidence and fresh independent authorization

Choose NEW external evidence and run-output directories; create them before preflight. Require at least 50 GB free on run and backup storage. The approved backup path must exist and be accessible. Human confirmation of independence is a trust assertion; software verifies bytes, not institutional independence or physical topology.

Set the existing asset variables from NEXT_VISIT_RUNBOOK.md: data root and reconciliation/review; oracle report; model manifest/blobs/model directory; tokenizer/provenance/link/revision; model review/identity report; original Accum-18 library/reference; pinned Sentence-BERT cache. Copy `engineering-500-config.example.json` to a NEW external config. Its variable references are expanded by the existing config loader. Leave `acceptance_export_path` empty for acceptance. In the production external config, replace it with the verified acceptance export directory. All other variable references must resolve even for acceptance.

Use these variables for either acceptance or production; choose different paths/run IDs for each:

```bash
: "${RQ1_SCALE_EVIDENCE:?New external evidence directory}"
: "${RQ1_SCALE_CONFIG:?External runtime config path}"
: "${RQ1_SCALE_RUN_ID:?Unique rq1-engineering-500-v1-prefixed run ID}"
export RQ1_SCALE_CANDIDATE="$RQ1_SCALE_EVIDENCE/candidate.json"
export RQ1_SCALE_APPROVAL="$RQ1_SCALE_EVIDENCE/approval.json"
export RQ1_SCALE_MANIFEST="$RQ1_SCALE_EVIDENCE/frozen.json"
# Acceptance: include --acceptance. Production: omit it. Never use the same output twice.
"$RQ1_PY" -m rq1.cli university-gpu engineering-scale candidate   --baseline-candidate university-gpu/prospective/feasibility-200-candidate.json   --output "$RQ1_SCALE_CANDIDATE"
"$RQ1_PY" -m rq1.cli university-gpu engineering-scale validate --manifest "$RQ1_SCALE_CANDIDATE"
```

An authorized human completes a new copy of `engineering-500-approval-template.json`. Acceptance must specify episode_count=4/maximum_attempts=4 unless prospective extra attempts are approved. Production specifies 500. Bind candidate SHA, reviewed source SHA, run ID, model identity, 200-action limit, both conditions, deadline (Unix UTC), physical attempt allowance, prior exposure and approved independent backup path. Set `backup_independence_confirmed=true` only after a real decision. Template `test_only=true`/blank signatures are deliberately invalid. No agent creates approval signatures.

Production approval must additionally bind the SHA-256 of `EXPORT-MANIFEST.json` from the separate complete four-worker acceptance export. The acceptance source, tasks and verified inputs must match production. Reusing the old 12-episode run does not satisfy this new acceptance gate.

A genuine independent reviewer communicates canonical approval SHA through an independently authenticated channel. Set its received value, never derive a trusted pin from an unreviewed local file:

```bash
export RQ1_ENGINEERING_AUTHORITY_MODE=independent
: "${RQ1_TRUSTED_ENGINEERING_APPROVAL_SHA256:?Received independent approval digest}"
"$RQ1_PY" -m rq1.cli university-gpu engineering-scale freeze   --candidate "$RQ1_SCALE_CANDIDATE" --run-id "$RQ1_SCALE_RUN_ID"   --data-report "$RQ1_FEASIBILITY_DATA_REPORT" --data-review "$RQ1_FEASIBILITY_DATA_REVIEW"   --oracle-report "$RQ1_FEASIBILITY_ORACLE_REPORT"   --identity-report "$RQ1_PROSPECTIVE_IDENTITY_REPORT" --identity-review "$RQ1_PROSPECTIVE_IDENTITY_REVIEW"   --accum18-library "$RQ1_ACCUM18_LIBRARY" --accum18-reference "$RQ1_ACCUM18_REFERENCE"   --sbert-cache "$RQ1_HUGGINGFACE_CACHE" --tokenizer "$RQ1_GEMMA_TOKENIZER_JSON"   --engineering-approval "$RQ1_SCALE_APPROVAL" --output "$RQ1_SCALE_MANIFEST"
"$RQ1_PY" -m rq1.cli university-gpu engineering-scale validate --manifest "$RQ1_SCALE_MANIFEST" --require-frozen
# Reviewer inspects the complete frozen manifest and independently transmits its digest:
: "${RQ1_TRUSTED_ENGINEERING_MANIFEST_SHA256:?Received independent frozen manifest digest}"
"$RQ1_PY" -m rq1.cli university-gpu engineering-scale preflight --config "$RQ1_SCALE_CONFIG" --manifest "$RQ1_SCALE_MANIFEST"
```

Revalidate existing data/model reviews against their reports. The same unchanged seed-11 task witnesses may be reused through the embedded original candidate ancestry. Every freeze/launch rehashes current assets. Freeze also records the installed Python/package-version inventory and its hash; preflight rejects dependency drift. Review this inventory in the independently pinned frozen manifest. No new inference is required to regenerate candidate metadata.

## Benchmark and launch (only with authentic authorization)

```bash
"$RQ1_PY" -m rq1.cli university-gpu engineering-scale benchmark   --config "$RQ1_SCALE_CONFIG" --manifest "$RQ1_SCALE_MANIFEST"   --approval-file "$RQ1_SCALE_APPROVAL" --yes --output "$RQ1_SCALE_EVIDENCE/benchmark.json"
: "${RQ1_ACCESS_END_EPOCH:?Actual future cutoff within the approved access window}"
"$RQ1_PY" -m rq1.cli university-gpu engineering-scale launch   --config "$RQ1_SCALE_CONFIG" --manifest "$RQ1_SCALE_MANIFEST"   --approval-file "$RQ1_SCALE_APPROVAL" --yes --cutoff-epoch "$RQ1_ACCESS_END_EPOCH"
```

Benchmark sends four eight-token requests, zero episodes. Launcher also performs its four-worker readiness smoke after every server restart. Scope those smoke requests and repeated segment restarts in the human approval. Four acceptance episodes are separately authorized. Never run them using a production-only approval. PID placement is preferred; the strict WSL fresh-log/PCI fallback records its method. Failure stops launch.

Launcher supervises 50-accepted-unit segments, drains workers, closes owned processes, validates exports at approved destination, writes immutable protection receipts, then resumes only a clean pause. It rechecks placement on every restart. Lost ownership, storage failure or monitoring failure stops assignments. Coordinator heartbeats and raw episode action activity are distinct evidence; inspect both. Coordinator `attempts_started` is per segment; the final summary records whole-run physical attempts. Reserve 75 minutes before cutoff for active work (response timeout up to one hour) and export. If measured export time needs more margin, set an earlier operational cutoff within the approved access window; never extend permission silently. GPU runtime is estimated 8–12 hours plus setup/reserve, not guaranteed by the earlier 9-minute batch.

Use an existing university-approved persistent session such as tmux only if installed/approved. Verify RDP-disconnection survival and WSL/Windows shutdown policy. Never assume overnight access. A crash stops automation; no automatic restart of unknown attempts.

## Status, stop, reviewed resume and protection recovery

```bash
: "${RQ1_SCALE_OUTPUT_ROOT:?Same run_output_dir as the config}"
export RQ1_SCALE_RUN_ROOT="$RQ1_SCALE_OUTPUT_ROOT/$RQ1_SCALE_RUN_ID"
"$RQ1_PY" -m rq1.cli university-gpu engineering-scale status   --output-root "$RQ1_SCALE_OUTPUT_ROOT" --run-root "$RQ1_SCALE_RUN_ROOT" --manifest "$RQ1_SCALE_MANIFEST"
# Cooperative stop only when intended:
"$RQ1_PY" -m rq1.cli university-gpu engineering-scale stop   --output-root "$RQ1_SCALE_OUTPUT_ROOT" --run-root "$RQ1_SCALE_RUN_ROOT" --reason "authorized cutoff"
```

Inspect `status-live.json`, `monitor/*.json`, per-attempt heartbeats and raw action logs. Monitor records are not proof of ongoing episode progress. Wait for active writers and owned processes to finish. Preserve failed/partial attempts, `.partial-*` checkpoints and `.incomplete-*` transfers. Do not delete a stale lock/takeover marker without reviewed ownership inspection.

If transfer failed after a clean segment, correct the actual storage error then run:

```bash
"$RQ1_PY" -m rq1.cli university-gpu engineering-scale protect   --output-root "$RQ1_SCALE_OUTPUT_ROOT" --run-root "$RQ1_SCALE_RUN_ROOT" --manifest "$RQ1_SCALE_MANIFEST"
```

This publishes a new destination copy; partial destinations remain intact. A corrupt already-receipted copy fails closed and needs reviewed recovery, never silent replacement. Normal resume re-verifies destination archive, inventory, anchor coverage and accepted-unit set before GPU startup:

```bash
"$RQ1_PY" -m rq1.cli university-gpu engineering-scale resume   --config "$RQ1_SCALE_CONFIG" --manifest "$RQ1_SCALE_MANIFEST"   --approval-file "$RQ1_SCALE_APPROVAL" --yes --cutoff-epoch "$RQ1_ACCESS_END_EPOCH"
```

Only add `--retry-failed` after reviewing an explicit safe/known latest failure and confirming stopped processes, unused approved attempt allowance and valid protection. A fully validated authoritative result published before its completion event remains accepted: after reviewed protection, coordinator resume repairs its missing checkpoint/event without reexecuting it. Unknown mutation state without matching validated authority stays blocked. Never rewrite historical run identity or results to resume.

## Final validation and reporting

```bash
"$RQ1_PY" -m rq1.cli university-gpu engineering-scale validate-run   --output-root "$RQ1_SCALE_OUTPUT_ROOT" --run-root "$RQ1_SCALE_RUN_ROOT" --manifest "$RQ1_SCALE_MANIFEST"
"$RQ1_PY" -m rq1.cli university-gpu engineering-scale merge   --output-root "$RQ1_SCALE_OUTPUT_ROOT" --run-root "$RQ1_SCALE_RUN_ROOT" --manifest "$RQ1_SCALE_MANIFEST"
"$RQ1_PY" -m rq1.cli university-gpu engineering-scale export-progress   --output-root "$RQ1_SCALE_OUTPUT_ROOT" --run-root "$RQ1_SCALE_RUN_ROOT" --manifest "$RQ1_SCALE_MANIFEST"   --destination "$RQ1_SCALE_EVIDENCE/final-export"
"$RQ1_PY" -m rq1.cli university-gpu engineering-scale verify-export --destination "$RQ1_SCALE_EVIDENCE/final-export"
"$RQ1_PY" -m rq1.cli university-gpu engineering-scale protect   --output-root "$RQ1_SCALE_OUTPUT_ROOT" --run-root "$RQ1_SCALE_RUN_ROOT" --manifest "$RQ1_SCALE_MANIFEST"
```

Retain all segment destinations and anchor snapshots. Raw outputs remain outside Git. A local export is not independently protected until destination readback succeeds. Final summary reports accepted/physical attempts, technical failures, retry history, per-condition/task outcomes and paired descriptive differences. No confirmatory p-values or inference treating repetitions as independent tasks. Acceptance records are never pooled into the 500.

## Storage and limits

Default local and independent reserves: 50 GB each. Production preflight also projects observed acceptance-run bytes per execution across the approved physical attempt allowance with a 12x copy/headroom factor, and requires the larger of that projection and 50 GB on both destinations. This is a conservative estimate, not a worst-case bound; monitor remaining space and retain the runtime reserve. Existing assets/model caches need additional space and separately approved preservation. Run exports contain code, results, traces and identity pins; they do not bundle model blobs or dataset caches. Prior transferred run evidence extrapolates to ~0.81 GB per 500 before omitted logs/checkpoints. Cumulative segment copies cost ~5.5x final evidence; actual lengths, retries, monitor logs and compression vary. Full snapshots every ten units remain unchanged for the old runner; scale uses per-unit checkpoints and final/explicit-stop bundles. Full boundary/export validation remains mandatory.

Launch readiness requires all offline checks, real four-GPU acceptance, independent or owner-reviewed authorization, approved persistent access, sufficient runtime/storage and verified independent readback. The software cannot establish institutional permission or physical backup independence by hashing a local path.
