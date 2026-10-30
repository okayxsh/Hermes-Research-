# Prospective valid-seen pilot audit — 2026-10-30

Status: **BLOCKED for scientific inference**. Offline engineering changes are implemented; real checkpoint eligibility, new model/tokenizer identity, human relevance ratings and approvals remain unverified or missing.

## Evidence and provenance

The local feature branch and remote feature branch both began at `7d65e47a15b958e81e8e8a1be8cab54a882563dc`; initial tracked/untracked status was clean. Origin was verified as `https://github.com/okayxsh/Hermes-Research-`. No other checkout was changed. The user reports the new model and real integrations; those reports are not independently reproduced here. The local proposal `artifacts/task_manifests/proposals/pilot-39cce59288e59c8d.json` and university `~/rq1-checkpoint-tests/` are absent from this checkout.

The existing Remote Desktop session was observed without entering commands. The terminal displayed `RECOVERY RESULTS: {"memory": true, "no_memory": true}` for engineering checks and a failed Git commit (`Author identity unknown`). These are not scientific six-arm results, and neither that screen nor the user report supplies all six task/state/hash/oracle records. The computer-use skill prohibits terminal automation; no remote commands, authentication, installation or inference were performed. No supported direct remote filesystem connection was available to inspect the logs. Preserve the remote staged/WIP engineering evidence, including the separate first pick-and-place test; do not rerun it merely to make a package.

## Recovery budget: verified defect and eligibility blockers

`src/rq1/evaluation/amended_protocol.py` sets `TOTAL_EPISODE_ACTION_BUDGET = 50`. The protocol defines recovery budget as **50 − prefix length − 1 detour**. `task_preparation.validate_controlled_failure` uses this bound and the real harness replays the unchanged reference continuation after the detour. Consequently its full scripted route must use at most 49 reference actions plus the detour. An oracle started with 200 actions proves neither scientific-budget feasibility nor eligibility.

| Reported family | Reference route | Prefix | Remaining recovery | Continuation | Assessment |
|---|---:|---:|---:|---:|---|
| clean_and_place | 92 | 48 | 1 | 44 | BLOCKED: frozen continuation exceeds budget by 43 |
| pick_two_and_place | 109 | 74 | -25 | 35 | BLOCKED: prefix and detour already exceed total budget; complete script exceeds it by 60 |
| pick_and_place | unavailable | unavailable | unavailable | unavailable | BLOCKED: separate original evidence needed |
| heat family | unavailable | unavailable | unavailable | unavailable | BLOCKED: raw route and oracle needed |
| cool family | unavailable | unavailable | unavailable | unavailable | BLOCKED: raw route and oracle needed |
| light family | unavailable | unavailable | unavailable | unavailable | BLOCKED: raw route and oracle needed |

The two lengths are user-provided concerns, not independently read logs. Their arithmetic is exact if those lengths are correct. This rejects these **task/route/checkpoint combinations**, not every possible plan for the underlying tasks. Moving the checkpoint alone cannot fix a route of 92 or 109: the full route plus one detour still exceeds 50. Do not shorten the route after seeing model recovery outcomes or silently enlarge the budget.

Before any scientific recovery outcomes, preserve the initial proposal and all exclusions, use its recorded family/seed/order policy to consider the next same-family candidate, and run `validate_controlled_failure(..., split="valid_seen", reference_actions=...)` with the existing 50-action protocol. Record every failed candidate and infrastructure failure; infrastructure faults must halt rather than change selection. No replacement task IDs can be nominated responsibly without the proposal and discovery/route records. If its policy does not define replacements, obtain a prospective amendment before selecting them. Record exposure from prior engineering model tests and prevent that exposure from driving selection. Do not visit valid-unseen tasks to resolve pilot eligibility.

The six-arm manifest validator previously checked only positive recovery budgets. It now also rejects inconsistent budget arithmetic, invalid reference prefixes and continuations that exceed 50 total actions. Existing historical inputs and constants retain their hashes.

## Candidate evidence and schema boundary

`python -m rq1.university_gpu.pilot_readiness` packages six normalized definitions from the existing task-preparation record format, with additional evidence references. It hashes source files, checks the same task/state oracle evidence and budget, and writes a create-only candidate review report. It never freezes a task manifest or populates approvals. A supplied JSON oracle record still requires authenticity review against raw logs and real replay; offline consistency is not independent real-world validation.

Each definition needs the original fields `task_id`, `task_family`, `split`, `checkpoint_id`, `reference_actions`, `prefix_actions`, `checkpoint_digest`, `detour_action`, `post_detour_digest`, `recovery_action_budget`; retain all available original observation and source metadata. Add `source_evidence: [{"path": "relative/task/file", "sha256": "..."}]` for the original task files, and `oracle_evidence: {"path": "relative/real-oracle.json", "sha256": "..."}`. The separate saved oracle summary must bind `task_id`, both state digests, `validated=true`, `success=true`, `total_action_limit=50`, and `total_actions` (prefix + detour + unchanged continuation). Summary fields must be extracted from real logs, never inferred from a claimed pass. Preserve the raw logs and their hashes alongside it. Run the existing task-manifest/schema validation and controlled-failure replay before any freeze. No validated six-task package was created here because the source evidence is unavailable.

## Hermes application runtime

`RQ1_HERMES_PYTHON` selects an explicit absolute interpreter in the installed Hermes application venv. Without it, discovery considers the legacy application path and `~/.hermes/installs/*/environments/*/venv/bin/python`; zero or multiple installations fail closed. A valid venv layout, actual venv execution, required Hermes plugin symbols, model-tools hook and project plugin imports are checked. Python symlinks are not resolved to base Python. The episode driver, real integration verifier and runtime preflights share this resolver. An import pass does not prove actual registry dispatch: retain and rerun the existing genuine integration check on the selected university installation without model inference. The local machine cannot verify university Hermes 0.22.5 compatibility.

## Prospective model and human prerequisites

`university-gpu/prospective/environment-6114515d.proposed.json` records the user-reported new digest separately and explicitly remains unapproved. Preserve the exact manifest bytes and referenced blobs, identify and hash tokenizer files, verify tokenizer revision/library/template/special-token behavior and Ollama token counts, and record the actual runtime versions before freezing a new environment. SBERT snapshot provenance does not verify the Gemma tokenizer. Existing launch checks intentionally continue to reject the new digest: a separately versioned, reviewed prospective protocol/launcher identity path is still required. Historical baseline files are references, not contemporaneous primary cells.

After six eligible contexts are finalized and independently reviewed, use the existing `make_rating_templates.py` to produce two blank independent forms, an adjudication worksheet and 108 task-skill pairs from frozen Accum-18. It requires a genuinely frozen task manifest and oracle definitions; do not fake a freeze to generate it. All task instructions must be present in the blinded worksheet. Exclude condition IDs, selected skills, recovery success, oracle routes and treatment allocation from human rating materials. Run `validate_rating_package.py --help` for the existing schema/coverage validator; missing ratings, adjudication, coverage, Gate calibration, filler/token matching and human approvals remain blockers. No human judgments or forms for unsettled contexts were generated.

## Durability and backups

The existing coordinator gives one authoritative result per unit, distinct retry attempts, exclusive ownership, heartbeat and corruption checks. Resume works at episode boundaries, not mid-episode. A Remote Desktop disconnect alone need not stop a Linux process, but WSL/workstation termination may lose the active attempt; use university-approved terminal persistence and do not promise uninterrupted execution. `export-progress` copies and hash-verifies incomplete runs only after the coordinator stops and refuses corrupt authority records. `verify-export` checks transferred files and authoritative results. Existing tests cover incomplete export, corruption, backup failure, retries, duplicates and resume.

A concurrency defect in create-only file publication was fixed: `os.link` now publishes the fsynced temporary file atomically without replacing a concurrent winner. Filesystems that do not support hard links fail closed. Local fsync and hash checks do not establish off-host backups or power-loss behavior on the university filesystem.

Use the operator commands below to create a verified external export, compress it, and checksum the archive. Store the full archive on separately approved storage; do not put it in Git. For GitHub, explicitly stage only reviewed small reports/manifests with safe paths and no secrets, personal data, traces, cache or model material. Verify the remote and feature branch again at transfer time. Publication permissions and university data-storage rules remain external dependencies. An upload is not a backup until the remote object and its checksum are independently read back.

## Operator commands (no scientific inference)

Run from the updated repository on Ubuntu. Supply actual existing paths; these commands do not manufacture missing evidence.

```bash
export RQ1_HERMES_PYTHON="/home/researcher/.hermes/installs/ACTUAL_INSTALL/environments/ACTUAL_ENV/venv/bin/python"
PYTHONPATH=src python3.11 -m rq1.hermes.runtime
PYTHONPATH=src python3.11 -m rq1.cli university-gpu check --json
PYTHONPATH=src python3.11 -m rq1.university_gpu.pilot_readiness \
  --failures "$RQ1_NORMALIZED_REAL_FAILURES" --evidence-root "$RQ1_RAW_EVIDENCE" \
  --output "$RQ1_NEW_CANDIDATE_REPORT"
```

After genuine task freeze and complete context verification, generate blank human materials:

```bash
PYTHONPATH=src python3.11 university-gpu/make_rating_templates.py \
  --tasks "$RQ1_FROZEN_PILOT_TASKS" --failures "$RQ1_PILOT_FAILURES" \
  --library "$RQ1_FROZEN_LIBRARY" --records "$RQ1_CONTEXT_RECORDS" \
  --output "$RQ1_NEW_RATING_DIR"
```

After a stopped run (engineering checks are not scientific results):

```bash
PYTHONPATH=src python3.11 -m rq1.cli university-gpu export-progress \
  --run-root "$RQ1_RUN_ROOT" --manifest "$RQ1_FROZEN_MANIFEST" --destination "$RQ1_NEW_EXPORT_DIR"
PYTHONPATH=src python3.11 -m rq1.cli university-gpu verify-export --destination "$RQ1_NEW_EXPORT_DIR"
test ! -e "$RQ1_NEW_ARCHIVE" && tar -czf "$RQ1_NEW_ARCHIVE" -C "$RQ1_NEW_EXPORT_DIR" .
sha256sum "$RQ1_NEW_ARCHIVE"
```

Save the printed checksum separately and verify the transferred archive against it. For small evidence already reviewed for GitHub publication:

```bash
git remote get-url origin
git branch --show-current
git status --short
# Continue only for okayxsh/Hermes-Research- and feat/rq1-university-4gpu.
git add -- "$RQ1_REVIEWED_SMALL_REPORT"
git diff --cached --stat
git diff --cached
git commit -m "Preserve reviewed pilot preparation evidence"
git push origin HEAD:refs/heads/feat/rq1-university-4gpu
git ls-remote origin refs/heads/feat/rq1-university-4gpu
```

The remote workstation currently needs a legitimate Git author email; do not enter the placeholder shown on its screen as an identity. Repository-local Git identity should be set by the owner. No push was performed by this audit.

Scientific launch commands remain in `OPERATIONS.md`, but are **not yet allowed**: finish eligible-task evidence, prospective identity implementation/review, ratings/adjudication, Gate/filler/tokenizer validation, and approvals first. Four GPU presence and successful engineering episodes alone do not satisfy these prerequisites.

## Local verification results

Python 3.11 unittest discovery: 349 tests run, 4 skipped, no failures or errors. After final modifications, the focused readiness plus university-package suite ran 38 tests with no skips or failures. These include fake-provider multiworker recovery, incomplete export, corruption, duplicate authority, safe cutoff and resumed completion. Python compilation and `git diff --check` passed. `pytest` was not installed in Python 3.11; no installation was attempted. Four frozen model/protocol identity files match the base commit after applying the existing Git checkout line-ending filters; their actual SHA-256 values are recorded in `university-gpu/reports/pilot-readiness-20261030.json`. Raw Git blobs use different line endings from the Windows checkout, so raw blob SHA-256 values must not be confused with working-file hashes. No GPU inference or real university oracle replay was run by this audit.
