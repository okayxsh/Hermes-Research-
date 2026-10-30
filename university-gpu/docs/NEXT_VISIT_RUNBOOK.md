# Next university visit runbook

**Current state: do not start the scientific pilot.** The transferred archive is now verified and audited; see `TRANSFERRED_EVIDENCE_REVIEW_20261030.md` for the six-task findings. The evidence-copy instructions below are historical workstation preservation steps, not a claim that the transfer is still missing.  The local feature branch includes an audit commit not yet pushed, and the deployed host's Gemma digest `6114515d…` differs from the frozen launch identity `4eb23ef…`. The first reported task and all pilot-context logs still need source-level review. This runbook separates no-inference checks from launch gates.

## 1. Verify workstation and checkout

From Ubuntu, in the existing checkout:

```bash
cd ~/Hermes-Research-
git status --short --branch
git log -1 --oneline
git remote get-url origin
git branch --show-current
python3.11 --version
PYTHONPATH=src python3.11 -m rq1.cli university-gpu check --json
nvidia-smi -L
nvidia-smi
ollama --version
ollama list
ollama ps
```

Expect branch `feat/rq1-university-4gpu`, Python 3.11 and four RTX 5000 Ada devices. Record command output in a new dated evidence directory outside the checkout. Do not run `setup` unless a separately reviewed dependency gap and installation authorization exist. Verify the repository commit and branch before transferring any new commit from GitHub.

Ollama endpoints 11534–11537 should be free before launch. `check --json` reports port readiness. The launch manager creates four isolated Ollama servers itself; do not start a second set manually. A successful generic Ollama request does not prove per-GPU placement.

## 2. Recover and inspect evidence before task selection

Copy the complete `~/rq1-checkpoint-tests/` tree and `artifacts/task_manifests/proposals/pilot-39cce59288e59c8d.json` to a new read-only review copy outside the repository. Keep the first pick-and-place evidence directory distinct from the five-task batch. Hash the source files, then inspect every task ID, split, family, reference action array, prefix, checkpoint/post-detour states and digests, detour, action limit, oracle completion and ALFWorld transcript. Do not infer missing fields from the prose report or from another test run.

The frozen six-arm protocol specifies a 50-action **total episode** cap. For a task with prefix length `p`, the recovery allowance is `50 − p − 1` because the detour consumes one action. Its expert continuation is `route length − p`; passing the frozen expert route requires `route length + 1 ≤ 50`. When that fails, mark the task/checkpoint/oracle combination blocked and state that the expert script did not fit. It does not prove no shorter solution exists. The present oracle replays the exact frozen reference continuation; it is not a bounded search for a shorter alternate plan. If shortest-path constrained solvability is required, commission a deterministic ALFWorld-state planner with completeness/bounds tests before using model outcomes. Never change the 50-action cap or select tasks using agent recovery outcomes.

The pushed prelaunch report identifies `valid_seen:pick_and_place_simple-Pencil-None-Desk-310/trial_T20190909_113054_894334`, a six-action route, and a four-action prefix. It records two-action memory and no-retrieval successes under `non_scientific_prelaunch`, but no detour, state digests, or 50-action oracle evidence. It is context only and does not replace its separate raw checkpoint test files.

After inspecting original selection order and exposure, assess replacements only before any prospective outcomes, within the same family and the prespecified selection rule. Record all exclusions. Finalize exactly one valid-seen context per family and validate each oracle with the scientific cap before any task freeze.

## 3. Freeze exact inputs and human decisions

Before manifest proposal, require: six eligible controlled-failure contexts; complete Accum-18 library identity and text; two independent ratings for all 108 task-skill pairs plus adjudication; outcome-blind task/relevance leakage review; approved neutral filler; valid-seen-only Gate calibration; and a tokenizer file, revision and hash verified against the running model's tokenization behavior. Current Gemma digest differs from the historical frozen identity. Preserve the historical files and obtain an explicitly reviewed prospective environment/protocol amendment before adding an approved launch path. The repository's current `university-gpu` manifest and worker hardcode the historical digest, so preflight is expected to reject the installed model until that amendment is implemented and reviewed.

Keep config, proposals, approvals, run roots and reports outside the Git checkout. No approval JSON should be filled by automation.

## 4. Preflight command

After authorized prospective compatibility code exists and all frozen inputs/config are complete:

```bash
: "${RQ1_CONFIG:?Set to the actual reviewed config file outside the checkout}"
: "${RQ1_EVIDENCE:?Set to the actual dated evidence directory outside the checkout}"
PYTHONPATH=src python3.11 -m rq1.cli university-gpu preflight \
  --config "$RQ1_CONFIG" --manifest "$RQ1_EVIDENCE/pilot-frozen.json"
```

Require all checks to pass, including model digest, tokenizer and source/input hashes, all four unique GPUs/ports and space. A static pass alone does not validate endpoint model identity or device placement.

## 5. Approved launch command

Only after eligible tasks, all human rating/adjudication, tokenizer/filler/Gate inputs, prospective model amendment, frozen manifest and signed human scientific/GPU launch approval have passed review:

```bash
: "${RQ1_ACCESS_END_EPOCH:?Set to the university access deadline in Unix UTC seconds}"
PYTHONPATH=src python3.11 -m rq1.cli university-gpu launch \
  --config "$RQ1_CONFIG" --manifest "$RQ1_EVIDENCE/pilot-frozen.json" \
  --approval-file "$RQ1_EVIDENCE/pilot-launch-approval.json" \
  --yes --cutoff-epoch "$RQ1_ACCESS_END_EPOCH"
```

This is the current CLI syntax, **not current authorization or a command to run now**. Today it is blocked by the identity mismatch and missing scientific prerequisites. Launch starts servers, runs four separately approved smoke inferences, verifies process-to-GPU UUID placement, and only then assigns the 36 pilot episodes.

## 6. Monitor, stop and resume

In another shell, with the same frozen paths:

```bash
PYTHONPATH=src python3.11 -m rq1.cli university-gpu status \
  --run-root "$RQ1_RUN_ROOT" --manifest "$RQ1_EVIDENCE/pilot-frozen.json"
PYTHONPATH=src python3.11 -m rq1.cli university-gpu progress-report \
  --run-root "$RQ1_RUN_ROOT" --manifest "$RQ1_EVIDENCE/pilot-frozen.json"
nvidia-smi
```

Request cooperative stop; active episodes are allowed to finish:

```bash
PYTHONPATH=src python3.11 -m rq1.cli university-gpu stop \
  --run-root "$RQ1_RUN_ROOT" --reason "scheduled access cutoff"
```

After confirming no active owner and reviewing status, resume the same immutable manifest and approval. A retry is a new physical attempt; completed authoritative units are retained. Interrupted episodes restart at unit boundaries, never mid-trajectory.

```bash
PYTHONPATH=src python3.11 -m rq1.cli university-gpu resume \
  --config "$RQ1_CONFIG" --manifest "$RQ1_EVIDENCE/pilot-frozen.json" \
  --approval-file "$RQ1_EVIDENCE/pilot-launch-approval.json" --yes \
  --retry-failed --cutoff-epoch "$RQ1_ACCESS_END_EPOCH"
```

## 7. Export and separate backup

After the run stops, validate and export partial progress to a new path outside the run tree. Verify the archive on the destination copy:

```bash
: "${RQ1_RUN_ROOT:?Read the actual run root from the reviewed config and run ID}"
: "${RQ1_EXPORT:?Choose a new export path outside the run tree on approved storage}"
PYTHONPATH=src python3.11 -m rq1.cli university-gpu export-progress \
  --run-root "$RQ1_RUN_ROOT" --manifest "$RQ1_EVIDENCE/pilot-frozen.json" \
  --destination "$RQ1_EXPORT"
PYTHONPATH=src python3.11 -m rq1.cli university-gpu verify-export --destination "$RQ1_EXPORT"
tar -czf "$RQ1_EXPORT.tar.gz" -C "$(dirname "$RQ1_EXPORT")" "$(basename "$RQ1_EXPORT")"
sha256sum "$RQ1_EXPORT.tar.gz" | tee "$RQ1_EXPORT.tar.gz.sha256"
```

Transfer the tarball and checksum through university-approved storage, verify SHA-256 at the destination, and retain a second independent copy. Keep full logs, task observations, model files, caches and raw ratings out of GitHub unless data/publication permissions explicitly allow them. Only reviewed small manifests and public-safe reports belong in Git. Confirm intended repository, branch and staged-file list before committing or pushing.

## 8. Blockers and owners

| Blocker | Owner |
|---|---|
| Export/checksum and source review of separate first task plus five-task batch and original proposal | Workstation operator + research engineer |
| Budget-feasible oracle result for each finalized task under total 50; possibly a tested constrained planner | Research engineer; protocol amendment by authors if needed |
| Selection/exposure review before same-family replacement | Authors + task reviewer |
| Current Gemma manifest/blobs and tokenizer provenance; prospective identity/protocol implementation | Research engineer; identity/budget decision by authors |
| 108 independent labels, disagreement adjudication, approved filler and Gate review | Two human raters + adjudicator/authors |
| Formal freeze and GPU launch authorization | Authorized authors |
| WSL persistence, export destination and independent backup readback | University workstation operator/IT |
| Git author identity and approved public artifact policy | Repository owner/authors |

No scientific inference is authorized by this runbook. The engineering recovery demonstration, historical study, and prospective pilot remain distinct evidence sets.

## Laptop Git handoff

The laptop origin is HTTPS; the university deploy-key setup must not be assumed to exist here. An SSH port-443 read probe failed host-key verification. Do not bypass verification or copy a private university key. The owner can configure a separate authorized laptop key and verify GitHub's published host fingerprint through an official trusted source, then run `git ls-remote ssh://git@ssh.github.com:443/okayxsh/Hermes-Research-.git refs/heads/feat/rq1-university-4gpu`. Only after that succeeds, push the preserved feature history with `git push ssh://git@ssh.github.com:443/okayxsh/Hermes-Research-.git HEAD:refs/heads/feat/rq1-university-4gpu` and read back its head. Never force push.

For a short future coordinator exercise, the real launch/resume CLI supports `--max-attempts 4`. It still requires every scientific/model/input and GPU approval gate; adding this flag does not turn an unapproved pilot into an engineering smoke run. A separate engineering-only manifest/approval path is not currently implemented. Continue independent per-device engineering checks under their existing explicit authorization; do not fake a pilot freeze to benchmark the coordinator.
