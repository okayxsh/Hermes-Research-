# Final pre-Monday reliability report

Date: 2026-10-31. Branch: `feat/rq1-university-4gpu`.

Latest existing commit before Task5 work:
`c9a26443c48cfbe9f27ced5ce1b7745d68435cfe`. Task5 code and docs are currently
uncommitted. At takeover, no test processes were running; the initial working
tree contained only the intentionally untracked OpenCode audit.

Evidence basis: this report records coordinator-executed checks and source/test
inspection. It did not rerun the test suites, did not run Git writes, and did not
modify research data or inputs. Test timings below are as supplied, not
regenerated here.

## 1. Executive verdict

- **A11: PASS locally, synthetic.** Deadline rejection, cooperative cutoff, and
  retry gates are exercised with a mocked clock and an event-controlled fake
  worker. No real OS process was killed.
- **A19: PASS in the synthetic offline path.** Partial export, external
  verification, anchor handling, and hash-preserving resume were exercised
  without real power loss or disk failure.
- **Task4 durability: PASS in regression tests.** Snapshot idempotency,
  final-snapshot resume, orphan handling, and retry gates are covered locally.
- **Trust compatibility: PASS in synthetic tests only.** No genuine human
  approval, external anchor, or GPU authorization pin was created or tested.
- **Workstation status: NOT VERIFIED.** The workstation was unavailable. No
  WSL2, GPU inference, Ollama, ALFWorld, process kill, power loss, disk error,
  or independent backup/readback was tested.
- **Monday posture:** read-only diagnostics are CONDITIONAL on an existing
  clone and a pre-provisioned `.venv`. Benchmark and the 12 engineering units
  are NO-GO until the section 10 blockers clear. The 36-episode pilot and 540
  confirmation are NO-GO on scientific prerequisites.

## 2. A11 — edge deadlines, cutoff, and retry

### Deadline validation and assignment cutoff

- `feasibility.py:1293` rejects a cutoff that is missing, nonfinite, or already
  expired, and rejects a cutoff less than 30 minutes away.
- `scheduler.py:565-566` starts work only when
  `time.time() + minimum_start_seconds < cutoff`. The cutoff is therefore a
  start gate, not a hard kill deadline.
- Active work is not killed at cutoff. The scheduler waits for its future,
  accepts a completed result, stops new assignment, and reports cutoff
  (`scheduler.py:583-587, 716`).
- `tests/test_university_gpu_package.py:435-481` uses a mocked clock and an
  event-controlled fake worker. One active unit completes after cutoff, no
  second unit starts, `attempts_started=1`, `cutoff_reached` is true, and
  active workers at return are 0.
- **Limitation:** this is cooperative scheduling behavior. No real process was
  terminated, and no GPU/Ollama worker was involved.

### Retry safety gates

- For a latest `attempt_failed`, `scheduler.py` requires explicit
  `retry_failed=True` **and** both `safe_to_continue is True` and
  `mutation_state_known is True`. A missing or false flag raises
  `SchedulerError` before any worker starts.
- A nonterminal `attempt_started` with no terminal event is not silently
  rerun. It requires the explicit retry flag after an operator inspects the
  process and attempt state.
- `tests/test_university_gpu_package.py:364-433` covers these paths.
- Physical work may repeat after an interruption. No exactly-once claim is
  made, even though result publication is create-only and accepted result
  hashes are preserved.

## 3. A19 — partial export, evidence, and resume

- `tests/test_per_unit_durability.py:57-96` runs a synthetic 12-unit
  feasibility case and interrupts it after accepted units **1, 5, and 11**.
- Each partial export contains exactly the accepted results and sets
  `complete=false`. No partial export is presented as complete.
- Each partial export is copied to a second temporary destination, and the
  production `verify-export` command returns success there.
- Additional passing tests cover missing or mutated raw logs, evidence
  anchors, unanchored IDs, and hash preservation across resume.
- Resume preserves prior result hashes and does not replace accepted results.
- **Limitation:** the interruption is in-process and synthetic. There was no
  real power loss, no filesystem corruption, no disk failure, and no
  off-device copy/readback. Durability against hardware failure remains
  UNVERIFIED.

## 4. Task4 regression fixes

- Same-count full snapshots are verified and reused. An internally valid but
  stale bundle is preserved beside a new create-only current bundle. A corrupt
  same-count bundle fails closed (`snapshot.py:263-340`).
- A missing final full snapshot failure was reproduced and then resumed. The
  retry publishes exactly one final full snapshot, and a repeated completed
  resume reuses it without creating another.
- An orphan checkpoint or a completion event missing its authoritative result
  returns `durability_failed` with 0 attempts. It is not rerun.
- Safe-retry and nonterminal-retry gates remain enforced, consistent with
  section 2.
- No research input or recorded result was changed. Historical prep files
  still match `SOURCE-HASHES.json`; the archive SHA-256 remains
  `d52ed82e01a073f8c3382711d03f65916cb41ca89bc0b773e6845817bd37e4fd`.
- Pinned hashes remain unchanged from HEAD `c9a2644`:
  - frozen protocol `27b945a673d95d01c0a62b06b94cda3bb53ef3c2209e0202f955c7b1d4e18901`
  - 200-action candidate `c60ead3eddcae90ffcac26fe0de55c12395fe8ee55bf782937ea6f5373590385`
  - model config `9a62efdd7ea5fa5d8e3552e6b60fa5d16f1dc985796877c75a36fd362050cab4`
  - amended config `9336e7f8fe3c55b0c3f8d880efe0cba7c027c99218c839de19d236740bcc7f11`
- `C:\ResearchEvidence` is absent. The broader 219-entry ledger was not
  rechecked in this session.

## 5. Trust compatibility (N01/N02 only synthetic evidence)

- `tests/test_engineering_authorization_trust.py` and
  `tests/test_evidence_anchor.py` pass.
- Missing, corrupt, or wrong anchors or pins are rejected. Missing anchors are
  reported as unanchored. `test_only` and scientific scopes behave as designed;
  test-only scope is rejected where scientific authorization is required.
- Static candidate validation without GPU authorization pins returns PASS with
  `episode_count=12` and `scientific_evidence=false`.
- **Limitation:** all pins and approvals are synthetic test fixtures. No
  genuine human approval was created, and no human identity or independent
  channel was verified. No scientific pilot or confirmation authorization is
  inferred.

## 6. Tests and static checks (as executed by the coordinator)

Focused A11/A19/N01/N02/N04/N08 suite command:

```text
.venv\Scripts\python.exe -m unittest tests.test_university_gpu_package tests.test_per_unit_durability tests.test_engineering_authorization_trust tests.test_evidence_anchor tests.test_feasibility_provenance_regression tests.test_feasibility_evidence_regression tests.test_feasibility_durability_regression tests.test_autopilot_path_safety tests.test_feasibility_path_guard tests.test_university_gpu_path_safety tests.test_process_platform_safety
```

| Check | Result | Exit |
|---|---|---|
| Focused 11-module suite (preceded the last orphan-checkpoint guard fix) | `Ran 123 tests in 509.718s; OK (skipped=5)` | 0 |
| Post-guard orphan-checkpoint test | 1 test OK | 0 |
| Post-guard retry tests (known-safe, unsafe terminal flags, explicit nonterminal retry) | 3 tests OK | 0 |
| Full suite after all source changes: `unittest discover -s tests` | `Ran 468 tests in 647.612s; OK (skipped=9)` | 0 |
| In-memory compile of 261 Python files under `src/` and `tests/` | success, no `.pyc` created | 0 |
| `git diff --check` | clean; LF/CRLF advisory only | 0 |

Sequencing matters: the 123-test focused run preceded the final small
orphan-checkpoint guard correction. The post-guard tests cover that change, and
the 468-test full run is authoritative because it followed all source changes.
These results are not a substitute for workstation execution.

## 7. Runbook command verification and setup gaps

- All runbook feasibility `--help` forms were checked: `validate`,
  `reconcile-data`, `reference-library`, `replay-oracles`, `freeze`,
  `preflight`, `benchmark`, `launch`, `resume`, `status`, `stop`,
  `validate-run`, `merge`, `export-progress`, and `verify-export`.
- `evidence_anchor create`, `evidence_anchor verify`, and
  `prospective_identity --help` also exited 0.
- Static candidate `validate` without GPU pins exited 0 and reported 12 units
  with `scientific_evidence=false`.
- Command forms match the parser. This does not prove execution against real
  hardware, models, data, or approvals.
- The runbook assumes an existing clone at `~/Hermes-Research-` and
  `.venv/bin/python`. It contains no clone command and no venv creation or
  install command. Pre-provision through an authorized procedure; do not
  install software without university authorization.
- On-site checks still required: hardware, ALFWorld data, model, tokenizer,
  approvals, independent backup, and the access deadline.
- `NEXT_VISIT_RUNBOOK.md:254` says “GitHub is supplementary backup.” That is
  ambiguous because experiment outputs are outside Git. A later wording fix
  should state that GitHub is a source backup only. No edit was made in Task5;
  this remains a documentation limitation.

## 8. Workstation-only checks and authorization

Not tested in this session: WSL2 behavior, GPU inference or placement, Ollama,
real ALFWorld execution, real process kill, power loss, disk failure, and
independent backup copy/readback. The Windows symlink check was skipped with
WinError 1314; other optional real tests were also skipped. These remain
authorized university/workstation checks. No mock result is promoted to a
hardware, GPU-readiness, or real-ALFWorld claim.

## 9. OpenCode audit status and disposition

- `university-gpu/docs/OPENCODE_RQ1_REDTEAM_AUDIT.md` remains untracked and
  unchanged at SHA-256
  `9bb6512ac700cb29585ff21b696dca248b1009a94de8bd41e15785710c4d028d`.
  It must not be staged, changed, moved, or committed.
- **Recommendation B:** leave it out of the source commit. Only after explicit
  approval, archive a byte-identical copy to an approved external location and
  verify its SHA-256 there. No copy has been made.
- Clean-checkout impact: while the audit is untracked and Task5 changes are
  uncommitted, the runbook's clean-source gate fails. Audit disposition plus an
  approved source commit is a pre-Monday blocker.

## 10. Monday GO/NO-GO matrix

| Activity | Decision | Condition / reason |
|---|---|---|
| Read-only diagnostics (`validate`, `status`, identity/reference checks) | **CONDITIONAL** | Requires an existing clone and pre-provisioned project `.venv`. |
| Benchmark and 12 engineering units | **NO-GO** | Needs audit disposition, an approved commit, host hardware/model/data verified on site, human independent authorization pins, and an independent backup procedure. |
| 36-episode pilot | **NO-GO** | Scientific prerequisites are unmet; human approvals are not established. |
| 540 confirmation | **NO-GO** | Same scientific and authorization prerequisites; untouched `valid_unseen` and final protocol conditions are not cleared. |

No “no issues” or overall-readiness claim is made. A11, A19, Task4, and trust
results hold only within their synthetic, local test envelopes. Workstation
behavior and real durability remain UNVERIFIED.
