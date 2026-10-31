# N04 and N08 durability remediation

## N04: run paths

On the pre-change `621b4a3961bb13a972a8af369f73cc56db30360e` source, a disposable `run_dir(root, "../../outside")` resolved outside `artifacts/autopilot`; no outside file was opened or changed. Autopilot creation and consumption now require canonical UUID IDs and check the resolved state, log and plan paths. Stored plan hashes and output paths are checked on load; the old five-field hash remains accepted only when all other plan fields equal their original defaults. New state publication is create-only. The generic university manifest validates run IDs on both construction and consumption and bounds launch paths under its configured output root.

Feasibility status, stop, validate, merge and progress export now require an explicit `--output-root`; they validate the selected run against that root and its frozen manifest before reading or writing. Anchor creation takes the same explicit output root. Generic stop no longer creates an arbitrary missing run directory, and its operational commands check a matching frozen manifest. The run store rejects traversal and symlinked path components for its managed results, attempts, events and backups. The Monday runbook documents the changed command forms.

This is an intentional command-interface change: feasibility run-inspection commands need the new flag, and older generic examples using `../` or `..\` in a run path must be replaced by a canonical operator-selected path. Existing valid files and directory layouts are not migrated or rewritten.

These checks reduce path and stale-lock races; ordinary path checks are not an OS-enforced sandbox against another actor able to replace ancestor directories between validation and use. Use an operator-controlled output root and verify filesystem ownership on WSL2. The locally supported Windows host could not create actual symlinks without additional privileges, so mocked symlink checks supplement tests; real WSL2 symlink behavior remains unverified here.

## N08: completed-unit durability

The baseline copied full snapshots only after 10 completed units and at the final unit. That did **not** mean a coordinator interruption erased the preceding nine: accepted result JSON and events were published atomically and fsynced, with their raw traces stored separately. A crash between result publication and a completion event could be reconciled on resume. The baseline did not establish that every raw trace and local backup had reached durable storage, and a failed disk or power loss could affect results and logs sharing that disk. No historical loss is inferred.

After each accepted result and completion event, the coordinator copies that unit's result and closed attempt files once into a create-only `backups/units/<unit-key>/` checkpoint. It verifies file inventory, result identity, hashes and, for feasibility runs, the copied raw trace semantics against the frozen task. Files and checkpoint directories are flushed where supported before atomic publication; I/O failures remain failures. Periodic full bundles remain separate. Validation can still reread previously completed traces, so actual storage and checkpoint overhead must be measured on the workstation. A reported `verified` checkpoint is local evidence, not proof of an independent backup.

On resume, existing checkpoints are verified and re-flushed. A missing checkpoint can be rebuilt only from still-valid authority; an invalid checkpoint or checkpoint with no authoritative result blocks new assignments and is never overwritten. A recorded completion without its result also blocks resume rather than being rerun; an existing STOP marker is retained when recovery checks fail. A crash during publication leaves a visibly partial directory, which is not counted as a checkpoint. Checkpoint failure preserves accepted results, stops further assignment and returns `durability_failed`; an already active worker may finish. A failure after result publication is not recorded as an ordinary worker failure. Final checkpoint verification failures also affect the durability status. Valid partial results can still be exported, with source checkpoint health disclosed.

Full-run bundles are also idempotent. Before publishing, the coordinator verifies an existing complete bundle for the same run and completed count against its manifest inventory and the current closed source files. A valid current bundle is reused without another publication; a valid but stale bundle is left intact while a new create-only current bundle is published; a partial, malformed or corrupt same-count bundle fails closed and is never overwritten. This removes repeated same-count bundles from ordinary complete resumes, but it is not an independent-backup claim and does not make physical execution exactly-once.

With concurrent workers, an episode may physically finish before its result has been received and published by the coordinator. An interruption in that window leaves a nonterminal attempt rather than an accepted result; its retained traces need review and a later retry can repeat physical work. The checkpoint guarantee begins with accepted authority, and does not claim exactly-once physical execution.

External per-unit evidence anchors remain separate: configured anchor conflicts fail validation, and missing anchors are reported as unanchored. Create-only local files and same-disk checkpoints cannot establish independent authentication or survive total workstation-storage failure. The operator must copy and verify protected anchors and a backup on separate storage. The runbook describes status inspection, stop, partial export, checkpoint failure and resume.

Stale run-lock recovery is serialized with a create-only takeover marker. Competing resumers fail closed rather than replacing a newly active lock. A takeover marker left after an interrupted takeover requires manual inspection; do not remove it until ownership and run state are established.

## Audit and protected inputs

The untracked OpenCode audit was read and scanned for common credential patterns, absolute personal paths, host names and GPU UUIDs; no matches were found in that bounded scan. Its SHA-256 remains `9bb6512ac700cb29585ff21b696dca248b1009a94de8bd41e15785710c4d028d`. It contains research findings, so the user should decide whether a separate documentation-only **local** commit is appropriate for the repository's intended audience. It was not modified, committed, ignored or published here. While it remains untracked, the clean-source freeze gate remains blocked.

The Oct 20 input ledger was checked before and after implementation: 219 MATCH, 0 MISMATCH, 0 MISSING each time, including six relocated archives. The tracked 50-action protocol, 200-action candidate profile, historical inputs, treatment definitions and model identities have no changes. Exact test outcomes are recorded after verification. No GPU inference or research episode was run. Actual WSL2 fsync behavior, reboot/power-loss recovery, device failure, university authorizations and off-machine backup/readback remain to be verified on the workstation.

## Final verification on current source

Verification ran on branch `feat/rq1-university-4gpu` at pre-Task-4 HEAD `621b4a3961bb13a972a8af369f73cc56db30360e`, using Python 3.11.9 in the local `.venv`. Task 4 remained uncommitted when this section was appended.

### Test results

- Initial full suite (before the final snapshot-retry patch; superseded by the final result below): `python -m unittest discover -s tests` -> `Ran 462 tests in 397.391s; OK (skipped=9)`, exit 0.
- N04/N08/trust focused modules after the final code patch, run with `PYTHONDONTWRITEBYTECODE=1`: `.venv\Scripts\python.exe -m unittest tests.test_autopilot_path_safety tests.test_feasibility_path_guard tests.test_university_gpu_path_safety tests.test_per_unit_durability tests.test_feasibility_durability_regression tests.test_feasibility_evidence_regression tests.test_feasibility_provenance_regression tests.test_evidence_anchor tests.test_engineering_authorization_trust tests.test_process_platform_safety` -> `Ran 89 tests in 192.866s; OK (skipped=5)`, exit 0.
- Durability module after the final code patch: `.venv\Scripts\python.exe -m unittest tests.test_per_unit_durability -v` -> `Ran 21 tests in 96.487s; OK (skipped=1)`, exit 0. These 21 tests are included in the 89 focused tests above, not additive.
- Final full suite after the final code patch: `.venv\Scripts\python.exe -m unittest discover -s tests` -> `Ran 463 tests in 426.317s; OK (skipped=9)`, exit 0.

Skips include host-inaccessible symlink creation (`WinError 1314`) and optional real-environment/profile tests. No real WSL2, GPU, Ollama or ALFWorld test was run.

Static checks: 15 modified/new Python files were compiled in memory, exit 0, with no bytecode artifacts; `git diff --check` exited 0.

### Reproduced durability defect

A fresh synthetic 12-unit feasibility run completed all 12 units, but an injected failure on the second full snapshot returned `durability_failed` with `full_snapshot_status=failed`. Resume previously returned `completed` with `full_snapshot_status=not_attempted`. This was fixed in the scheduler and is covered by `test_resume_retries_missing_final_full_snapshot_without_rerunning_units`; the targeted module and final suite pass. After the fix, a resume with no current final bundle publishes exactly one new create-only full snapshot without starting another worker or changing result hashes. A second complete resume verifies and reuses that current bundle, so the final-snapshot count stays unchanged. This is not an independent-backup or exactly-once physical-work claim.

### Protected inputs

- The historical preparation `historical-inputs` hashes for `evaluation-task-manifest.json`, `controlled-failures.json` and `evaluation-libraries.json` all match `SOURCE-HASHES.json`. The original `rq1-evaluation-360-final-20260914.tar.gz` SHA-256 matches `d52ed82e01a073f8c3382711d03f65916cb41ca89bc0b773e6845817bd37e4fd`.
- `university-gpu/docs/PROTOCOL.md`, `university-gpu/prospective/feasibility-200-candidate.json`, `configs/models/gemma4-12b.yaml` and `configs/evaluation/amended.yaml` have no diff from pre-Task-4 HEAD. Current SHA-256 values: protocol `27b945a673d95d01c0a62b06b94cda3bb53ef3c2209e0202f955c7b1d4e18901`; candidate `c60ead3eddcae90ffcac26fe0de55c12395fe8ee55bf782937ea6f5373590385`; model config `9a62efdd7ea5fa5d8e3552e6b60fa5d16f1dc985796877c75a36fd362050cab4`.
- `C:\ResearchEvidence` does not exist. The three SOURCE-HASHES entries and the archive hash above were verified through the alternative existing research-workspace paths. The broader 219-entry ledger was not recomputed in this session.
- CLI `--help` exited 0 for runbook feasibility `status`, `stop`, `validate-run`, `merge`, `export-progress`, and `evidence_anchor create` / `verify` with their exact current flags.
- The OpenCode audit SHA-256 remains `9bb6512ac700cb29585ff21b696dca248b1009a94de8bd41e15785710c4d028d`; `git ls-files --error-unmatch` confirms it is untracked. It was not altered and must not be included in the Task 4 commit.

### Task 4 commit contents (17 paths)

Twelve modified tracked paths plus four new tests and this report:

1. `src/rq1/autopilot/cli.py`
2. `src/rq1/autopilot/executor.py`
3. `src/rq1/autopilot/state.py`
4. `src/rq1/university_gpu/cli.py`
5. `src/rq1/university_gpu/evidence_anchor.py`
6. `src/rq1/university_gpu/feasibility.py`
7. `src/rq1/university_gpu/manifest.py`
8. `src/rq1/university_gpu/ops.py`
9. `src/rq1/university_gpu/scheduler.py`
10. `src/rq1/university_gpu/snapshot.py`
11. `tests/test_process_platform_safety.py`
12. `university-gpu/docs/NEXT_VISIT_RUNBOOK.md`
13. `tests/test_autopilot_path_safety.py`
14. `tests/test_feasibility_path_guard.py`
15. `tests/test_per_unit_durability.py`
16. `tests/test_university_gpu_path_safety.py`
17. `university-gpu/docs/N04_N08_DURABILITY_REMEDIATION.md`

`university-gpu/docs/OPENCODE_RQ1_REDTEAM_AUDIT.md` is explicitly excluded: it remains untracked and outside the Task 4 commit.

### Limitations

1. No real symlink operations were performed on this host where those tests were skipped.
2. No actual WSL2, GPU, process-kill, power-loss, disk-failure or independent-backup verification was performed.
3. Full snapshots are supplemental to per-unit checkpoints, and the status CLI reports per-unit checkpoint health rather than historical full-bundle health; the immediate launch/resume outcome reports `full_snapshot_status`.
4. No persisted autopilot state was found or loaded, so legacy state compatibility was exercised only by synthetic tests, not genuine prior run records.
5. No independent copy of anchors or backup was verified.
6. The preserved OpenCode audit contains internal scientific, security and operational findings; no credential value was visible on direct read. I recommend option B: keep it out of the Task 4 commit and, only after explicit user approval, archive a byte-identical copy in an approved external location with SHA-256 verification. No copy was made, and the existing file remains untracked and unchanged.
