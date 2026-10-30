# RQ1 audit remediation â€” 2026-10-30 through 2026-10-31

Scope: targeted engineering remediation of audited commit `dab5ed243d04c98f389fe809064a10ecd689bfec` on `feat/rq1-university-4gpu`. The prior untracked independent audit is preserved unchanged and included with this remediation. No model downloads, installations, GPU inference, real episodes, ratings, human approvals or remote pushes were performed. The existing Python 3.11 virtual environment was used with explicit `PYTHONPATH=src`.

## Baseline and reproduction

Baseline: 370 tests ran in 123.522 seconds, 366 passed and four optional real integrations skipped, exit 0. The baseline suite was started before implementation edits. Prior audit reproductions were retained; the provenance engineer additionally loaded the original audited module from Git read-only and reproduced candidate oracle acceptance/frozen rejection and authentic historical library-reference rejection. New regressions exercise the other demonstrated defects using temporary synthetic assets and providers.

Historical SHA comparison before remediation: all 219 authentic Oct 20 reference files matched (six relocated archive files resolved in `03_Experiment_Archives`). Final comparison is recorded below. Scientific 50-action constants, historical digest, archived evidence, labels and candidate task selection are protected; no test fixture is presented as human approval.

## Remediation matrix

Test names below resolve under `tests/`; implementation filenames resolve under `src/rq1/university_gpu/`. FIXED means the specific laptop-resolvable defect has direct regression evidence. It does not attest real external assets or university hardware.

| Finding | Exact change and proving evidence | Remaining uncertainty | Status |
|---|---|---|---|
| A01 oracle binding | `feasibility.py` preserves full original candidate and its hash; validates original task records and candidate ancestry; oracle checks frozen `candidate_manifest_sha256`. `test_feasibility_provenance_regression`: candidateâ†’referenceâ†’freezeâ†’preflightâ†’partial exportâ†’resumeâ†’full export using the same frozen output; negative ancestry/hash/checkpoint mutation. Also repaired missing tokenizer imports exposed by this previously untested path. | Synthetic oracle rows/tokenizer are not real environment validation. | FIXED |
| A02 historical library ancestry | `reference-library --archive --archive-manifest` verifies pinned archive SHA, unique regular member, ledger size/hash, exact supplied library bytes, 18 distinct original IDs/texts. Source ancestry is reverified at freeze/preflight. Real original archive/member/library accepted without patched constants; wrong/missing provenance rejected in regressions. | University transfer must preserve archive/ledger/library paths and bytes. | FIXED |
| A03 raw log completeness | New `evidence.py`; worker supplies mandatory portable inventory; publication/completed reads/export verify required retrieval/failure/recovery/episode traces, bounded paths, byte hashes and identity. Missing/mutated/substituted logs fail. Raw environment responses are now checked against recovery steps and summaries; retrieval records bind run/episode identity. Nine negative/boundary regressions independently passed. | Real producer/environment integration remains target-only. | FIXED |
| A04 real assets | Fail-closed checks retained; real model/tokenizer/SBERT/data/placement proof still required. | University-only verification and human authenticity review. | BLOCKED |
| A05 sample exposure | No scientific sample or methodology changed. | Authors must choose fixed-task follow-up versus fresh confirmation. | BLOCKED |
| A06 analysis/power | No inferential design changes. | Primary contrasts, multiplicity, missingness, clustering and simulations required. | BLOCKED |
| A07 relevance/filler/Gate | No ratings fabricated or task selections approved. | Independent human labels, coverage, semantic filler review and calibration required for science. | BLOCKED |
| A08 result semantics | Scheduler checks actual recovery action counts/budgets/steps, termination, unique library IDs, finite cosine values and arm shape. Evidence checks tie raw actions/counts/retrieval to result. Negative and boundary regressions in `test_feasibility_evidence_regression`. `actions` means recovery actions in existing schema; total includes scripted prefix+one detour. | No claim of real model outcomes. | FIXED |
| A09 live snapshots | Automatic snapshots copy completed attempt evidence only; failed/time-out writers may still be live and are explicitly omitted along with active attempts. Current-root resolution supports relocated runs. Stopped exports preserve failed/partial evidence. Concurrent writer, closed mutation, relocation and failed-writer exclusion regressions. | Operator must stop all writers before full/partial export; host filesystem semantics untested. | FIXED |
| A10 backup status | Separate execution/backup/export/independent-copy states; final backup failure gives `durability_failed`, preserves accepted results and does not claim independent copy. Injected final backup failure regression. | Independent target copy/readback must be performed externally. | FIXED |
| A11 cutoff | Reject boolean/nonfinite/expired/insufficient-reserve deadlines before assets. Existing finite assignment cutoff preserved; documentation explains active episodes can overrun and need measured reserve. Negative deadline regressions. | No hard forced-termination guarantee; target timing unknown. | FIXED |
| A12 interpreter | Runbook consistently uses `~/Hermes-Research-/.venv/bin/python` via `RQ1_PY`, checks 3.11/import locations and pip dependencies. `test_monday_runbook_cli` parses actual documented command forms. | Host venv/dependency installation and lock agreement must be checked. | FIXED |
| A13 run containment | Same scoped run-ID regex at freeze/consumption; resolved run root must remain below approved output root and reject symlink escape. Traversal/symlink regressions. | Approved output-root ownership remains host gate. | FIXED |
| A14 construct scope | Engineering-only labeling preserved; methodology untouched. | Broader scientific claims need narrowing/design evidence. | BLOCKED |
| A15 noise estimand | No treatment/analysis redesign authorized here. | Retrieved versus injected relevance and policy contrast interpretation remain scientific work. | BLOCKED |
| A16 WSL/session/storage | Runbook retains explicit target-only lifecycle/supervision and independent backup requirements. | RDP disconnect, WSL termination, reboot, institutional permissions unresolved. | BLOCKED |
| A17 shared model storage | Fail-closed startup model checks retained. | Protected model storage/ownership and tag-mutation prevention require operator/IT evidence. | BLOCKED |
| A18 tokenizer revision | Identity report exposes provenance revision; freeze requires reviewed/report equality; preflight requires provenance/config/report/review/frozen equality plus exact-byte checks. Mismatched-revision regressions. | Authentic tokenizer/model linkage remains human review, not inferred from test samples. | FIXED |
| A19 partial export runbook | Separate monitor/stop/resume, complete validation/merge and incomplete direct export paths; unique export/archive names with no-clobber archive creation. Actual documented CLI forms parsed offline. | Backup destination and readback permission still external. | FIXED |

## Offline integration meaning

The provenance regression creates a real temporary Git checkout, synthetic ALFWorld source tree, archive/member ledger, library, model manifest/blobs and cached SBERT snapshot. It exercises actual hashing/discovery/reconciliation and actual freeze/preflight. Expected synthetic asset digests are patched in test scope; the tokenizer implementation, approvals, oracle rows and inference provider are explicitly fixtures. The same resulting frozen manifest proceeds through partial execution/export/verification and resume/full validation/merge/export. Accepted result bytes are preserved across resume; mutated model bytes fail preflight.

This closes the former component-boundary gap. It does not prove real ALFWorld game behavior, real tokenizer correspondence, Ollama compatibility, four-GPU placement, throughput or host persistence. The four real optional integration tests remain disabled.

## Adversarial verification

Four actual review assignments challenged other engineers' changes: A provenance/frozen preflight, B raw evidence/result correctness, C concurrency/resume/backup safety, D tests/history integrity. They share a filesystem and AI limitations and are not external human certification. No target scores were used.

A independently accepted the authentic historical archive, ledger and recovered library: 18 skills, SHA `e01d3a569ee8373d8cc235c4343f2b52f92c9bf41751b5d3e435ced70648e1f5`. B and D identified missing raw environment-response consistency and missing retrieval episode binding, rather than simply endorsing the changes. C identified absolute attempt paths on relocation and failed attempts whose timed-out worker could still append; automatic checkpoints now derive current-root paths and omit unresolved failed writers. Regression and final review outcomes are added below.

## Remaining Monday gates and human decisions

Monday is **NO-GO for episodes until actual host gates pass**, even when all local regressions pass. The package can then be used for real university validation in the corrected runbook:

1. Transfer/verify the explicitly reviewed clean commit; preserve branch history and host-key verification. No push is authorized by this remediation request.
2. Verify project Python 3.11 venv, dependency versions, original data and all scoped identity differences; obtain genuine data review.
3. Verify actual prospective Gemma manifest/blobs/Q4_K_M, authentic tokenizer linkage and exact revision equality, pinned SBERT cache; obtain genuine model review. Historical digest is not substituted.
4. Verify historical Accum-18 archive/member/library ancestry and replay all six exact CPU oracle witnesses under cap 200. Do not regenerate checkpoints or swap tasks on failure.
5. Obtain candidate-bound engineering-only GPU approval from authorized authors, acknowledging prior Pencil/Desk outcomes. Test-only fixture approvals are never usable here.
6. Confirm four available GPU UUIDs and ports, protected model storage/permissions, authorized finite access deadline, measured active-work reserve, approved session supervision and persistent WSL storage. Run the approved smoke benchmark and inspect actual process/GPU placement before scheduling any episode.
7. Execute only the 12-unit D0/D18 engineering matrix. Keep retries/infra failures; stop on any mismatch or evidence/durability failure. Stopped progress export and independent copy/readback are required whether the matrix completes or not.

No scientific six-arm pilot/confirmation authorization follows from these steps. A05â€“A07/A14â€“A15, authorship/disclosures, live Overleaf status, institutional/public artifact permissions and later scientific model/budget/selection decisions remain pending. The 2,400-action single-attempt matrix allowance is not a measured runtime; retries/smoke/oracles add work.

## Verification completion

Final focused regressions: **21 passed in 44.884 seconds** (6 provenance/integration, 9 evidence/semantics, 5 durability, 1 runbook parser). Compilation and working/staged whitespace checks passed. Ruff is unavailable; no tool was installed. A bounded staged private-key/GitHub/AWS pattern scan found zero hits across 15 files.

Intermediate checks: 388 tests passed with four skipped during implementation. Later stricter run/episode checks exposed a stale synthetic fixture run ID: the 21-case focused suite and the 391-case full suite each failed the chained test (0 versus 4 completed). The fixture now records the actual frozen run ID; validators were not weakened. A clean combined 15-case provenance/evidence rerun passed, followed by the final 21-case focused pass above. Final full-suite result is recorded below.

Post-change historical comparison: **219 MATCH, 0 MISMATCH, 0 MISSING**. Protected configs/data/decisions/protocol and candidate task manifest have no diff against the audited commit. The original audit was preserved; its working-file SHA-256 is `14f6ab59ab67a0c49144312096dc24c5f8cddd3974ea3b5b7fd23156608d54a2`. Test temporaries were isolated; ignored bytecode caches may have refreshed.

Final independent review: A verified authentic historical ancestry; B/D challenged environment-response consistency, now corrected; D independently reran nine evidence regressions successfully; C confirmed relocated snapshot resolution and exclusion of possibly live failed writers. No additional unresolved laptop blocker was established by these bounded passes. The matrix records **11 FIXED and 8 BLOCKED**; scientific and host-only findings remain pending.

Changed execution files under `src/rq1/university_gpu/`: cli.py, feasibility.py, feasibility_worker.py, prospective_identity.py, scheduler.py, snapshot.py and new evidence.py. Existing feasibility fixtures were strengthened; four new test modules cover provenance, evidence, durability and actual runbook CLI forms. Documentation includes the runbook, this report and preserved audit.

The verified remediation will be committed locally on the requested branch after the final full-suite result. No push is authorized. Commit identity is reported in delivery; this report is included in that commit.


Final full suite on stable implementation: **391 tests ran in 180.196 seconds: 387 passed, 4 skipped, 0 failures/errors; exit 0**. Command: `PYTHONPATH=src .venv/Scripts/python.exe -m unittest discover -s tests` on the laptop (PowerShell set PYTHONPATH explicitly). Four skips remain optional real pilot/Hermes/ALFWorld/recovery integrations. The corrected runbook parser test was rerun after adding an explicit reviewed-source-commit check and passed in 0.392 seconds. Local interpreter is Python 3.11.9 from the project venv; `rq1` resolves to this checkout. No unresolved local test failure remains.
