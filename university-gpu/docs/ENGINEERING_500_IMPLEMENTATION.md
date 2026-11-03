# Engineering 500 implementation and verification

Offline release integration PASSED: 500 synthetic executions, four workers and ten protected segments in 3157.503 seconds (about 52.6 minutes on native Windows). Machine-readable results are in `engineering-500-offline-validation.json`. Hardware launch readiness is BLOCKED until the external gates below pass. This document records software work and synthetic validation; it is not a protocol approval.

## Source and preserved work

Development branch: `feat/rq1-engineering-500`, created directly from verified remote `6af25622464142beef25d2c248655946389e4f8b`. The integrated baseline differed from laptop `e916581` only in authorization-mode support. No unrelated branch was merged/rebased. Existing `feasibility.py`, `feasibility_worker.py` and the six-task candidate have identical Git blobs before/after; scientific configurations, data, protocols and historical records were not edited.

The separate versioned orchestration and worker retain the baseline's behavior while reusing its scheduler, executor, model server, evidence and export primitives. Shared code recognizes an explicit new schema; unknown engineering schemas do not gain a scientific fallback. The old 12-unit contract remains unchanged.

## Implemented behavior

- Exactly 500 logical executions, 250 pairs, D0/D18=250 each; original task order receives 42,42,42,42,41,41 pairs. Environment/model seed 11, 200 TOTAL actions and exact frozen task witnesses/assets are preserved. Repetition/pair IDs are bound into unit identity, results, evidence, events, checkpoints and anchors.
- Separate four-execution acceptance manifest and run identity; production preflight requires its approved export fingerprint, four distinct workers/placements, matching clean source and exact verified inputs. Recovery success is not required for infrastructure acceptance.
- Fresh independent approval and externally pinned frozen manifest. Source, scope, deadline, attempt allowance and backup path are bound. Self-attestation is rejected for scale. Source dependency manifests and installed Python/package-version inventory are frozen; drift blocks preflight. Production storage requirements are recalculated from the approved acceptance export with a 12x copy/headroom factor and 50 GB floor; this remains an estimate rather than a worst-case bound. Approval templates remain invalid/test-only.
- Four isolated Ollama endpoints, exact prospective model/tokenizer identity, original PID-first placement plus strict WSL fallback, RTX 5000 Ada hardware checks and bounded four-worker concurrency.
- Fifty-accepted-unit segments, per-unit atomic checkpoints, full final/explicit-stop snapshots, create-only authoritative results, source/destination/anchor verification, immutable receipts and clean-pause-only supervision. Failed/incomplete transfers remain visible. Source traces are checked against protected anchors, not just against a self-consistent copy.
- Sixty-second monitor samples, separate coordinator/attempt/action liveness, low-storage/lost-worker stop, fixed 75-minute cutoff reserve and bounded worker response timeout. Physical execution is not claimed exactly-once.
- Unknown mutation states are quarantined. Matching validated authority published before its event is retained and reconciled without reexecution. Safe retries require explicit review, true safety flags and unused prospectively approved allowance; default allowance is 500 (or four acceptance executions). Optional prospectively approved additional attempts are bounded to 20. Corrupt receipts/checkpoints are never silently overwritten.
- Final create-only descriptive summary includes physical attempts, per-condition/task counts and paired differences. Six contexts and fixed-seed repetitions cannot establish independent-sample or confirmatory effects.

## Transferred run audit

The November 3 repeat archive remains byte-identical, SHA-256 `8d6f2ce4acd7f30875052cd497b10a4cb92f5ef412c398adc20e3906e05c6455`. Its checksum, 415 listed files, 12 result hashes and 48 required raw-log hashes match. The normal export semantic verifier passes: 12/12 units, correct matrix, event/result correspondence and raw action accounting. Events include 12 starts/completions and no failed attempts. Source records `6af2562`; all 211 archived source files match the raw Git blobs of that commit. Raw objects were used to avoid native Windows archive line-ending conversion. Checkpoint directories and independently authenticated anchors are absent; source-reported checkpoint status is not promoted into independently verified durability. The first successful run remains user-reported in this execution.

## Tests and checks

The broad suite passed 520 tests with nine optional/platform skips. Final focused coverage and later targeted tests cover strict protocols, profile validation, synthetic freeze-to-preflight, installed dependency drift, acceptance provenance, raw/source corruption, duplicate results, lock contention, disk-copy/flush failure, accepted-result recovery, unknown states and supervisor protection barriers. The final release 500-unit scenario also passed checkpoints, exports, anchor verification, validation and merge. See machine-readable counts and elapsed time. Broad/focused run counts overlap and must not be summed. The CPU freeze/preflight path, dependency drift and supervisor restart/protection barriers also passed synthetic tests.

A full temporary-filesystem 500-unit/four-worker scenario exercises all ten segments, checkpoints, exports, destination checksums, anchors, validation and merge. Development runs were interrupted to address repeated whole-run validation cost; they are not claimed as successful 500-unit checks. The completed development integration took about 42 minutes on native Windows. Scale per-unit checkpoint checks now validate the selected authority rather than revalidating every older unit; whole-run checks remain at segment/resume/export/final boundaries. Directory anchor checks avoid repeating full manifest reconstruction for every unit. The old runner keeps its existing verification cadence.

Python compilation and diff checks pass. Ruff is unavailable and was not installed. No real GPU requests, ALFWorld research episodes, new model downloads, human labels or genuine approvals were produced.

## Exact file inventory

Existing files extended:
- `src/rq1/university_gpu/authorization.py`
- `src/rq1/university_gpu/cli.py`
- `src/rq1/university_gpu/evidence.py`
- `src/rq1/university_gpu/evidence_anchor.py`
- `src/rq1/university_gpu/scheduler.py`
- `src/rq1/university_gpu/snapshot.py`
- `src/rq1/university_gpu/worker_process.py`

New files:
- `src/rq1/university_gpu/engineering_scale.py`
- `src/rq1/university_gpu/engineering_scale_worker.py`
- `src/rq1/university_gpu/scale_protection.py`
- `tests/test_engineering_scale.py`
- `tests/test_engineering_scale_recovery.py`
- `university-gpu/engineering-500-config.example.json`
- `university-gpu/prospective/engineering-500-approval-template.json`
- `university-gpu/docs/ENGINEERING_500_RUNBOOK.md`
- `university-gpu/docs/ENGINEERING_500_IMPLEMENTATION.md`
- `university-gpu/docs/engineering-500-offline-validation.json`

## Remaining launch blockers and critical path

1. Review/deploy the committed software on the university workstation; verify Python/ALFWorld/Ollama/Sentence-BERT/tokenizer assets and real four-GPU availability.
2. Obtain genuine independent acceptance approval and trusted approval/frozen-manifest hashes; verify approved independent storage and session/access policy.
3. Run the separately authorized benchmark and four-episode acceptance; verify its complete export and protected destination.
4. Obtain fresh production approval binding that acceptance export, freeze the 500 matrix on the same reviewed source, preflight and launch with the actual access cutoff.

The 8–12-hour elapsed allowance plus setup and cutoff reserve remains an estimate based on a small prior batch; 500 completion is not guaranteed within eight-hour access. Synthetic Windows test timing does not establish WSL/GPU throughput. For multiple visits, approval must genuinely cover the permitted access scope; an expired frozen approval cannot be silently extended. No unattended/overnight access or physical backup independence has been verified here. Scientific confirmatory approval is separate.

Mandatory tomorrow: reviewed deployment, human authority/trust pins, real acceptance, adequate time/storage and verified independent readback. Dashboards, alerts, automatic performance tuning and automatic unknown-state recovery are deferred. Use `ENGINEERING_500_RUNBOOK.md` for exact commands.
