# RQ1 independent scientific and engineering audit — 2026-10-30

## Scope and evidence standard

Audited branch: `feat/rq1-university-4gpu`, commit `dab5ed243d04c98f389fe809064a10ecd689bfec`. Read-only remote lookup returned this same SHA. Scientific base resolves to `c052e2cbdaa63419db4122cf08b1ccc64b38b3b4`. The tree was clean before the audit. This report is the only intended working-tree change. No installation, download, inference, scientific episode, approval, source fix, commit, or push was performed.

The root reviewer inspected provenance, the runbook, source and tests. Four actual subagents independently reviewed: A scientific validity/causal inference; B code/test quality; C runtime/durability; D provenance/security/reproducibility. They worked in separate agent contexts against the same checkout and returned source-supported findings. They share the same filesystem, model limitations and project evidence; this is an independent AI review procedure, not human peer review or externally certified assurance. No target scores were given. Findings below reconcile overlaps rather than counting each reviewer's version separately.

Line references are for the audited commit and use repository-relative paths. Archive member line numbers refer to newline-delimited JSON. Paths outside the checkout are identified explicitly. Severity distinguishes demonstrated source defects from unresolved launch evidence and scientific limitations. A passing mock test is never treated as target-host validation.

Source-reference key: unqualified university execution filenames (`feasibility.py`, `feasibility_worker.py`, `scheduler.py`, `snapshot.py`, `models.py`, `ollama.py`, `worker_process.py`, `hardware.py`, `ops.py`, `cli.py`, `manifest.py`, `analysis.py`, `interventions.py`, `prospective_identity.py`) mean `src/rq1/university_gpu/<filename>`. `episode_driver.py` means `src/rq1/hermes/episode_driver.py`; `amended_executor.py` and `recovery_executor.py` mean `src/rq1/evaluation/<filename>`; `controlled_failure.py` means `src/rq1/recovery/controlled_failure.py`; `retriever.py` means `src/rq1/retrieval/retriever.py`. Unqualified audit/protocol/runbook document names mean `university-gpu/docs/<filename>`, script names mean `university-gpu/<filename>`, candidate JSON means `university-gpu/prospective/feasibility-200-candidate.json`, and `test_*.py` means `tests/<filename>`.

## 1. Executive verdict

| Area | Verdict | Basis |
|---|---|---|
| Repository correctness | CONDITIONAL | Clean, synchronized history; concrete freeze/preflight and provenance-command defects remain. |
| Historical integrity | PASS within checked scope | All 219 Oct 20 reference files matched recorded SHA-256 values, including six relocated archive/checksum/manifest files. No changes to tracked `configs/`, `data/`, `docs/decisions/`, or `protocol/` from scientific base. This does not attest unavailable original model weights. |
| Experimental validity | BLOCKED | Budget/model/selection amendment unresolved; labels, relevance controls, target sample and statistical plan unfinished. |
| Feasibility runner correctness | BLOCKED | Valid candidate-bound oracle reports fail frozen preflight; documented library-reference route rejects authentic available historical provenance. |
| Four-GPU readiness | NOT VERIFIED | No target-host process/GPU placement, installed assets, throughput or WSL lifecycle checked here. |
| Reproducibility | BLOCKED for launch | Historical artifacts are preserved; prospective manifest/blobs, tokenizer linkage, dataset reconciliation and runtime environment need verification. |
| Data durability | CONDITIONAL | Atomic authoritative results and episode-level resume exist; raw trace completeness is not enforced, live snapshots can race, independent backup not demonstrated. |
| Security | CONDITIONAL | Bounded credential scan clean; consumer run-ID containment defect and target permissions remain unresolved. |
| Monday engineering launch | BLOCKED / NO-GO now | See queue A. Approval alone cannot repair the source defects. |
| Later scientific launch | BLOCKED / NO-GO | See queue B; engineering success does not satisfy these gates. |

Judgmental readiness scores: **engineering 4/10; scientific 3/10**. Rubric is five dimensions worth 0–2 each. Engineering: historical/profile separation 1.5, complete launch-path correctness 0.5, target hardware evidence 0, durability 1, offline verification 1. Scientific: historical evidence preservation 2, prospective identity/approved protocol 0, causal estimand/controls 0.5, task eligibility/leakage 0.5, confirmatory analysis/power specification 0. These are audit judgments about present launch readiness, not objective measurements of paper quality.

**Findings: 0 CRITICAL, 7 HIGH, 12 MEDIUM, 0 LOW.** No actual historical corruption, credential compromise or budget-overrun episode was established.

## 2. Findings, ordered by severity and impact

“Monday” means the 12 D0/D18 engineering episodes. “Science” means later scientific inference. Conditional blockers require the stated mitigation or evidence; they are not blanket requirements for read-only workstation diagnostics.

| ID | Severity / subsystem | Precise issue and evidence | Consequence, remedy and gates |
|---|---|---|---|
| A01 | HIGH / oracle binding | `feasibility.py:449` accepts oracle report against candidate; `526–552` changes manifest hash at freeze; `714–715` requires report hash equal supplied manifest hash; `923–926` supplies the frozen manifest. | Every genuine candidate-bound oracle report is rejected at frozen preflight. Validate against frozen `candidate_manifest_sha256`, retaining exact task/state/action checks; add full transition regression. **Monday: blocks; Science: shared assurance lesson, separate path.** |
| A02 | HIGH / library provenance | `feasibility.py:559–566` requires a frozen new university six-arm manifest; runbook `70,84–86` requires this before engineering freeze. Authentic historical task/library files fail that schema. | The prescribed route cannot establish Accum-18 ancestry from the actual historical archive. Add a reviewed archive/member-ledger verification route; never fabricate scientific approval to unblock engineering. **Monday: blocks documented route unless independently verified accepted reference exists; Science: blocks unsupported provenance.** |
| A03 | HIGH / raw evidence | `scheduler.py:209–224,422–447`; `snapshot.py:90–97,147–160` validate result envelopes and files present, but do not require a result-bound inventory/hash of raw logs. Fixture `test_university_gpu_feasibility.py:111` has empty `log_paths`. | A complete fake run remains validated/exportable after deleting all 12 mock raw logs. Require mandatory logs, safe relative references, hashes and consistency checks before evidential completion. **Monday: blocks evidence-quality acceptance; Science: blocks inference from incomplete traces.** |
| A04 | HIGH / prospective assets | `NEXT_VISIT_RUNBOOK.md:197,201–207`; `prospective_identity.py:19–62`; unfilled review templates. Manifest/blob, tokenizer-link, selected task files, SBERT and target placement evidence have not been supplied to this audit. | Source gates exist, but target identity/readiness cannot be certified. Verify actual assets and human reviews; do not equate declared constants with observations. **Monday: blocks; Science: blocks.** |
| A05 | HIGH / target population | `PROTOCOL.md:65` acknowledges historical outcomes on 30 follow-up tasks; `FINAL_BUDGET_SELECTION_RECOMMENDATION.md:46,54,56` proposes fresh confirmation. | There is no settled independent confirmation sample/claim. Authors must freeze either a transparent fixed-task follow-up estimand or genuinely unexposed prespecified sample and selection policy. **Monday: no; Science: blocks.** |
| A06 | HIGH / inferential plan | `PROTOCOL.md:63`; `SCIENTIFIC_PREREQUISITES.md:26,38–42`; `analysis.py:16–17,148–158` outputs nine contrasts without a settled confirmatory primary family. | 36/540 are counts, not power evidence. Specify minimum effect/precision, primary contrasts, multiplicity, task clustering, missingness and eligibility loss; simulate operating characteristics. **Monday: no; Science: blocks.** |
| A07 | HIGH / treatment validity | `PROTOCOL.md:77–81`; `manifest.py:75–95`; `interventions.py:156–186`. Relevant/irrelevant selection, neutral filler and Gate depend on unverified human semantics/blinding. | Code cannot prove relevance, independence or absence of leakage. Freeze contexts/rubric, independently rate/adjudicate, assess library coverage, review filler, calibrate on eligible development data. **Monday: no relevance-rating requirement for D0/D18; Science: blocks.** |
| A08 | MEDIUM / result semantics | `scheduler.py:32–69` checks declared budgets and retrieval list lengths, not actual action bounds, unique frozen IDs or finite scores. | A synthetic result with 10,000 actions, duplicate nonexistent IDs and NaN scores is accepted. Require numeric bounds/count consistency, library membership, unique IDs, finite cosine values and trace consistency. **Monday: resolve before accepting batch evidence; Science: blocks reliable analysis.** |
| A09 | MEDIUM / checkpoint race | `snapshot.py:34–46` copies all attempts including active append-only logs; `scheduler.py:389–399` snapshots at 10 and final completions; `episode_driver.py:517–520` appends logs. | Concurrent append can fail snapshot stability checks and pause assignment. Snapshot only closed attempts or quiesce writers; a stopped verified export is a limited operational mitigation. **Monday: fix or demonstrate mitigation; Science: same.** |
| A10 | MEDIUM / durability status | `scheduler.py:389–399,417–420` prioritizes completed result count over backup failure. | Injected backup failure yields `status=completed` with a backup failure in `failures`. Surface durability failure separately; require explicit export/readback regardless of status. **Monday: operator mitigation required; Science: same.** |
| A11 | MEDIUM / cutoff | `feasibility.py:1170`; `cli.py:161`; `scheduler.py:345–368`; `worker_process.py:20–22,84–88`. Nonfinite floats pass feasibility cutoff validation; new assignments stop 30 min early but active work may wait 3600 s. | `+inf` disables a finite deadline; ordinary finite deadlines still are not hard completion cutoffs. Reject nonfinite values, verify real epoch, measure episode duration and reserve time or define authorized interruption. **Monday: required mitigation/fix; Science: same.** |
| A12 | MEDIUM / interpreter/runbook | Runbook `42–43,49,77–86` and all later commands use `python3.11` without selecting the project environment; `WORKSTATION_SETUP.md:35–38` only says activate venv first. | System Python may lack project/dependencies. Fresh shell reproduction failed imports. Use explicit project `.venv/bin/python`, verify 3.11 and dependency imports, then use that executable throughout. **Monday: blocks if interpreter not proven; Science: same.** |
| A13 | MEDIUM / path safety | Frozen consumer `feasibility.py:366–367` accepts any nonempty run ID; creator `505–506` has a safe regex; launch appends ID at `1196` after checking only output root at `1189–1195`. | Rehashed externally supplied manifest accepts `../../outside-approved-run-root`. Apply creator regex at consumption and resolved containment/symlink checks. Normal freeze CLI cannot generate this value. **Monday: fix before external manifest use; Science: same.** |
| A14 | MEDIUM / construct validity | Scripted prefixes `recovery_executor.py:95–96`; reversible navigation disturbance `controlled_failure.py:130–154`; expert exploration loops discussed in `FINAL_BUDGET_SELECTION_RECOMMENDATION.md:16,25–36,50`. | Expert route length does not establish intrinsic difficulty, autonomous long-horizon planning or general failure recovery. Narrow claims to recovery after scripted history and this disturbance; broader claims need new designs. **Monday: no; Science: blocks broader claims.** |
| A15 | MEDIUM / noise estimand | `manifest.py:75–95,149–184`; `interventions.py:156–186`; `analysis.py:161–184` measures retrieved IDs while interventions change injected IDs/content/filler. | Rel–Irr estimates selected single-skill policies, not an isolated continuous noise dose; top-3 retrieved noise may be unchanged across arms. Report retrieved and injected relevance separately; name causal contrasts precisely. **Monday: no; Science: blocks stronger mechanism inference.** |
| A16 | MEDIUM / host/session durability | Foreground commands `NEXT_VISIT_RUNBOOK.md:159–162`; local checkpoint location `snapshot.py:26`; process cleanup `ollama.py:118–129`. No target disconnection/reboot/persistence evidence. | RDP survival, process orphan cleanup and independent storage remain unverified. Verify host policy, approved supervision, owned-process recovery and independent copy/readback before unattended execution. **Monday: unattended batch blocked until verified; Science: same.** |
| A17 | MEDIUM / shared model storage | Startup checks `feasibility_worker.py:46–60`; inference uses mutable tag `episode_driver.py:597–599`; isolated servers share model directory `ollama.py:33–46`. | Startup identity checks do not attest every later request if another user changes the tag. Verify protected model storage/ownership, prohibit tag mutation during run, and add suitable episode-boundary attestation before scientific use. No substitution observed. **Monday: protected-storage mitigation required; Science: blocks unresolved identity drift.** |
| A18 | MEDIUM / tokenizer metadata | `feasibility.py:465–474` validates a reviewed 40-character revision but does not compare it to provenance revision; `prospective_identity.py:58` records only provenance hash. | Correct tokenizer bytes may carry inconsistent declared revision. Expose provenance revision and require equality across review/config/report. Bytes remain hash-checked; this is not proof of wrong tokenization. **Monday: independently cross-check; Science: resolve metadata binding.** |
| A19 | MEDIUM / interrupted-run backup procedure | Runbook `170,181–188` enables `set -e` and invokes complete-run validation/merge before progress export; CLI `273–286` returns failure for incomplete validation. | On the interrupted run for which a progress backup is needed, the pasted sequence stops before export. Branch complete-run validation from incomplete progress export, and verify the independent copy in both cases. **Monday: correct procedure before use; Science: same.** |

### Reproductions and adjudication of findings

A01 was reproduced without writing files: six synthetic rows were populated from the checked-in candidate, including expected sequence hash, states, action count, split and seed. `validate_oracle_report(report, candidate)` returned `[]`; the same report with the existing fake-frozen test manifest returned `['oracle report schema or candidate manifest binding differs']`. Frozen manifest identity necessarily differs after status, run/source/assets/approval are added. Replaying against the frozen manifest is not the documented solution and would replace the intended candidate binding; correct the validator explicitly.

A02 was reproduced by calling `build_accum18_reference` with the recovered historical `evaluation-libraries.json` and `evaluation-task-manifest.json`: it raises `FeasibilityError: source manifest is not an approved frozen university-GPU manifest`. The historical library exists and hashes correctly. Missing new scientific approval must not be “reconstructed.” The alternative accepted reference format at `feasibility.py:489–497` checks its self-hash, file hash and 18 IDs, but does not itself prove archive ancestry. A reviewed authentic reference can resolve this without approving the six-arm experiment.

A03 reproduction used existing fake manifest/workers in a temporary directory, completed 12 units, deleted only their 12 generated `raw-attempt.jsonl` files, then called existing validators/export: `validate.passed=true`, `export.status=EXPORTED`, `export.complete=true`, `verify_export.status=PASS`. Temporary fixtures were cleaned. This demonstrates validator incompleteness, not loss of historical real traces. A08 similarly mutates a fake result to `actions=steps=post_failure_actions=10000`, duplicate nonexistent IDs and NaN scores; `_validate_feasibility_result` returns `[]`. Production executor caps are separate and do not excuse missing saved-evidence checks.

A10 reproduction injected `OSError('audit injected backup failure')` into the backup function with four fake workers: all 12 results completed, status was `completed`, and `failures` contained a backup failure. A13 reproduction changed only an in-memory fake frozen manifest's run ID to `../../outside-approved-run-root` and recomputed its content hash; frozen validation returned no problems. These fixtures do not represent authorized assets or approvals.

## 3. Methodology review

Historical acquisition (240 episodes) establishes the provenance of an accumulated library under its own acquisition process. Historical evaluation (360 episodes) remains the frozen numerical study. Neither is replaced by feasibility results. Current six-arm code retains the historical model digest and 50-action policy; the feasibility profile explicitly uses a prospective digest and 200 cap. A new digest cannot be called equivalent because the visible tag and quantization agree. Historical NoLib/Accum-18 are separate references; no accepted evidence permits replacing contemporaneous D0/D18 cells.

The 12 episodes are six fixed valid_seen contexts × D0/D18 × seed 11. They can diagnose execution, checkpoint replay, retrieval, logging and floor/ceiling behavior. They cannot independently confirm population treatment effects. One pair per family supplies no within-family task replication. Reusing the same fixed contexts for engineering is reasonable with disclosure; later task selection must not silently depend on observed outcomes.

The 36-episode scientific pilot has six task clusters and one seed per arm. It develops protocol estimates and checks intervention validity. The 540-episode plan has 30 task clusters, three seeds, six arms. Seeds do not create 540 independent tasks. A task-cluster estimator and explicit repeated-seed structure are required; family estimates with five tasks per family are descriptive. Primary contrasts and multiplicity must be settled before interpreting nine reported contrasts. Missing infrastructure outcomes cannot be coded as ordinary agent failures or dropped without a prespecified rule.

D0/D18 holds checkpoint, detour and permitted recovery actions constant within task. D0 removes library content; D18 injects genuine Accum-18 deterministic top-3. This identifies a library-policy contrast conditional on the chosen contexts, not a relevance-only mechanism. Rel/Irr hold a selected single-skill policy structure while varying content; filler quantity and skill count differ from D18. Gate filters the same top-3 without new retrieval and pads the memory block. Token matching controls the serialized memory-block dose, not every whole-prompt token boundary or semantic effect of filler. A continuous “retrieval noise causes degradation” claim needs careful estimand wording or a separately justified design.

The disturbance is a reversible navigation detour after scripted expert history. It does not experimentally establish autonomous accumulation during recovery, irreversible failure handling, or disturbance cost relative to an undisturbed episode. The absence of an undisturbed arm is acceptable for paired comparisons conditional on this disturbance, but forbids claims about its causal cost. Random retrieval is not automatically necessary when irrelevant controls are valid; coverage and semantic controls must be demonstrated first. Gate is a selective retrieval extension, distinct from RQ2 archival/forgetting experiments.

## 4. Code and test review

### Actual execution results

The initial literal command `py -3.11 -m unittest discover -s tests` used global Python with `PYTHONPATH` unset: **344 tests run; 1 failure, 2 errors, 4 skipped**, 143.599 s, exit 1. Both import errors were `ModuleNotFoundError: rq1` in acquisition modules. The rating-template subprocess failed to import `rq1`, causing its expected-message assertion to fail. This is an interpreter/source-path reproducibility failure, not a university hardware failure.

The corrected environment uses PowerShell `$env:PYTHONPATH=(Resolve-Path src).Path` and `.venv/Scripts/python.exe`. Final full-suite and focused outcomes are recorded in the completion addendum below. No packages were installed. Tests only exercised offline/fake providers and local subprocesses.

`py -3.11 -m compileall -q src tests` and `git diff --check` completed without errors. Ruff is unavailable (`No module named ruff`); formatting/lint verification is NOT VERIFIED. Candidate CLI validation passed: 12 units, candidate hash `61f01fe25a78d45d4e51c8dcf958a2f3103fa8665030ec28ced9e4e98490bff0`, no structural problems. Every one of the 15 feasibility command forms in the runbook was parsed through the actual `build_parser()` without executing handlers. Parsing establishes argument validity, not asset availability or runtime success.

### Coverage limits

The eleven feasibility tests never invoke `freeze_engineering_manifest`, `verify_frozen_feasibility_assets`, `FeasibilityEpisodeWorkerRuntime`, or `run_feasibility_benchmark`. The only launch invocation rejects an unapproved candidate before external checks (`test_university_gpu_feasibility.py:310–316`). The export/resume test constructs an artificial frozen manifest and fake workers (`39–111,318–352`). The oracle test patches task-source/state checks and substitutes a harness (`365–407`). It tests orchestration and action-sequence recording; it does not prove real ALFWorld behavior or real cap-200 model recovery.

Useful covered behaviors include immutable result publication, duplicate/corrupt authority rejection, mocked quarantine, retry attempt IDs, low-storage assignment stopping, checkpoint/export partial-publication rejection, and source/model/tokenizer mismatch rejection in isolated units. The test at `test_university_gpu_package.py:431–436` hashes one model config around a rejected merge; it cannot prove all historical inputs are unchanged. This audit therefore independently checked authentic reference files.

Required coverage additions after this audit: full candidate→freeze→preflight transition; genuine historical-library reference path; real worker constructor with local dependency adapters; actions at 0/limit/limit+1, early success and interrupted attempts; corrupt/missing raw logs; invalid IDs/NaN/counts; mutated data/model/tokenizer on resume; live-log snapshot races; backup-failed final status; finite cutoff and path traversal; process/port cleanup after coordinator death. Real GPU/WSL tests remain separately authorized target-host work. These gaps are not resolved by increasing fake-worker test counts.

## 5. Scientific integrity and provenance

### Fresh integrity checks

The Oct 20 `audit/original-input-manifest.json` was read and all **219** listed byte hashes compared. Six archive-related references had moved into `03_Experiment_Archives`; each was located by its exact basename and checked against its original digest, not marked missing from its former path. **219 MATCH, 0 MISMATCH, 0 MISSING after relocation.** This includes historical evaluation outputs, original ratings/adjudication files and frozen-source files within that manifest's scope. These checks do not prove that a document's scientific interpretation is correct or verify future workstation copies.

The recovered historical preparation ledger was independently checked:

| File under `experiment/university-friday-preparation-20261029/historical-inputs/` | Observed SHA-256, matching ledger |
|---|---|
| `evaluation-task-manifest.json` | `340f52ce660f703d9b98e1cf2a58ac195e2fe7c1c380f08b98d0419929bdb429` |
| `controlled-failures.json` | `69e1d7e924db899131727e002e058c9206987d507c8d3751a8405aad3fabf622` |
| `evaluation-libraries.json` | `e01d3a569ee8373d8cc235c4343f2b52f92c9bf41751b5d3e435ced70648e1f5` |

Ledger archive reference: `d52ed82e01a073f8c3382711d03f65916cb41ca89bc0b773e6845817bd37e4fd`. `extract_archived_inputs.py:12–49` verifies archive/member bytes and uses create-only extraction. No new extraction was needed here. Git comparison from `c052e2c` to HEAD found no changes to the tracked frozen configuration/data/decision directories. Changes to executor/runtime source do exist and must not be confused with frozen historical source: 86 files differ overall, across 14 feature commits. No historical recomputation with changed code was performed.

History inspected: initial GPU package `65002ab`, hardening `153d7e7`, Gate preparation `153f5cf`, AGENTS restriction clarification `b571eb4`, Friday package `7d65e47`, university engineering summary `2ba4ea8`, runtime audit `b1a57c2`, merge `1087bfb`, runbook/evidence updates `1a13bee,6971094,85f32c0`, amendment/provenance work `cfad4b9,76bf36d`, and feasibility implementation `dab5ed2`. The initial clean status and remote SHA were checked live; no fetch/push mutation was needed.

### Six-task evidence actually available

`C:\ResearchEvidence` was not available locally. Workspace search found the original proposal, candidate and eligibility JSON under `rq1-evidence-audit-20261030-v2`, plus the actual transferred archive `<local-research-workspace>/rq1-university-evidence-20261030.tar.gz`. Reviewer A read all six raw oracle event-log members rather than relying on the previous report.

| Family / exact `valid_seen:` task suffix | Route | Prefix | Detoured terminal actions | Recovery allowance `200-prefix-1` |
|---|---:|---:|---:|---:|
| `pick_clean_then_place_in_recep-ButterKnife-None-Drawer-30/trial_T20190908_052007_212776` | 95 | 48 | 96 | 151 |
| `pick_cool_then_place_in_recep-Cup-None-SideTable-21/trial_T20190906_192823_237997` | 13 | 6 | 14 | 193 |
| `pick_heat_then_place_in_recep-Tomato-None-Fridge-23/trial_T20190909_082320_103350` | 14 | 6 | 15 | 193 |
| `look_at_obj_in_light-Bowl-None-DeskLamp-316/trial_T20190908_230702_962704` | 11 | 5 | 12 | 194 |
| `pick_and_place_simple-Pencil-None-Desk-310/trial_T20190909_113054_894334` | 6 | 4 | 7 | 195 |
| `pick_two_obj_and_place-Book-None-Desk-302/trial_T20190906_181233_363875` | 146 | 74 | 147 | 125 |

All six logs contain reset requests `valid_seen`, seed 11, action limit 200, and terminal responses reporting success. Pick/place member: `rq1-checkpoint-tests/205f7af291c44b5b8fc4bf3652b68ab2/4/solvability-oracle/episode-events.jsonl` lines 1 and 9. Other five: `rq1-checkpoint-tests/758210385a12406194886aa9df20e2a1/<family>/<prefix>/solvability-oracle/episode-events.jsonl`, terminal lines respectively 98,16,17,14,149. Navigation detours are `go to cabinet 1` for clean/cool/heat and `go to bed 1` for light/pick/place/pick-two. This corroborates real recorded successful cap-200 witnesses. It does not establish shortest routes, intrinsic difficulty, cap-50 success, authenticity of original task/game bytes, or actual environment seeding beyond the requested/echoed interface.

The existing transferred audit records proposal/discovery identity `43be7c24c054c3b2299b5054f293c21eaca08734554310094c5654783a8d7aa1` versus adapter identity `5fd5b9193ea361d62e877e93157ab567e63a95cf2dece0859db36cf4f534a0c0`. Those are different scopes until actual source-file reconciliation and human review establish otherwise. Original ALFWorld files were not found in these transferred evidence inputs and were **not independently rehashed**. Candidate `oracle.validated=false` remains intentional historical evidence; new profile replay and review must not edit it.

The Pencil/Desk task has prior Gemma memory/no-memory outcomes in committed `artifacts/prelaunch/prelaunch-a718b4e4d9644b01aa4861e6905cd587/pilot-report.json:20–53,66–68`: both successful with two post-failure actions; non-scientific mode at line 33 and inference seed 42 at `34–40`. This is not proof the genuine historical Accum-18 library was used. The candidate discloses this exposure (`feasibility-200-candidate.json:510`). No further Gemma exposure was found for the other five in inspected evidence; this bounded search cannot prove its universal absence.

### Model, tokenizer and retrieval

Historical digest remains `4eb23ef187e2c5462566d6a1d3bbbc2f1346d0b4327cbb66d58fffbcc9b2b05c`. Prospective digest is `6114515d63c17436a7c0417d82820ac65ad643e2806c5a3c89cb62846436ed0b`. Both use declared tag `gemma4:12b`, Q4_K_M; equality of tag/quantization is insufficient for equivalence.

`prospective_identity.py:19–34` verifies saved manifest hash and all config/layer blob digests/sizes. The CLI defaults to the prospective digest and production preflight supplies it explicitly (`feasibility.py:959–963`). Offline report quantization is populated from a constant (`prospective_identity.py:54–56`); live endpoint checking is the later actual Q4_K_M check. No prospective manifest/blob set was verified here. Tokenizer bytes, full revision format, five reference samples and hashed linkage evidence are checked, but sample authenticity/link semantics still require independent review. No model-derived all-input tokenizer equivalence has been demonstrated.

SBERT is pinned to `sentence-transformers/all-mpnet-base-v2`, revision `e8c3b32edf5434bc2275fc9bab85f82640a19130`, snapshot `bbfecb04a834241144af31e9200a3dd0df83ff0fbb2453d56a4bb383a6bfbe11`. Worker loads local-only (`feasibility_worker.py:107–110`); snapshot hash checks exact cached revision (`acquisition/environment.py:85–91`). Python worker processes hide CUDA (`worker_process.py:50–51`); the four embeddings do not intentionally consume Ollama GPUs. No local/target SBERT asset was newly certified.

Static source supports one retrieval after perturbation (`recovery_executor.py:95–149`, SingleRetrievalBoundary), normalized cosine ranking with deterministic skill-ID tie breaks (`retriever.py:28–38`), genuine frozen 18-skill content checks and no persistent writes. D0 uses an empty library; both arms share requested recovery settings. Actual Ollama request uses temperature 0, episode inference seed, 2048 output tokens, 32768 context and `think=False` (`episode_driver.py:272–282`). Feasibility seed 11 differs from historical default seed 42 and must remain disclosed. Backend/platform numerical determinism and real runtime configuration remain untested here.

## 6. GPU execution, interruption and security review

| Event | Source-supported behavior | Tested level / unresolved risk |
|---|---|---|
| RDP disconnect | Completed files remain on disk; no dependence on Git push. | Foreground lifetime depends on Windows/WSL policy; no disconnect test or supervision proof. |
| Worker crash | Failed attempt recorded/quarantined; healthy workers continue; missing unit can be retried with a new attempt ID. | Fake exception/quarantine tests; actual process death and resource cleanup unverified. |
| Ollama crash | HTTP failure leads to failed attempt; explicit restart/resume possible. | No automatic model-process restart or mid-episode restoration. |
| Coordinator dies | Published results can reconcile missing completion events (`scheduler.py:305–311`). | Orphaned child servers can retain ports/VRAM; no parent-death process-group cleanup. Verify ownership before stopping processes. |
| Partial result write | Same-directory atomic create-only hard-link publication, file and directory fsync (`models.py:33–64`). | Mock corruption/partial tests; host power-loss semantics unverified. Malformed records fail validation rather than being silently overwritten. |
| Low disk/full disk | Reserve check stops new work while pending exists (`scheduler.py:333–344`). | Active logging continues; full disk can prevent logs/failure-event publication. Disk checks are not a guarantee of complete traces. |
| Other Windows login | No algorithmic recovery guarantee applies. | Session/GPU ownership policy and shared-account behavior unknown; university IT must resolve. |
| WSL terminates/reboot | Intended episode-level resume from durable completed authority. | Active episodes lost/retried; no mid-episode resume. Filesystem survival/backup readback not exercised. |

Retries may execute the same unit physically more than once; one authoritative accepted result per unit is the intended guarantee. A failed attempt is preserved, not relabeled as an agent outcome. Completed result files use create-only publication; conflicting/duplicate authority is rejected. This is useful evidence protection, but A03/A08 limit what “validated” means.

Each Ollama server is given one GPU UUID via `CUDA_VISIBLE_DEVICES`, loopback endpoint, one parallel request and one loaded model (`ollama.py:33–46`). Startup probes and live PID-descendant/GPU UUID checks exist (`hardware.py:156–189`). Real WSL PID/GPU visibility, startup/port ownership and concurrent VRAM use have not been established. A port probe cannot guarantee availability at later bind; startup failure must remain a stop, not permission to attach to another user's service. `OllamaServer.stop` targets its own tracked process rather than global process names. Coordinator hard death can still orphan it.

Automatic checkpoints live under the same run filesystem. They are not an independent backup. Verified stopped export, archive checksum, independently stored copy and readback remain mandatory. GitHub is correctly supplementary (`NEXT_VISIT_RUNBOOK.md:193,217`). Do not push raw traces or credentials merely to make a backup.

Security scan: all 86 current changed files and the diffs of 14 feature commits were scanned for private-key headers and GitHub/AWS token patterns; no hits. This bounded pattern scan does not certify all history or all secret formats. No private keys were opened/copied, authentication permissions changed, or credential values printed. Deployment-key scope/token storage are NOT VERIFIED. `.gitignore` excludes `.env`, venv, logs, runs/artifacts/results, caches; ignored inventory had 396 entries (mostly Python bytecode/venv, plus `.hermes`), not inspected as a blanket source of trusted research evidence. No initial untracked files were present.

Snapshot/export reject symlink files and check export containment (`snapshot.py:80–95,141–143`). A13 remains a separate run-root consumer gap. Existing file creation respects process umask; shared host output/model permissions and independent backup ACLs are not verified. Use university-approved storage and check permissions before storing restricted material. No compromise is alleged.

## 7. Monday runbook audit and launch gates

All fifteen feasibility subcommand forms parse correctly, including `validate-run` versus generic scientific `validate`. `prospective_identity` CLI arguments match its parser. The commands are not collectively READY: schema/asset and environment defects remain.

| Runbook lines | Audit result / corrected proposal |
|---|---|
| 10–26 | Branch/clean-state checks and fetch/fast-forward logic are sensible, but an ahead local branch is allowed without proving it is the reviewed commit. Also compare HEAD to the specifically approved implementation SHA after future fixes. No force push or SSH bypass is proposed. |
| 38–43 | New evidence directory creation fails safely on reuse. Set `umask 077` if approved for project evidence; verify directory ownership/backup ACLs. Select the venv executable explicitly rather than relying on `python3.11`. |
| 49,77–86 | Command names/options parse. Reconcile requires real original files. Identity command needs actual manifest, blobs, tokenizer and authentic reference samples. Reference-library command is blocked by A02. |
| 98–104 | Oracle replay is CPU environment work; real dependencies/source files needed. Preserve exact prefix/detour/routes; do not substitute a successful task after failure. |
| 112–135 | Templates are intentionally unapproved; no human values may be fabricated. `RQ1_HUGGINGFACE_CACHE` and run root must already exist/be reviewed. Freeze succeeds only with genuine evidence, but current frozen preflight then fails A01. |
| 144–152 | Preflight/benchmark cannot run successfully until A01/A02 and asset gates are resolved. Benchmark does consume GPU for four smoke requests; it needs explicit approval. No measured target benchmark exists. |
| 159–165 | Launch requires real finite deadline and verified reserve. “Exactly 12” means 12 authoritative units; startup smoke and retries add requests/work. There is no hard finish guarantee. |
| 172–190 | These are individual operator actions, not a script to paste wholesale: current block immediately requests STOP after status and then resumes with the same deadline. Use a later valid cutoff after checking active state. Complete validation/merge before export can fail under `set -e` on a partial run, preventing the intended progress backup. On incomplete runs, call status then **export-progress/verify-export directly**, without complete-run validate/merge. Tar/checksum outputs need unique names and existence checks to prevent overwriting prior evidence. |
| 193,209,217 | Independent backup/WSL policies correctly remain external gates. Git synchronization is not evidence backup verification. |

Suggested interpreter correction, to be applied in the runbook after this audit (not executed on university hardware here):

```bash
cd ~/Hermes-Research-
export PYTHONPATH="$PWD/src"
RQ1_PY="$PWD/.venv/bin/python"
test -x "$RQ1_PY"
"$RQ1_PY" -c 'import sys; assert sys.version_info[:2] == (3,11); import rq1, alfworld, sentence_transformers, tokenizers'
"$RQ1_PY" -m pip check
"$RQ1_PY" -m rq1.cli university-gpu feasibility validate --manifest university-gpu/prospective/feasibility-200-candidate.json
```

Use `"$RQ1_PY"` for **every** subsequent module command. Import/version checks do not establish lockfile equality; record installed package versions and compare the approved lock/environment. `pyproject.toml` restricts Python to 3.11; base dependencies are empty, ALFWorld/retrieval are extras, and `uv.lock` exists. System Python/pip success does not prove those extras are installed in the executing environment. Do not install anything automatically to resolve a failed check.

| Gate | Current status | Owner / exact verification action | Stop condition |
|---|---|---|---|
| Source fixes and reviewed clean SHA | BLOCKED | Engineer: repair A01/A02 plus accepted evidence checks; run regression suite; record approved clean SHA. | Any unexpected diff, failed regression or unreviewed source. |
| Runtime/interpreter | NOT VERIFIED on host | Operator: explicit venv imports/version and installed-package/lock inventory. | Missing/wrong interpreter or dependency. |
| Data/task identity | BLOCKED | Operator/data reviewer: run reconcile-data on actual data root, compare selected files and complete scoped difference list, bind report review. | Hash/split/task mismatch or unresolved difference. |
| Exact witnesses | Recorded success; new profile replay pending | Operator/task reviewer: run replay-oracles for all six, inspect state/action/source and terminal evidence. | Any mismatch/failure; no replacement task. |
| Prospective model/tokenizer | BLOCKED | Model owner/independent reviewer: actual manifest + all blobs, endpoint Q4_K_M, authentic tokenizer linkage and samples, exact revision equality. | Missing blob/link, drift or unsupported correspondence. |
| Accum-18 | Historical file exists; documented route BLOCKED | Engineer/reviewer: bind exact archive/member hash, original 18 IDs/text and accepted reference. | Unauthenticated source or fabricated approval. |
| SBERT | NOT VERIFIED on host | Operator: exact local revision/snapshot hash and 768-dimensional local-only load. | Hash/load mismatch or network fallback. |
| Human engineering approval | BLOCKED | Authors: candidate-bound engineering-only record acknowledging prior Pencil outcomes. | Blank/mismatched scope/hash/person/reference. |
| Four GPUs and endpoints | NOT VERIFIED | Approved operator: verify free ports/devices, run approved smoke benchmark and actual PID→UUID evidence. | Wrong digest, shared GPU/endpoint or missing process evidence. |
| Session/access deadline | NOT VERIFIED | University IT/operator: approved session supervision, finite epoch/reserve, ownership and restart procedure. | Unclear permission/lifetime or insufficient reserve. |
| Trace validation and backup | BLOCKED | Engineer/operator: required log/hash checks; stopped export; copy/checksum/readback from independent target. | Missing trace, failed snapshot/export/readback or low disk. |

The report itself makes the checkout dirty. Production clean-source gates should therefore refuse launch until the audit/future fixes are reviewed and committed through a separately authorized workflow. This audit intentionally does not commit or push.

## 8. Fix prioritization

### A. Before Monday's 12 engineering episodes

1. Correct A01 candidate/frozen oracle binding and A02 authentic historical-library reference path. Add a complete freeze→preflight regression; retain fail-closed behavior.
2. Make completion evidence meaningful: required raw-log inventory/hashes, actual action-count consistency, valid unique retrieval IDs and finite scores. Test missing/corrupt logs and out-of-budget synthetic outcomes. Do not modify historical traces.
3. Repair the runbook's interpreter and partial-export sequence; distinguish monitoring, intentional stop and later resume; use new archive names. Reject nonfinite cutoff and unsafe consumed run IDs. Resolve tokenizer revision equality or explicitly verify it in reviewed evidence before freeze.
4. Fix live-log snapshot race and backup-failed status, or demonstrate a reviewed quiescent snapshot/stopped-export mitigation; a result count alone is insufficient.
5. On the target, verify all real assets and all six CPU oracle witnesses, obtain authentic engineering approval, and verify protected model storage/session lifetime, deadline reserve and independent backup. Then perform the authorized four-GPU smoke benchmark. Stop on any failure.
6. Conduct supervised initial engineering execution only after these gates pass; retain attempts, infrastructure failures and exposure disclosures. No scientific claims or automatic task substitutions.

### B. Before future six-arm scientific inference

1. Resolve budget and task eligibility prospectively with author approval. Preserve the frozen 50-action history; neither 200-action engineering nor failed oracle routes activate a scientific amendment. Establish bounded expert/oracle eligibility without using model treatment outcomes for selection.
2. Resolve prospective scientific model identity and historical comparability explicitly. Rerun contemporaneous D0/D18 under the approved new identity/design unless strict compatibility evidence and author approval justify a sensitivity reuse. Current evidence does not justify anchor substitution.
3. Settle fixed previously observed follow-up versus fresh confirmation; freeze target sample, leakage policy, sample-selection procedure and eligible-task counts.
4. Acquire independent blinded relevance ratings/adjudication; audit library coverage/contamination; review filler and exact memory-block dosing; calibrate Gate on valid_seen development labels only. No fabricated labels or outcome-tuned thresholds.
5. Freeze estimands and analysis: primary contrasts, minimum meaningful effect/precision, multiple comparisons, task-cluster uncertainty, seed dependence, family description, missingness/infrastructure censoring. Simulate planned operating characteristics with Monte Carlo uncertainty before claiming adequacy.
6. Resolve A14/A15 claim scope, validate manipulation using injected relevance, and document original-rater sensitivity. Assess whether additional controls answer an explicit scientific question before adding workload.
7. Complete target reliability tests and reproducible environment/runtime capture; settle authorship, AI/tool disclosures, live Overleaf synchronization and institutional/public artifact permissions before dissemination. These are external dependencies, not fabricated completion flags.

Items in queue B can wait until after Monday's strictly engineering work. A04 hardware/assets, genuine approval, trace protection and working launch path cannot.

## 9. Time and compute risks

The planned batch permits **2,400 total environment actions for one attempt per unit**. Prefix/detour replay consumes 298 actions across both arms, leaving **2,102 recovery actions**. Maximum is an allowance, not expected usage or wall time. Each new retry can add up to another 200 actions; launch/benchmark smoke requests, oracle replay, embedding initialization and checkpoint/export overhead add work. No measured 200-action university runtime or throughput is available, so no completion-time estimate or 8-hour sufficiency claim is justified.

A worker's 3600-second response timeout and a 30-minute assignment reserve do not prove it will finish before access ends. Process startup/model loading, long generated actions, disk reserve/snapshot pauses and backup transfer can dominate. Measure approved smoke/early engineering attempts and retain timeout/failure distributions before scheduling unattended work. Do not infer model recovery cost from successful expert witness length.

The 36+540 design is 576 scientific episodes, distinct from these 12 engineering episodes. Counts of 450/360 that reuse anchors remain hypothetical. Do not approve 540 automatically without eligible tasks, target sample and task-cluster operating characteristics.

## 10. Final GO / NO-GO recommendations

| Activity | Recommendation now | Exact prerequisites to change verdict |
|---|---|---|
| A. Four-GPU engineering benchmark | CONDITIONAL in principle; current packaged route BLOCKED | Repair A01/A02 or use a separately reviewed genuine evidence path, verify exact real assets, interpreter, clean reviewed source, hardware availability/permissions, model storage and explicit smoke approval. Run zero ALFWorld episodes; inspect placement/request results. |
| B. Twelve D0/D18 feasibility episodes | **NO-GO currently** | Complete queue A, successful approved smoke/placement, source/data/oracle/model/library review, correct trace validation, session/deadline and independent-backup gates, authentic candidate-bound engineering approval. |
| C. Thirty-six six-arm scientific pilot | **NO-GO** | Queue A plus approved scientific budget/task/model policy, genuine task eligibility, blinded relevance/coverage, filler and Gate calibration, prespecified pilot outcomes/stop criteria. Engineering batch success is insufficient. |
| D. Five-hundred-forty confirmatory study | **NO-GO** | Queue B completed, settled exposure/sample interpretation, pilot-go evidence, complete task-cluster precision/power simulation, primary contrasts/multiplicity/missingness plan and explicit author approval. Neither 540 nor historical-anchor reuse is automatically sufficient. |

No production corrections have been made in this audit. The next authorized action should be a focused engineering repair and regression pass on queue A, followed by target-host evidence collection. Scientific decisions remain author decisions supported by the documented evidence.

## Completion addendum

Correctly scoped full suite:

```powershell
$env:PYTHONPATH=(Resolve-Path src).Path
& .venv/Scripts/python.exe -m unittest discover -s tests
```

**370 tests run in 180.769 s; 366 passed, 4 skipped, 0 failures/errors; exit 0.** `unittest` includes skipped cases in “Ran”; the previous phrasing “370 passed, 4 skipped” overstates passing cases. The four skipped optional integrations are real pilot (`RQ1_RUN_REAL_PILOT_TESTS`), real Hermes profiles (`RQ1_RUN_REAL_HERMES_PROFILE_TESTS`), real ALFWorld (`RQ1_RUN_REAL_ALFWORLD_TESTS`), and real recovery (`RQ1_RUN_REAL_RECOVERY_TESTS`). They were intentionally not enabled; passing offline tests do not replace them.

Focused suite, same environment:

```powershell
& .venv/Scripts/python.exe -m unittest -v tests.test_university_gpu_feasibility tests.test_university_gpu_package tests.test_prospective_identity tests.test_transferred_evidence
```

**48 tests passed in 77.548 s; 0 skipped/failures/errors; exit 0.** These include the eleven feasibility tests, 29 GPU-package tests, six identity tests and two transferred-evidence tests. Concurrent execution of these offline test suites adds timing variability; these wall times are not GPU throughput measurements.

Compilation/static/candidate/parser checks are described in section 4. Test/compilation runs created or refreshed ignored bytecode/cache files and temporary synthetic test outputs; reviewer reproductions used automatically cleaned Windows TEMP directories. No unrelated files were deleted. Post-test Git status showed only this untracked requested audit document; no tracked source/evidence changes. `git diff --check` passed. This document has not been committed or pushed.
