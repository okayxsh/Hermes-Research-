# Prospective budget amendment decision — 2026-10-30

Status: TWO UNAPPROVED PROPOSALS. Recommendation: A for the next prospective pilot, with narrowed claims. Neither option changes historical files or authorizes inference. Evidence basis: `reports/transferred-evidence-audit-20261030.json`; no checkpoint audit was repeated.

## Evidence and decision boundary

Frozen total action cap is 50, including prefix and one detour. Remaining recovery budget is total minus prefix minus one. Verified scripts: clean 95/prefix48/96 total with detour; pick-two 146/prefix74/147 total. Earlier 92/109 narratives are superseded by actual arrays. Other complete traces use 7, 12, 14 and 15 actions. All were run with cap 200. Expert exploration loops do not establish intrinsically long-horizon planning requirements. Rejecting a script is not proof of shortest-plan impossibility. All six source records remain unchanged.

## A — fixed 50, budget-eligible selection

**Estimand:** paired effects of six memory interventions on recovery within 50 total actions among valid-seen tasks that pass a prespecified deterministic oracle eligibility procedure. It is conditional on that eligible population, not all long-route ALFWorld tasks.

**Selection algorithm to approve before running it:** reconstruct the complete original valid-seen inventory and verify its task hashes. Exclude the six previously inspected candidates from the new pilot selection to avoid retaining the exposed successful demonstration or selectively discarding its failures. Archive this exclusion as an exposure rule applied to all six, not a performance rule. Use the existing `select_tasks` seeded round-robin family procedure with seed 29, requested_count equal to the remaining inventory size, on that disjoint inventory. Seed 29 is a proposed fixed value, not an approved or outcome-tuned choice. Write the complete ordered ledger before any oracle run. For each family take the first candidate whose unchanged deterministic expert script has length at most 49, admits a nonterminal navigation checkpoint under the existing midpoint ordering, and passes real prefix/detour/continuation replay with total cap 50. Keep scanning the locked within-family order on a scientific eligibility failure; halt on infrastructure failure. Preserve every reason and evidence hash. Take exactly one eligible task per family; a depleted family blocks the pilot rather than relaxing eligibility. Do not inspect Gemma outcomes during selection. Do not combine a shortest-path search for some tasks with unchanged expert scripts for others.

**Validity/bias:** deterministic eligibility removes outcome-driven replacement, but selects on oracle-script efficiency and environment complexity. The screen changes the target population and may favor easy tasks; disclose excluded proportions and route lengths by family. Seed 29 and uniform exclusion require author review. Historical longest-route selection governed valid-unseen evaluation, while the actual unapproved pilot proposal used seeded round-robin; this proposal resolves that discrepancy explicitly for the pilot only.

**Long-horizon question:** the pilot can test intervention mechanics and bounded recovery, with limited evidence for long-horizon claims. Do not use it to claim efficacy on tasks whose reference routes exceed 49. Long-horizon confirmation requires its own eligibility and precision review; do not automatically propagate this screen to the 30 historical evaluation tasks.

**Comparability:** retains the historical cap, retrieval and arm definitions, but new model/runtime and new selected tasks still prevent pooling historical baselines. Run prospective D0/D18 cells contemporaneously.

**Compute:** 36 pilot episodes unchanged, maximum 1800 environment actions; oracle screening adds one expert route and rejected checkpoint replays per candidate and is CPU-only. Limit screen to the finite inventory ledger; report attempted candidates/action counts. No promised runtime or GPU sufficiency. Confirmation remains 540 hypothetical episodes, subject to separate eligibility/power approval.

**Inputs/approvals:** create a new prospective selection amendment, disjoint task proposal, eligibility/exclusion ledger, capped oracle definitions, data hashes, task contexts, new 108-pair human ratings, Gate calibration, model/tokenizer/filler identity, frozen manifest and launch approval. Preserve old proposal and definitions. Authors approve target population, seed/exposure rule, selection order, cap and replacement policy before execution; task reviewer approves eligibility; raters/adjudicator and authorized authors supply judgments and freezes. No signatures are populated here.

## B — retain current six tasks, 200 total actions

**Estimand:** paired recovery effects on the six existing valid-seen tasks with cap 200; remaining task-specific recovery allowances are 151 clean, 125 pick-two, 193 cool/heat, 194 light and 195 pick-and-place. These allowances are equal across arms within a task, not across tasks. Declare this estimand before prospective recovery outcomes. The proposal does not change the historical study's 50-action definition.

**Justification:** 200 is the cap actually used in all six saved oracle tests and covers the longest observed 147-action detoured script with 53 actions of remaining headroom. It is not statistically optimized or claimed minimal. A cap of 150 leaves only three actions of headroom on that script; no evidence supports that as a robust recovery allowance. These facts justify proposing 200 for author review, not silently adopting it.

**Validity/bias:** holding tasks fixed avoids replacing difficult scripts with favorable tasks, but one task has exposed engineering Gemma outcomes, and the decision to increase the budget was informed by oracle failures under 50. Disclose both. Freeze all six tasks together without outcome-based substitution and treat this pilot as development/calibration evidence. A fresh independently selected confirmatory sample and separate protocol are required. Longer exploratory expert loops are not established task difficulty; independently define structural horizon/required subgoals before calling these intrinsically challenging tasks.

**Oracle eligibility:** the transferred traces already provide successful route witnesses within 200, but rehash original data, reconcile adapter/discovery identity definitions, and replay exact task/seed/checkpoint/detour at the amended cap on the current runtime. Verify no early terminal states or invalid actions, stable state digests and budget accounting. A replay must not be used to tune model-facing recovery behavior. No bounded search is needed to show these observed scripts fit 200; no shortest-path optimality claim is made.

**Long-horizon question:** retains the two long-script cases and affords useful recovery time after prefix74. Larger budgets may hide early degradation and change the success endpoint; prespecify secondary actions/latency and success-by-action curves without promoting a post-hoc cutoff to the primary outcome. Do not extend timeouts after seeing failures.

**Comparability:** budget and current model both differ from historical evaluation. Historical cells are separate references, not primary controls or replications. All six arms need contemporaneous prospective execution under the same 200 cap. If both budget questions are later studied, prespecify a separate factorial budget study; do not pool the two pilot designs.

**Compute:** 36 pilot episodes still, worst-case 7200 environment actions versus 1800 under A, a fourfold cap increase, not a measured fourfold runtime increase. At 540 episodes, caps would be 108000 versus 27000 actions; confirmation at 200 remains a separate decision. Saved six oracle scripts sum to 291 actions including detours, but this is no forecast of agent behavior, model calls or wall time. Benchmark throughput only after approved engineering gates, then compute a credible access-window/cutoff and timeout plan.

**Inputs/approvals:** new prospective protocol version with 200-action arithmetic, per-task budgets, timeout/maximum-selection-attempt review, data/oracle hashes, manifest validation, worker accounting and analysis provenance. Historical constants remain 50. A separate versioned launch path must bind the approved budget and model identity through manifest, worker, executor, resume and export; changing one constant or bypassing a preflight is forbidden. Regenerate rating contexts if task/checkpoint state changes, otherwise review genuine context compatibility rather than assume it. Authors approve budget rationale, exposure disclosure, success estimand, compute/time limits and statistical plan; task reviewer approves oracle evidence; raters and adjudicator supply all labels; authors supply separate freeze and GPU approvals.

## Recommendation

Choose A for the next technical/scientific pilot if authors accept the narrower eligible-population claim. It preserves the existing action endpoint, limits compute, and avoids treating expert search inefficiency as intrinsic horizon. Choose B instead only if retaining these specific cases is a scientific priority and authors accept the new endpoint and development-pilot exposure limitations. Neither is approved. Do not generate task-specific rating packages or threshold calibration before that decision and eligible contexts are fixed.

## Remaining blockers and executable handoff

| Item | State / owner |
|---|---|
| Budget/selection/exposure amendment | Human authors choose A or B before selection or new freeze |
| Original task files and deterministic cap-specific replay | University operator + task reviewer |
| Current model manifest, all blobs, tokenizer linkage and reference-token samples | University operator supplies exact local assets; engineer verifies |
| New versioned identity/budget launch route | Engineer after approved scientific schema; historical route remains fail-closed |
| Library/context hashes and 108-pair pilot rating package | Engineer prepares blank materials after task approval; humans provide 216 judgments and adjudication |
| Gate, filler, precision/power, freezes and launch permissions | Authors/raters; no inferred threshold or sufficiency |
| GPU placement/coordinator and backup readback | University workstation validation pending |
| GitHub sync | Five prior commits verified; SSH must pass strict host verification before push |

`PYTHONPATH=src python3.11 -m rq1.university_gpu.prospective_identity --help` exposes the new offline identity-evidence verifier. Provide the exact saved manifest, its local blob directory, tokenizer.json, and provenance JSON. Provenance requires model_manifest_sha256, tokenizer_sha256, source_repository, full revision, model_tokenizer_link_evidence_sha256 and at least five distinct samples with text/token_ids/reference_source. Use JSON serialization, Unicode, empty input and task/skill text samples from an independently recorded reference tokenizer with documented special-token behavior. No source URL/hash/revision is guessed. Export exact files from the workstation/model owner; verify any license/access terms before an independently approved acquisition. This tool checks local byte identity and finite reference samples, emits only CANDIDATE_REVIEW_REQUIRED and never activates the historical launcher.

For operator-supplied real paths, use the command below after setting each variable to an existing evidence file (output must be new):

```bash
: "${RQ1_MODEL_MANIFEST:?Set to the exact saved Ollama manifest}"
: "${RQ1_MODEL_BLOBS:?Set to its existing blobs directory}"
: "${RQ1_GEMMA_TOKENIZER_JSON:?Set to the exact local tokenizer.json}"
: "${RQ1_TOKENIZER_PROVENANCE:?Set to reviewed source and reference-sample JSON}"
: "${RQ1_TOKENIZER_LINK_EVIDENCE:?Set to the actual model-tokenizer linkage evidence file}"
: "${RQ1_NEW_IDENTITY_EVIDENCE:?Choose a new output file outside the checkout}"
PYTHONPATH=src python3.11 -m rq1.university_gpu.prospective_identity \
  --manifest "$RQ1_MODEL_MANIFEST" --blobs "$RQ1_MODEL_BLOBS" \
  --tokenizer "$RQ1_GEMMA_TOKENIZER_JSON" --provenance "$RQ1_TOKENIZER_PROVENANCE" \
  --link-evidence "$RQ1_TOKENIZER_LINK_EVIDENCE" \
  --output "$RQ1_NEW_IDENTITY_EVIDENCE"
PYTHONPATH=src python3.11 -m rq1.cli university-gpu check --json
```

Runtime defaults proposed for review remain temperature0, seed42, think=false, num_predict2048, num_ctx32768 with actual Ollama/Hermes versions recorded. Quantization and template/config semantics still require manifest/config inspection. Token counts match the serialized recovery-memory block, not whole prompts. Rating infrastructure exists but emits no forms for this unresolved selection. Statistical confirmation still needs task-cluster precision/power and multiplicity review. See NEXT_VISIT_RUNBOOK.md for stop/resume/export syntax; none of those commands grants scientific launch authorization.

## Verified Git handoff and test record

The five original unpushed commits are b1a57c2, 1087bfb (merge preserving 2ba4ea8), 1a13bee, 6971094 and 85f32c0. The verified remote remains 2ba4ea8. All ancestry is preserved. Strict SSH verification fails on this laptop because the port-443 host key is unknown. No key/config change or push was made.

A bundle named `rq1-feature-handoff-20261030-amendments.bundle` will be created outside the checkout after this change is committed. Transfer that file and its SHA-256 over approved storage. On the university machine use the existing authorized identity; do not copy its private key. The following requires operator-supplied existing paths and a clean feature checkout. If remote source has advanced incompatibly, stop and reconcile; never force a merge or push.

```bash
cd ~/Hermes-Research-
: "${RQ1_HANDOFF_BUNDLE:?Set to the transferred bundle file}"
git branch --show-current
git status --short
# Continue only with a clean feat/rq1-university-4gpu checkout.
git bundle verify "$RQ1_HANDOFF_BUNDLE"
git fetch "$RQ1_HANDOFF_BUNDLE" feat/rq1-university-4gpu
git merge --ff-only FETCH_HEAD
git remote get-url origin
git ls-remote origin refs/heads/feat/rq1-university-4gpu
# Continue only after the existing SSH remote is verified as okayxsh/Hermes-Research-.
git push origin HEAD:refs/heads/feat/rq1-university-4gpu
git ls-remote origin refs/heads/feat/rq1-university-4gpu
```

Tests: 44 targeted tests passed (identity verifier, readiness, GPU package); after the last token-ID validation refinement the six identity tests passed again. Python compilation, CLI help and whitespace checks passed. Historical config and evaluation protocol files have no changes in this work. No model inference, task selection or scientific approval occurred. Exact assets remain unavailable, so the identity verifier is tested engineering, not verified compatibility of the actual prospective model.
