# Final budget and pilot selection recommendation

**Evidence cut-off: 2026-10-30.** This review supersedes the *recommendation* in the earlier `BUDGET_AMENDMENT_DECISION.md`; that earlier file remains unchanged as historical evidence.

**Decision status: pending authors. No amendment is approved or active.** The recommendation is Option B for a development and feasibility pilot, because Option A risks selecting away the long-horizon feature named in RQ1. Option B cannot be presented as an uncontaminated treatment-effect pilot: one proposed task has prior Gemma memory/no-memory outcomes. A remains defensible if authors prefer the historical endpoint and accept a narrower target population.

This review uses the verified transfer audit and prior source inspections. It does not repeat checkpoint hashing or change any frozen file.

## A/B comparison

| Dimension | A: retain total cap 50, select eligible tasks | B: amend pilot total cap to 200, retain tasks |
|---|---|---|
| RQ1 question | Answers bounded recovery for tasks with an expert witness fitting 49 actions including one detour. Claims become conditional on eligibility. | Better tests recovery after the demonstrated long routes and leaves 125 recovery actions after the 74-action prefix task. It tests the original long-horizon motivation more directly. |
| Internal validity | Keeps the historical action endpoint. Eligibility selection changes the task population and selects partly on expert-script efficiency. | One cap applies to every arm and task; paired comparisons remain possible. Endpoint changes from historical evaluation, and a task has prior model-outcome exposure. |
| Selection bias | A route screen can favor shorter/easier tasks. A fixed seeded rule prevents outcome-driven replacement but cannot remove this construct-selection bias. | Retains the proposed six tasks, avoiding post-hoc task replacement. Actual proposal is seeded round-robin, unlike the historical longest-route valid-unseen rule. |
| External validity / difficulty | Supports claims only for the eligible valid-seen population. A pilot cannot claim coverage of the long-route cases it excludes. | Retains observed long expert scripts; route length alone does not prove intrinsic task difficulty because the pick-two witness contains substantial exploration. |
| Retrieval-noise sensitivity | A 1-action recovery allowance for clean-and-place risks a floor; short tasks may ceiling. | More opportunity to observe whether noise can be recovered from, but a 200 cap can hide slow recovery. Prespecify action-use and latency summaries. |
| Historical comparability | The 50 endpoint matches, but tasks and the current model identity do not. No pooling or replication claim. | Both action endpoint and model identity differ. All arms need contemporaneous controls; no historical pooling or replication claim. |
| Maximum workload | 36 episodes; 1,800 total episode-action cap. Oracle screening adds CPU environment work for candidates. | 36 episodes; 7,200 total episode-action cap. This is four times the maximum action allowance, not a measured fourfold runtime. |
| Statistical precision | Six task clusters and one seed remain a feasibility pilot. | Same six clusters and one seed; the larger cap adds no independent observations or precision. |
| Floor/ceiling and triviality | Eligibility may remove long routes and still yield easy tasks; describe the selected route distribution. | Cap can reduce budget exhaustion but may ceiling on simpler families. The pilot tests feasibility, not population effects. |

## Evidence and budget rule

The frozen evaluation protocol defines total actions as prefix + one controlled detour + recovery actions, with recovery allowance `50 - prefix_length - 1`. Transferred evidence reports the following. All six oracle traces succeeded under a recorded action limit of 200; the four traces totaling 7, 12, 14, and 15 actions fit within a 50-action witness. These runs were not performed with the 50 limit, and original ALFWorld task files were not in the transfer for independent rehashing.

| Family | Route | Prefix | Recovery at 50 | Recovery at 200 | Detoured expert witness | Decision |
|---|---:|---:|---:|---:|---:|---|
| clean_and_place | 95 | 48 | 1 | 151 | 96 | Blocked as constructed |
| cool_and_place | 13 | 6 | 43 | 193 | 14 | Witness fits; cap-50 replay pending |
| heat_and_place | 14 | 6 | 43 | 193 | 15 | Witness fits; cap-50 replay pending |
| look_at_object | 11 | 5 | 44 | 194 | 12 | Witness fits; cap-50 replay pending |
| pick_and_place | 6 | 4 | 45 | 195 | 7 | Witness fits; cap-50 replay pending; Gemma outcome exposed |
| pick_two_and_place | 146 | 74 | -25 | 125 | 147 | Blocked as constructed |

“Blocked as constructed” means the particular frozen prefix, detour and unchanged expert continuation cannot satisfy the cap. It does not prove that no shorter ALFWorld plan exists. Four “witness fits” are action-count evidence only; the harness needs an actual 50-cap replay to verify the runtime enforces the cap with the same split, seed, checkpoint and state digests. Never mark oracle eligibility from a 200-cap test alone.

## Selection rules and exposure

The historical amended **valid-unseen** policy is five longest hand-coded expert routes per family, with deterministic task-ID tie-breaks and replacements when no checkpoint supports a valid controlled failure. The actual **pilot proposal** is status `proposed`, seed 11, round-robin across families (`task-selection-v1`) with one task per family. Neither pilot proposal nor amendments are author-approved. Do not silently substitute the historical policy for the pilot policy. The earlier provisional Option A used seed 29 and a blanket exclusion of all six oracle-inspected tasks; this review rejects that blanket exclusion because the evidence distinguishes oracle inspection from Gemma model-outcome exposure, and it instead requires an outcome-independent rule that excludes the one model-exposed context before selection.

Prior oracle validation exposed route/checkpoint/detour outcomes for all six. That is not Gemma outcome exposure. The evidence identifies prior Gemma recovery success in the pick-and-place task for memory and no-memory conditions. Avoid the earlier proposal’s unnecessary exclusion of all six tasks solely because their oracle routes were checked.

For Option A, the minimal outcome-independent policy is: verify the complete valid-seen inventory and task hashes; predeclare exclusion of the one exposed Gemma task irrespective of its success; use the proposal’s seed 11 and round-robin family order on the remaining inventory; within each family choose the first locked-order task whose deterministic expert witness, midpoint checkpoint, detour and unchanged continuation fit and pass under cap 50. Record every candidate, replay, failure, route length and hash; infra errors halt. Require one per family or block. Publish inclusion and exclusion route-length distributions by family against the full inventory. The result is not representative if eligibility removes most longer tasks. Do not call route length recovery difficulty.

For Option B, retain the six proposal identities and never replace one after seeing model behavior. Treat the pilot as development/feasibility evidence because the pick-and-place task already has model outcomes. Do not include those old attempts as new cells, and do not claim an uncontaminated six-task effect. Before a later confirmatory study, use fresh valid-unseen tasks and decide whether to carry 200 forward; a 50-cap confirmation would answer a different endpoint.

## Recommended prospective amendment: Option B, cap 200

Propose 200 total actions for the prospective six-task pilot only. This is the bound used by all six transferred oracle validations and accommodates the longest observed detoured witness of 147 actions with 53 actions of headroom. It is not a minimal or optimized cap. A 150 cap would leave only three actions beyond that witness, with no evidence that this is robust. The cap is not a conclusion that any task is intrinsically 147-step difficult.

**Retain as invariants:** six arms D0/D18/Rel/Irr/Fill/Gate; paired task and seed assignments; single post-detour retrieval; same frozen retrieval/library rules; blinded condition labels; no skill writes; failure-state digest checks; no outcome-dependent reordering; task-clustered interpretation; independent human relevance labels; historical data and protocol unchanged.

**Necessarily version:** prospective total and per-task recovery budgets; success-within-200 primary endpoint; task/failure manifests and hashes; timeout and selection retry limits; protocol identity and launch validation; run analysis denominators; current model identity/runtime. Keep the historical 50 constant intact. Do not change one constant or weaken preflight. If a confirmatory design later adopts 200, it needs its own author-approved version, fresh valid-unseen oracle evidence, task-cluster power/precision simulation and contemporaneous D0/D18 cells.

**Interpretation:** the development pilot can establish that the system can execute the interventions and can expose operational/outcome distributions on these contexts. Because of one prior outcome-exposed task, six tasks and one seed, report no confirmatory causal effect or inferential generalization from it. A later fresh confirmatory sample can estimate the causal contrasts under its specified 200-action endpoint. Historical results remain reference evidence only; current Gemma digest differs from historical digest.

**Censoring and metrics:** prespecify binary success by action 200 as the endpoint; record every post-detour action, infrastructure failure, retry, cap exhaustion and censoring reason. Report the action-to-success distribution and latency as secondary summaries; do not choose a cutoff after inspecting outcomes. Infrastructure failures are not recovery failures and require the prespecified missing-cell/retry rule. Analyze paired contrasts at task level and treat 30 tasks, not 540 episodes, as the independent task clusters. Prespecify primary contrasts/multiplicity and run task-cluster precision/power simulations before confirmation; 540 is a design count, not evidence of sufficiency.

## Required decisions and validation

Authors must explicitly approve or reject: Option B and cap 200; development-only status of the six-task pilot; disclosure/exclusion handling for prior pick-and-place conditions; whether confirmation also uses 200 or a narrower cap-50 population; target estimand, primary contrasts, multiplicity and censored/infrastructure denominator rules. No author name, approval date or signature is prefilled.

Before any scientific launch, complete these steps in order: (1) authors record a signed selection and budget decision; (2) the operator supplies original ALFWorld source/game files and an independent reviewer reconciles the two data-identity algorithms and source hashes; (3) the reviewer checks task IDs, family membership, prefix/checkpoint state and performs same-seed/split/detour/state-digest replay at the selected cap; (4) authors explicitly classify the pilot as feasibility-only or approve a distinct inferential estimand; (5) verify prospective Gemma manifest, blobs, tokenizer linkage and runtime while preserving the historical digest; (6) after exact contexts are frozen, obtain independent relevance ratings and adjudication, then review filler and calibrate Gate; (7) complete token-matching and four-GPU coordinator/placement/backup checks; and (8) issue a new versioned manifest, statistical analysis plan and launch approval. No scientific episodes are authorized by this proposal.

## Smallest engineering implementation

After approval, add a **versioned** budget field and approved protocol identifier to the prospective manifest; validate `total_action_budget = prefix + 1 + recovery_budget`; pass it explicitly through `EvaluationTask`, worker configuration, `AmendedEvaluationExecutor`/`RealRecoveryHarness`, oracle replay, resume identity, result rows and export validation. Keep historical defaults fixed at 50. Add fixtures for approved-200, historical-50, mixed-budget rejection, resume drift and cutoff accounting. CPU tests can verify arithmetic and gates; only the university can validate real ALFWorld and live GPU placement. Do not add an alternate route search unless authors make that a common, independently validated eligibility rule.

The proposed Gemma identity verifier is offline provenance preparation only. Actual model manifest, blobs, tokenizer, tokenizer-model linkage and reference-token samples are still unavailable locally. Existing worker/preflight intentionally rejects digest `6114515d...`; no launcher activation is included. Rating templates already exist, but no 108-pair form should be generated until exact six task contexts and the budget decision are approved. Gate threshold remains uncomputed until human labels exist.

## Author decision template

Complete a copy of `BUDGET_AMENDMENT_AUTHOR_DECISION_TEMPLATE.md`. The two choices, consequences, provenance and approval fields are defined there. Blank signature and date fields are intentional.

## Remaining uncertainty and external work

| Item | Owner / status |
|---|---|
| Choose A or B and confirm whether pilot is development-only | Authors; pending |
| Rehash task/game source and reconcile identity definitions | University operator + engineer; workstation required |
| Same-state cap-specific oracle validation | University operator + task reviewer; no replay yet |
| Prospective model/tokenizer compatibility | Model owner + engineer; assets/linkage pending |
| Independent 108-pair pilot ratings and adjudication | Human raters/adjudicator; not started |
| Gate, filler, approvals, task-cluster precision/power | Authors/raters; pending |
| Four-GPU coordinator/placement and independent backup readback | University PC; pending |
| GitHub sync | Seven local commits ahead after this decision; read-only HTTPS lookup confirmed the public branch at `2ba4ea8`. Local history is a fast-forward descendant; no push was performed. |

## Safe Git handoff

A transfer bundle was created outside the checkout at `rq1-feature-handoff-20261030-budget-decision-v2.bundle` with a SHA-256 sidecar. It contains the feature ref and declares `2ba4ea8` and `7d65e47` as prerequisites. The university checkout should have both after fetching the current public feature branch and retaining its original `7d65e47` history. Transfer the bundle and sidecar over approved storage; do not copy credentials.

On the university workstation, first verify the checkout and transfer, then fast-forward the local feature branch. Stop if the branch, cleanliness check, prerequisite, ancestry, or checksum differs:

```bash
cd ~/Hermes-Research-
git branch --show-current
git status --short
# Continue only on feat/rq1-university-4gpu with a clean worktree.
git remote get-url origin
git fetch origin refs/heads/feat/rq1-university-4gpu:refs/remotes/origin/feat/rq1-university-4gpu
RQ1_HANDOFF_BUNDLE=/path/to/rq1-feature-handoff-20261030-budget-decision-v2.bundle
cd "$(dirname "$RQ1_HANDOFF_BUNDLE")"
sha256sum -c "$(basename "$RQ1_HANDOFF_BUNDLE").sha256"
git -C ~/Hermes-Research- bundle verify "$RQ1_HANDOFF_BUNDLE"
git -C ~/Hermes-Research- fetch "$RQ1_HANDOFF_BUNDLE" refs/heads/feat/rq1-university-4gpu
git -C ~/Hermes-Research- merge-base --is-ancestor HEAD FETCH_HEAD
git -C ~/Hermes-Research- merge --ff-only FETCH_HEAD
git -C ~/Hermes-Research- status --short --branch
```

This prepares a local fast-forward only. Publishing is a separate operator action: re-fetch and inspect the live remote first, then use a normal `git push origin HEAD:refs/heads/feat/rq1-university-4gpu` only if the remote ref remains an ancestor of local `HEAD`. If it is not, stop and reconcile; never force-push.

The Monday **2026-11-02** 36-episode scientific pilot is not launchable under the present state. At best, after author approval and the required source/model/human/host checks, Monday can be used for an explicitly authorized engineering feasibility session. The current system must not be described as ready for scientific launch.
