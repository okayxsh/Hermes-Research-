# Scientific prerequisites (read-only status)

Captured 2026-10-29 from
`university-gpu/check_scientific_prerequisites.py` against the current checkout.
This is a status report, not an approval, label, threshold, or validation record.
The checker is read-only and exits **1** while any requirement is not `PASS`.

Reproduce:

```powershell
.venv\Scripts\python.exe university-gpu\check_scientific_prerequisites.py
.venv\Scripts\python.exe university-gpu\check_scientific_prerequisites.py --json
```

## Status table

| # | Requirement | Status | File examined | Reason / next human action |
|---|---|---|---|---|
| a | Pilot manifest approved and hash-bound | **BLOCKED** | `university-gpu/approval-templates/pilot-manifest.json` | `approved` is false and `manifest_sha256` is blank. A responsible human must approve the exact frozen pilot manifest. |
| b | Pilot relevance completeness | **BLOCKED** | `../university-friday-preparation-20261029/friday-rating-package` | Existing validator reports **1620 outstanding ratings**. Rater A, rater B, and adjudication must be completed by humans. |
| c | Historical reuse decision | **BLOCKED** | `.../friday-rating-package/historical-reuse-decision.md` | Selected option, rationale, two author signatures, and adjudicator acknowledgement are blank. A checked option box is not a decision. |
| d | Gate calibration | **BLOCKED** | `artifacts/gate-frozen.json` (expected) | No frozen valid-seen threshold and no adjudicated valid-seen Gate scores/labels. Needs CPU Sentence-BERT scores, adjudicated labels, and human calibration approval. |
| e | Neutral filler | **BLOCKED** | `artifacts/approved-neutral-filler.json` (expected) | No approved filler record exists in the checkout; the approved filler is an external evidence input. |
| f | Tokenizer | **BLOCKED** | `artifacts/gemma-tokenizer-record.json` (expected) | No hash-pinned Gemma tokenizer record exists in the checkout; the tokenizer is an external evidence input. |
| g | Leakage | **BLOCKED** | `data/task_lists/pilot_seen.json` | Pilot list is empty (`"tasks": []`). An empty list is **not** PASS-by-vacuity; no pilot tasks are frozen. |
| h | Sample size / precision | **NOT_IMPLEMENTED** | `university-gpu/docs/FRIDAY_FINAL_READINESS.md` | Prospective task-cluster power/precision is not implemented. `../university-friday-preparation-20261029/historical-cluster-simulation.json` is historical context only, not a power result or sufficiency claim. |
| i | Historical baseline compatibility | **BLOCKED** | `university-gpu/docs/HISTORICAL_BASELINE_COMPATIBILITY.md` (expected) | No compatibility review record found. The readiness report states historical NoLib/Accum-18 equivalence is not established. |

Summary: **0 PASS, 8 BLOCKED, 1 NOT_IMPLEMENTED**; exit code 1.

## Scientific requirements that cannot be completed tonight

- **Adjudicated pilot relevance labels need real pilot tasks.** The six-task
  `valid_seen` pilot list is unfrozen and no ALFWorld data is present locally, so
  the 108 pilot task–skill pairs cannot be generated, rated, or adjudicated. The
  confirmatory 540 pairs also remain blank (1620 outstanding cells across the
  three rating files).
- **Prospective power/precision needs an author-approved decision.** The planned
  contrast, minimum relevant effect, task heterogeneity, seed correlation,
  eligibility loss, multiplicity handling, and target precision are not
  prespecified by code. The historical 30-task simulation must not be treated as
  prospective power or as evidence that 576 units are adequate.
- **Historical baseline compatibility needs a review.** A fair side-by-side check
  of task/checkpoint identity, perturbation, model/runtime, memory and token
  handling, retrieval timing, eligibility, seed/order, and missingness has not
  been recorded or approved; historical cells remain references only.
- **Gate calibration, neutral filler, and tokenizer need external artifacts and
  human review.** The frozen Sentence-BERT scores, adjudicated valid-seen labels,
  exact Gemma tokenizer file/revision/hash, and approved filler are not in the
  checkout. None may be guessed or synthesized here.
- **Leakage cannot be reviewed before the task lists exist.** With no frozen pilot
  tasks and no frozen `valid_unseen` IDs, the overlap check is unverifiable, not
  passed.
