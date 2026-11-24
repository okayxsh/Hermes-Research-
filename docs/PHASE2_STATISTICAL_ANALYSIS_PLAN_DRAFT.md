# RQ1 Phase 2B: statistical analysis plan (DRAFT, PROPOSED)

**Status: DRAFT, not approved.**
- The implementation is in `src/rq1/five_arm/analysis.py`, `stats.py`, `power.py` and `pilot.py`.
- It is tested only on synthetic data.
- The governing decisions are SD-08, SD-10, SD-11, SD-20 and SD-21 in `PHASE2_DECISION_REGISTER_DRAFT.md`. All await approval.

## 1. Hierarchy, estimand and primary test

**Primary comparison (single, confirmatory): D18 − D0, over every roster task.**

- **Outcome:** environment-reported success within the approved total action cap after the controlled disruption.
- **Unit:** the task. Per task, the outcome is the mean over the inference seeds with VALID outcomes in **both** arms (pairwise complete case).
- **Estimand:** the task-weighted mean within-task difference, reported in percentage points.
  - It describes the **finite screened population** of roster tasks.
  - It is the total effect of natural Accum-18 retrieval versus no memory, not an isolated relevance effect.
- **Test:** a two-sided task-level sign-flip permutation test at α = 0.05, with no multiplicity adjustment because there is one primary comparison.
  - It is exact by enumeration for ≤ 20 non-zero differences; otherwise a seeded Monte-Carlo with 100,000 draws.
  - With one binary pair per task it equals the exact McNemar/sign test.
- **Primary interval (SD-21):** a template-cluster bootstrap percentile 95% CI (10,000 replicates).
  - Goal templates are resampled with replacement, keeping every task of a drawn template; the task-weighted ratio estimator is used.
  - Sensitivity interval: the task bootstrap CI.

**Secondary, exploratory comparisons** (Holm *within* this family only; never pooled with the primary; reported as exploratory):
- D18 − Fill, Irr − Fill, Rel − Irr, Rel − Fill, over their frozen subsets.
- Restricted-mean recovery actions (failures counted at the full allowance), invalid actions and latency (failures included).
- Cumulative success by recovery action.
- Retrieved Precision@3 (strict; lenient sensitivity) and injected relevance by arm.
- Acquisition-template-overlap exclusion.

Rel − D18 is never treated as a relevance contrast (1 versus 3 active skills).

## 2. Audit of the task-level sign-flip test

**Null hypothesis tested.** The *sharp* null: within every task, the two arms' outcome distributions are identical, so the arm labels are exchangeable within the task. Given task-level latent factors, tasks are conditionally independent; each episode is a fresh session from the frozen checkpoint.

Under this null each task difference d_i is symmetric about zero and independent across tasks, conditional on |d_i|. Enumerating sign flips is therefore exact.

**Template dependence:**
- Shared template or scene effects that act on **both** arms cancel inside d_i. They do not invalidate the test of the sharp null.
- The test is **not** an exact test of the weak null "average effect = 0" when effects are heterogeneous and correlated within templates. The code and docs no longer describe it as exact for that case.
- **Template-aware sensitivity:**
  - the template-level sign-flip (unit = template mean difference; 27 units);
  - the template-cluster bootstrap interval, which is the primary interval.
- The simulation (§4) shows the template-level test has less power, sharply so under an uneven template allocation. It is therefore a sensitivity analysis, not the primary test.

**Scenes.** There are only three scenes, so per-scene and leave-one-scene-out estimates are **descriptive**. No scene-cluster asymptotics are used.

**Family/scene confounding.** Family and scene are confounded, and a family × scene table is reported with every contrast.

**Generalisation.** The estimate describes the screened finite population. Inference to broader ALFWorld tasks assumes exchangeability that the screening does not provide. The template-cluster interval reflects only the dependence among the observed templates.

## 3. Missingness and retries

- **Infrastructure failures** (provider, environment, interruption, including process death detected on resume) are retried up to **2** times with the same unit identity.
  - Retrying anything else aborts the analysis.
  - So do exceeding the limit, gaps in attempt numbering, or two VALID outcomes for one unit.
- **Model failures, budget exhaustion and selection exhaustion are outcomes.** They are never retried or treated as missing.
- **INVALID_MEASUREMENT** (context limit reached, provider/local token discrepancy beyond tolerance, missing provider count in real runs) is missing, never a failure.
- **Primary analysis:** pairwise complete case within task. Denominators and status counts are reported by arm; tasks with no complete pair are listed.
- **Sensitivity:** worst-case and best-case bounds. Records for unplanned cells abort the analysis.
- **Blocked filler cells** are excluded prospectively from Fill/Rel/Irr subsets and never remove a task's D18/D0 pair.

## 4. Prospective power and precision for D18 − D0 (synthetic; not evidence of any effect)

**Model:**
- logit random effects: template SD 0.8, scene SD 0.4 (three scenes), task SD 1.0;
- treatment-effect heterogeneity: template-level SD 0.3, task-level SD 0.5;
- the effect is calibrated so that the **marginal** success difference equals Δ at marginal reference success p_ref.

**Assumptions:**
- *Templates:* 27, with a balanced allocation of 79 tasks (25×3 + 2×2). This is an assumption; real counts come from the frozen roster. A skewed allocation is run as a sensitivity.
- *Seeds:* with probability 0.7 all seeds of a task/arm share one outcome (temperature-0 near-determinism); otherwise they are independent given the task probability.
- *Missingness:* 2% of cells are MCAR.

**Simulated test:** the task-level sign-flip via its conditional normal approximation (z = Σd / √Σd²).

**Settings:**
- 2,000 replications per scenario, seed 20261124; Monte-Carlo SE ≤ 0.011.
- The +33.2 pp engineering difference is **not** used as an expected effect.
- Output `power-simulation.json`, `content_sha256` `db540edeed7e3a5518bde3b9fa0bbb4dc9e59e986ee34276b12e2658197254ec`, not committed. Regenerate with `python -m rq1.five_arm power --reps 2000 --grid all`.

**79 tasks, α = 0.05, power with Monte-Carlo SE in brackets** (precision = 1.96 × sampling SD of the estimate):

| p_ref | Δ | 1 seed | 3 seeds | 95% precision ±, 1 seed / 3 seeds |
|---|---|---|---|---|
| 0.3 | 0.05 | 0.13 (0.01) | 0.14 (0.01) | 0.13 / 0.12 |
| 0.3 | 0.10 | 0.31 (0.01) | 0.42 (0.01) | 0.14 / 0.12 |
| 0.3 | 0.15 | 0.59 (0.01) | 0.68 (0.01) | 0.13 / 0.12 |
| 0.3 | 0.20 | **0.82** (0.01) | **0.90** (0.01) | 0.14 / 0.12 |
| 0.3 | 0.25 | 0.95 | 0.99 | 0.14 / 0.12 |
| 0.3 | 0.30 | 0.99 | 1.00 | 0.14 / 0.12 |
| 0.5 | 0.05 | 0.12 (0.01) | 0.13 (0.01) | 0.14 / 0.12 |
| 0.5 | 0.10 | 0.28 (0.01) | 0.36 (0.01) | 0.13 / 0.12 |
| 0.5 | 0.15 | 0.55 (0.01) | 0.65 (0.01) | 0.14 / 0.12 |
| 0.5 | 0.20 | **0.83** (0.01) | **0.90** (0.01) | 0.14 / 0.12 |
| 0.5 | 0.25 | 0.96 | 0.98 | 0.13 / 0.12 |
| 0.5 | 0.30 | 1.00 | 1.00 | 0.13 / 0.12 |
| 0.7 | 0.05 | 0.13 (0.01) | 0.14 (0.01) | 0.13 / 0.11 |
| 0.7 | 0.10 | 0.36 (0.01) | 0.45 (0.01) | 0.12 / 0.11 |
| 0.7 | 0.15 | 0.70 (0.01) | **0.79** (0.01) | 0.12 / 0.11 |
| 0.7 | 0.20 | **0.92** (0.01) | 0.97 | 0.12 / 0.11 |
| 0.7 | 0.25 | 0.99 | 1.00 | 0.13 / 0.11 |

(p_ref + Δ > 0.95 is not simulated.)

**Dependence sensitivity** (79 tasks, 3 seeds, p_ref 0.5):

| Assumption changed | Power Δ = 0.10 | Power Δ = 0.15 |
|---|---|---|
| default (seed determinism 0.7) | 0.36 | 0.65 |
| seeds fully independent (determinism 0) | 0.61 | 0.94 |
| seeds fully deterministic (determinism 1) | 0.32 | 0.59 |
| task SD 0.5 (more discordant pairs) / 2.0 | 0.32 / 0.45 | 0.63 / 0.75 |
| template SD 0 / 1.5 | 0.36 / 0.40 | 0.62 / 0.73 |
| template effect heterogeneity SD 0 / 0.8 | 0.38 / 0.37 | 0.68 / 0.64 |
| 0% / 10% missing cells | 0.38 / 0.33 | 0.69 / 0.66 |
| skewed template allocation | 0.35 | 0.66 |
| template-level sign-flip, skewed allocation | 0.22 | 0.44 |

**Other observations:**
- **Paired discordance:** 0.30–0.44 of task-seed pairs are discordant across scenarios, depending on p_ref and the task SD.
- **Interval coverage:** Wald coverage of the marginal Δ is about 0.92–0.95 with task SEs and 0.92–0.95 with template-cluster SEs in all scenarios except near ceiling (p_ref 0.7, Δ 0.25: about 0.89–0.90). The percentile cluster bootstrap is used for reporting.

**Smaller populations** (1 seed, p_ref 0.5):

| n | Δ 0.10 | Δ 0.15 | Δ 0.20 | Δ 0.25 |
|---|---|---|---|---|
| 40 | 0.16 | 0.33 | 0.52 | 0.74 |
| 60 | 0.25 | 0.46 | 0.72 | 0.89 |

**Interpretation:**
- With all 79 tasks, the single primary comparison reaches 80% power for an absolute difference of about **20 percentage points** (about 15 points at p_ref 0.7 with three variable seeds).
- The 95% precision is about **±12–14 points**.
- Effects of 5–10 points cannot be reliably detected. They can only be bounded by the interval.
- Mechanism subsets of 40–60 tasks are underpowered for anything below about 25 points. This is why they are exploratory.

## 5. Inference-seed policy (SD-08; not approved)

**Episode counts per confirmatory run:**
- one seed per task: D0 (79) + D18 (79) + Fill (≤ 79) + |Rel| + |Irr| ≤ **395**;
- three seeds: up to **1,185**;
- one primary seed plus a reproducibility subset (12 tasks × 2 extra seeds × D0/D18): ≤ 395 + 48 = **443**.

For example, with |Rel| = 40, |Irr| = 50 and Fill unblocked, the counts are 327, 981 and 375 respectively.

| Policy | Episodes (upper bound) | Precision / power | Assumption for the gain |
|---|---|---|---|
| One seed | 395 | ±0.13–0.14; 80% at Δ ≈ 0.20 | none |
| Three seeds | 1,185 (×3) | ±0.11–0.12; power +0.07–0.10 at default determinism, up to +0.39 if seeds are fully independent | seeds must produce **different** trajectories; at temperature 0 they may not |
| One primary seed + reproducibility subset | ≈ 443 | as one seed; the subset measures seed variability without changing the primary analysis | none for the primary; the subset is descriptive |

Seeds are never treated as extra tasks: the analysis averages within task.

**Recommendation:** one primary seed plus a prospectively selected reproducibility subset.
- If the diagnostic pilot shows low trajectory identity across seeds, adopting three seeds before confirmation is the documented alternative. In that case the decision must be taken before any confirmatory outcome.
- Per-episode compute time must come from the pilot. It is not estimated here.

## 6. Decisions required before approval

SD-08 (seeds), SD-10 (hierarchy and primary test), SD-11 (retry and missingness), SD-20 (secondary multiplicity) and SD-21 (primary interval). See the register.
