# RQ1 Phase 2B: statistical analysis plan (DRAFT, PROPOSED)

**Status: DRAFT, not approved.**
- The implementation is in `src/rq1/five_arm/analysis.py`, `stats.py`, `power.py` and `pilot.py`.
- It is tested only on synthetic data.
- The governing decisions are SD-08, SD-10, SD-11, SD-20 and SD-21 in `PHASE2_DECISION_REGISTER_DRAFT.md`. All await approval.

## 0. Phase 3C additions (DRAFT; for author approval)

### 0.1 Primary test: task-level McNemar versus a template-level paired test (revised in Phase 3D)

**Unchanged.**
- The single primary comparison stays D18 − D0 over all 79 tasks, at α = 0.05, two-sided.
- The **estimand and point estimate stay task-weighted**: the mean of the 79 within-task differences.
- The primary interval stays the template-cluster bootstrap 95% CI.

Only the choice of the primary *test* is open.

**Option A, status quo: task-level sign flip (= exact McNemar).**
- With one inference seed, each task gives one binary pair, and the task-level sign-flip p-value equals the exact McNemar p-value.
- *Exchangeability needed:* within every task the D18 and D0 labels are exchangeable (sharp null: no effect in any task), and the 79 pairs are independent given task-level factors.
- Template or scene effects acting equally on both arms cancel inside each pair.
- It is **not** valid for the weak null ("average effect = 0") when effects vary and that variation is shared within goal templates. Discordances from the same template then move together, and the test treats them as independent.

**Option B: template-level paired test with a task-weighted statistic (implemented in Phase 3D).**
- The D18/D0 labels of **all tasks of a goal template are flipped jointly**.
- The statistic is |Σ over templates of the template *sum* of task differences|. This equals |Σ task differences|, i.e. 79 × the task-weighted estimate, so test and estimate address the same quantity. The test is not the template-mean version.
- *Exchangeability needed:*
  1. **Template-level exchangeability (null):** within every template, the joint distribution of its tasks' paired outcomes is unchanged when the D18 and D0 labels are swapped for all of its tasks together. No independence among tasks within a template is needed, so template-correlated effects and shared within-template state are allowed.
  2. **Independence across templates:** the 27 template sums are independent given their absolute values.
- *Not covered, and to be stated in the paper:*
  - Dependence across templates through shared scenes (3) and families (5). There are too few to resample, so these stay descriptive.
  - Asymmetric distributions of template-level effects under a weak null. Like option A, option B is exact only for its sharp null. It is approximately valid, not exact, for the weak null.
- *Mechanics:*
  - 27 templates (25 × 3 tasks + 2 × 2 tasks).
  - More than 20 non-zero template sums use a seeded Monte Carlo with 100,000 draws (Monte Carlo SE ≈ 0.0007 at p = 0.05); otherwise the test is exact.
  - The smallest attainable p is 2 / 2^k for k non-zero templates, so k ≥ 6 is needed to reach p < 0.05.
- *Weighting:* the two 2-task templates get weight 2 rather than 3. Under this near-balanced allocation the task-sum and template-mean statistics almost coincide.

**Planning simulation** (`docs/evidence/phase3d_template_test_power.py`; synthetic, 2,000 replicates per row, exact sign-flip p-values on the real 27-template allocation)
- *Model:* p(D0) = 0.25; task random effect SD 2 and template random effect SD 1 on the logit scale, both acting on both arms. The "het" rows add a template-shared effect.
- *Monte Carlo SE:* ≈ 0.005 for the rejection rates near 0.05 and ≈ 0.009 for the power values near 0.8.

| Scenario | Task-weighted effect | McNemar (A) | Template mean | **Template sum (B)** |
|---|---|---|---|---|
| Sharp null | 0.00 | 0.028 | 0.029 | 0.027 |
| Weak null, template effects ± (h = 0.5) | 0.00 | 0.042 | 0.036 | 0.032 |
| Weak null, template effects ± (h = 1.0) | 0.00 | **0.070** | 0.038 | 0.035 |
| Weak null, template effects ± (h = 1.5) | 0.00 | **0.104** | 0.041 | 0.035 |
| Effect ≈ 11 points, homogeneous | 0.11 | 0.388 | 0.379 | 0.364 |
| Effect ≈ 11 points, het SD 1 | 0.11 | 0.387 | 0.344 | 0.330 |
| Effect ≈ 20 points, homogeneous | 0.20 | 0.851 | 0.845 | 0.833 |
| Effect ≈ 20 points, het SD 1 | 0.19 | 0.811 | 0.778 | 0.762 |

**Reading:**
- When template-correlated effects average to zero, McNemar's type-I error rises to 0.07–0.10. The template-level tests stay at or below 0.041.
- Option B costs about 1.5 to 5 points of power.
- Part of McNemar's extra power in the heterogeneous rows comes from the same anti-conservatism.

**Recommendation for author decision (decision sheet item 9): adopt option B as the primary test.**
- It is valid under the dependence that the primary interval already assumes (template-cluster CI). This makes the test and the interval coherent and removes the need for the A/B disagreement rule.
- It keeps the task-weighted estimand and estimate.
- The power loss is small relative to the type-I protection.
- Exact McNemar is still computed and reported as a secondary, task-level analysis, and the template-mean sign flip stays as a sensitivity analysis.

**Implementation.**
- The test is selected by `design["primary_test"]` in `analysis.analyze`: `"task_sign_flip"` (default, status quo) or `"template_sign_flip_task_weighted"`.
- Both p-values are always reported: `test` / `mcnemar_exact` (A) and `template_level_task_weighted` (B).
- The code default is unchanged until the author records the decision. The chosen value must be fixed in the approved analysis design **before** any confirmatory outcome is seen.
- If option A is kept, the existing disagreement rule applies: report McNemar as the prespecified test and state that the conclusion is sensitive to template dependence.

### 0.2 Factorial estimates (secondary, estimation only)

- D18U − D0U: continuation benefit without perturbation.
- D0 − D0U: perturbation effect without memory. This is the manipulation check on the 79 confirmatory tasks.
- D18 − D18U: perturbation effect with the same memory block.
- Interaction (D18 − D0) − (D18U − D0U).

Each is a task-weighted estimate with template-cluster and task bootstrap 95% intervals; **no p-values**. The interaction's half-width is expected to be about ±15–20 points. **Interpretation rule:** no disruption-specific claim unless the interaction's template-cluster interval excludes 0. D18U shows the frozen post-detour D18 block (protocol §0).

### 0.3 Interface reporting (descriptive; parser unchanged)

Per condition:
- episodes;
- model decisions (one per recovery-trace entry);
- selection attempts (decisions + rejections);
- rejected selections, as counts and as rates per episode, per decision and per attempt;
- rejection categories;
- selection-limit terminations and their rate.

Selection-limit terminations remain outcomes (failures), never missing data.

### 0.4 Cap and hardware transition

The cap is **50**. It was chosen by the frozen budget rule from cap-100 diagnostic trajectories truncated to cap 50, which uses baseline feasibility only.
- **What changed:** the confirmatory and pilot runs execute on **GPU** provider lanes, whereas the diagnostic ran sequentially on CPU inference.
- **What it must not be read as:** a difference between cap-50 GPU trajectories and truncated cap-100 CPU trajectories is **not** evidence of a cap effect. Hardware, numeric kernels and provider builds differ, and temperature-0 decoding is not guaranteed bit-identical across them.
- **Prospective comparability checks** (no extra scientific episodes; see the parallel-execution design):
  1. The seven-condition pilot's D0 cell for each pilot task is compared descriptively with the diagnostic's DETOURED seed-11 trajectory truncated to cap 50. Agreement is reported as a CPU-to-GPU comparability statistic, not as a cap test.
  2. Lane provenance (GPU UUID, full offload, provider version, model digest) must be identical within a run. Any CPU fallback halts execution.
  3. The 48-episode reproducibility subset documents within-hardware seed determinism under memory.

### 0.5 Revised episode counts

- **Confirmatory:** ≤ 601 episodes (316 for D0/D18/D0U/D18U, ≤ 237 for Fill/Rel/Irr, 48 reproducibility).
- **Pilot:** ≤ 84.
- **Time:** at the diagnostic's measured 8.2 minutes per episode on CPU, sequential execution would take ≤ 82 h. With four GPU lanes the wall-clock time is expected to fall roughly fourfold. This must be measured in the pilot.

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
