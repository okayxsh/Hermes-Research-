# RQ1 Phase 2: statistical analysis plan (DRAFT, PROPOSED)

**Status: DRAFT, not approved.**
- The implementation is `src/rq1/five_arm/analysis.py`, `stats.py` and `power.py`.
- It is tested only on synthetic data.

## 1. Outcome, unit and hypotheses

- **Primary outcome:** environment-reported success within the approved total action budget.
- **Unit of analysis: task.**
  - Seeds are repeated inference measurements within a task.
  - Per task and contrast, the outcome is the mean over seeds where both arms have a VALID outcome.
- **Primary family (Holm across the three):**

  | Contrast | Population |
  |---|---|
  | D18 − D0 | all eligible tasks |
  | D18 − Fill | all eligible tasks |
  | Irr − Fill | frozen Irr subset |

- **Test:** a two-sided sign-flip permutation test on task-level paired differences.
  - It is exact when there are 20 or fewer non-zero differences, and otherwise a seeded Monte-Carlo with 100,000 draws.
  - With one binary outcome per task, this is the exact McNemar/sign test.
  - McNemar is reported only in that case, never on correlated seed-level outcomes.
- **Estimates:** the mean task-level difference, with a task-bootstrap 95% interval.
- **Secondary contrasts:**
  - Rel − Irr (the Rel ∩ Irr subset) and Rel − Fill (the Rel subset);
  - recovery actions with failures held at the budget, which is a restricted mean, not success-only;
  - invalid actions;
  - descriptive latency (wall clock and inference, failures included);
  - cumulative success by recovery action;
  - retrieval relevance: Precision@3, strict and lenient, plus injected relevance by arm;
  - acquisition-template-overlap exclusion sensitivity.
- **Rel vs D18 is not a relevance contrast,** because it compares 1 active skill against 3.

## 2. Dependence

- **Templates:** the 79 tasks share 27 goal templates. Template-aware sensitivity averages task differences within each template, then applies the sign-flip test and a template bootstrap over the templates.
- **Scenes:** there are only three. Results by scene and leave-one-scene-out are **descriptive only**; no scene-cluster asymptotics are used, and family and scene are confounded.
- **Generalisation:** the 79 tasks are not independent draws from ALFWorld. Claims are restricted to the screened population.

## 3. Missingness and retries

- **Infrastructure failures** (provider, environment, interruption) are retried up to 2 times with the same unit identity.
  - Retrying anything else, exceeding the limit, gaps in attempt numbers, or two valid outcomes for one unit all abort the analysis.
- **Model failures, budget exhaustion and invalid-action limits are outcomes.** They are never retried and never treated as missing.
- **Primary analysis:** pairwise complete cases within task.
  - Denominators and status counts are reported by arm.
  - Tasks with no complete pair are listed.
- **Sensitivity:** worst-case and best-case bounds, setting missing cells to the least or most favourable outcome.
- **Records for unplanned cells** (for example, Rel outside its frozen subset) abort the analysis.

## 4. Prospective power and precision (synthetic simulation; not evidence of any effect)

**Simulation model:**
- logit random effects for template (SD 0.8), task (SD 1.0) and treatment-effect heterogeneity (SD 0.5);
- the effect calibrated on the marginal probability scale;
- a 70% chance that all seeds share one draw (temperature-0 near-determinism);
- 2% of cells missing completely at random;
- 400 replications per scenario, seed 20261124.

The paired t-test approximates the planned sign-flip test. Bonferroni α/3 is a conservative lower bound for Holm. The +33.2 pp engineering difference (six repeated contexts) is **not** used as an expected effect. The output file has `content_sha256` `d6d2b8d1893b80534315adf35c3ae7c2835ae6da8434d3dc8193b1d2ff9bc248` and is not committed. Regenerate it with `python -m rq1.five_arm power --reps 400`.

**79 tasks:**

| p_ref | Δ | seeds | Power α=.05 | Power α/3 (Holm bound) | Median 95% CI half-width |
|---|---|---|---|---|---|
| 0.50 | 0.05 | 3 | 0.14 | 0.06 | 0.124 |
| 0.50 | 0.10 | 3 | 0.38 | 0.23 | 0.123 |
| 0.50 | 0.15 | 3 | 0.68 | 0.53 | 0.122 |
| 0.50 | 0.20 | 3 | 0.88 | 0.78 | 0.121 |
| 0.50 | 0.30 | 3 | 0.99 | 0.99 | 0.115 |
| 0.75 | 0.10 | 3 | 0.38 | 0.20 | 0.111 |
| 0.75 | 0.15 | 3 | 0.79 | 0.63 | 0.107 |
| 0.75 | 0.20 | 3 | 0.97 | 0.95 | 0.102 |
| 0.50 | 0.20 | 1 | 0.84 | 0.66 | 0.138 |

**Smaller populations** (mechanism subsets may be this size):

| n | p_ref | Δ | seeds | Power α/3 |
|---|---|---|---|---|
| 40 | 0.50 | 0.20 | 3 | 0.50 |
| 40 | 0.50 | 0.30 | 3 | 0.87 |
| 40 | 0.75 | 0.20 | 3 | 0.67 |
| 20 | 0.50 | 0.20 | 3 | 0.20 |

**Sensitivity** (79 tasks, three seeds, p_ref 0.5, Δ 0.10, at the α/3 Holm bound):

| Condition | Power |
|---|---|
| Independent seeds, no missing cells | 0.48 |
| Fully deterministic seeds | 0.17 |
| 10% missing cells (independent seeds) | 0.37 |
| Template SD 0 | 0.19 |
| Template SD 1.5 | 0.25 |

**Interpretation:**
- With all 79 tasks, the design reliably detects (≥ 80% at the Holm bound) only absolute success differences of about **0.20 (p_ref 0.75) to 0.30 (p_ref 0.5)**.
- The interval half-width is about ±0.10 to ±0.12.
- Extra seeds help only if outputs actually vary across seeds at temperature 0.
- Mechanism subsets of 20–40 tasks cannot support confirmatory claims for effects below about 0.30.

**Recommendations:**
- Frame D18 − D0 as the one adequately powered confirmatory contrast, or present all contrasts as estimation with intervals.
- Treat the Irr − Fill and Rel − Irr mechanism contrasts as exploratory unless their pre-frozen subsets are large.
- Measure seed variability in the pilot before keeping three seeds.

## 5. Decisions required before approval

1. Primary family composition (keep three contrasts, or one primary plus secondaries).
2. α and multiplicity.
3. Retry limit.
4. Missing-data bounds as sensitivity only.
5. Seeds.
6. Whether the confirmatory claim is hypothesis-testing or estimation-only.

See `PHASE2_FIVE_ARM_PROTOCOL_DRAFT.md` §11.
