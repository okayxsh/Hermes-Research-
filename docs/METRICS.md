# Metrics

Primary final metrics are calculated only by `rq1 analysis` from a validated, activated real evaluation report. Conditional recovery rate is successful valid, solvable, plan-invalidated post-failure episodes divided by eligible budget-complete episodes.

Retrieval quality is measured against human relevance labels, never cosine similarity: `Precision@3` is the number of adjudicated-relevant skills among the retrieved top-3 divided by 3, and `Retrieval Noise = 1 - Precision@3`. No-retrieval episodes are reported separately and never coerced to a noise of 0 or 1. Cohen's kappa is computed from the original independent ratings and reported with its sample size (no pass/fail threshold).

The legacy Hermes-native `skill_loaded`/`skill_view` noise definition and relevant-skill hit rate are deprecated and retained only for compatibility. Actions, invalid actions, tool/model calls, latency, and first-useful/relevant-event time retain null when unavailable.
