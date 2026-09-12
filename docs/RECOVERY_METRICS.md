# Recovery metrics boundary

Later evaluation will calculate conditional recovery rate, post-failure retrieval quality (Precision@3 and Retrieval Noise = 1 − Precision@3 against human labels), Cohen's kappa, actions, invalid actions, latency, and degradation onset from saved recovery/retrieval logs. Phase 5 records the pre-failure/post-failure boundary but does not run evaluation or calculate scientific results.

Phase 6 calculates only pilot instrumentation and runtime projections. Metrics from fake or mini pilot conditions are labelled simulated/pilot-only and must not enter final RQ analysis. Final analysis is paired by task, checkpoint digest, perturbation digest, recovery-context digest, repetition, and seed; it uses deterministic cluster bootstrap intervals over paired units and reports associations as non-causal.
