# Evaluation protocol

Freeze `valid_unseen` task IDs and the deterministic queue before evaluation. For each task × snapshot × repetition, use the correct isolated read-only profile, a fresh session, and a fresh ALFWorld reset. Reconcile raw logs before analysis.
