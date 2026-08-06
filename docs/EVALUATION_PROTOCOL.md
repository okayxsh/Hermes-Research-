# Controlled recovery evaluation protocol

Freeze `valid_unseen` task IDs, checkpoints, perturbations, recovery context, and the deterministic queue before evaluation. For each task × perturbation × snapshot × repetition, use the correct isolated read-only recovery profile, a fresh session, and a fresh ALFWorld reset/replay. Validate state equality before perturbation, preserve the same post-failure state across paired snapshots, and reconcile raw logs before analysis.
