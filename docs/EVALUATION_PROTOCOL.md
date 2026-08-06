# Controlled recovery evaluation protocol

Freeze `valid_unseen` task IDs, checkpoints, perturbations, recovery context, and the deterministic queue before evaluation. For each task × perturbation × snapshot × repetition, use the correct isolated read-only recovery profile, a fresh session, and a fresh ALFWorld reset/replay. Validate state equality before perturbation, preserve the same post-failure state across paired snapshots, and reconcile raw logs before analysis.

`valid_unseen` metadata discovery is blocked until the approved real pilot and environment/protocol freezes exist. The repository never reads unseen instructions, outcomes, or trajectories for pilot calibration, relevance-rule design, or perturbation design.

Final execution additionally requires an immutable evaluation activation manifest, `--yes`, and `RQ1_RUN_FINAL_EVALUATION=1`. The default configuration remains disabled; a changed YAML boolean cannot authorize an unseen task. Any evidence or capability drift invalidates activation and requires fresh manual approval.
