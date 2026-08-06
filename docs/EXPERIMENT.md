# Experiment boundary

The canonical RQ concerns recovery after a controlled plan-invalidating failure, not ordinary task success. Acquire skills only on `train`; use `valid_seen` for pilots and recovery-protocol calibration; reserve untouched `valid_unseen` for final paired controlled-recovery evaluation. Each paired condition must replay the same checkpoint, apply the same solvable perturbation, receive the same recovery context, and differ only by frozen read-only skill snapshot/profile.

Phase 5 implements this pairing only against a deterministic fake environment. Real ALFWorld checkpoint replay, state equality, target relocation, and solvability are capability-gated and remain `TO_BE_VERIFIED_BY_RECOVERY_PILOT`.

Phase 6 implements the complete 37-test pilot runner. Fake mode exercises orchestration, disposable mini acquisition/snapshots, paired recovery, failures, resume, and reporting without contributing scientific evidence. Phase 7 performs the real university pilot and manually approves the environment and recovery-protocol freeze.
