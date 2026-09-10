# Checkpoint protocol

Checkpoint policies may select a valid prefix by length, index, fraction, or externally frozen prefix. A checkpoint must follow at least one valid action and precede completion. The final policy, source trajectory, and placement are deliberately not frozen in Phase 5.

This scientific recovery checkpoint is distinct from the crash-recovery state
used by acquisition/evaluation orchestration. The latter is documented in
`docs/EXPERIMENT_RECOVERY.md`; it restarts unfinished ALFWorld episodes and does
not claim to restore internal mid-episode state.
