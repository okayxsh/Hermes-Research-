# Acquisition protocol

Use only frozen `train` tasks. Start a fresh Hermes session for each task. A successful episode may create or patch at most one general skill; failed episodes may not create a positive skill. Validate source metadata, duplicates, and task-specific leakage before snapshots.

The frozen queue is generated only from deterministic installed-data discovery. It is rejected if it overlaps a pilot/evaluation manifest or includes a task-specific source/game identity already assigned elsewhere.

Every terminal task attempt is journalled and checkpointed immediately. Resume
uses the canonical task/order/profile identity and preserves one-worker
chronology. Unknown skill-write outcomes block further acquisition rather than
advancing the library from uncertain state. See `EXPERIMENT_RECOVERY.md`.
