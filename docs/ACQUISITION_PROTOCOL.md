# Acquisition protocol

Use only frozen `train` tasks. Start a fresh Hermes session for each task. A successful episode may create or patch at most one general skill; failed episodes may not create a positive skill. Validate source metadata, duplicates, and task-specific leakage before snapshots.
