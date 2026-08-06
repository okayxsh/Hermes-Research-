# Implementation order

1. Foundation (this pass).
2. ALFWorld bridge with a real adapter behind the existing interface and contract tests.
3. Hermes plugin and capability adapter.
4. Isolated profile lifecycle and contamination controls after CLI capabilities are observed.
5. Recovery controller: checkpointing, replay, controlled perturbation, solvability validation, and recovery logging.
6. Recovery-aware pilot framework and manual evidence capture.
7. Acquisition, snapshots, paired controlled-recovery evaluation, and analysis only after the recovery protocol is frozen.
