# Implementation order

1. Foundation (this pass).
2. ALFWorld bridge with a real adapter behind the existing interface and contract tests.
3. Hermes plugin and capability adapter.
4. Isolated profile lifecycle and contamination controls after CLI capabilities are observed.
5. Recovery controller: checkpointing, replay, controlled perturbation, solvability validation, and recovery logging.
6. Recovery-aware pilot framework and manual evidence capture (implemented in code; fake verified, real evidence pending).
7. Real university pilot, environment/model/recovery-protocol approval, and freeze.
8. Acquisition, snapshots, paired controlled-recovery evaluation, and analysis only after the recovery protocol is frozen.
