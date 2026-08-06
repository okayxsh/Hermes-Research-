from __future__ import annotations
from pathlib import Path
from rq1.evaluation.activation import log_task_access, require_runtime_opt_in
def run_final_evaluation(root: Path, activation_manifest: Path) -> None:
    """Validate deliberate activation before any unseen task can be loaded."""
    require_runtime_opt_in(root, activation_manifest)
    # No task list is opened before the activation verification above.
    raise RuntimeError("real final evaluation remains blocked until observed recovery-profile, perturbation, solvability, and Hermes dispatch adapters are available")
