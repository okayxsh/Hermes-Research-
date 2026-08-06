from __future__ import annotations
from pathlib import Path
from rq1.freeze.validation import validate_final_gates
def run_final_evaluation(root: Path) -> None:
    gates=validate_final_gates(root)
    if not gates.valid: raise RuntimeError("final gate blocked: " + "; ".join(gates.reasons))
    raise RuntimeError("real final evaluation is blocked until observed recovery-profile and perturbation adapters are available")
