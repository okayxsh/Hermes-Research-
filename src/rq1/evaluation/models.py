from __future__ import annotations
from dataclasses import asdict, dataclass
@dataclass(frozen=True)
class EvaluationQueueItem:
    run_id: str; task_id: str; task_family: str; checkpoint_id: str; checkpoint_digest: str
    perturbation_id: str; perturbation_digest: str; recovery_context_digest: str
    snapshot_id: str; snapshot_hash: str; repetition: int; seed: int; profile: str
    expected_phase: str = "post_failure"; claim_status: str = "planned"
    def to_dict(self): return asdict(self)
