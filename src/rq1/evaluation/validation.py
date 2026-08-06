from __future__ import annotations
from rq1.evaluation.models import EvaluationQueueItem
def validate_queue(items: list[EvaluationQueueItem]) -> list[str]:
    errors=[]; paired={}; ids=set()
    for item in items:
        if not item.task_id.startswith("valid_unseen:") and "valid_unseen" not in item.task_id: errors.append("non-valid_unseen item")
        identity=(item.task_id,item.checkpoint_digest,item.perturbation_digest,item.repetition,item.seed)
        value=(item.recovery_context_digest, item.snapshot_id)
        paired.setdefault(identity, []).append(value)
        full=(identity,item.snapshot_id)
        if full in ids: errors.append("duplicate queue item")
        ids.add(full)
    for values in paired.values():
        if len({context for context, _ in values}) != 1: errors.append("paired recovery context mismatch")
    return errors
