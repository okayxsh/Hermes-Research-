from __future__ import annotations
from rq1.evaluation.models import EvaluationQueueItem
def build_paired_queue(*, run_id: str, tasks: list[dict], snapshots: list[dict], checkpoint: dict, perturbation: dict, context_digest: str, repetitions: int, seeds: list[int]) -> list[EvaluationQueueItem]:
    if any(task.get("split") != "valid_unseen" for task in tasks): raise ValueError("final evaluation accepts valid_unseen only")
    if repetitions < 1 or len(seeds) != repetitions: raise ValueError("frozen repetition/seeds mismatch")
    items=[]
    for task in sorted(tasks, key=lambda x: x["task_id"]):
      for repetition, seed in enumerate(seeds, 1):
       for snapshot in sorted(snapshots, key=lambda x: x["snapshot_id"]):
        items.append(EvaluationQueueItem(run_id, task["task_id"], task["task_family"], checkpoint["checkpoint_id"], checkpoint["observable_state_digest"], perturbation["perturbation_id"], perturbation["observable_post_state_digest"], context_digest, snapshot["snapshot_id"], snapshot["directory_sha256"], repetition, seed, "rq1-recovery-"+snapshot["snapshot_id"]))
    return items
