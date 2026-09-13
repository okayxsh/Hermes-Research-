from __future__ import annotations
from dataclasses import asdict, dataclass

@dataclass(frozen=True)
class AcquisitionPlan:
    run_id: str
    task_ids: tuple[str, ...]
    split: str = "train"
    profile: str = "rq1-acquisition"
    task_families: tuple[str, ...] = ()
    queue_sha256: str | None = None
    # Continuation of a completed parent acquisition: unit N is logical position
    # ``logical_index_offset + N`` of the combined acquisition.
    parent_run_id: str | None = None
    logical_index_offset: int = 0

    def to_dict(self): return asdict(self)

@dataclass(frozen=True)
class AcquisitionAttempt:
    run_id: str; task_id: str; attempt_id: str; status: str
    episode_log: str | None; session_id: str | None; exclusion_reason: str | None = None
    def to_dict(self): return asdict(self)

@dataclass(frozen=True)
class SkillOperation:
    operation_index: int; operation: str; skill_id: str; content_sha256: str
    source_task_id: str; source_attempt_id: str; source_episode_log: str
    def to_dict(self): return asdict(self)
