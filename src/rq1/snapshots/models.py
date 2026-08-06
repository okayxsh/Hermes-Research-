from __future__ import annotations
from dataclasses import asdict, dataclass
@dataclass(frozen=True)
class SnapshotManifest:
    schema_version: int; snapshot_id: str; source_acquisition_run: str; operation_cutoff: int
    skills: tuple[dict, ...]; skill_count: int; directory_sha256: str; parent_snapshot_sha256: str | None
    created_from_git_commit: str; read_only: bool = True
    def to_dict(self): return asdict(self)
