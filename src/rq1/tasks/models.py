from __future__ import annotations
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

class ManifestState(str, Enum):
    PLACEHOLDER = "placeholder"; DISCOVERED = "discovered"; PROPOSED = "proposed"; APPROVED = "approved"; FROZEN = "frozen"; INVALIDATED = "invalidated"

@dataclass(frozen=True)
class TaskRecord:
    task_id: str; split: str; family: str; source_identifier: str; source_sha256: str; game_sha256: str | None; order_index: int = 0
    def to_dict(self) -> dict[str, Any]: return asdict(self)

@dataclass(frozen=True)
class DiscoveryResult:
    schema_version: int; data_root_identity: str; split: str; records: tuple[TaskRecord, ...]; exclusions: tuple[dict[str, str], ...]; parse_errors: tuple[str, ...]
    def to_dict(self) -> dict[str, Any]: return {**asdict(self), "records": [item.to_dict() for item in self.records]}

@dataclass(frozen=True)
class SelectionPolicy:
    version: str; seed: int; requested_count: int | None; balancing: str = "round_robin_families"
    def to_dict(self) -> dict[str, Any]: return asdict(self)

@dataclass(frozen=True)
class TaskManifest:
    schema_version: int; manifest_type: str; status: str; split: str; alfworld_version: str | None; data_root_identity: str
    repository_commit: str | None; selection_policy: dict[str, Any]; requested_count: int | None; actual_count: int
    family_counts: dict[str, int]; tasks: tuple[TaskRecord, ...]; exclusions: tuple[dict[str, str], ...]
    duplicate_resolution: tuple[dict[str, str], ...]; generated_at: str; approved_at: str | None; approval_reference: str | None; manifest_sha256: str
    def to_dict(self) -> dict[str, Any]:
        value = asdict(self); value["tasks"] = [item.to_dict() for item in self.tasks]; return value
