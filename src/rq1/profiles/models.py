"""Typed, capability-gated contracts for isolated Hermes experiment profiles."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class ProfileLifecycleState(str, Enum):
    PLANNED = "planned"
    CREATED = "created"
    VALIDATED = "validated"
    CONTAMINATED = "contaminated"
    ARCHIVED = "archived"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class ProfilePlan:
    name: str
    purpose: str
    repository_path: str
    allow_skill_writes: bool
    allowed_skill_prefixes: tuple[str, ...] = ()
    snapshot_id: str | None = None
    snapshot_hash: str | None = None
    read_only_snapshot: bool = False
    instantiate: bool = True
    plugin_name: str = "alfworld-experiment"
    plugin_opt_in: bool = True
    memory_required_disabled: bool = True
    curator_required_disabled: bool = True
    unrelated_tools_required_disabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProfileInspection:
    name: str
    profile_path: str | None
    exists: bool
    configuration: dict[str, Any] = field(default_factory=dict)
    skills: tuple[str, ...] = ()
    sessions: tuple[str, ...] = ()
    memory_entries: tuple[str, ...] = ()
    plugins: tuple[str, ...] = ()
    enabled_toolsets: tuple[str, ...] = ()
    curator_enabled: bool | None = None
    profile_database_path: str | None = None
    source: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ContaminationResult:
    clean: bool
    findings: tuple[str, ...]
    configuration_hash: str | None
    skill_directory_hash: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProfileManifest:
    schema_version: int
    profile_name: str
    purpose: str
    lifecycle_state: str
    creation_mechanism: str
    repository_path: str
    profile_path: str | None
    hermes_version: str | None
    enabled_capabilities: tuple[str, ...]
    disabled_capabilities: tuple[str, ...]
    plugin_configuration: dict[str, Any]
    skill_count: int
    bundled_skill_count: int
    unexpected_skill_count: int
    memory_status: str
    curator_status: str
    session_count_at_creation: int
    configuration_hash: str | None
    skill_directory_hash: str | None
    created_at: str
    git_commit: str | None
    machine_manifest_id: str | None
    validation_result: dict[str, Any]
    contamination_result: dict[str, Any]
    snapshot_id: str | None = None
    snapshot_hash: str | None = None
    read_only_snapshot: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
