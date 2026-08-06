from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class RunMode(str, Enum): BOOTSTRAP="bootstrap"; FINAL="final"
class TopStatus(str, Enum): RUNNING="RUNNING"; PILOT_GO="PILOT_GO"; SUCCESS="SUCCESS"; BLOCKED="BLOCKED"; FAILED="FAILED"; STOPPED="STOPPED_BY_USER"
class StageStatus(str, Enum): PENDING="pending"; RUNNING="running"; PASSED="passed"; BLOCKED="blocked"; FAILED="failed"; INTERRUPTED="interrupted"; INVALIDATED="invalidated"
class MutationClass(str, Enum): READ_ONLY="read_only"; CERTAIN_MUTATION="certain_mutation"; UNCERTAIN_MUTATION="uncertain_mutation"
class ErrorClass(str, Enum): TRANSIENT="transient_operational_failure"; UNCERTAIN="uncertain_mutating_operation_failure"; COMPATIBILITY="compatibility_failure"; SCIENTIFIC="scientific_protocol_failure"; MEASUREMENT="current_rq_measurement_failure"; CONTAMINATION="data_contamination"; RESOURCE="resource_exhaustion"; USER_STOP="user_stop"; DEFECT="code_defect"; ARCHIVE="archive_integrity_failure"

def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

@dataclass(frozen=True)
class RunPlan:
    schema_version: int; run_plan_id: str; mode: str; repository_commit: str | None
    primary_model: str = "hermes3:8b"; fallback_model: str = "llama3.1:8b"; model_digest: str | None = None
    hermes_version: str | None = None; ollama_version: str | None = None; alfworld_version: str | None = None; alfworld_data_digest: str | None = None; python_version: str | None = None
    task_splits: dict[str, str] = field(default_factory=lambda: {"acquisition":"train", "pilot":"valid_seen", "evaluation":"valid_unseen"})
    acquisition_task_count: int | None = None; evaluation_task_count: int | None = None; task_family_balancing_policy: str | None = None
    checkpoint_policy: str | None = None; perturbation_policy: str | None = None; solvability_policy: str | None = None; snapshot_policy: str | None = None
    repetition_count: int | None = None; random_seeds: tuple[int, ...] = (); action_limit: int | None = None; timeout_values: dict[str, int] = field(default_factory=dict)
    relevance_rule_hash: str | None = None; exclusion_rule_hash: str | None = None; profile_templates: tuple[str, ...] = ("rq1-acquisition", "rq1-recovery-<snapshot-id>")
    worker_policy: dict[str, Any] = field(default_factory=lambda: {"acquisition_workers":1,"evaluation_workers":1,"two_worker_threshold":.30})
    output_directory: str = "results/final"; health_thresholds: dict[str, Any] = field(default_factory=dict); archive_policy: str = "local-redacted-zip-v1"; approval_references: tuple[str, ...] = (); content_hash: str = ""
    def to_dict(self) -> dict[str, Any]: return asdict(self)

@dataclass(frozen=True)
class StageDefinition:
    name: str; prerequisites: tuple[str, ...]; mutation_class: MutationClass; idempotent: bool; safe_resume: str; owned_artifacts: tuple[str, ...] = (); invalidates: tuple[str, ...] = ()

@dataclass(frozen=True)
class ContingencyRecord:
    schema_version: int; run_id: str; stage: str; attempt_id: str | None; timestamp: str; error_code: str; error_class: str; message: str; safe_to_retry: bool; retry_policy: str; last_successful_stage: str | None; affected_artifacts: tuple[str, ...]; invalidated_stages: tuple[str, ...]; log_paths: tuple[str, ...]; evidence_paths: tuple[str, ...]; remediation: str; approval_required: bool; resumable: bool; scientific_impact: str
    def to_dict(self) -> dict[str, Any]: return asdict(self)

@dataclass(frozen=True)
class RuntimeForecast:
    schema_version: int; pilot_report: str; evaluation_cells: int | None; optimistic_hours: float | None; median_hours: float | None; conservative_hours: float | None; serial_hours: float | None; approved_parallel_hours: float | None; expected_gpu_hours: float | None; expected_disk_gib: float | None; expected_log_gib: float | None; timeout_upper_hours: float | None; evidence_available: bool
    def to_dict(self) -> dict[str, Any]: return asdict(self)
