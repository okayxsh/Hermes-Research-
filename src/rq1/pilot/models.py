"""Typed contracts for the Phase 6 pilot runner."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class PilotMode(str, Enum):
    FAKE = "fake"
    REAL = "real"


class PilotStatus(str, Enum):
    NOT_STARTED = "not_started"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    BLOCKED = "blocked"
    SKIPPED = "skipped"
    INTERRUPTED = "interrupted"
    INVALIDATED = "invalidated"


class EvidenceLevel(str, Enum):
    STATIC = "static"
    MOCK = "mock"
    INSTALLED = "installed"
    REAL_COMPONENT = "real_component"
    REAL_INTEGRATED = "real_integrated"
    EXPERIMENTAL_READY = "experimental_ready"


EVIDENCE_RANK = {level: index for index, level in enumerate(EvidenceLevel)}


class BlockingClass(str, Enum):
    BLOCKING = "blocking"
    ADVISORY = "advisory"


class RetryPolicy(str, Enum):
    NEVER_IN_ATTEMPT = "never_in_attempt"
    READ_ONLY_ONCE = "read_only_once"
    NEW_ATTEMPT_ONLY = "new_attempt_only"


class CleanupPolicy(str, Enum):
    NONE = "none"
    AUTOMATIC_FAKE = "automatic_fake"
    EXPLICIT_DESTRUCTIVE = "explicit_destructive"


@dataclass(frozen=True)
class CapabilityRequirement:
    name: str
    minimum_evidence: EvidenceLevel


@dataclass(frozen=True)
class PilotTestSpec:
    test_id: str
    name: str
    purpose: str
    group: str
    prerequisites: tuple[str, ...]
    supported_modes: tuple[PilotMode, ...]
    required_capabilities: tuple[CapabilityRequirement, ...]
    timeout_seconds: int
    retry_policy: RetryPolicy
    cleanup_policy: CleanupPolicy
    blocking: BlockingClass
    fake_evidence: EvidenceLevel
    real_evidence: EvidenceLevel

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvidenceReference:
    path: str
    sha256: str
    level: EvidenceLevel
    validator: str
    created_at: str
    simulated: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PilotAttemptResult:
    schema_version: int
    pilot_run_id: str
    pilot_test_id: str
    attempt_id: str
    mode: PilotMode
    status: PilotStatus
    evidence_level: EvidenceLevel
    started_at: str
    completed_at: str
    input_fingerprint: str
    evidence: list[EvidenceReference] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    remediation: str | None = None
    next_allowed_test: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RuntimeSample:
    metric: str
    value: float
    unit: str
    source: str
    simulated: bool


@dataclass(frozen=True)
class ProtocolRecommendation:
    checkpoint_source: str
    checkpoint_policy: str
    perturbation_type: str
    solvability_method: str
    recovery_context: str
    action_limit: int
    timeout_seconds: int
    exclusion_policy: str
    approval_state: str = "unapproved"


@dataclass(frozen=True)
class GoNoGoDecision:
    decision: str
    experimental_ready: bool
    reasons: tuple[str, ...]
    generated_from_mode: PilotMode

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RuntimeExecution:
    status: PilotStatus
    evidence_level: EvidenceLevel
    details: dict[str, Any]
    error: str | None = None
    remediation: str | None = None
