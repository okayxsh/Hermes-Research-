from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


SETUP_STATUSES = {"pending", "running", "passed", "failed", "blocked", "skipped"}


@dataclass(frozen=True)
class SetupOptions:
    dry_run: bool = False
    yes: bool = False
    resume: bool = False
    skip_system_packages: bool = False
    skip_model: bool = False
    skip_alfworld_data: bool = False
    install_fallback_model: bool = False
    force_stage: str | None = None
    verbose: bool = False


@dataclass(frozen=True)
class CommandResult:
    command: tuple[str, ...]
    returncode: int
    stdout: str = ""
    stderr: str = ""

    @property
    def ok(self) -> bool:
        return self.returncode == 0


@dataclass(frozen=True)
class ProbeResult:
    name: str
    available: bool
    details: str
    version: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SetupStageResult:
    stage: str
    status: str
    started_at: str
    completed_at: str
    run_id: str
    attempt_id: str
    dry_run: bool
    input_fingerprint: str
    commands: list[list[str]] = field(default_factory=list)
    probes: list[dict[str, Any]] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    remediation: str | None = None
    skip_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in SETUP_STATUSES:
            raise ValueError(f"Invalid setup status: {self.status}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SetupStage:
    name: str
    prerequisites: tuple[str, ...] = ()


SETUP_STAGES: tuple[SetupStage, ...] = (
    SetupStage("preflight"),
    SetupStage("system-packages", ("preflight",)),
    SetupStage("python-environment", ("system-packages",)),
    SetupStage("ollama", ("python-environment",)),
    SetupStage("hermes", ("python-environment",)),
    SetupStage("alfworld-package", ("python-environment",)),
    SetupStage("alfworld-data", ("alfworld-package",)),
    SetupStage("candidate-models", ("ollama",)),
    SetupStage("base-profiles", ("hermes", "ollama")),
    SetupStage(
        "installation-verification",
        ("alfworld-data", "candidate-models", "base-profiles"),
    ),
)
SETUP_STAGE_MAP = {stage.name: stage for stage in SETUP_STAGES}


@dataclass
class SetupState:
    status: str = "pending"
    attempt_id: str | None = None
    report: str | None = None
    completed_at: str | None = None
    input_fingerprint: str | None = None

    def __post_init__(self) -> None:
        if self.status not in SETUP_STATUSES:
            raise ValueError(f"Invalid setup status: {self.status}")


def path_for_report(root: Path, stage: str, attempt_id: str) -> Path:
    return root / "artifacts" / "stage_reports" / f"setup-{stage}-{attempt_id}.json"
