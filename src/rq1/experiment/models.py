from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonical_hash(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ExperimentUnit:
    """One scientifically meaningful, independently journalled execution unit."""

    phase: str
    task_id: str
    task_index: int
    condition: str
    seed: int | None
    identity: Mapping[str, Any]
    payload: Mapping[str, Any] = field(default_factory=dict)
    library_name: str | None = None
    library_size: int | None = None
    library_hash: str | None = None

    @property
    def run_key(self) -> str:
        return canonical_hash(dict(self.identity))

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["identity"] = dict(self.identity)
        value["payload"] = dict(self.payload)
        value["run_key"] = self.run_key
        return value


@dataclass(frozen=True)
class RunOutcome:
    """A normally terminated run, including scientifically valid failure."""

    success: bool
    measurements: Mapping[str, Any] = field(default_factory=dict)
    recovery_success: bool | None = None
    retrieved_skill_ids: tuple[str, ...] = ()
    retrieval_similarity_scores: tuple[float, ...] = ()
    actions: int | None = None
    steps: int | None = None
    invalid_actions: int | None = None
    latency_ms: float | None = None
    runtime_seconds: float | None = None
    log_paths: tuple[str, ...] = ()
    skill_library_hash_after: str | None = None
    library_size_after: int | None = None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["measurements"] = dict(self.measurements)
        return value


@dataclass(frozen=True)
class RunExecutionContext:
    experiment_id: str
    attempt_id: str
    attempt_index: int
    output_dir: Path
    stop_requested: Callable[[], bool]


class RunExecutor(Protocol):
    def __call__(self, unit: ExperimentUnit, context: RunExecutionContext) -> RunOutcome: ...


class RunFailure(RuntimeError):
    """A failed attempt whose mutation/cleanup state is explicitly known."""

    def __init__(
        self,
        message: str,
        *,
        safe_to_continue: bool = False,
        mutation_state_known: bool = False,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.safe_to_continue = safe_to_continue
        self.mutation_state_known = mutation_state_known
        self.details = dict(details or {})
