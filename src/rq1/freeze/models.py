from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class FreezeManifest:
    schema_version: int
    kind: str
    created_at: str
    repository_commit: str
    pilot_run_id: str
    pilot_report_sha256: str
    inputs: dict[str, Any]
    input_fingerprint: str
    approval: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FreezeValidation:
    valid: bool
    reasons: tuple[str, ...]
    environment: FreezeManifest | None = None
    protocol: FreezeManifest | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "reasons": list(self.reasons),
            "environment": self.environment.to_dict() if self.environment else None,
            "protocol": self.protocol.to_dict() if self.protocol else None,
        }
