from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from rq1.bridge.models import AdapterState, EpisodeStartRequest


@dataclass(frozen=True)
class CapabilityResult:
    available: bool
    version: str | None
    details: str


@runtime_checkable
class ALFWorldAdapter(Protocol):
    def start(self, request: EpisodeStartRequest) -> AdapterState: ...
    def step(self, action: str) -> AdapterState: ...
    def status(self) -> AdapterState: ...
    def abort(self, reason: str | None = None) -> AdapterState: ...
    def reset(self) -> AdapterState: ...


@runtime_checkable
class HermesAdapter(Protocol):
    def probe(self) -> CapabilityResult: ...


@runtime_checkable
class ProfileManager(Protocol):
    def probe(self) -> CapabilityResult: ...
    def materialize(self, profile_name: str, dry_run: bool = True) -> None: ...


@runtime_checkable
class OllamaAdapter(Protocol):
    def probe(self) -> CapabilityResult: ...


class UnverifiedAdapter:
    """Fails closed until an installed version is capability-probed in the pilot."""
    def __init__(self, integration: str) -> None:
        self.integration = integration

    def probe(self) -> CapabilityResult:
        return CapabilityResult(False, None, f"{self.integration} integration is unverified; pilot capability probe required.")
