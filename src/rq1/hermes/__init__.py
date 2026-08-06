"""Capability-gated Hermes-to-local-bridge integration boundary."""

from rq1.hermes.adapter import FakeHermesAdapter, HermesAdapter, RealHermesAdapter
from rq1.hermes.capabilities import HermesCapabilityReport, probe_hermes_capabilities
from rq1.hermes.models import HermesToolResult, ToolValidationError
from rq1.hermes.reconcile import reconcile_evidence

__all__ = [
    "FakeHermesAdapter",
    "HermesAdapter",
    "HermesCapabilityReport",
    "HermesToolResult",
    "RealHermesAdapter",
    "ToolValidationError",
    "probe_hermes_capabilities",
    "reconcile_evidence",
]
