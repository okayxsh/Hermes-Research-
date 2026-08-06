"""Recovery-aware Phase 6 pilot orchestration.

Fake mode validates this package's orchestration contract.  Real evidence is
always capability-gated and is never inferred from fake execution.
"""

from rq1.pilot.catalog import PILOT_TESTS, PILOT_TEST_MAP
from rq1.pilot.models import EvidenceLevel, PilotMode, PilotStatus

__all__ = ["PILOT_TESTS", "PILOT_TEST_MAP", "EvidenceLevel", "PilotMode", "PilotStatus"]
