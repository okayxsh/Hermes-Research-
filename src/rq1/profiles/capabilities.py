from rq1.hermes.capabilities import HermesCapabilityReport, probe_hermes_capabilities


def probe_hermes() -> HermesCapabilityReport:
    """Read-only profile-relevant capability evidence from the Phase 3 probe."""
    return probe_hermes_capabilities()
