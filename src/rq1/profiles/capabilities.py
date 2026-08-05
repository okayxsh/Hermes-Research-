from rq1.integrations.contracts import CapabilityResult, UnverifiedAdapter


def probe_hermes() -> CapabilityResult:
    return UnverifiedAdapter("Hermes").probe()
