from rq1.integrations.contracts import HermesAdapter, UnverifiedAdapter


def unverified_hermes_adapter() -> UnverifiedAdapter:
    return UnverifiedAdapter("Hermes")
