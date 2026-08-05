from rq1.integrations.contracts import OllamaAdapter, UnverifiedAdapter


def unverified_ollama_adapter() -> UnverifiedAdapter:
    return UnverifiedAdapter("Ollama")
