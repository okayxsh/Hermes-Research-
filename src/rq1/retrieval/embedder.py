"""Capability-gated Sentence-BERT embedder (all-mpnet-base-v2).

The embedder is the only part of the retrieval subsystem that requires the
optional ``retrieval`` extra. Everything else is dependency-free. When the
extra is absent or the model cannot be constructed, construction raises
``RetrievalUnavailable`` instead of falling back to any fake behavior.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

DEFAULT_MODEL_NAME = "all-mpnet-base-v2"


class RetrievalUnavailable(RuntimeError):
    """Raised when the Sentence-BERT embedding stack cannot be constructed."""


@dataclass(frozen=True)
class EmbedderIdentity:
    model_name: str
    available: bool
    details: str

    def to_dict(self) -> dict[str, Any]:
        return {"model_name": self.model_name, "available": self.available, "details": self.details}


def probe_retrieval(model_name: str = DEFAULT_MODEL_NAME) -> EmbedderIdentity:
    try:
        import sentence_transformers  # noqa: F401
    except Exception as exc:
        return EmbedderIdentity(model_name, False, f"sentence-transformers unavailable: {type(exc).__name__}")
    return EmbedderIdentity(model_name, True, "sentence-transformers importable")


class SentenceBERTEmbedder:
    """Thin wrapper exposing a deterministic, normalized ``encode``."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        *,
        cache_folder: Path | None = None,
        revision: str | None = None,
        local_files_only: bool = False,
    ) -> None:
        identity = probe_retrieval(model_name)
        if not identity.available:
            raise RetrievalUnavailable(identity.details)
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                model_name,
                cache_folder=str(cache_folder) if cache_folder is not None else None,
                revision=revision,
                local_files_only=local_files_only,
            )
        except Exception as exc:
            raise RetrievalUnavailable(
                f"could not load Sentence-BERT model {model_name}: {type(exc).__name__}"
            ) from exc
        self.model_name = model_name

    @property
    def model_identity(self) -> str:
        return self.model_name

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self._model.encode(list(texts), convert_to_numpy=True, normalize_embeddings=True)
        return [list(map(float, vector)) for vector in vectors]
