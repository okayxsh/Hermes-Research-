"""Dependency-free cosine similarity and deterministic top-k ranking."""
from __future__ import annotations

import math
from typing import Sequence

from rq1.retrieval.models import RetrievalCandidate, RetrievalQuery, RetrievalResult


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError("embedding dimension mismatch")
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


def rank_candidates(
    query_embedding: Sequence[float],
    skill_embeddings: dict[str, Sequence[float]],
) -> tuple[RetrievalCandidate, ...]:
    scored = [
        RetrievalCandidate(skill_id, cosine_similarity(query_embedding, embedding))
        for skill_id, embedding in skill_embeddings.items()
    ]
    scored.sort(key=lambda item: (-item.score, item.skill_id))
    return tuple(scored)


class TopKRetriever:
    """Deterministic retriever over precomputed skill embeddings.

    The embedder is injected so the ranking/cosine logic stays dependency-free
    and unit-testable without torch/sentence-transformers.
    """

    def __init__(self, embedder: object, skills: Sequence[tuple[str, str]]) -> None:
        self.embedder = embedder
        self.library_size = len(skills)
        self._skill_embeddings: dict[str, Sequence[float]] = {}
        if skills:
            vectors = embedder.encode([text for _skill_id, text in skills])  # type: ignore[attr-defined]
            for (skill_id, _text), vector in zip(skills, vectors):
                self._skill_embeddings[skill_id] = tuple(float(value) for value in vector)

    def retrieve(self, query: RetrievalQuery, top_k: int = 3) -> RetrievalResult:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if not self._skill_embeddings:
            return RetrievalResult(
                query=query, ranking=(), top_k=top_k, retrieved=False, library_size=self.library_size
            )
        query_embedding = self.embedder.encode([query.text()])[0]  # type: ignore[attr-defined]
        ranking = rank_candidates(tuple(float(value) for value in query_embedding), self._skill_embeddings)
        return RetrievalResult(
            query=query, ranking=ranking, top_k=top_k, retrieved=True, library_size=self.library_size
        )
