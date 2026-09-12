"""Single-retrieval boundary for controlled recovery episodes.

This is the only place the scientific Sentence-BERT retrieval may be invoked
during evaluation. It runs exactly once per recovery episode: after the
checkpoint replay and perturbation/solvability validation, and before the first
recovery action. Acquisition and pilot code must not use this path.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence
from uuid import uuid4

from rq1.retrieval.injection import RecoveryMemory, build_recovery_memory
from rq1.retrieval.logging import RetrievalEvent
from rq1.retrieval.models import RetrievalQuery, RetrievalResult
from rq1.retrieval.retriever import TopKRetriever
from rq1.utils.time import utc_now


class RetrievalBoundaryError(RuntimeError):
    """Raised when the single-retrieval invariant is violated."""


@dataclass(frozen=True)
class RetrievalContext:
    run_id: str
    attempt_id: str
    task_id: str
    condition: str
    library_name: str
    library_size: int
    library_hash: str | None
    episode_id: str | None = None


@dataclass(frozen=True)
class RetrievalOutcome:
    result: RetrievalResult
    recovery_memory: RecoveryMemory
    event: RetrievalEvent


class SingleRetrievalBoundary:
    """Performs the scientific retrieval exactly once per recovery episode."""

    def __init__(
        self,
        retriever: TopKRetriever,
        *,
        skill_texts: Mapping[str, str],
        embedding_model: str,
        top_k: int = 3,
        embedding_model_revision: str | None = None,
        embedding_model_file_hashes: Mapping[str, str] | None = None,
    ) -> None:
        self._retriever = retriever
        self._skill_texts = dict(skill_texts)
        self._embedding_model = embedding_model
        self._top_k = top_k
        self._embedding_model_revision = embedding_model_revision
        self._embedding_model_file_hashes = dict(embedding_model_file_hashes or {})
        self._consumed = False

    def retrieve(
        self,
        context: RetrievalContext,
        query: RetrievalQuery,
        *,
        top_k: int | None = None,
    ) -> RetrievalOutcome:
        if self._consumed:
            raise RetrievalBoundaryError(
                "scientific retrieval has already been performed for this recovery episode"
            )
        self._consumed = True
        k = top_k if top_k is not None else self._top_k
        result = self._retriever.retrieve(query, top_k=k)
        recovery_memory = build_recovery_memory(result, self._skill_texts)
        event = RetrievalEvent(
            run_id=context.run_id,
            attempt_id=context.attempt_id,
            task_id=context.task_id,
            episode_id=context.episode_id,
            condition=context.condition,
            library_name=context.library_name,
            library_size=context.library_size,
            library_hash=context.library_hash,
            embedding_model=self._embedding_model,
            embedding_model_revision=self._embedding_model_revision,
            embedding_model_file_hashes=self._embedding_model_file_hashes,
            result=result,
            event_id=str(uuid4()),
            timestamp=utc_now(),
        )
        return RetrievalOutcome(result=result, recovery_memory=recovery_memory, event=event)


def build_retrieval_boundary(
    skills: Sequence[tuple[str, str]],
    embedder: object,
    *,
    embedding_model: str,
    top_k: int = 3,
    embedding_model_revision: str | None = None,
    embedding_model_file_hashes: Mapping[str, str] | None = None,
) -> SingleRetrievalBoundary:
    """Construct a single-retrieval boundary from (skill_id, text) pairs."""
    texts = {skill_id: text for skill_id, text in skills}
    retriever = TopKRetriever(embedder, skills)
    return SingleRetrievalBoundary(
        retriever,
        skill_texts=texts,
        embedding_model=embedding_model,
        top_k=top_k,
        embedding_model_revision=embedding_model_revision,
        embedding_model_file_hashes=embedding_model_file_hashes,
    )
