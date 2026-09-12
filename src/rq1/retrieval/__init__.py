"""Sentence-BERT top-3 retrieval subsystem (the RQ1 scientific retrieval path).

The ranking and cosine logic are dependency-free and unit-testable; only the
embedder requires the optional ``retrieval`` extra (``sentence-transformers``).
The embedder is capability-gated and fails closed when the extra is missing.
"""
from __future__ import annotations

from rq1.retrieval.embedder import (
    DEFAULT_MODEL_NAME,
    EmbedderIdentity,
    RetrievalUnavailable,
    SentenceBERTEmbedder,
    probe_retrieval,
)
from rq1.retrieval.controller import (
    RetrievalBoundaryError,
    RetrievalContext,
    RetrievalOutcome,
    SingleRetrievalBoundary,
    build_retrieval_boundary,
)
from rq1.retrieval.injection import (
    RecoveryMemory,
    RecoveryMemorySkill,
    build_recovery_memory,
)
from rq1.retrieval.logging import RetrievalEvent, RetrievalEventLog
from rq1.retrieval.models import (
    RetrievalCandidate,
    RetrievalQuery,
    RetrievalResult,
    SkillDocument,
)
from rq1.retrieval.query import (
    CANONICAL_FAILURE_MESSAGE,
    EMPTY_INVENTORY_MARKER,
    QUERY_TEMPLATE_VERSION,
    build_query_text,
    query_template_hash,
)
from rq1.retrieval.retriever import TopKRetriever, cosine_similarity, rank_candidates
from rq1.retrieval.text import SKILL_TEXT_VERSION, build_skill_text, skill_text_hash

__all__ = [
    "CANONICAL_FAILURE_MESSAGE",
    "DEFAULT_MODEL_NAME",
    "EMPTY_INVENTORY_MARKER",
    "QUERY_TEMPLATE_VERSION",
    "SKILL_TEXT_VERSION",
    "EmbedderIdentity",
    "RecoveryMemory",
    "RecoveryMemorySkill",
    "RetrievalBoundaryError",
    "RetrievalCandidate",
    "RetrievalContext",
    "RetrievalEvent",
    "RetrievalEventLog",
    "RetrievalOutcome",
    "RetrievalQuery",
    "RetrievalResult",
    "RetrievalUnavailable",
    "SentenceBERTEmbedder",
    "SingleRetrievalBoundary",
    "SkillDocument",
    "TopKRetriever",
    "build_query_text",
    "build_recovery_memory",
    "build_retrieval_boundary",
    "build_skill_text",
    "cosine_similarity",
    "probe_retrieval",
    "query_template_hash",
    "rank_candidates",
    "skill_text_hash",
]

