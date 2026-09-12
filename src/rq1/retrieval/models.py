"""Typed retrieval contracts for the Sentence-BERT top-3 subsystem."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from rq1.retrieval.query import (
    CANONICAL_FAILURE_MESSAGE,
    QUERY_TEMPLATE_VERSION,
    build_query_text,
    query_template_hash,
)
from rq1.retrieval.text import skill_text_hash


@dataclass(frozen=True)
class RetrievalQuery:
    """A frozen post-failure query. The text representation is deterministic."""

    task_instruction: str
    observation: str
    inventory: tuple[str, ...] = ()
    failure_message: str = CANONICAL_FAILURE_MESSAGE
    version: str = QUERY_TEMPLATE_VERSION

    def text(self) -> str:
        return build_query_text(
            task_instruction=self.task_instruction,
            observation=self.observation,
            inventory=self.inventory,
            failure_message=self.failure_message,
        )

    def text_hash(self) -> str:
        return hashlib.sha256(self.text().encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "task_instruction": self.task_instruction,
            "observation": self.observation,
            "inventory": list(self.inventory),
            "failure_message": self.failure_message,
            "query_text_hash": self.text_hash(),
            "query_template_hash": query_template_hash(),
        }


@dataclass(frozen=True)
class SkillDocument:
    """One embeddable skill: a stable ID plus its deterministic retrieval text."""

    skill_id: str
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "text": self.text,
            "text_hash": skill_text_hash(self.text),
        }


@dataclass(frozen=True)
class RetrievalCandidate:
    """One ranked skill with its cosine similarity score (range [0, 1])."""

    skill_id: str
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {"skill_id": self.skill_id, "score": self.score}


@dataclass(frozen=True)
class RetrievalResult:
    """The complete deterministic ranking; ``top`` exposes only the top-k."""

    query: RetrievalQuery
    ranking: tuple[RetrievalCandidate, ...] = ()
    top_k: int = 3
    retrieved: bool = False
    library_size: int = 0

    @property
    def top(self) -> tuple[RetrievalCandidate, ...]:
        return self.ranking[: self.top_k]

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query.to_dict(),
            "top_k": self.top_k,
            "retrieved": self.retrieved,
            "library_size": self.library_size,
            "ranking": [item.to_dict() for item in self.ranking],
            "top": [item.to_dict() for item in self.top],
        }
