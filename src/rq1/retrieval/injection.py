"""Build the Hermes recovery-memory context message from a retrieval result.

Scores are deliberately excluded: the agent sees only rank, skill identifier,
and retrieval text. Cosine scores are logged for analysis only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from rq1.retrieval.models import RetrievalResult


@dataclass(frozen=True)
class RecoveryMemorySkill:
    rank: int
    skill_id: str
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {"rank": self.rank, "skill_id": self.skill_id, "text": self.text}


@dataclass(frozen=True)
class RecoveryMemory:
    retrieved_skills: tuple[RecoveryMemorySkill, ...]
    no_retrieval: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "recovery_memory": {
                "retrieved_skills": [item.to_dict() for item in self.retrieved_skills],
                "no_retrieved_skills_available": self.no_retrieval,
            }
        }


def build_recovery_memory(result: RetrievalResult, skill_texts: Mapping[str, str]) -> RecoveryMemory:
    """Return a score-free structured context message for the top-k skills.

    For NoLib (no retrieved skills), the same structure is returned with an
    explicit ``no_retrieved_skills_available`` marker.
    """
    if not result.retrieved:
        return RecoveryMemory((), no_retrieval=True)
    skills = tuple(
        RecoveryMemorySkill(rank, candidate.skill_id, skill_texts[candidate.skill_id])
        for rank, candidate in enumerate(result.top, 1)
    )
    return RecoveryMemory(skills, no_retrieval=False)
