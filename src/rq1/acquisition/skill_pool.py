"""Append-only acquisition skill pool whose authority is the committed results journal.

Every accepted skill is embedded in the completed result row of the episode that
created it.  The pool is rebuilt from those rows in queue order, so a crash can
never leave a skill without a committed source episode.  ``skill_pool.json`` is a
derived atomic snapshot; it may lag the journal but must never disagree with it.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from rq1.experiment.models import canonical_hash
from rq1.experiment.persistence import ExperimentStateError, atomic_write_json
from rq1.retrieval.text import build_skill_text
from rq1.skills.library import TASK_FAMILIES
from rq1.utils.hashing import sha256_text

SNAPSHOT_NAME = "skill_pool.json"


class SkillPoolError(ExperimentStateError):
    """Scientific skill-pool state is inconsistent; it is never silently repaired."""


@dataclass(frozen=True)
class PoolSkill:
    pool_index: int
    skill_id: str
    title: str
    body: str
    text: str
    text_sha256: str
    task_family: str
    source_task_id: str
    source_task_index: int
    source_run_key: str
    source_attempt_id: str
    created_at: str
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def identity(self) -> dict[str, Any]:
        # Attempt IDs and timestamps are provenance, not content identity.
        return {
            "pool_index": self.pool_index,
            "skill_id": self.skill_id,
            "task_family": self.task_family,
            "text": self.text,
            "source_task_id": self.source_task_id,
            "source_task_index": self.source_task_index,
            "source_run_key": self.source_run_key,
        }

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["provenance"] = dict(self.provenance)
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PoolSkill":
        try:
            return cls(**{**dict(value), "provenance": dict(value.get("provenance") or {})})
        except TypeError as exc:
            raise SkillPoolError(f"malformed pool skill record: {exc}") from exc


def pool_hash(skills: Sequence[PoolSkill]) -> str:
    return canonical_hash([skill.identity() for skill in skills])


EMPTY_POOL_HASH = pool_hash(())


def skill_id_for(source_task_id: str, title: str, body: str) -> str:
    """Same identifier convention as ``rq1.acquisition.real_executor``."""
    return "skill_" + sha256_text(f"{source_task_id}\0{title}\0{body}")[:16]


def rebuild_pool(records: Iterable[Mapping[str, Any]]) -> tuple[PoolSkill, ...]:
    completed = [
        record for record in records
        if record.get("phase") == "acquisition" and record.get("status") == "completed"
    ]
    completed.sort(key=lambda record: int(record.get("task_index", 0)))
    skills: list[PoolSkill] = []
    texts: set[str] = set()
    identifiers: set[str] = set()
    for record in completed:
        where = f"task_index={record.get('task_index')}"
        if record.get("skill_pool_hash_before") != pool_hash(skills) or record.get("skill_pool_size_before") != len(skills):
            raise SkillPoolError(f"skill pool chronology mismatch before {where}")
        candidate = record.get("skill_candidate")
        if not isinstance(candidate, Mapping):
            raise SkillPoolError(f"completed acquisition result lacks skill_candidate evidence at {where}")
        if candidate.get("status") == "accepted":
            if record.get("success") is not True:
                raise SkillPoolError(f"accepted skill without a successful episode at {where}")
            skill = PoolSkill.from_dict(candidate.get("skill") or {})
            problems = []
            if skill.pool_index != len(skills) + 1:
                problems.append("pool_index")
            if (skill.source_run_key, skill.source_task_id, skill.source_task_index, skill.source_attempt_id) != (
                record.get("run_key"), record.get("task_id"), record.get("task_index"), record.get("attempt_id")
            ):
                problems.append("source provenance")
            if skill.task_family != record.get("task_family") or skill.task_family not in TASK_FAMILIES:
                problems.append("task_family")
            if skill.text != build_skill_text(title=skill.title, body=skill.body) or skill.text_sha256 != sha256_text(skill.text):
                problems.append("text identity")
            if skill.skill_id in identifiers or skill.text in texts:
                problems.append("duplicate")
            if problems:
                raise SkillPoolError(f"accepted skill is inconsistent at {where}: {', '.join(problems)}")
            skills.append(skill)
            identifiers.add(skill.skill_id)
            texts.add(skill.text)
        elif candidate.get("skill") is not None:
            raise SkillPoolError(f"non-accepted candidate carries a pool skill at {where}")
        if record.get("skill_library_hash_after") != pool_hash(skills) or record.get("library_size_after") != len(skills):
            raise SkillPoolError(f"skill pool identity mismatch after {where}")
    return tuple(skills)


def snapshot_payload(skills: Sequence[PoolSkill]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "authority": "results.jsonl",
        "pool_size": len(skills),
        "pool_hash": pool_hash(skills),
        "skills": [skill.to_dict() for skill in skills],
    }


def write_snapshot(directory: Path, skills: Sequence[PoolSkill]) -> Path:
    path = directory / SNAPSHOT_NAME
    atomic_write_json(path, snapshot_payload(skills))
    return path


def verify_snapshot(directory: Path, skills: Sequence[PoolSkill]) -> None:
    path = directory / SNAPSHOT_NAME
    if not path.is_file():
        return
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        recorded = [PoolSkill.from_dict(item) for item in value.get("skills", [])]
    except (OSError, ValueError, AttributeError) as exc:
        raise SkillPoolError(f"unreadable skill pool snapshot: {path}") from exc
    consistent = (
        len(recorded) <= len(skills)
        and [item.identity() for item in recorded] == [item.identity() for item in skills[: len(recorded)]]
        and value.get("pool_hash") == pool_hash(recorded)
        and value.get("pool_size") == len(recorded)
    )
    if not consistent:
        raise SkillPoolError("skill_pool.json disagrees with committed results; refusing to repair scientific state")
