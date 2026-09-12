"""Authoritative RQ1 named skill-library construction (NoLib / Clean-24 / Accum-60 / Accum-96).

This replaces the legacy chronological snapshot design (``L0/L25/L50/L75/L100``).
Skill selection is deterministic and chronological per task family and is
deliberately independent of any final-evaluation outcome.

Rules
-----
- ``Clean-24``: within each family, the earliest four skills that pass the
  frozen human quality-validation rubric form the 24-skill core.
- ``Accum-60`` / ``Accum-96``: append the earliest 6 / 12 additional successful
  acquired skills per family (after the core). The core subset is byte-identical
  across the three non-empty conditions.
- Near-duplicates and competing skills are preserved, never deduplicated.
- If a required per-family quota cannot be met, construction fails closed.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Sequence

from rq1.retrieval.text import build_skill_text, skill_text_hash

TASK_FAMILIES: tuple[str, ...] = (
    "pick_and_place",
    "pick_two_and_place",
    "look_at_object",
    "clean_and_place",
    "heat_and_place",
    "cool_and_place",
)

CONDITIONS: tuple[str, ...] = ("NoLib", "Clean-24", "Accum-60", "Accum-96")

CORE_PER_FAMILY = 4
CORE_TOTAL = len(TASK_FAMILIES) * CORE_PER_FAMILY  # 24

ACCUM_EXTRAS_PER_FAMILY: dict[str, int] = {"Accum-60": 6, "Accum-96": 12}

LIBRARY_SIZES: dict[str, int] = {
    "NoLib": 0,
    "Clean-24": CORE_TOTAL,
    "Accum-60": CORE_TOTAL + len(TASK_FAMILIES) * ACCUM_EXTRAS_PER_FAMILY["Accum-60"],
    "Accum-96": CORE_TOTAL + len(TASK_FAMILIES) * ACCUM_EXTRAS_PER_FAMILY["Accum-96"],
}


class LibraryConstructionError(ValueError):
    """Raised when the frozen library quotas cannot be satisfied."""


@dataclass(frozen=True)
class AcquiredSkill:
    """One skill produced by a successful acquisition episode."""

    skill_id: str
    title: str
    body: str
    task_family: str
    acquisition_index: int
    validated: bool = False

    def retrieval_text(self) -> str:
        return build_skill_text(title=self.title, body=self.body)


@dataclass(frozen=True)
class LibrarySkill:
    skill_id: str
    task_family: str
    text: str
    core: bool

    @property
    def text_hash(self) -> str:
        return skill_text_hash(self.text)

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "task_family": self.task_family,
            "text_hash": self.text_hash,
            "core": self.core,
        }


@dataclass(frozen=True)
class LibraryManifest:
    condition: str
    skills: tuple[LibrarySkill, ...]

    @property
    def library_size(self) -> int:
        return len(self.skills)

    @property
    def core_skills(self) -> tuple[LibrarySkill, ...]:
        return tuple(item for item in self.skills if item.core)

    def content_sha256(self) -> str:
        payload = [
            {"skill_id": item.skill_id, "text": item.text, "task_family": item.task_family, "core": item.core}
            for item in self.skills
        ]
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def core_sha256(self) -> str:
        payload = [
            {"skill_id": item.skill_id, "text": item.text, "task_family": item.task_family}
            for item in self.core_skills
        ]
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "condition": self.condition,
            "library_size": self.library_size,
            "content_sha256": self.content_sha256(),
            "core_sha256": self.core_sha256(),
            "skills": [item.to_dict() for item in self.skills],
        }


def _validate_input(skills: Sequence[AcquiredSkill]) -> list[AcquiredSkill]:
    ordered = sorted(skills, key=lambda item: (item.acquisition_index, item.skill_id))
    seen: set[str] = set()
    for item in ordered:
        if item.task_family not in TASK_FAMILIES:
            raise LibraryConstructionError(f"unknown task family: {item.task_family}")
        if not item.skill_id:
            raise LibraryConstructionError("acquired skill has an empty skill_id")
        if item.skill_id in seen:
            raise LibraryConstructionError(f"duplicate acquired skill id: {item.skill_id}")
        seen.add(item.skill_id)
    return ordered


def _by_family(ordered: list[AcquiredSkill]) -> dict[str, list[AcquiredSkill]]:
    result: dict[str, list[AcquiredSkill]] = {family: [] for family in TASK_FAMILIES}
    for item in ordered:
        result[item.task_family].append(item)
    return result


def _core_per_family(by_family: dict[str, list[AcquiredSkill]]) -> dict[str, list[AcquiredSkill]]:
    core: dict[str, list[AcquiredSkill]] = {}
    for family in TASK_FAMILIES:
        validated = [item for item in by_family[family] if item.validated]
        if len(validated) < CORE_PER_FAMILY:
            raise LibraryConstructionError(
                f"family {family!r} has {len(validated)} validated skills; "
                f"need {CORE_PER_FAMILY} to build Clean-24"
            )
        core[family] = validated[:CORE_PER_FAMILY]
    return core


def _extra_pool_per_family(
    by_family: dict[str, list[AcquiredSkill]], core: dict[str, list[AcquiredSkill]]
) -> dict[str, list[AcquiredSkill]]:
    core_ids = {family: {item.skill_id for item in skills} for family, skills in core.items()}
    extras: dict[str, list[AcquiredSkill]] = {}
    for family in TASK_FAMILIES:
        extras[family] = [item for item in by_family[family] if item.skill_id not in core_ids[family]]
    return extras


def _library_skills(
    core: dict[str, list[AcquiredSkill]],
    extras: dict[str, list[AcquiredSkill]],
    extras_per_family: int,
) -> tuple[LibrarySkill, ...]:
    result: list[LibrarySkill] = []
    for family in TASK_FAMILIES:
        for item in core[family]:
            result.append(LibrarySkill(item.skill_id, item.task_family, item.retrieval_text(), True))
        for item in extras[family][:extras_per_family]:
            result.append(LibrarySkill(item.skill_id, item.task_family, item.retrieval_text(), False))
    return tuple(result)


def build_libraries(skills: Sequence[AcquiredSkill]) -> dict[str, LibraryManifest]:
    """Build all four frozen conditions from the acquired skill chronology.

    Raises :class:`LibraryConstructionError` if any per-family quota cannot be
    satisfied, so the caller fails closed rather than silently changing the
    protocol.
    """
    ordered = _validate_input(skills)
    by_family = _by_family(ordered)
    core = _core_per_family(by_family)
    extras = _extra_pool_per_family(by_family, core)

    clean = LibraryManifest("Clean-24", _library_skills(core, extras, 0))
    accum_60 = LibraryManifest("Accum-60", _library_skills(core, extras, ACCUM_EXTRAS_PER_FAMILY["Accum-60"]))
    accum_96 = LibraryManifest("Accum-96", _library_skills(core, extras, ACCUM_EXTRAS_PER_FAMILY["Accum-96"]))

    # Fail closed if the accumulated quotas cannot be met.
    for manifest, condition in ((accum_60, "Accum-60"), (accum_96, "Accum-96")):
        needed = LIBRARY_SIZES[condition]
        if manifest.library_size != needed:
            raise LibraryConstructionError(
                f"{condition} requires {needed} skills but only {manifest.library_size} could be assembled"
            )

    core_hashes = {clean.core_sha256(), accum_60.core_sha256(), accum_96.core_sha256()}
    if len(core_hashes) != 1:
        raise LibraryConstructionError("Clean-24 core is not identical across conditions")

    return {
        "NoLib": LibraryManifest("NoLib", ()),
        "Clean-24": clean,
        "Accum-60": accum_60,
        "Accum-96": accum_96,
    }
