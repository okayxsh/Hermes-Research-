"""Rule A construction of the amended nested libraries (NoLib / Core-6 / Accum-12 / Accum-18).

- Core: per family, the chronologically earliest skill a human marked PASS.
- Extras: per family, the chronologically earliest acquired skills not in the core,
  never filtered by review outcome and never deduplicated.

The constructor is pure: it reads the raw 50-skill pool snapshot and the completed
fast core-review file and fails closed on any gap, tampering, or missing PASS.
"""
from __future__ import annotations

import csv
import io
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from rq1.evaluation.amended_protocol import CONDITIONS, CORE_PER_FAMILY, PER_FAMILY_SIZES, RAW_POOL_HASH
from rq1.experiment.models import canonical_hash
from rq1.retrieval.text import build_skill_text, skill_text_hash
from rq1.skills.library import TASK_FAMILIES

REVIEW_COLUMNS = ("reviewer_quality_pass", "reviewer_quality_reason", "reviewer_notes", "reviewed_at")
IMMUTABLE_REVIEW_COLUMNS = (
    "task_family", "family_chronological_rank", "pool_index", "logical_acquisition_index", "origin", "skill_id",
    "source_run_id", "source_task_id", "source_episode_events_log", "skill_text", "skill_text_sha256",
)
_UTC = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z$")


class LibraryAmendmentError(ValueError):
    pass


@dataclass(frozen=True)
class PoolEntry:
    pool_index: int
    skill_id: str
    task_family: str
    family_rank: int
    logical_acquisition_index: int
    origin: str
    source_run_id: str
    source_task_id: str
    source_episode_events_log: str | None
    text: str
    text_sha256: str


def load_raw_pool(path: Path) -> tuple[PoolEntry, ...]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("pool_hash") != RAW_POOL_HASH or value.get("pool_size") != 50:
        raise LibraryAmendmentError("raw pool snapshot is not the closed 50-skill acquisition pool")
    entries = []
    for item in value["entries"]:
        skill = item["skill"]
        if skill["text"] != build_skill_text(title=skill["title"], body=skill["body"]) or skill["text_sha256"] != skill_text_hash(skill["text"]):
            raise LibraryAmendmentError(f"raw pool skill text identity is broken: {skill['skill_id']}")
        entries.append(PoolEntry(
            pool_index=int(item["pool_index"]), skill_id=skill["skill_id"], task_family=skill["task_family"],
            family_rank=int(item["family_chronological_rank"]), logical_acquisition_index=int(item["logical_acquisition_index"]),
            origin=item["origin"], source_run_id=item["source_run_id"], source_task_id=skill["source_task_id"],
            source_episode_events_log=item.get("source_episode_events_log"), text=skill["text"], text_sha256=skill["text_sha256"],
        ))
    ordered = tuple(sorted(entries, key=lambda entry: entry.pool_index))
    if [entry.pool_index for entry in ordered] != list(range(1, 51)):
        raise LibraryAmendmentError("raw pool indices are not the contiguous chronological order 1-50")
    return ordered


def by_family(pool: Sequence[PoolEntry]) -> dict[str, list[PoolEntry]]:
    families: dict[str, list[PoolEntry]] = {family: [] for family in TASK_FAMILIES}
    for entry in sorted(pool, key=lambda item: item.pool_index):
        families[entry.task_family].append(entry)
    for family, entries in families.items():
        if [entry.family_rank for entry in entries] != list(range(1, len(entries) + 1)):
            raise LibraryAmendmentError(f"family chronological ranks are not contiguous: {family}")
    return families


def raw_feasibility(pool: Sequence[PoolEntry]) -> dict[str, Any]:
    counts = {family: len(entries) for family, entries in by_family(pool).items()}
    largest = max(PER_FAMILY_SIZES.values())
    return {"raw_counts": counts, "largest_per_family": largest, "feasible": all(count >= largest for count in counts.values()),
            "families_below": {family: count for family, count in counts.items() if count < largest}}


def review_rows(pool: Sequence[PoolEntry]) -> list[dict[str, Any]]:
    rows = []
    for family, entries in by_family(pool).items():
        for entry in entries:
            rows.append({
                "task_family": family,
                "family_chronological_rank": entry.family_rank,
                "pool_index": entry.pool_index,
                "logical_acquisition_index": entry.logical_acquisition_index,
                "origin": entry.origin,
                "skill_id": entry.skill_id,
                "source_run_id": entry.source_run_id,
                "source_task_id": entry.source_task_id,
                "source_episode_events_log": entry.source_episode_events_log or "",
                "skill_text": entry.text,
                "skill_text_sha256": entry.text_sha256,
                "core_review_instruction": "REVIEW" if entry.family_rank == 1 else "REVIEW ONLY IF EVERY EARLIER SKILL IN THIS FAMILY IS FAIL",
                **{column: "" for column in REVIEW_COLUMNS},
            })
    return rows


def render_review_csv(rows: Sequence[Mapping[str, Any]]) -> bytes:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0].keys()), lineterminator="\r\n")
    writer.writeheader()
    writer.writerows(rows)
    return ("﻿" + buffer.getvalue()).encode("utf-8")


def parse_review_csv(data: bytes) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(data.decode("utf-8-sig"))))


@dataclass(frozen=True)
class CoreSelection:
    complete: bool
    core: dict[str, str]
    problems: tuple[str, ...]
    pending_families: tuple[str, ...]
    failed_families: tuple[str, ...]
    decisions: tuple[dict[str, str], ...]

    def to_dict(self) -> dict[str, Any]:
        return {"complete": self.complete, "core": dict(self.core), "problems": list(self.problems),
                "pending_families": list(self.pending_families), "failed_families": list(self.failed_families),
                "decisions": [dict(item) for item in self.decisions]}


def select_core(pool: Sequence[PoolEntry], rows: Iterable[Mapping[str, str]]) -> CoreSelection:
    """Earliest PASS per family from a human core review; fails closed on any inconsistency."""
    expected = {row["skill_id"]: row for row in review_rows(pool)}
    problems: list[str] = []
    seen: dict[str, Mapping[str, str]] = {}
    for row in rows:
        skill_id = row.get("skill_id", "")
        if skill_id not in expected:
            problems.append(f"unknown skill in review: {skill_id!r}")
            continue
        if skill_id in seen:
            problems.append(f"duplicate review row: {skill_id}")
        seen[skill_id] = row
        for column in IMMUTABLE_REVIEW_COLUMNS:
            if str(row.get(column, "")) != str(expected[skill_id][column]):
                problems.append(f"immutable column changed for {skill_id}: {column}")
    if set(seen) != set(expected):
        problems.append(f"review must contain exactly the {len(expected)} raw-pool skills")
    core: dict[str, str] = {}
    pending: list[str] = []
    failed: list[str] = []
    decisions: list[dict[str, str]] = []
    for family, entries in by_family(pool).items():
        found = None
        for entry in entries:
            row = seen.get(entry.skill_id) or {}
            verdict = str(row.get("reviewer_quality_pass", "")).strip()
            if not verdict:
                break
            if verdict not in {"PASS", "FAIL"}:
                problems.append(f"{entry.skill_id}: reviewer_quality_pass must be PASS or FAIL")
                break
            if not str(row.get("reviewer_quality_reason", "")).strip():
                problems.append(f"{entry.skill_id}: reviewer_quality_reason is required")
            if not _UTC.match(str(row.get("reviewed_at", "")).strip()):
                problems.append(f"{entry.skill_id}: reviewed_at must be a UTC ISO-8601 time ending in Z")
            decisions.append({"task_family": family, "family_chronological_rank": str(entry.family_rank), "skill_id": entry.skill_id,
                              "verdict": verdict, "reason": str(row.get("reviewer_quality_reason", "")).strip(),
                              "reviewed_at": str(row.get("reviewed_at", "")).strip()})
            if verdict == "PASS":
                found = entry.skill_id
                break
        if found:
            core[family] = found
        elif all(str((seen.get(entry.skill_id) or {}).get("reviewer_quality_pass", "")).strip() == "FAIL" for entry in entries):
            failed.append(family)
        else:
            pending.append(family)
    if failed:
        problems.append("no PASS core candidate in: " + ", ".join(failed) + " (evaluation must not start)")
    complete = not problems and not pending and len(core) == len(TASK_FAMILIES)
    return CoreSelection(complete, core, tuple(problems), tuple(pending), tuple(failed), tuple(decisions))


@dataclass(frozen=True)
class AmendedLibrary:
    condition: str
    skills: tuple[dict[str, Any], ...]

    @property
    def size(self) -> int:
        return len(self.skills)

    @property
    def content_sha256(self) -> str:
        return canonical_hash([{key: skill[key] for key in ("skill_id", "task_family", "text", "role")} for skill in self.skills])

    @property
    def core_sha256(self) -> str:
        return canonical_hash([{key: skill[key] for key in ("skill_id", "task_family", "text")} for skill in self.skills if skill["role"] == "core"])

    def retrieval_documents(self) -> tuple[tuple[str, str], ...]:
        return tuple((skill["skill_id"], skill["text"]) for skill in self.skills)

    def to_dict(self) -> dict[str, Any]:
        return {"condition": self.condition, "size": self.size, "content_sha256": self.content_sha256,
                "core_sha256": self.core_sha256 if self.size else None, "skills": [dict(skill) for skill in self.skills]}


def build_amended_libraries(pool: Sequence[PoolEntry], core: Mapping[str, str]) -> dict[str, AmendedLibrary]:
    families = by_family(pool)
    if set(core) != set(TASK_FAMILIES):
        raise LibraryAmendmentError("a core skill is required for every family")
    libraries: dict[str, AmendedLibrary] = {}
    for condition in CONDITIONS:
        per_family = PER_FAMILY_SIZES[condition]
        skills: list[dict[str, Any]] = []
        for family in TASK_FAMILIES:
            entries = families[family]
            if per_family == 0:
                continue
            chosen = next((entry for entry in entries if entry.skill_id == core[family]), None)
            if chosen is None:
                raise LibraryAmendmentError(f"core skill {core[family]} is not a {family} skill of the raw pool")
            extras = [entry for entry in entries if entry.skill_id != chosen.skill_id]
            selection = [chosen, *extras[: per_family - CORE_PER_FAMILY]]
            if len(selection) != per_family:
                raise LibraryAmendmentError(f"{condition} needs {per_family} {family} skills but only {len(selection)} exist")
            for position, entry in enumerate(selection):
                skills.append({"skill_id": entry.skill_id, "task_family": family, "text": entry.text, "text_sha256": entry.text_sha256,
                               "role": "core" if position == 0 else "extra", "family_chronological_rank": entry.family_rank,
                               "pool_index": entry.pool_index, "logical_acquisition_index": entry.logical_acquisition_index})
        libraries[condition] = AmendedLibrary(condition, tuple(skills))
    problems = nesting_problems(libraries)
    if problems:
        raise LibraryAmendmentError("; ".join(problems))
    return libraries


def nesting_problems(libraries: Mapping[str, AmendedLibrary]) -> list[str]:
    problems = []
    for condition in CONDITIONS:
        library = libraries.get(condition)
        if library is None:
            problems.append(f"missing library: {condition}")
            continue
        expected = PER_FAMILY_SIZES[condition]
        counts = {family: sum(skill["task_family"] == family for skill in library.skills) for family in TASK_FAMILIES}
        if any(count != expected for count in counts.values()):
            problems.append(f"{condition} is not balanced at {expected} per family: {counts}")
        if len({skill["skill_id"] for skill in library.skills}) != library.size:
            problems.append(f"{condition} repeats a skill")
    non_empty = [libraries[condition] for condition in CONDITIONS if condition in libraries and PER_FAMILY_SIZES[condition]]
    for smaller, larger in zip(non_empty, non_empty[1:]):
        if not {skill["skill_id"] for skill in smaller.skills} <= {skill["skill_id"] for skill in larger.skills}:
            problems.append(f"{smaller.condition} is not nested in {larger.condition}")
    if len({library.core_sha256 for library in non_empty}) > 1:
        problems.append("core skills differ between non-empty libraries")
    return problems


def library_freeze_payload(libraries: Mapping[str, AmendedLibrary], *, selection: CoreSelection, review_sha256: str,
                           pool_snapshot_sha256: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "rq1-amended-evaluation-libraries",
        "rule": "A",
        "raw_pool_hash": RAW_POOL_HASH,
        "raw_pool_snapshot_sha256": pool_snapshot_sha256,
        "core_review_sha256": review_sha256,
        "core_selection": selection.to_dict(),
        "libraries": {condition: libraries[condition].to_dict() for condition in CONDITIONS},
        "library_hashes": {condition: libraries[condition].content_sha256 for condition in CONDITIONS},
        "nesting_verified": not nesting_problems(libraries),
    }
