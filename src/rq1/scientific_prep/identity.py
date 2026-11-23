"""Stable ALFWorld task identity parsing and template keys.

A task ID is ``<split>:<task_type>-<object>-<mrecep>-<recep>-<scene>/<trial>`` exactly as built by
``rq1.tasks.discovery``.  Distinct trials can share one template, so independence is never inferred from
distinct IDs: every overlap check uses an explicit, recorded template definition.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Callable

from rq1.tasks.discovery import CANONICAL_FAMILIES

SPLITS = ("train", "valid_seen", "valid_unseen")
_TEMPLATE_DIR = re.compile(r"^(?P<task_type>[a-z_]+)-(?P<object>[^-/]+)-(?P<mrecep>[^-/]+)-(?P<recep>[^-/]+)-(?P<scene>\d+)$")
_TRIAL = re.compile(r"^trial_[A-Za-z0-9_]+$")
_REFERENCE = re.compile(r"(?:^|/)(?:json_2\.1\.1/)?(?P<split>train|valid_seen|valid_unseen)[:/](?P<template>[^/]+)/(?P<trial>trial_[A-Za-z0-9_]+)")


class TaskIdentityError(ValueError):
    pass


@dataclass(frozen=True)
class TaskIdentity:
    task_id: str
    split: str
    template_dir: str
    task_type: str
    family: str | None
    object: str
    mrecep: str
    recep: str
    scene: str
    trial: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_task_id(task_id: str) -> TaskIdentity:
    split, sep, relative = task_id.partition(":")
    if not sep or split not in SPLITS:
        raise TaskIdentityError(f"not a split-qualified task ID: {task_id!r}")
    template_dir, slash, trial = relative.partition("/")
    match = _TEMPLATE_DIR.fullmatch(template_dir)
    if not slash or match is None or not _TRIAL.fullmatch(trial):
        raise TaskIdentityError(f"task ID does not follow <type>-<object>-<mrecep>-<recep>-<scene>/<trial>: {task_id!r}")
    # ALFRED also contains task types ALFWorld 0.4.2 does not support (e.g. pick_and_place_with_movable_recep, which
    # has no installed game files).  They keep a stable identity with family None; they are never runnable candidates.
    family = CANONICAL_FAMILIES.get(match["task_type"])
    return TaskIdentity(task_id, split, template_dir, match["task_type"], family, match["object"], match["mrecep"],
                        match["recep"], match["scene"], trial)


def normalize_task_reference(value: str) -> str | None:
    """Map a historical reference (task ID, ``json_2.1.1/<split>/.../traj_data.json``, ``$ALFWORLD_DATA/...``) to a
    task ID.  Returns None when the reference cannot be resolved unambiguously (it is then reported, never guessed)."""
    text = str(value).strip().replace("\\", "/")
    if not text:
        return None
    match = _REFERENCE.search(text)
    if match is None:
        return None
    candidate = f"{match['split']}:{match['template']}/{match['trial']}"
    try:
        parse_task_id(candidate)
    except TaskIdentityError:
        return None
    return candidate


# Template definitions.  "goal_template" ignores the scene: it reproduces the historical evaluation/preparation
# screening (94 -> 79) from the frozen manifests.  "scene_template" is the trial directory (scene included).
TEMPLATE_DEFINITIONS: dict[str, Callable[[TaskIdentity], str]] = {
    "goal_template": lambda i: f"{i.task_type}|{i.object}|{i.mrecep}|{i.recep}",
    "scene_template": lambda i: f"{i.task_type}|{i.object}|{i.mrecep}|{i.recep}|{i.scene}",
    "type_object": lambda i: f"{i.task_type}|{i.object}",
}


def template_key(identity: TaskIdentity, definition: str) -> str:
    try:
        return TEMPLATE_DEFINITIONS[definition](identity)
    except KeyError as exc:
        raise TaskIdentityError(f"unknown template definition {definition!r}; known: {sorted(TEMPLATE_DEFINITIONS)}") from exc
