from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Stage:
    name: str
    prerequisites: tuple[str, ...] = ()
    external: bool = False


STAGES: tuple[Stage, ...] = (
    Stage("preflight"),
    Stage("install", ("preflight",), True),
    Stage("configure", ("install",), True),
    Stage("pilot", ("configure",), True),
    Stage("freeze", ("pilot",), True),
    Stage("acquisition", ("freeze",), True),
    Stage("validate-acquisition", ("acquisition",)),
    Stage("snapshots", ("validate-acquisition",)),
    Stage("validate-snapshots", ("snapshots",)),
    Stage("evaluation", ("validate-snapshots",), True),
    Stage("validate-evaluation", ("evaluation",)),
    Stage("analysis", ("validate-evaluation",)),
    Stage("report-assets", ("analysis",)),
    Stage("archive", ("report-assets",)),
)
STAGE_MAP = {stage.name: stage for stage in STAGES}
VALID_STAGE_STATUSES = {"not_started", "running", "passed", "failed", "blocked", "interrupted", "invalidated"}


def next_stage(name: str) -> str | None:
    names = [stage.name for stage in STAGES]
    index = names.index(name)
    return names[index + 1] if index + 1 < len(names) else None
