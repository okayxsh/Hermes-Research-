"""Deterministic, manifest-driven index for installed ALFWorld text data."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from rq1.bridge.adapters.base import IndexedTask


ALLOWED_REAL_SPLITS = frozenset({"train", "valid_seen"})
TASK_TYPES = {
    1: "pick_and_place_simple", 2: "look_at_obj_in_light", 3: "pick_clean_then_place_in_recep",
    4: "pick_heat_then_place_in_recep", 5: "pick_cool_then_place_in_recep", 6: "pick_two_obj_and_place",
}


class TaskIndexError(ValueError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _data_identity(entries: list[tuple[str, str, str]]) -> str:
    encoded = "".join(f"{path}\0{source}\0{game}\n" for path, source, game in entries).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class TaskIndex:
    data_root: Path
    entries: tuple[IndexedTask, ...]
    identity: str

    def resolve(self, task_id: str, split: str) -> IndexedTask:
        if split not in ALLOWED_REAL_SPLITS:
            raise TaskIndexError("Real ALFWorld permits only train or valid_seen; valid_unseen is never available here.")
        matches = [entry for entry in self.entries if entry.task_id == task_id]
        if not matches:
            raise TaskIndexError(f"Unknown task_id: {task_id}")
        if len(matches) != 1:
            raise TaskIndexError(f"Ambiguous task_id: {task_id}")
        if matches[0].split != split:
            raise TaskIndexError(f"task_id {task_id} belongs to split {matches[0].split}, not {split}")
        return matches[0]

    def for_split(self, split: str) -> tuple[IndexedTask, ...]:
        if split not in ALLOWED_REAL_SPLITS:
            raise TaskIndexError("Only train and valid_seen can be indexed by this command.")
        return tuple(entry for entry in self.entries if entry.split == split)

    def to_dict(self, split: str | None = None) -> dict[str, object]:
        entries = self.entries if split is None else self.for_split(split)
        return {"schema_version": 1, "data_identity": self.identity, "task_count": len(entries), "tasks": [item.to_dict() for item in entries]}


def build_task_index(data_root: Path, *, splits: tuple[str, ...] = ("train", "valid_seen")) -> TaskIndex:
    if any(split not in ALLOWED_REAL_SPLITS for split in splits):
        raise TaskIndexError("valid_unseen is intentionally excluded from real adapter indexing.")
    root = data_root.expanduser().resolve()
    base = root / "json_2.1.1"
    if not base.is_dir():
        raise TaskIndexError("ALFWorld data is missing json_2.1.1.")
    raw: list[tuple[str, str, str, str, Path, Path]] = []
    for split in sorted(set(splits)):
        split_root = base / split
        if not split_root.is_dir():
            raise TaskIndexError(f"ALFWorld split directory is missing: {split}")
        for source in sorted(split_root.rglob("traj_data.json")):
            game = source.with_name("game.tw-pddl")
            if not game.is_file():
                continue
            try:
                payload = json.loads(source.read_text(encoding="utf-8"))
                game_payload = json.loads(game.read_text(encoding="utf-8"))
                task_type = payload["task_type"]
            except (OSError, ValueError, KeyError, TypeError) as exc:
                raise TaskIndexError(f"Malformed ALFWorld task data: {source.name}") from exc
            if task_type not in TASK_TYPES or game_payload.get("solvable") is not True:
                continue
            relative = source.parent.relative_to(split_root).as_posix()
            if not relative or relative.startswith("../"):
                raise TaskIndexError("Task path cannot produce a stable task identifier.")
            task_id = f"{split}:{relative}"
            raw.append((task_id, split, TASK_TYPES[task_type], relative, source, game))
    identifiers = [item[0] for item in raw]
    if len(identifiers) != len(set(identifiers)):
        raise TaskIndexError("Duplicate stable task ID detected in ALFWorld data.")
    fingerprints = [(relative, _sha256(source), _sha256(game)) for _, _, _, relative, source, game in raw]
    identity = _data_identity(fingerprints)
    entries = tuple(IndexedTask(task_id, split, family, source.relative_to(root), game.relative_to(root), source_hash, game_hash, identity)
                    for (task_id, split, family, _relative, source, game), (_p, source_hash, game_hash) in zip(raw, fingerprints))
    return TaskIndex(root, entries, identity)
