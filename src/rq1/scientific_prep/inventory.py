"""Deterministic, read-only ALFWorld task inventory.

For each requested split under ``<data-root>/json_2.1.1/<split>`` every directory holding ``traj_data.json`` is an
*indexed entry*; it is *installed/runnable* only if ``game.tw-pddl`` is present.  Every source file is SHA-256 hashed,
duplicate file hashes are grouped (distinct IDs are not assumed independent), and the split identity is computed with
the SAME function as ``rq1.tasks.discovery`` so it can be compared with historical manifests' ``data_root_identity``.
Only ``task_type`` is read from ``traj_data.json``; expert plans and annotations are never read or copied.
"""
from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

from rq1.bridge.adapters.task_index import _resolve_task_family
from rq1.scientific_prep.identity import SPLITS, TEMPLATE_DEFINITIONS, TaskIdentityError, parse_task_id, template_key
from rq1.tasks.discovery import CANONICAL_FAMILIES, _identity
from rq1.tasks.models import TaskRecord
from rq1.utils.hashing import sha256_file

INVENTORY_SCHEMA = "rq1-sci-prep-task-inventory-v1"


class InventoryError(ValueError):
    pass


def _split_entries(data_root: Path, split: str) -> list[dict[str, Any]]:
    split_root = data_root / "json_2.1.1" / split
    if not split_root.is_dir():
        raise InventoryError(f"missing ALFWorld split directory: json_2.1.1/{split}")
    entries: list[dict[str, Any]] = []
    for dirpath, dirnames, filenames in os.walk(split_root):
        dirnames.sort()
        here = Path(dirpath)
        has_traj, has_game = "traj_data.json" in filenames, "game.tw-pddl" in filenames
        if not (has_traj or has_game):
            continue
        relative = here.relative_to(split_root).as_posix()
        task_id = f"{split}:{relative}"
        row: dict[str, Any] = {"task_id": task_id, "split": split, "relative_dir": relative,
                               "trajectory_path": f"json_2.1.1/{split}/{relative}/traj_data.json" if has_traj else None,
                               "game_path": f"json_2.1.1/{split}/{relative}/game.tw-pddl",
                               "indexed": has_traj, "game_present": has_game, "installed": has_traj and has_game,
                               "trajectory_sha256": sha256_file(here / "traj_data.json") if has_traj else None,
                               "game_sha256": sha256_file(here / "game.tw-pddl") if has_game else None,
                               "declared_task_type": None, "problems": []}
        if not has_traj:
            row["problems"].append("orphan_game_without_trajectory")
        else:
            try:
                native = _resolve_task_family(json.loads((here / "traj_data.json").read_text(encoding="utf-8"))["task_type"])
                row["declared_task_type"] = native
                if native not in CANONICAL_FAMILIES:
                    row["problems"].append("unmapped_task_type")
            except (OSError, ValueError, KeyError, TypeError):
                row["problems"].append("malformed_trajectory_metadata")
        try:
            identity = parse_task_id(task_id)
            row.update({k: v for k, v in identity.to_dict().items() if k not in ("task_id", "split")})
            if identity.family is None:
                row["problems"].append("unsupported_task_type")
            row["templates"] = {name: template_key(identity, name) for name in sorted(TEMPLATE_DEFINITIONS)}
            if row["declared_task_type"] and row["declared_task_type"] != identity.task_type:
                row["problems"].append("task_type_differs_from_directory_name")
        except TaskIdentityError as exc:
            row["problems"].append(f"unparseable_task_id: {exc}")
        if not has_game and has_traj:
            row["problems"].append("missing_game_file")
        entries.append(row)
    entries.sort(key=lambda r: r["task_id"])
    return entries


_EXPECTED_WHEN_NOT_INSTALLED = {"missing_game_file", "unmapped_task_type", "unsupported_task_type"}


def _is_anomaly(row: dict[str, Any]) -> bool:
    expected = _EXPECTED_WHEN_NOT_INSTALLED if not row["installed"] else {"missing_game_file"}
    return any(p not in expected for p in row["problems"])


def _duplicates(entries: Sequence[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for row in entries:
        if row.get(field):
            groups[row[field]].append(row["task_id"])
    return [{"sha256": sha, "task_ids": ids} for sha, ids in sorted(groups.items()) if len(ids) > 1]


def _discovery_identity(entries: Sequence[dict[str, Any]]) -> str:
    """Identical to rq1.tasks.discovery._identity over installed, mapped records (sorted by task ID)."""
    records = [TaskRecord(r["task_id"], r["split"], r.get("family", ""), r["relative_dir"], r["trajectory_sha256"],
                          r["game_sha256"], 0) for r in entries if r["installed"] and "family" in r]
    return _identity(records)


def build_inventory(data_root: Path, splits: Sequence[str]) -> dict[str, Any]:
    root = Path(data_root).expanduser().resolve()
    if not (root / "json_2.1.1").is_dir():
        raise InventoryError(f"{root} does not contain json_2.1.1")
    unknown = [s for s in splits if s not in SPLITS]
    if unknown or not splits:
        raise InventoryError(f"splits must be a non-empty subset of {SPLITS}")
    per_split: dict[str, Any] = {}
    all_entries: list[dict[str, Any]] = []
    for split in sorted(set(splits), key=SPLITS.index):
        entries = _split_entries(root, split)
        installed = [r for r in entries if r["installed"]]
        per_split[split] = {
            "indexed_entries": sum(r["indexed"] for r in entries),
            "installed_runnable_games": len(installed),
            "missing_game_files": sum(1 for r in entries if r["indexed"] and not r["game_present"]),
            "orphan_games": sum(1 for r in entries if not r["indexed"]),
            # missing games (and unsupported ALFRED task types WITHOUT games) are expected categories, not anomalies
            "entries_with_problems": sum(1 for r in entries if _is_anomaly(r)),
            "installed_by_family": dict(sorted(Counter(r.get("family", "UNPARSED") for r in installed).items())),
            "installed_scenes": len({r.get("scene") for r in installed}),
            "installed_templates": {name: len({r["templates"][name] for r in installed if "templates" in r})
                                    for name in sorted(TEMPLATE_DEFINITIONS)},
            "duplicate_trajectory_hashes": _duplicates(entries, "trajectory_sha256"),
            "duplicate_game_hashes": _duplicates(entries, "game_sha256"),
            "discovery_identity": _discovery_identity(entries),
        }
        all_entries += entries
    cross = {"duplicate_trajectory_hashes_across_splits": [g for g in _duplicates(all_entries, "trajectory_sha256")
                                                          if len({t.split(':')[0] for t in g["task_ids"]}) > 1],
             "duplicate_game_hashes_across_splits": [g for g in _duplicates(all_entries, "game_sha256")
                                                    if len({t.split(':')[0] for t in g["task_ids"]}) > 1]}
    return {"inventory_schema": INVENTORY_SCHEMA, "data_root": str(root), "splits": per_split, "cross_split": cross,
            "entries": all_entries,
            "notes": ["indexed entries include trials without installed game files",
                      "distinct task IDs may share a template or identical files; see duplicate groups and templates",
                      "discovery_identity reproduces rq1.tasks.discovery data_root_identity for comparison"]}


def load_inventory(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("inventory_schema") != INVENTORY_SCHEMA or "entries" not in value:
        raise InventoryError(f"{path} is not a {INVENTORY_SCHEMA} file")
    return value
