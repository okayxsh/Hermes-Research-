"""Metadata-only discovery. It never reads instructions, trajectories, or outcomes."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
from rq1.bridge.adapters.task_index import TASK_TYPES, _sha256
from rq1.tasks.models import DiscoveryResult, TaskRecord

CANONICAL_FAMILIES = {
    "pick_and_place_simple": "pick_and_place", "pick_two_obj_and_place": "pick_two_and_place",
    "look_at_obj_in_light": "look_at_object", "pick_clean_then_place_in_recep": "clean_and_place",
    "pick_heat_then_place_in_recep": "heat_and_place", "pick_cool_then_place_in_recep": "cool_and_place",
}

class TaskDiscoveryError(ValueError): pass

def _identity(records: list[TaskRecord]) -> str:
    text = "".join(f"{r.task_id}\0{r.source_sha256}\0{r.game_sha256 or ''}\n" for r in records)
    return hashlib.sha256(text.encode()).hexdigest()

def discover_tasks(data_root: Path, split: str, *, allow_unseen_metadata: bool = False) -> DiscoveryResult:
    if split not in {"train", "valid_seen", "valid_unseen"}: raise TaskDiscoveryError("unsupported ALFWorld split")
    if split == "valid_unseen" and not allow_unseen_metadata: raise TaskDiscoveryError("valid_unseen metadata discovery is blocked before the final evaluation gate")
    root = data_root.expanduser().resolve(); split_root = root / "json_2.1.1" / split
    if not split_root.is_dir(): raise TaskDiscoveryError(f"missing ALFWorld split directory: {split}")
    records: list[TaskRecord] = []; exclusions: list[dict[str, str]] = []; errors: list[str] = []
    for source in sorted(split_root.rglob("traj_data.json")):
        game = source.with_name("game.tw-pddl")
        if not game.is_file(): exclusions.append({"source": source.relative_to(root).as_posix(), "reason": "missing_game_file"}); continue
        try:
            payload = json.loads(source.read_text(encoding="utf-8")); native = TASK_TYPES[payload["task_type"]]; family = CANONICAL_FAMILIES[native]
        except (OSError, ValueError, KeyError, TypeError): errors.append(source.relative_to(root).as_posix()); continue
        relative = source.parent.relative_to(split_root).as_posix()
        if not relative or relative.startswith("../"): errors.append(source.relative_to(root).as_posix()); continue
        records.append(TaskRecord(f"{split}:{relative}", split, family, relative, _sha256(source), _sha256(game), 0))
    if errors: raise TaskDiscoveryError("malformed or unmapped task metadata: " + ", ".join(errors))
    if len({item.task_id for item in records}) != len(records): raise TaskDiscoveryError("duplicate stable task IDs")
    records.sort(key=lambda item: item.task_id)
    ordered = tuple(TaskRecord(**{**item.to_dict(), "order_index": index}) for index, item in enumerate(records, 1))
    return DiscoveryResult(1, _identity(list(ordered)), split, ordered, tuple(exclusions), ())
