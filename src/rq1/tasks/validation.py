from __future__ import annotations
import hashlib, json
from collections import Counter
from typing import Iterable
from rq1.tasks.models import ManifestState, TaskManifest

def manifest_hash(payload: dict) -> str:
    value = dict(payload); value.pop("manifest_sha256", None)
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

def validate_manifest(manifest: TaskManifest, *, require_frozen: bool = False) -> list[str]:
    errors: list[str] = []
    if manifest.status not in {item.value for item in ManifestState}: errors.append("invalid manifest status")
    if require_frozen and manifest.status != ManifestState.FROZEN.value: errors.append("manifest is not frozen")
    if manifest.manifest_type not in {"pilot", "acquisition", "evaluation"}: errors.append("invalid manifest type")
    expected = {"pilot": "valid_seen", "acquisition": "train", "evaluation": "valid_unseen"}.get(manifest.manifest_type)
    if manifest.split != expected: errors.append("manifest type/split mismatch")
    if manifest.actual_count != len(manifest.tasks): errors.append("actual_count mismatch")
    if [item.order_index for item in manifest.tasks] != list(range(1, len(manifest.tasks) + 1)): errors.append("task order indices are not contiguous")
    if any(item.split != manifest.split for item in manifest.tasks): errors.append("wrong-split task record")
    if len({item.task_id for item in manifest.tasks}) != len(manifest.tasks): errors.append("duplicate task IDs")
    if len({item.source_identifier for item in manifest.tasks}) != len(manifest.tasks): errors.append("duplicate source identifiers")
    if len({item.game_sha256 for item in manifest.tasks if item.game_sha256}) != len([item for item in manifest.tasks if item.game_sha256]): errors.append("duplicate game identities")
    if manifest.family_counts != dict(sorted(Counter(item.family for item in manifest.tasks).items())): errors.append("family counts mismatch")
    if manifest.manifest_sha256 != manifest_hash(manifest.to_dict()): errors.append("manifest content hash mismatch")
    return errors

def overlap_errors(manifests: Iterable[TaskManifest]) -> list[str]:
    seen: dict[str, str] = {}; sources: dict[str, str] = {}; games: dict[str, str] = {}; errors: list[str] = []
    for manifest in manifests:
        for task in manifest.tasks:
            for value, table, label in ((task.task_id, seen, "task ID"), (task.source_identifier, sources, "source"), (task.game_sha256, games, "game")):
                if value and value in table and table[value] != manifest.manifest_type: errors.append(f"cross-manifest {label} overlap: {value}")
                elif value: table[value] = manifest.manifest_type
    return errors
