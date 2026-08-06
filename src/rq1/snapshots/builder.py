from __future__ import annotations
import hashlib, json
from pathlib import Path
from rq1.acquisition.models import SkillOperation
from rq1.snapshots.models import SnapshotManifest

def _hash(value: object) -> str: return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

def build_snapshots(*, acquisition_run: str, operations: list[SkillOperation], cutoffs: list[tuple[str, int]], commit: str, destination: Path) -> list[SnapshotManifest]:
    ordered = sorted(operations, key=lambda value: value.operation_index)
    if [item.operation_index for item in ordered] != list(range(1, len(ordered) + 1)): raise ValueError("acquisition operations are not chronological")
    if not cutoffs or cutoffs[0] != ("L0", 0): raise ValueError("snapshot policy must start with L0 cutoff 0")
    result: list[SnapshotManifest] = []; parent: str | None = None
    for snapshot_id, cutoff in cutoffs:
        if cutoff < 0 or cutoff > len(ordered): raise ValueError("snapshot cutoff is outside acquisition history")
        selected = tuple(item.to_dict() for item in ordered[:cutoff])
        payload = {"skills": selected, "cutoff": cutoff}
        digest = _hash(payload)
        path = destination / snapshot_id
        if path.exists(): raise FileExistsError(f"immutable snapshot already exists: {path}")
        manifest = SnapshotManifest(1, snapshot_id, acquisition_run, cutoff, selected, len(selected), digest, parent, commit)
        path.mkdir(parents=True); (path / "manifest.json").write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True)+'\n', encoding='utf-8')
        parent = digest; result.append(manifest)
    return result
