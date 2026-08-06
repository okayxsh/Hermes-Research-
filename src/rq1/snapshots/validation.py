from __future__ import annotations
from rq1.snapshots.models import SnapshotManifest
def validate_snapshot_chain(manifests: list[SnapshotManifest]) -> list[str]:
    errors: list[str] = []; previous: set[str] = set(); parent = None
    for index, manifest in enumerate(manifests):
        skills = {str(item.get("skill_id")) for item in manifest.skills}
        if index == 0 and (manifest.snapshot_id != "L0" or manifest.skill_count != 0): errors.append("L0 must contain no learned skills")
        if not previous.issubset(skills): errors.append("snapshots are not nested chronologically")
        if manifest.parent_snapshot_sha256 != parent: errors.append("parent snapshot hash mismatch")
        if manifest.skill_count != len(manifest.skills): errors.append("skill count mismatch")
        previous, parent = skills, manifest.directory_sha256
    return errors
