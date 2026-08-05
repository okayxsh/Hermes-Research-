from __future__ import annotations

from collections.abc import Mapping


REQUIRED_SNAPSHOT_FIELDS = {"snapshot_id", "skill_count", "skills", "directory_sha256", "created_from_git_commit"}


def validate_snapshot_manifest(manifest: Mapping[str, object]) -> list[str]:
    errors = [f"Missing field: {field}" for field in sorted(REQUIRED_SNAPSHOT_FIELDS - set(manifest))]
    skills = manifest.get("skills")
    if not isinstance(skills, list):
        errors.append("skills must be a list")
    elif manifest.get("skill_count") != len(skills):
        errors.append("skill_count does not match number of skills")
    return errors
