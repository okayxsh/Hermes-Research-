from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    pass


def load_json_yaml(path: Path) -> dict[str, Any]:
    """Load JSON-compatible YAML without adding an unverified runtime dependency."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"{path}: configuration must be JSON-compatible YAML: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigError(f"{path}: root must be an object")
    return value


def validate_config_tree(config_root: Path) -> list[str]:
    errors: list[str] = []
    for path in sorted(config_root.rglob("*.yaml")):
        try:
            load_json_yaml(path)
        except ConfigError as exc:
            errors.append(str(exc))
    return errors
