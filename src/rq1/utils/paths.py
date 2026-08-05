from __future__ import annotations

from pathlib import Path


def repository_root(start: Path | None = None) -> Path:
    path = (start or Path.cwd()).resolve()
    for candidate in (path, *path.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise FileNotFoundError("Could not locate repository root (pyproject.toml missing).")
