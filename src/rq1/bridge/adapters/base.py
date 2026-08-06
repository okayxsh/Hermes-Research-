"""Typed local contracts used by version-specific adapter implementations."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


class RealALFWorldUnavailable(RuntimeError):
    """A required installed ALFWorld capability is unavailable or unsupported."""


@dataclass(frozen=True)
class IndexedTask:
    task_id: str
    split: str
    task_family: str
    source_path: Path
    game_file: Path
    source_sha256: str
    game_sha256: str
    data_identity: str

    def to_dict(self) -> dict[str, object]:
        return {
            "task_id": self.task_id, "split": self.split, "task_family": self.task_family,
            "source_path": "$ALFWORLD_DATA/" + self.source_path.as_posix(),
            "game_file": "$ALFWORLD_DATA/" + self.game_file.as_posix(),
            "source_sha256": self.source_sha256, "game_sha256": self.game_sha256,
            "data_identity": self.data_identity,
        }
