from __future__ import annotations

import os
from pathlib import Path


class StageLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.acquired = False

    def __enter__(self) -> "StageLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise RuntimeError(f"Stage lock already exists: {self.path}") from exc
        os.write(descriptor, str(os.getpid()).encode("utf-8"))
        os.close(descriptor)
        self.acquired = True
        return self

    def __exit__(self, *_: object) -> None:
        if self.acquired:
            self.path.unlink(missing_ok=True)
