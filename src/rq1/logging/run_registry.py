from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from contextlib import contextmanager
from collections.abc import Iterator

from rq1.utils.ids import new_attempt_id
from rq1.utils.time import utc_now


RUN_STATUSES = {"planned", "claimed", "running", "completed", "failed", "interrupted", "retry_planned", "excluded", "invalid"}


@dataclass(frozen=True)
class Run:
    run_id: str
    task_id: str
    split: str
    snapshot: str
    profile: str
    repetition: int
    status: str
    attempt_id: str | None = None


class RunRegistry:
    def __init__(self, database: Path) -> None:
        self.database = database

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        self.database.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.database, timeout=10, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        try:
            yield connection
        finally:
            connection.close()

    def initialize(self) -> None:
        with self.connection() as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY, task_id TEXT NOT NULL, split TEXT NOT NULL,
                    snapshot TEXT NOT NULL, profile TEXT NOT NULL, repetition INTEGER NOT NULL,
                    status TEXT NOT NULL, attempt_id TEXT, machine_id TEXT,
                    claimed_at TEXT, started_at TEXT, completed_at TEXT,
                    result_path TEXT, error_type TEXT, retry_of TEXT, exclusion_reason TEXT
                )
            """)

    def plan(self, run: Run) -> None:
        if run.status != "planned":
            raise ValueError("New runs must be planned")
        self.initialize()
        with self.connection() as connection:
            connection.execute(
                "INSERT INTO runs (run_id, task_id, split, snapshot, profile, repetition, status) VALUES (?, ?, ?, ?, ?, ?, 'planned')",
                (run.run_id, run.task_id, run.split, run.snapshot, run.profile, run.repetition),
            )

    def claim_next(self, machine_id: str) -> Run | None:
        """Atomically claim one planned run; safe for multiple local workers."""
        self.initialize()
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT * FROM runs WHERE status='planned' ORDER BY run_id LIMIT 1").fetchone()
            if row is None:
                connection.execute("COMMIT")
                return None
            attempt_id = new_attempt_id()
            result = connection.execute(
                "UPDATE runs SET status='claimed', attempt_id=?, machine_id=?, claimed_at=? WHERE run_id=? AND status='planned'",
                (attempt_id, machine_id, utc_now(), row["run_id"]),
            )
            if result.rowcount != 1:
                connection.execute("ROLLBACK")
                return None
            connection.execute("COMMIT")
            return Run(
                run_id=row["run_id"], task_id=row["task_id"], split=row["split"],
                snapshot=row["snapshot"], profile=row["profile"], repetition=row["repetition"],
                status="claimed", attempt_id=attempt_id,
            )

    def transition(self, run_id: str, expected: str, target: str, **fields: str | None) -> None:
        if target not in RUN_STATUSES:
            raise ValueError(f"Invalid run status: {target}")
        assignments = ["status=?"]
        values: list[str | None] = [target]
        for key, value in fields.items():
            if key not in {"result_path", "error_type", "exclusion_reason", "completed_at", "started_at"}:
                raise ValueError(f"Unsupported field: {key}")
            assignments.append(f"{key}=?")
            values.append(value)
        values.extend([run_id, expected])
        with self.connection() as connection:
            changed = connection.execute(
                f"UPDATE runs SET {', '.join(assignments)} WHERE run_id=? AND status=?", values
            )
        if changed.rowcount != 1:
            raise RuntimeError(f"Could not transition {run_id} from {expected} to {target}")

    def rows(self) -> list[sqlite3.Row]:
        self.initialize()
        with self.connection() as connection:
            return connection.execute("SELECT * FROM runs ORDER BY run_id").fetchall()
