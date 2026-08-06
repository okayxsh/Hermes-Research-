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


@dataclass(frozen=True)
class EpisodeBinding:
    run_id: str
    attempt_id: str
    episode_id: str
    session_id: str | None
    profile: str | None
    hermes_log_path: str | None
    bridge_log_path: str | None


@dataclass(frozen=True)
class PilotAttemptBinding:
    pilot_run_id: str
    pilot_test_id: str
    attempt_id: str
    mode: str
    status: str
    report_path: str


@dataclass(frozen=True)
class RecoveryEvidenceBinding:
    run_id: str
    attempt_id: str
    checkpoint_id: str | None
    perturbation_id: str | None
    profile: str | None
    snapshot: str | None
    recovery_log_path: str | None
    result_path: str | None


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
            connection.execute("""
                CREATE TABLE IF NOT EXISTS episode_bindings (
                    run_id TEXT NOT NULL, attempt_id TEXT NOT NULL, episode_id TEXT NOT NULL,
                    session_id TEXT, profile TEXT, hermes_log_path TEXT, bridge_log_path TEXT,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (run_id, attempt_id, episode_id)
                )
            """)
            connection.execute("""
                CREATE TABLE IF NOT EXISTS pilot_attempt_bindings (
                    pilot_run_id TEXT NOT NULL, pilot_test_id TEXT NOT NULL, attempt_id TEXT NOT NULL,
                    mode TEXT NOT NULL, status TEXT NOT NULL, report_path TEXT NOT NULL, created_at TEXT NOT NULL,
                    PRIMARY KEY (pilot_run_id, pilot_test_id, attempt_id)
                )
            """)
            connection.execute("""
                CREATE TABLE IF NOT EXISTS recovery_evidence_bindings (
                    run_id TEXT NOT NULL, attempt_id TEXT NOT NULL, checkpoint_id TEXT,
                    perturbation_id TEXT, profile TEXT, snapshot TEXT, recovery_log_path TEXT,
                    result_path TEXT, created_at TEXT NOT NULL,
                    PRIMARY KEY (run_id, attempt_id)
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

    def bind_episode(self, binding: EpisodeBinding) -> None:
        self.initialize()
        with self.connection() as connection:
            connection.execute(
                """INSERT OR REPLACE INTO episode_bindings
                (run_id, attempt_id, episode_id, session_id, profile, hermes_log_path, bridge_log_path, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    binding.run_id,
                    binding.attempt_id,
                    binding.episode_id,
                    binding.session_id,
                    binding.profile,
                    binding.hermes_log_path,
                    binding.bridge_log_path,
                    utc_now(),
                ),
            )

    def episode_bindings(self, run_id: str | None = None) -> list[sqlite3.Row]:
        self.initialize()
        with self.connection() as connection:
            if run_id is None:
                return connection.execute("SELECT * FROM episode_bindings ORDER BY created_at, episode_id").fetchall()
            return connection.execute(
                "SELECT * FROM episode_bindings WHERE run_id=? ORDER BY created_at, episode_id", (run_id,)
            ).fetchall()

    def bind_pilot_attempt(self, binding: PilotAttemptBinding) -> None:
        self.initialize()
        with self.connection() as connection:
            connection.execute(
                """INSERT INTO pilot_attempt_bindings
                (pilot_run_id, pilot_test_id, attempt_id, mode, status, report_path, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (binding.pilot_run_id, binding.pilot_test_id, binding.attempt_id, binding.mode,
                 binding.status, binding.report_path, utc_now()),
            )

    def pilot_attempt_bindings(self, pilot_run_id: str | None = None) -> list[sqlite3.Row]:
        self.initialize()
        with self.connection() as connection:
            if pilot_run_id is None:
                return connection.execute("SELECT * FROM pilot_attempt_bindings ORDER BY created_at").fetchall()
            return connection.execute(
                "SELECT * FROM pilot_attempt_bindings WHERE pilot_run_id=? ORDER BY created_at", (pilot_run_id,)
            ).fetchall()

    def bind_recovery_evidence(self, binding: RecoveryEvidenceBinding) -> None:
        self.initialize()
        with self.connection() as connection:
            connection.execute(
                """INSERT OR REPLACE INTO recovery_evidence_bindings
                (run_id, attempt_id, checkpoint_id, perturbation_id, profile, snapshot,
                 recovery_log_path, result_path, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (binding.run_id, binding.attempt_id, binding.checkpoint_id, binding.perturbation_id,
                 binding.profile, binding.snapshot, binding.recovery_log_path, binding.result_path, utc_now()),
            )

    def recovery_evidence_bindings(self, run_id: str | None = None) -> list[sqlite3.Row]:
        self.initialize()
        with self.connection() as connection:
            if run_id is None:
                return connection.execute("SELECT * FROM recovery_evidence_bindings ORDER BY created_at").fetchall()
            return connection.execute(
                "SELECT * FROM recovery_evidence_bindings WHERE run_id=? ORDER BY created_at", (run_id,)
            ).fetchall()
