"""Append-only retrieval-event logging for reproducible audit.

The agent receives only the top-3 text; the complete ranking and scores are
logged here for analysis/audit only.
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rq1.retrieval.models import RetrievalResult
from rq1.retrieval.query import query_template_hash
from rq1.utils.time import utc_now

RETRIEVAL_EVENT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RetrievalEvent:
    run_id: str
    attempt_id: str
    task_id: str
    condition: str
    library_name: str
    library_size: int
    library_hash: str | None
    embedding_model: str
    result: RetrievalResult
    episode_id: str | None = None
    embedding_model_revision: str | None = None
    embedding_model_file_hashes: dict[str, str] = field(default_factory=dict)
    event_id: str | None = None
    timestamp: str | None = None

    def to_dict(self) -> dict[str, Any]:
        query = self.result.query
        return {
            "schema_version": RETRIEVAL_EVENT_SCHEMA_VERSION,
            "event_id": self.event_id or "",
            "timestamp": self.timestamp or utc_now(),
            "run_id": self.run_id,
            "attempt_id": self.attempt_id,
            "task_id": self.task_id,
            "episode_id": self.episode_id,
            "condition": self.condition,
            "library_name": self.library_name,
            "library_size": self.library_size,
            "library_hash": self.library_hash,
            "query_version": query.version,
            "query_text_hash": query.text_hash(),
            "query_template_hash": query_template_hash(),
            "embedding_model": self.embedding_model,
            "embedding_model_revision": self.embedding_model_revision,
            "embedding_model_file_hashes": self.embedding_model_file_hashes,
            "retrieved": self.result.retrieved,
            "no_retrieval": not self.result.retrieved,
            "top_k": self.result.top_k,
            "ranking": [item.to_dict() for item in self.result.ranking],
            "top": [item.to_dict() for item in self.result.top],
        }


class RetrievalEventLog:
    """Small append-only JSONL writer for retrieval events."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()

    def append(self, event: RetrievalEvent) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event.to_dict(), sort_keys=True) + "\n")
