"""Tests for the single-retrieval boundary, injection, and logging."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rq1.retrieval import (
    RetrievalBoundaryError,
    RetrievalContext,
    RetrievalEventLog,
    RetrievalQuery,
    build_retrieval_boundary,
)


class DictEmbedder:
    def __init__(self, table: dict[str, list[float]], default: list[float] | None = None) -> None:
        self.table = table
        self.default = default or [0.0, 0.0, 0.0]

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [self.table.get(text, list(self.default)) for text in texts]


def _skills() -> list[tuple[str, str]]:
    return [
        ("heat", "TITLE: heat object\nBODY: use the microwave"),
        ("clean", "TITLE: clean object\nBODY: use a cloth and sink"),
        ("cool", "TITLE: cool object\nBODY: put it in the fridge"),
        ("look", "TITLE: examine object\nBODY: pick it up and look"),
    ]


def _query() -> RetrievalQuery:
    return RetrievalQuery(
        task_instruction="heat the mug and put it on the counter",
        observation="You are in the kitchen. The mug is not on the counter.",
        inventory=("mug",),
    )


def _boundary() -> object:
    skills = _skills()
    embedder = DictEmbedder(
        {
            "TITLE: heat object\nBODY: use the microwave": [1.0, 0.0],
            "TITLE: clean object\nBODY: use a cloth and sink": [0.0, 1.0],
            "TITLE: cool object\nBODY: put it in the fridge": [0.9, 0.1],
            "TITLE: examine object\nBODY: pick it up and look": [0.8, 0.2],
            _query().text(): [1.0, 0.0],
        }
    )
    return build_retrieval_boundary(
        skills, embedder, embedding_model="all-mpnet-base-v2", top_k=3
    )


class BoundaryTests(unittest.TestCase):
    def test_top3_memory_exposes_rank_id_text_only(self) -> None:
        outcome = _boundary().retrieve(_context(), _query())
        payload = outcome.recovery_memory.to_dict()["recovery_memory"]
        self.assertEqual(len(payload["retrieved_skills"]), 3)
        self.assertFalse(payload["no_retrieved_skills_available"])
        self.assertEqual(payload["retrieved_skills"][0]["skill_id"], "heat")
        self.assertEqual(payload["retrieved_skills"][0]["rank"], 1)
        self.assertIn("text", payload["retrieved_skills"][0])
        # Scores must never reach the agent.
        self.assertNotIn("score", payload["retrieved_skills"][0])

    def test_no_lib_yields_explicit_no_retrieval_marker(self) -> None:
        boundary = build_retrieval_boundary([], DictEmbedder({}), embedding_model="all-mpnet-base-v2")
        outcome = boundary.retrieve(_context("NoLib", 0, None), _query())
        payload = outcome.recovery_memory.to_dict()["recovery_memory"]
        self.assertEqual(payload["retrieved_skills"], [])
        self.assertTrue(payload["no_retrieved_skills_available"])
        self.assertFalse(outcome.result.retrieved)

    def test_retrieval_runs_exactly_once(self) -> None:
        boundary = _boundary()
        boundary.retrieve(_context(), _query())
        with self.assertRaises(RetrievalBoundaryError):
            boundary.retrieve(_context(), _query())

    def test_event_records_audit_fields(self) -> None:
        outcome = _boundary().retrieve(_context(), _query())
        event = outcome.event.to_dict()
        self.assertEqual(event["condition"], "Accum-60")
        self.assertEqual(event["library_size"], 4)
        self.assertEqual(event["embedding_model"], "all-mpnet-base-v2")
        self.assertEqual(event["retrieved"], True)
        self.assertEqual(event["no_retrieval"], False)
        self.assertEqual(len(event["ranking"]), 4)
        self.assertEqual(len(event["top"]), 3)
        self.assertEqual(len(event["query_text_hash"]), 64)
        self.assertEqual(len(event["query_template_hash"]), 64)

    def test_event_log_appends_valid_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "retrieval.jsonl"
            log = RetrievalEventLog(path)
            outcome = _boundary().retrieve(_context(), _query())
            log.append(outcome.event)
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)
            record = json.loads(lines[0])
            self.assertEqual(record["condition"], "Accum-60")
            self.assertEqual(record["schema_version"], 1)


def _context(condition: str = "Accum-60", size: int = 4, library_hash: str | None = "abc123") -> RetrievalContext:
    return RetrievalContext(
        run_id="run-1",
        attempt_id="att-1",
        task_id="valid_unseen_042",
        condition=condition,
        library_name=condition,
        library_size=size,
        library_hash=library_hash,
        episode_id="ep-1",
    )


if __name__ == "__main__":
    unittest.main()
