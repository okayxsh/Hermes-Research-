"""Tests for the dependency-free Sentence-BERT top-3 retrieval subsystem."""
from __future__ import annotations

import unittest

from rq1.retrieval import (
    CANONICAL_FAILURE_MESSAGE,
    EMPTY_INVENTORY_MARKER,
    QUERY_TEMPLATE_VERSION,
    RetrievalCandidate,
    RetrievalQuery,
    RetrievalResult,
    TopKRetriever,
    build_query_text,
    build_skill_text,
    cosine_similarity,
    query_template_hash,
    rank_candidates,
    skill_text_hash,
)


class DictEmbedder:
    """Deterministic fake embedder that resolves text to a fixed vector."""

    def __init__(self, table: dict[str, list[float]], default: list[float] | None = None) -> None:
        self.table = table
        self.default = default or [0.0, 0.0, 0.0]

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [self.table.get(text, list(self.default)) for text in texts]


class CosineSimilarityTests(unittest.TestCase):
    def test_identical_vectors_are_one(self) -> None:
        self.assertAlmostEqual(cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]), 1.0)

    def test_orthogonal_vectors_are_zero(self) -> None:
        self.assertAlmostEqual(cosine_similarity([1.0, 0.0], [0.0, 1.0]), 0.0)

    def test_dimension_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            cosine_similarity([1.0, 0.0], [1.0])

    def test_zero_vector_is_zero(self) -> None:
        self.assertEqual(cosine_similarity([0.0, 0.0], [1.0, 1.0]), 0.0)


class RankingTests(unittest.TestCase):
    def test_rank_candidates_orders_by_score_desc_then_id(self) -> None:
        query = [1.0, 0.0]
        embeddings = {"b": [0.5, 0.0], "a": [1.0, 0.0], "c": [0.5, 0.0]}
        result = rank_candidates(query, embeddings)
        self.assertEqual([item.skill_id for item in result], ["a", "b", "c"])

    def test_top_k_retriever_returns_top_three(self) -> None:
        query = RetrievalQuery(task_instruction="heat a mug", observation="in kitchen")
        skills = [("s1", "TITLE: heat\nBODY: put in microwave"), ("s2", "TITLE: clean\nBODY: use cloth")]
        embedder = DictEmbedder(
            {
                "TITLE: heat\nBODY: put in microwave": [1.0, 0.0],
                "TITLE: clean\nBODY: use cloth": [0.0, 1.0],
                query.text(): [1.0, 0.0],
            }
        )
        retriever = TopKRetriever(embedder, skills)
        result = retriever.retrieve(query, top_k=3)
        self.assertTrue(result.retrieved)
        self.assertEqual(len(result.ranking), 2)
        self.assertEqual(result.top[0].skill_id, "s1")

    def test_empty_library_is_no_retrieval(self) -> None:
        query = RetrievalQuery(task_instruction="heat a mug", observation="in kitchen")
        retriever = TopKRetriever(DictEmbedder({}), [])
        result = retriever.retrieve(query, top_k=3)
        self.assertFalse(result.retrieved)
        self.assertEqual(result.ranking, ())
        self.assertEqual(result.library_size, 0)


class QueryTextTests(unittest.TestCase):
    def test_query_text_uses_frozen_order(self) -> None:
        text = build_query_text(
            task_instruction="heat the mug",
            observation="You are in the kitchen.",
            inventory=("mug",),
            failure_message=CANONICAL_FAILURE_MESSAGE,
        )
        self.assertEqual(
            text,
            "TASK:\nheat the mug\nOBSERVATION:\nYou are in the kitchen.\n"
            f"INVENTORY:\nmug\nFAILURE:\n{CANONICAL_FAILURE_MESSAGE}",
        )

    def test_empty_inventory_uses_marker(self) -> None:
        text = build_query_text(
            task_instruction="heat the mug",
            observation="kitchen",
            inventory=(),
        )
        self.assertIn(f"INVENTORY:\n{EMPTY_INVENTORY_MARKER}", text)

    def test_query_text_normalizes_whitespace(self) -> None:
        text = build_query_text(
            task_instruction="  heat   the \n mug  ",
            observation=" kitchen ",
            inventory=("mug",),
        )
        self.assertTrue(text.startswith("TASK:\nheat the mug\nOBSERVATION:\nkitchen\n"))

    def test_query_template_hash_is_stable(self) -> None:
        self.assertEqual(query_template_hash(), query_template_hash())
        self.assertEqual(len(query_template_hash()), 64)


class SkillTextTests(unittest.TestCase):
    def test_skill_text_uses_title_and_body_only(self) -> None:
        text = build_skill_text(title="  Heat object ", body=" Put it in the microwave. ")
        self.assertEqual(text, "TITLE: Heat object\nBODY: Put it in the microwave.")

    def test_skill_text_hash_is_stable(self) -> None:
        text = build_skill_text(title="Heat", body="microwave")
        self.assertEqual(skill_text_hash(text), skill_text_hash(text))


class RetrievalResultTests(unittest.TestCase):
    def test_top_slices_ranking(self) -> None:
        query = RetrievalQuery(task_instruction="x", observation="y")
        ranking = (RetrievalCandidate("a", 0.9), RetrievalCandidate("b", 0.8), RetrievalCandidate("c", 0.7), RetrievalCandidate("d", 0.6))
        result = RetrievalResult(query=query, ranking=ranking, top_k=3, retrieved=True)
        self.assertEqual([item.skill_id for item in result.top], ["a", "b", "c"])

    def test_to_dict_exposes_full_ranking_and_top(self) -> None:
        query = RetrievalQuery(task_instruction="x", observation="y")
        result = RetrievalResult(query=query, ranking=(RetrievalCandidate("a", 0.9),), top_k=3, retrieved=True)
        payload = result.to_dict()
        self.assertIn("ranking", payload)
        self.assertIn("top", payload)
        self.assertEqual(payload["query"]["version"], QUERY_TEMPLATE_VERSION)


if __name__ == "__main__":
    unittest.main()
