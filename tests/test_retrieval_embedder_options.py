from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from rq1.retrieval.embedder import SentenceBERTEmbedder


class SentenceBERTEmbedderOptionTests(unittest.TestCase):
    def test_forwards_explicit_frozen_cache_controls(self) -> None:
        captured: dict[str, object] = {}

        class FakeSentenceTransformer:
            def __init__(self, model_name: str, **kwargs: object) -> None:
                captured["model_name"] = model_name
                captured.update(kwargs)

        fake_module = types.ModuleType("sentence_transformers")
        fake_module.SentenceTransformer = FakeSentenceTransformer
        with patch.dict(sys.modules, {"sentence_transformers": fake_module}):
            SentenceBERTEmbedder(
                "sentence-transformers/all-mpnet-base-v2",
                cache_folder=Path("/persistent/frozen-hf-cache"),
                revision="frozen-revision",
                local_files_only=True,
            )

        self.assertEqual("sentence-transformers/all-mpnet-base-v2", captured["model_name"])
        self.assertEqual("/persistent/frozen-hf-cache", captured["cache_folder"])
        self.assertEqual("frozen-revision", captured["revision"])
        self.assertIs(captured["local_files_only"], True)


if __name__ == "__main__":
    unittest.main()
