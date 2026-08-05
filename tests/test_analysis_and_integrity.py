from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rq1.analysis.metrics import invalid_action_rate, retrieval_noise_rate, task_success_rate
from rq1.skills.integrity import validate_snapshot_manifest
from rq1.skills.leakage import find_leakage


class AnalysisAndIntegrityTests(unittest.TestCase):
    def test_metrics(self) -> None:
        self.assertEqual(0.5, task_success_rate([{"success": True}, {"success": False}]))
        self.assertEqual(0.5, invalid_action_rate([{"action_valid": True}, {"action_valid": False}]))
        self.assertEqual(0.5, retrieval_noise_rate([{"event": "skill_view", "relevant": True}, {"event": "skill_view", "relevant": False}]))

    def test_snapshot_and_leakage(self) -> None:
        manifest = {"snapshot_id": "L0", "skill_count": 0, "skills": [], "directory_sha256": "a", "created_from_git_commit": "b"}
        self.assertEqual([], validate_snapshot_manifest(manifest))
        self.assertIn("task_id", find_leakage("Use valid_unseen_42 only"))
        self.assertIn("room_number", find_leakage("Go to kitchen 3"))

    def test_schemas_are_json(self) -> None:
        schema_root = Path(__file__).resolve().parents[1] / "data" / "schemas"
        schemas = list(schema_root.glob("*.json"))
        self.assertEqual(8, len(schemas))
        for schema in schemas:
            self.assertIsInstance(json.loads(schema.read_text(encoding="utf-8")), dict)
