"""Structural checks for the blocked pilot-readiness document.

These assert only document/table shape against the canonical family set; they say
nothing about ALFWorld data, task validity, or recovery success.
"""
from __future__ import annotations

import re
import unittest
from pathlib import Path

from rq1.tasks.discovery import CANONICAL_FAMILIES

DOC = Path(__file__).resolve().parents[1] / "university-gpu" / "docs" / "PILOT_READINESS.md"
FAMILY_ROW = re.compile(r"^\|\s*`([a-z_]+)`\s*\|")
EXPECTED_CELLS = (
    "TO BE SELECTED",
    "NOT AVAILABLE",
    "NOT AVAILABLE",
    "NOT AVAILABLE",
    "REQUIRES REAL VALIDATION",
)


class PilotReadinessDocTests(unittest.TestCase):
    def setUp(self) -> None:
        self.assertTrue(DOC.is_file(), f"missing {DOC}")
        self.text = DOC.read_text(encoding="utf-8")

    def _family_rows(self) -> list[list[str]]:
        rows = []
        for line in self.text.splitlines():
            if FAMILY_ROW.match(line):
                rows.append([cell.strip() for cell in line.strip().strip("|").split("|")])
        return rows

    def test_family_table_matches_canonical_families(self) -> None:
        rows = self._family_rows()
        names = [row[0].strip("`") for row in rows]
        self.assertEqual(len(names), 6, f"expected six family rows, got {names}")
        self.assertEqual(set(names), set(CANONICAL_FAMILIES.values()))
        self.assertEqual(len(set(names)), len(names), "duplicate family rows")

    def test_no_row_claims_selection_or_validation(self) -> None:
        rows = self._family_rows()
        self.assertTrue(rows)
        for row in rows:
            self.assertEqual(len(row), 6, f"unexpected column count: {row}")
            self.assertEqual(tuple(row[1:]), EXPECTED_CELLS, row)

    def test_status_is_blocked_and_not_validated(self) -> None:
        self.assertIn("**BLOCKED.**", self.text)
        self.assertIn("Not produced and not validated", self.text)
        self.assertIn("| result | |", self.text)


if __name__ == "__main__":
    unittest.main()
