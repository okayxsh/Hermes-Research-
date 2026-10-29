"""Offline tests for the read-only rating-package validator.

Every fixture here is synthetic and lives in a temporary directory that is deleted when the
test ends. The real rater files are never opened, copied, or written by these tests.
"""
from __future__ import annotations

import csv
import importlib.util
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "university-gpu" / "validate_rating_package.py"
_SPEC = importlib.util.spec_from_file_location("validate_rating_package", MODULE_PATH)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover - defensive
    raise RuntimeError(f"cannot load {MODULE_PATH}")
validator = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = validator
_SPEC.loader.exec_module(validator)

RATER_HEADER = ["task_id", "skill_id", "label"]
ADJUDICATION_HEADER = ["task_id", "skill_id", "adjudicated"]


def write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def run_cli(yash: Path, hani: Path, adjudication: Path) -> tuple[int, str]:
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        code = validator.main([
            "--yash", str(yash),
            "--hani", str(hani),
            "--adjudication", str(adjudication),
        ])
    return code, buffer.getvalue()


def blank_rows(tasks: int, skills: int) -> list[list[str]]:
    return [
        [f"fake-task-{task:02d}", f"fake-skill-{skill:02d}", ""]
        for task in range(tasks)
        for skill in range(skills)
    ]


def filled_rows(tasks: int, skills: int) -> list[list[str]]:
    labels = ["RELEVANT", "IRRELEVANT"]
    return [
        [f"fake-task-{task:02d}", f"fake-skill-{skill:02d}", labels[(task + skill) % 2]]
        for task in range(tasks)
        for skill in range(skills)
    ]


class ValidateRatingPackageTests(unittest.TestCase):
    def test_blank_package_reports_every_missing_judgment(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            yash, hani, adjudication = root / "a.csv", root / "b.csv", root / "final.csv"
            write_csv(yash, RATER_HEADER, blank_rows(30, 18))
            write_csv(hani, RATER_HEADER, blank_rows(30, 18))
            write_csv(adjudication, ADJUDICATION_HEADER, blank_rows(30, 18))

            code, output = run_cli(yash, hani, adjudication)

            self.assertNotEqual(code, 0)
            self.assertIn("outstanding ratings: 1620", output)
            report = validator.validate(yash, hani, adjudication)
            self.assertFalse(report["ok"])
            self.assertEqual(report["files"]["Yash (rater A)"]["missing"], 540)
            self.assertEqual(report["files"]["Hani (rater B)"]["missing"], 540)
            self.assertEqual(report["files"]["Adjudication"]["missing"], 540)
            self.assertEqual(report["files"]["Yash (rater A)"]["unique_pairs"], 540)

    def test_complete_valid_fixture_is_ready_and_untouched(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            yash, hani, adjudication = root / "a.csv", root / "b.csv", root / "final.csv"
            rows = filled_rows(3, 2)
            write_csv(yash, RATER_HEADER, rows)
            write_csv(hani, RATER_HEADER, rows)
            write_csv(adjudication, ADJUDICATION_HEADER, rows)
            before = {path: path.read_bytes() for path in (yash, hani, adjudication)}

            code, output = run_cli(yash, hani, adjudication)

            self.assertEqual(code, 0)
            self.assertIn("outstanding ratings: 0", output)
            self.assertIn("disagreements between Yash and Hani: 0", output)
            self.assertIn("status: READY", output)
            for path, content in before.items():
                self.assertEqual(path.read_bytes(), content, f"validator modified {path}")

    def test_duplicate_key_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            yash, hani, adjudication = root / "a.csv", root / "b.csv", root / "final.csv"
            rows = filled_rows(3, 2)
            write_csv(yash, RATER_HEADER, rows + [rows[0]])
            write_csv(hani, RATER_HEADER, rows)
            write_csv(adjudication, ADJUDICATION_HEADER, rows)

            code, output = run_cli(yash, hani, adjudication)

            self.assertNotEqual(code, 0)
            self.assertIn("duplicates task_id/skill_id", output)
            self.assertIn("duplicate(s)", output)

    def test_invalid_label_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            yash, hani, adjudication = root / "a.csv", root / "b.csv", root / "final.csv"
            rows = filled_rows(3, 2)
            rows[0][2] = "MAYBE"
            write_csv(yash, RATER_HEADER, rows)
            write_csv(hani, RATER_HEADER, filled_rows(3, 2))
            write_csv(adjudication, ADJUDICATION_HEADER, filled_rows(3, 2))

            code, output = run_cli(yash, hani, adjudication)

            self.assertNotEqual(code, 0)
            self.assertIn("is not RELEVANT or IRRELEVANT", output)

    def test_mismatched_pairs_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            yash, hani, adjudication = root / "a.csv", root / "b.csv", root / "final.csv"
            write_csv(yash, RATER_HEADER, filled_rows(3, 2))
            write_csv(hani, RATER_HEADER, filled_rows(3, 2)[:-1])
            write_csv(adjudication, ADJUDICATION_HEADER, filled_rows(3, 2))

            code, output = run_cli(yash, hani, adjudication)

            self.assertNotEqual(code, 0)
            self.assertIn("missing", output)


if __name__ == "__main__":
    unittest.main()
