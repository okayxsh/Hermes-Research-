"""Offline tests for the read-only scientific-prerequisite checker.

Every fixture is synthetic and lives in a temporary directory. The real rating
package, approval templates, task lists, and readiness documents are never
opened or written by these tests.
"""
from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "university-gpu"))
import check_scientific_prerequisites as checker


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def blank_root(root: Path) -> None:
    """Create a checkout with blank/unfrozen scientific inputs only."""
    write_json(
        root / "university-gpu" / "approval-templates" / "pilot-manifest.json",
        {"approved": False, "approved_by": "", "approval_reference": "", "manifest_sha256": ""},
    )
    write_json(
        root / "data" / "task_lists" / "pilot_seen.json",
        {"status": "TO_BE_FROZEN_BY_PILOT", "split": "valid_seen", "tasks": []},
    )
    write_json(
        root / "data" / "task_lists" / "evaluation_unseen.json",
        {"status": "TO_BE_FROZEN_BEFORE_EVALUATION", "split": "valid_unseen", "tasks": []},
    )
    package = root / "rating-package"
    package.mkdir(parents=True, exist_ok=True)
    (package / "historical-reuse-decision.md").write_text(
        "# Historical rating reuse decision (blank template)\n\n"
        "- [ ] **Option 1**\n- [ ] **Option 2**\n- [ ] **Option 3**\n\n"
        "- Selected option (1 / 2 / 3): ______\n"
        "- Rationale: ____________________________________________________________\n"
        "- Author 1 name: ______________________  Signature: ______________________  Date (UTC): __________\n"
        "- Author 2 name: ______________________  Signature: ______________________  Date (UTC): __________\n"
        "- Adjudicator acknowledgement: ______________________  Date (UTC): __________\n",
        encoding="utf-8",
    )


def run_json(root: Path, package: Path) -> tuple[int, dict[str, object]]:
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        code = checker.main(
            ["--json", "--root", str(root), "--rating-package", str(package)]
        )
    return code, json.loads(buffer.getvalue())


class BlankStateTests(unittest.TestCase):
    def test_blank_state_exits_one_and_never_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "repo"
            blank_root(root)
            package = root / "rating-package"
            code, report = run_json(root, package)

            self.assertEqual(code, 1)
            self.assertEqual(report["status"], "BLOCKED")
            requirements = report["requirements"]
            self.assertEqual(set(requirements), set(checker.REQUIREMENT_KEYS))
            self.assertTrue(all(item["status"] != "PASS" for item in requirements.values()))

            blocked = {
                "pilot_manifest_approval",
                "pilot_relevance_completeness",
                "historical_reuse_decision",
                "gate_calibration",
                "neutral_filler",
                "tokenizer",
                "leakage",
                "historical_baseline_compatibility",
            }
            for key in blocked:
                self.assertEqual(requirements[key]["status"], "BLOCKED", key)
                self.assertTrue(requirements[key]["reason"].strip(), key)
                self.assertTrue(requirements[key]["path"].strip(), key)
            self.assertEqual(requirements["sample_size_precision"]["status"], "NOT_IMPLEMENTED")

    def test_empty_pilot_list_is_never_pass(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "repo"
            blank_root(root)
            finding = checker.check_leakage(root)
            self.assertEqual(finding["status"], "BLOCKED")
            self.assertIn("no pilot tasks frozen", finding["reason"])

    def test_pilot_list_without_frozen_evaluation_ids_is_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "repo"
            blank_root(root)
            write_json(
                root / "data" / "task_lists" / "pilot_seen.json",
                {"status": "frozen", "split": "valid_seen", "tasks": ["valid_seen:pick_and_place:1"]},
            )
            finding = checker.check_leakage(root)
            self.assertEqual(finding["status"], "BLOCKED")
            self.assertIn("leakage cannot be verified", finding["reason"])

    def test_json_schema_has_nine_requirement_keys(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "repo"
            blank_root(root)
            package = root / "rating-package"
            _, report = run_json(root, package)
            requirements = report["requirements"]
            self.assertIsInstance(requirements, dict)
            self.assertEqual(len(requirements), 9)
            self.assertEqual(
                list(requirements),
                [
                    "pilot_manifest_approval",
                    "pilot_relevance_completeness",
                    "historical_reuse_decision",
                    "gate_calibration",
                    "neutral_filler",
                    "tokenizer",
                    "leakage",
                    "sample_size_precision",
                    "historical_baseline_compatibility",
                ],
            )
            for finding in requirements.values():
                self.assertIn(finding["status"], {"PASS", "BLOCKED", "NOT_IMPLEMENTED"})
                self.assertIn("reason", finding)
                self.assertIn("path", finding)

    def test_reviewed_option_alone_is_not_a_decision(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "repo"
            blank_root(root)
            package = root / "rating-package"
            decision = package / "historical-reuse-decision.md"
            decision.write_text(
                decision.read_text(encoding="utf-8").replace("- [ ] **Option 2**", "- [x] **Option 2**"),
                encoding="utf-8",
            )
            finding = checker.check_historical_reuse(package)
            self.assertEqual(finding["status"], "BLOCKED")
            self.assertIn("checkbox", finding["reason"])


class ResultTests(unittest.TestCase):
    def test_all_pass_exits_zero(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "repo"
            blank_root(root)
            write_json(
                root / "university-gpu" / "approval-templates" / "pilot-manifest.json",
                {"approved": True, "approved_by": "R", "approval_reference": "A-1", "manifest_sha256": "a" * 64},
            )
            write_json(
                root / "data" / "task_lists" / "pilot_seen.json",
                {"tasks": ["valid_seen:pick_and_place:1"]},
            )
            write_json(
                root / "data" / "task_lists" / "evaluation_unseen.json",
                {"tasks": ["valid_unseen:pick_and_place:9"]},
            )
            # The remaining checks are stubbed so the exit-code contract is tested
            # without inventing approval, threshold, filler, or tokenizer records.
            passing = {"status": "PASS", "reason": "synthetic", "path": "synthetic"}
            stubs = {
                "check_relevance": passing,
                "check_historical_reuse": passing,
                "check_gate_calibration": passing,
                "check_neutral_filler": passing,
                "check_tokenizer": passing,
                "check_sample_size": passing,
                "check_historical_compatibility": passing,
            }
            with patch.multiple(checker, **{name: (lambda *a, **k: value) for name, value in stubs.items()}):
                report = checker.check_all(root, rating_package=root / "rating-package")
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(report["counts"]["PASS"], 9)


if __name__ == "__main__":
    unittest.main()
