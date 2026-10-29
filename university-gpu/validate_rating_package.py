"""Validate the three independent human-rating files without writing or filling them.

This checker is deliberately read-only. It never edits, fills, or creates a rating file;
it only reports whether the package is ready to hand to ``combine_ratings.py``.

Usage (from the repository root):

    python university-gpu/validate_rating_package.py \
        --yash friday-rating-package/yash_rater_a.csv \
        --hani friday-rating-package/hani_rater_b.csv \
        --adjudication friday-rating-package/adjudication-final-labels.csv

Exit status is 0 only when every file is well formed, labels are complete and valid,
and all three files cover the same nonempty set of task-skill pairs.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

RATER_HEADER = ["task_id", "skill_id", "label"]
ADJUDICATION_HEADER = ["task_id", "skill_id", "adjudicated"]
VALID_LABELS = frozenset({"RELEVANT", "IRRELEVANT"})

Key = tuple[str, str]


@dataclass
class ParsedRatingFile:
    """One rating file's read-only parse result."""

    role: str
    path: str
    label_column: str
    labels: dict[Key, str] = field(default_factory=dict)
    rows: int = 0
    missing: int = 0
    invalid: int = 0
    duplicates: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def unique_pairs(self) -> int:
        return len(self.labels)


def _parse(
    path: Path,
    role: str,
    label_column: str,
    expected_header: list[str],
) -> ParsedRatingFile:
    parsed = ParsedRatingFile(role=role, path=str(path), label_column=label_column)
    if not path.is_file():
        parsed.errors.append(f"{role}: file not found: {path}")
        return parsed
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            header = list(reader.fieldnames or [])
            if header != expected_header:
                parsed.errors.append(
                    f"{role}: header must be exactly {','.join(expected_header)}; "
                    f"found {','.join(header) if header else '<empty>'}"
                )
                return parsed
            for line_number, row in enumerate(reader, start=2):
                parsed.rows += 1
                if None in row:
                    parsed.errors.append(f"{role}: line {line_number} has more fields than the header")
                task_id = (row.get("task_id") or "").strip()
                skill_id = (row.get("skill_id") or "").strip()
                if not task_id or not skill_id:
                    parsed.errors.append(f"{role}: line {line_number} has a blank task_id or skill_id")
                    continue
                key = (task_id, skill_id)
                if key in parsed.labels:
                    parsed.duplicates += 1
                    parsed.errors.append(
                        f"{role}: line {line_number} duplicates task_id/skill_id {task_id} + {skill_id}"
                    )
                    continue
                raw = (row.get(label_column) or "").strip()
                if not raw:
                    parsed.missing += 1
                    parsed.labels[key] = ""
                    continue
                label = raw.upper()
                if label not in VALID_LABELS:
                    parsed.invalid += 1
                    parsed.errors.append(
                        f"{role}: line {line_number} label {raw!r} is not RELEVANT or IRRELEVANT"
                    )
                    parsed.labels[key] = ""
                    continue
                parsed.labels[key] = label
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        parsed.errors.append(f"{role}: could not read {path}: {exc}")
    return parsed


def _coverage_errors(
    files: list[ParsedRatingFile],
) -> list[str]:
    errors: list[str] = []
    for parsed in files:
        if not parsed.labels:
            errors.append(f"{parsed.role}: covers no task-skill pairs")
    if len(files) != 3 or any(not parsed.labels for parsed in files):
        return errors
    reference = set(files[0].labels)
    for parsed in files[1:]:
        extra = set(parsed.labels) - reference
        absent = reference - set(parsed.labels)
        if extra:
            errors.append(
                f"{parsed.role}: {len(extra)} pair(s) absent from {files[0].role} "
                f"(example: {sorted(extra)[0][0]} + {sorted(extra)[0][1]})"
            )
        if absent:
            errors.append(
                f"{parsed.role}: {len(absent)} pair(s) from {files[0].role} are missing "
                f"(example: {sorted(absent)[0][0]} + {sorted(absent)[0][1]})"
            )
    return errors


def _disagreements(yash: ParsedRatingFile, hani: ParsedRatingFile) -> list[Key]:
    keys = set(yash.labels) & set(hani.labels)
    return sorted(
        key
        for key in keys
        if yash.labels[key] and hani.labels[key] and yash.labels[key] != hani.labels[key]
    )


def validate(yash_path: Path, hani_path: Path, adjudication_path: Path) -> dict[str, object]:
    """Return a JSON-serialisable report; performs no writes."""
    files = [
        _parse(yash_path, "Yash (rater A)", "label", RATER_HEADER),
        _parse(hani_path, "Hani (rater B)", "label", RATER_HEADER),
        _parse(adjudication_path, "Adjudication", "adjudicated", ADJUDICATION_HEADER),
    ]
    errors = [error for parsed in files for error in parsed.errors]
    errors.extend(_coverage_errors(files))
    disagreements = _disagreements(files[0], files[1])
    outstanding = sum(parsed.missing for parsed in files)

    summary = [
        f"{parsed.role}: {parsed.rows} row(s), {parsed.unique_pairs} unique pair(s), "
        f"{parsed.missing} missing, {parsed.invalid} invalid, {parsed.duplicates} duplicate(s)"
        for parsed in files
    ]
    summary.append(f"disagreements between Yash and Hani: {len(disagreements)}")
    summary.append(f"outstanding ratings: {outstanding}")
    # Disagreements are expected before adjudication, so they do not by themselves
    # make the package invalid; only structural/nonempty-coverage problems do.
    ok = not errors and outstanding == 0
    summary.append("status: READY" if ok else "status: NOT READY")
    if errors:
        summary.append("rejections:")
        summary.extend(f"  - {error}" for error in errors)

    return {
        "ok": ok,
        "files": {
            parsed.role: {
                "path": parsed.path,
                "label_column": parsed.label_column,
                "rows": parsed.rows,
                "unique_pairs": parsed.unique_pairs,
                "missing": parsed.missing,
                "invalid": parsed.invalid,
                "duplicates": parsed.duplicates,
            }
            for parsed in files
        },
        "disagreement_count": len(disagreements),
        "disagreements": [{"task_id": task, "skill_id": skill} for task, skill in disagreements],
        "outstanding ratings": outstanding,
        "errors": errors,
        "summary_lines": summary,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Read-only validator for the three human-rating files.",
    )
    parser.add_argument("--yash", required=True, type=Path, help="rater A CSV (task_id,skill_id,label)")
    parser.add_argument("--hani", required=True, type=Path, help="rater B CSV (task_id,skill_id,label)")
    parser.add_argument(
        "--adjudication",
        required=True,
        type=Path,
        help="adjudicated CSV (task_id,skill_id,adjudicated)",
    )
    parser.add_argument("--json", action="store_true", help="emit a single JSON document on stdout")
    args = parser.parse_args(argv)

    report = validate(args.yash, args.hani, args.adjudication)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print("\n".join(str(line) for line in report["summary_lines"]))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
