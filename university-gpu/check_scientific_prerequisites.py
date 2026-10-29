#!/usr/bin/env python3
"""Read-only scientific-prerequisite checker for the university RQ1 package.

Every requirement is reported as ``PASS``, ``BLOCKED`` or ``NOT_IMPLEMENTED``
with a reason and the file that was examined. The checker never writes, fills,
signs, or creates an approval, label, threshold, filler, tokenizer, or task
record. It only reads local files and the existing rating validator.

Usage (from the repository root)::

    .venv\\Scripts\\python.exe university-gpu\\check_scientific_prerequisites.py
    .venv\\Scripts\\python.exe university-gpu\\check_scientific_prerequisites.py --json

Exit status is 0 only when all nine requirements are ``PASS``; otherwise 1.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA = "rq1-scientific-prerequisites-v1"
STATUS_PASS = "PASS"
STATUS_BLOCKED = "BLOCKED"
STATUS_NOT_IMPLEMENTED = "NOT_IMPLEMENTED"

REQUIREMENT_KEYS: tuple[str, ...] = (
    "pilot_manifest_approval",
    "pilot_relevance_completeness",
    "historical_reuse_decision",
    "gate_calibration",
    "neutral_filler",
    "tokenizer",
    "leakage",
    "sample_size_precision",
    "historical_baseline_compatibility",
)

_HEX64 = re.compile(r"^[0-9a-fA-F]{64}$")
_PLACEHOLDERS = {
    "",
    "tbd",
    "todo",
    "n/a",
    "na",
    "none",
    "unset",
    "unverified",
    "set_to_the_exact_frozen_tokenizer_revision",
}


def _blank(value: Any) -> bool:
    if not isinstance(value, str):
        return True
    text = value.strip()
    if not text:
        return True
    if set(text) <= set("_-.\u2013\u2014 "):
        return True
    return text.lower() in _PLACEHOLDERS


def _digest_ok(value: Any) -> bool:
    return isinstance(value, str) and bool(_HEX64.match(value))


def _finding(status: str, reason: str, path: Path | str, context: str | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {"status": status, "reason": reason, "path": str(path)}
    if context is not None:
        result["context"] = context
    return result


def _read_json(path: Path) -> tuple[Any | None, str | None]:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig")), None
    except OSError as exc:
        return None, f"cannot read {path}: {exc}"
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return None, f"invalid JSON in {path}: {exc}"


def _first_existing(candidates: Sequence[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# a. Pilot manifest approval
# ---------------------------------------------------------------------------
def check_pilot_manifest(root: Path) -> dict[str, Any]:
    path = root / "university-gpu" / "approval-templates" / "pilot-manifest.json"
    if not path.is_file():
        return _finding(STATUS_BLOCKED, "pilot manifest approval template is missing", path)
    payload, error = _read_json(path)
    if error:
        return _finding(STATUS_BLOCKED, error, path)
    if not isinstance(payload, Mapping):
        return _finding(STATUS_BLOCKED, "pilot manifest approval is not a JSON object", path)

    issues: list[str] = []
    if payload.get("approved") is not True:
        issues.append("approved is false or not the boolean true")
    digest = payload.get("manifest_sha256")
    if not _digest_ok(digest):
        issues.append("manifest_sha256 is blank or not a SHA-256 binding")
    if issues:
        return _finding(STATUS_BLOCKED, "pilot manifest is not approved: " + "; ".join(issues), path)
    return _finding(STATUS_PASS, "pilot manifest is approved and bound to a SHA-256 manifest", path)


# ---------------------------------------------------------------------------
# b. Pilot relevance completeness (reuses the existing read-only validator)
# ---------------------------------------------------------------------------
def _load_rating_validator(root: Path) -> Any | None:
    path = root / "university-gpu" / "validate_rating_package.py"
    if not path.is_file():
        return None
    spec = importlib.util.spec_from_file_location("_scientific_prereq_rating_validator", path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        return None
    return module


def check_relevance(root: Path, rating_package: Path) -> dict[str, Any]:
    package = rating_package
    if not package.is_dir():
        return _finding(
            STATUS_BLOCKED,
            "rating package is missing; outstanding ratings cannot be counted",
            package,
        )
    validator = _load_rating_validator(root)
    if validator is None:
        return _finding(
            STATUS_BLOCKED,
            "existing validate_rating_package.py could not be loaded; completeness not established",
            root / "university-gpu" / "validate_rating_package.py",
        )
    try:
        report = validator.validate(
            package / "yash_rater_a.csv",
            package / "hani_rater_b.csv",
            package / "adjudication-final-labels.csv",
        )
    except Exception as exc:  # pragma: no cover - defensive for malformed external files
        return _finding(STATUS_BLOCKED, f"rating validator failed: {exc}", package)

    outstanding = report.get("outstanding ratings")
    errors = [str(item) for item in (report.get("errors") or [])]
    unique_pairs = 0
    files = report.get("files")
    if isinstance(files, Mapping):
        for entry in files.values():
            if isinstance(entry, Mapping):
                unique_pairs += int(entry.get("unique_pairs") or 0)
    ok = report.get("ok") is True and outstanding == 0 and unique_pairs > 0 and not errors
    if ok:
        return _finding(STATUS_PASS, "all three rating files are complete with 0 outstanding ratings", package)
    reason = f"outstanding ratings: {outstanding}"
    if errors:
        reason += "; " + "; ".join(errors[:3])
    return _finding(STATUS_BLOCKED, "pilot relevance labels are incomplete: " + reason, package)


# ---------------------------------------------------------------------------
# c. Historical reuse decision
# ---------------------------------------------------------------------------
_SELECTED_RE = re.compile(r"Selected option\s*\(1\s*/\s*2\s*/\s*3\)\s*:\s*(.*)$", re.IGNORECASE)
_AUTHOR_RE = re.compile(
    r"Author\s+([12])\s+name\s*:\s*(.*?)\s+Signature\s*:\s*(.*?)\s+Date\s*\(UTC\)\s*:\s*(.*)$",
    re.IGNORECASE,
)
_ADJUDICATOR_RE = re.compile(
    r"Adjudicator acknowledgement\s*:\s*(.*?)\s+Date\s*\(UTC\)\s*:\s*(.*)$",
    re.IGNORECASE,
)
_RATIONALE_RE = re.compile(r"Rationale\s*:\s*(.*)$", re.IGNORECASE)
_CHECKED_OPTION_RE = re.compile(r"^\s*-\s*\[[xX]\]")


def check_historical_reuse(rating_package: Path) -> dict[str, Any]:
    path = rating_package / "historical-reuse-decision.md"
    if not path.is_file():
        return _finding(STATUS_BLOCKED, "historical reuse decision record is missing", path)
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        return _finding(STATUS_BLOCKED, f"cannot read decision record: {exc}", path)

    selected = ""
    rationale = ""
    authors: dict[str, dict[str, str]] = {name: {} for name in ("1", "2")}
    adjudicator: dict[str, str] = {}
    checked_box = False
    for line in lines:
        if _CHECKED_OPTION_RE.match(line):
            checked_box = True
        match = _SELECTED_RE.search(line)
        if match:
            selected = match.group(1).strip()
            continue
        match = _AUTHOR_RE.search(line)
        if match:
            authors[match.group(1)] = {
                "name": match.group(2).strip(),
                "signature": match.group(3).strip(),
                "date": match.group(4).strip(),
            }
            continue
        match = _ADJUDICATOR_RE.search(line)
        if match:
            adjudicator = {"acknowledgement": match.group(1).strip(), "date": match.group(2).strip()}
            continue
        match = _RATIONALE_RE.search(line)
        if match:
            rationale = match.group(1).strip()

    issues: list[str] = []
    if selected not in {"1", "2", "3"}:
        issues.append("Selected option is blank or not 1/2/3")
        if checked_box:
            issues.append("a checkbox is marked but the signed Selected option field is blank")
    if _blank(rationale):
        issues.append("Rationale is blank")
    for name in ("1", "2"):
        entry = authors.get(name, {})
        if _blank(entry.get("name")):
            issues.append(f"Author {name} name is blank")
        if _blank(entry.get("signature")):
            issues.append(f"Author {name} signature is blank")
        if _blank(entry.get("date")):
            issues.append(f"Author {name} date is blank")
    if _blank(adjudicator.get("acknowledgement")):
        issues.append("Adjudicator acknowledgement is blank")
    if _blank(adjudicator.get("date")):
        issues.append("Adjudicator date is blank")
    if issues:
        return _finding(
            STATUS_BLOCKED,
            "historical reuse decision is undecided; option checkboxes alone are not a decision: "
            + "; ".join(issues),
            path,
        )
    return _finding(STATUS_PASS, f"historical reuse decision records option {selected} with signatures", path)


# ---------------------------------------------------------------------------
# d. Gate calibration
# ---------------------------------------------------------------------------
def _validate_gate_payload(path: Path) -> list[str]:
    payload, error = _read_json(path)
    if error:
        return [error]
    if not isinstance(payload, Mapping):
        return ["frozen Gate calibration is not a JSON object"]
    issues: list[str] = []
    if str(payload.get("status", "")).lower() != "frozen":
        issues.append("status is not frozen")
    if payload.get("split") != "valid_seen":
        issues.append("split is not valid_seen")
    try:
        threshold = float(payload.get("threshold"))
    except (TypeError, ValueError):
        issues.append("threshold is not numeric")
    else:
        if not -1.0 <= threshold <= 1.0:
            issues.append("threshold is outside cosine range [-1, 1]")
    approval = payload.get("approval")
    if not isinstance(approval, Mapping):
        issues.append("human approval record is missing")
    else:
        if approval.get("approved") is not True:
            issues.append("calibration approval is not approved")
        if _blank(approval.get("approved_by")):
            issues.append("calibration approver is blank")
        if _blank(approval.get("approval_reference")):
            issues.append("calibration approval reference is blank")
    return issues


def _read_csv(path: Path, required_columns: set[str]) -> tuple[list[dict[str, str]], list[str]]:
    issues: list[str] = []
    rows: list[dict[str, str]] = []
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            header = set(reader.fieldnames or [])
            missing = required_columns - header
            if missing:
                issues.append(f"missing columns: {', '.join(sorted(missing))}")
                return rows, issues
            rows = [dict(row) for row in reader]
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        issues.append(f"cannot read {path.name}: {exc}")
    return rows, issues


def check_gate_calibration(
    root: Path,
    calibration_override: Path | None,
    scores_override: Path | None,
) -> dict[str, Any]:
    calibration_candidates = (
        [calibration_override]
        if calibration_override is not None
        else [
            root / "artifacts" / "gate-frozen.json",
            root / "artifacts" / "gate-calibration-frozen.json",
            root / "evidence" / "gate-frozen.json",
            root / "university-gpu" / "evidence" / "gate-frozen.json",
        ]
    )
    calibration = _first_existing(calibration_candidates)
    if calibration is None:
        return _finding(
            STATUS_BLOCKED,
            "no frozen Gate threshold file found",
            calibration_candidates[0],
        )
    issues = _validate_gate_payload(calibration)
    if issues:
        return _finding(STATUS_BLOCKED, "frozen Gate threshold is invalid: " + "; ".join(issues), calibration)

    if scores_override is not None:
        scores_path = scores_override / "scores.csv"
        labels_path = scores_override / "labels.csv"
    else:
        score_dirs = (
            root / "artifacts" / "gate-scores",
            root / "evidence" / "gate-scores",
            root / "university-gpu" / "evidence" / "gate-scores",
        )
        score_dir = next((candidate for candidate in score_dirs if candidate.is_dir()), score_dirs[0])
        scores_path = score_dir / "scores.csv"
        labels_path = score_dir / "labels.csv"

    if not scores_path.is_file() or not labels_path.is_file():
        return _finding(
            STATUS_BLOCKED,
            "adjudicated valid-seen Gate scores/labels are absent",
            scores_path if not scores_path.is_file() else labels_path,
            context=str(calibration),
        )

    score_rows, score_issues = _read_csv(scores_path, {"task_id", "skill_id", "cosine_similarity"})
    label_rows, label_issues = _read_csv(labels_path, {"task_id", "skill_id", "adjudicated"})
    issues = score_issues + label_issues
    if not score_rows:
        issues.append("scores.csv has no score rows")
    if not label_rows:
        issues.append("labels.csv has no adjudicated rows")
    score_keys: set[tuple[str, str]] = set()
    for row in score_rows:
        task_id = str(row.get("task_id") or "").strip()
        skill_id = str(row.get("skill_id") or "").strip()
        if not task_id.startswith("valid_seen:"):
            issues.append("scores.csv contains a non-valid_seen task")
            break
        score_keys.add((task_id, skill_id))
    label_keys: set[tuple[str, str]] = set()
    for row in label_rows:
        task_id = str(row.get("task_id") or "").strip()
        skill_id = str(row.get("skill_id") or "").strip()
        label = str(row.get("adjudicated") or "").strip().upper()
        if not task_id.startswith("valid_seen:"):
            issues.append("labels.csv contains a non-valid_seen task")
            break
        if label not in {"RELEVANT", "IRRELEVANT"}:
            issues.append("labels.csv contains a blank or invalid adjudicated label")
            break
        label_keys.add((task_id, skill_id))
    if score_keys and label_keys and score_keys != label_keys:
        issues.append("scores and adjudicated labels do not cover identical pairs")
    if issues:
        return _finding(STATUS_BLOCKED, "Gate calibration evidence is incomplete: " + "; ".join(issues), calibration)
    return _finding(
        STATUS_PASS,
        "frozen valid-seen Gate threshold and adjudicated scores/labels are present",
        calibration,
        context=str(scores_path.parent),
    )


# ---------------------------------------------------------------------------
# e. Neutral filler
# ---------------------------------------------------------------------------
def check_neutral_filler(root: Path, override: Path | None) -> dict[str, Any]:
    candidates = (
        [override]
        if override is not None
        else [
            root / "artifacts" / "approved-neutral-filler.json",
            root / "artifacts" / "neutral-filler.json",
            root / "university-gpu" / "evidence" / "neutral-filler.json",
            root / "university-gpu" / "approval-templates" / "filler.json",
        ]
    )
    path = _first_existing(candidates)
    if path is None:
        return _finding(
            STATUS_BLOCKED,
            "no approved neutral filler record found in the checkout "
            "(the approved filler is an external evidence input)",
            candidates[0],
        )
    payload, error = _read_json(path)
    if error:
        return _finding(STATUS_BLOCKED, error, path)
    if not isinstance(payload, Mapping):
        return _finding(STATUS_BLOCKED, "neutral filler record is not a JSON object", path)

    issues: list[str] = []
    if payload.get("approved") is not True:
        issues.append("approved is not true")
    if _blank(payload.get("approved_by")):
        issues.append("approved_by is blank")
    if _blank(payload.get("approval_reference")):
        issues.append("approval_reference is blank")
    digest = payload.get("filler_sha256", payload.get("sha256"))
    if not _digest_ok(digest):
        issues.append("filler_sha256 is not a SHA-256 digest")
    text = payload.get("filler_text", payload.get("neutral_filler"))
    source = payload.get("filler_path", payload.get("path"))
    if not _blank(source):
        source_path = Path(str(source))
        if not source_path.is_absolute():
            source_path = path.parent / source_path
        if not source_path.is_file():
            issues.append(f"filler file does not exist: {source_path}")
        elif _digest_ok(digest) and _sha256(source_path) != str(digest).lower():
            issues.append("filler file hash does not match filler_sha256")
    elif _blank(text):
        issues.append("no filler text or filler file is recorded")
    if issues:
        return _finding(STATUS_BLOCKED, "neutral filler is not approved: " + "; ".join(issues), path)
    return _finding(STATUS_PASS, "approved neutral filler record is present and hash-bound", path)


# ---------------------------------------------------------------------------
# f. Tokenizer
# ---------------------------------------------------------------------------
def check_tokenizer(root: Path, override: Path | None) -> dict[str, Any]:
    candidates = (
        [override]
        if override is not None
        else [
            root / "artifacts" / "gemma-tokenizer-record.json",
            root / "artifacts" / "tokenizer-record.json",
            root / "university-gpu" / "evidence" / "tokenizer-record.json",
            root / "university-gpu" / "approval-templates" / "tokenizer.json",
        ]
    )
    path = _first_existing(candidates)
    if path is None:
        return _finding(
            STATUS_BLOCKED,
            "no hash-pinned Gemma tokenizer record found in the checkout "
            "(the tokenizer is an external evidence input)",
            candidates[0],
        )
    payload, error = _read_json(path)
    if error:
        return _finding(STATUS_BLOCKED, error, path)
    if not isinstance(payload, Mapping):
        return _finding(STATUS_BLOCKED, "tokenizer record is not a JSON object", path)

    issues: list[str] = []
    model = payload.get("model", payload.get("tokenizer_model"))
    if _blank(model) or "gemma" not in str(model).lower():
        issues.append("model identity is not a Gemma tokenizer")
    revision = payload.get("revision")
    if _blank(revision) or "unverified" in str(revision).lower() or str(revision).startswith("SET_TO_"):
        issues.append("exact tokenizer revision is blank or a placeholder")
    digest = payload.get("sha256", payload.get("tokenizer_sha256"))
    if not _digest_ok(digest):
        issues.append("tokenizer sha256 is missing or invalid")
    source = payload.get("tokenizer_path", payload.get("path"))
    if _blank(source):
        issues.append("tokenizer_path is missing; the pinned file cannot be verified")
    else:
        source_path = Path(str(source))
        if not source_path.is_absolute():
            source_path = path.parent / source_path
        if not source_path.is_file():
            issues.append(f"tokenizer file does not exist: {source_path}")
        elif _digest_ok(digest) and _sha256(source_path) != str(digest).lower():
            issues.append("tokenizer file hash does not match the recorded sha256")
    if issues:
        return _finding(STATUS_BLOCKED, "tokenizer is not hash-pinned: " + "; ".join(issues), path)
    return _finding(STATUS_PASS, "hash-pinned Gemma tokenizer record and file match", path)


# ---------------------------------------------------------------------------
# g. Leakage
# ---------------------------------------------------------------------------
def _task_ids(payload: Any) -> list[str]:
    if not isinstance(payload, Mapping):
        return []
    raw = payload.get("tasks")
    if not isinstance(raw, list):
        raw = payload.get("task_ids")
    if not isinstance(raw, list):
        return []
    ids: list[str] = []
    for item in raw:
        if isinstance(item, str):
            if item.strip():
                ids.append(item.strip())
        elif isinstance(item, Mapping):
            for key in ("task_id", "id"):
                value = item.get(key)
                if isinstance(value, str) and value.strip():
                    ids.append(value.strip())
                    break
    return ids


def check_leakage(root: Path) -> dict[str, Any]:
    pilot_path = root / "data" / "task_lists" / "pilot_seen.json"
    eval_path = root / "data" / "task_lists" / "evaluation_unseen.json"
    pilot_payload, pilot_error = _read_json(pilot_path) if pilot_path.is_file() else (None, "pilot list is missing")
    if pilot_error:
        return _finding(STATUS_BLOCKED, f"pilot task list cannot be read: {pilot_error}", pilot_path)
    pilot_ids = _task_ids(pilot_payload)
    if not pilot_ids:
        return _finding(
            STATUS_BLOCKED,
            "no pilot tasks frozen; an empty pilot list is never PASS-by-vacuity",
            pilot_path,
        )

    if not eval_path.is_file():
        return _finding(
            STATUS_BLOCKED,
            "valid_unseen task list is missing; leakage cannot be checked",
            eval_path,
            context=f"pilot tasks: {len(pilot_ids)}",
        )
    eval_payload, eval_error = _read_json(eval_path)
    if eval_error:
        return _finding(STATUS_BLOCKED, f"valid_unseen task list cannot be read: {eval_error}", eval_path)
    eval_ids = _task_ids(eval_payload)
    if not eval_ids:
        return _finding(
            STATUS_BLOCKED,
            "valid_unseen task list contains no frozen IDs; leakage cannot be verified",
            eval_path,
            context=f"pilot tasks: {len(pilot_ids)}",
        )

    overlap = sorted(set(pilot_ids) & set(eval_ids))
    if overlap:
        return _finding(
            STATUS_BLOCKED,
            f"{len(overlap)} valid_unseen task ID(s) appear in the pilot list: " + ", ".join(overlap[:3]),
            pilot_path,
        )
    return _finding(
        STATUS_PASS,
        "no valid_unseen task ID appears in the frozen pilot list",
        pilot_path,
        context=f"pilot tasks: {len(pilot_ids)}; valid_unseen tasks: {len(eval_ids)}",
    )


# ---------------------------------------------------------------------------
# h. Sample size / precision
# ---------------------------------------------------------------------------
def check_sample_size(root: Path) -> dict[str, Any]:
    context = (
        root.parent
        / "university-friday-preparation-20261029"
        / "historical-cluster-simulation.json"
    )
    reason = (
        "prospective task-cluster power/precision is not implemented; the readiness report "
        "requires an author-approved simulation before any sufficiency claim"
    )
    context_text = str(context) + " exists but is a historical simulation, not a prospective power result"
    if not context.is_file():
        context_text = str(context) + " (historical simulation output not found)"
    examined = root / "university-gpu" / "docs" / "FRIDAY_FINAL_READINESS.md"
    return _finding(STATUS_NOT_IMPLEMENTED, reason, examined, context=context_text)


# ---------------------------------------------------------------------------
# i. Historical baseline compatibility
# ---------------------------------------------------------------------------
def check_historical_compatibility(root: Path, override: Path | None) -> dict[str, Any]:
    candidates: list[Path] = []
    if override is not None:
        candidates.append(override)
    else:
        candidates.extend(
            [
                root / "university-gpu" / "docs" / "HISTORICAL_BASELINE_COMPATIBILITY.md",
                root / "artifacts" / "historical-baseline-compatibility.json",
                root / "university-gpu" / "evidence" / "historical-baseline-compatibility.md",
            ]
        )
        for directory in (root / "university-gpu" / "docs", root / "artifacts"):
            if directory.is_dir():
                candidates.extend(
                    sorted(
                        item
                        for item in directory.iterdir()
                        if item.is_file() and "compat" in item.name.lower()
                    )
                )
    record = _first_existing(candidates)
    if record is None:
        return _finding(
            STATUS_BLOCKED,
            "no compatibility review record found; the readiness report states historical "
            "NoLib/Accum-18 equivalence is not established",
            candidates[0],
        )
    return _finding(
        STATUS_BLOCKED,
        f"compatibility record found at {record.name}, but the readiness report states historical "
        "NoLib/Accum-18 equivalence is not established; an author-approved compatibility review "
        "and preserved historical results are still required",
        record,
    )


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------
def check_all(
    root: Path,
    *,
    rating_package: Path | None = None,
    gate_calibration: Path | None = None,
    gate_scores: Path | None = None,
    filler: Path | None = None,
    tokenizer_record: Path | None = None,
    compatibility_record: Path | None = None,
) -> dict[str, Any]:
    """Return the full read-only report for ``root``."""
    root = Path(root)
    package = (
        Path(rating_package)
        if rating_package is not None
        else root.parent / "university-friday-preparation-20261029" / "friday-rating-package"
    )
    requirements = {
        "pilot_manifest_approval": check_pilot_manifest(root),
        "pilot_relevance_completeness": check_relevance(root, package),
        "historical_reuse_decision": check_historical_reuse(package),
        "gate_calibration": check_gate_calibration(root, gate_calibration, gate_scores),
        "neutral_filler": check_neutral_filler(root, filler),
        "tokenizer": check_tokenizer(root, tokenizer_record),
        "leakage": check_leakage(root),
        "sample_size_precision": check_sample_size(root),
        "historical_baseline_compatibility": check_historical_compatibility(root, compatibility_record),
    }
    counts = {status: 0 for status in (STATUS_PASS, STATUS_BLOCKED, STATUS_NOT_IMPLEMENTED)}
    for finding in requirements.values():
        counts[finding["status"]] = counts.get(finding["status"], 0) + 1
    overall = STATUS_PASS if all(item["status"] == STATUS_PASS for item in requirements.values()) else STATUS_BLOCKED
    return {
        "schema": SCHEMA,
        "root": str(root),
        "status": overall,
        "counts": counts,
        "requirements": requirements,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Read-only scientific-prerequisite checker for the university RQ1 package.",
    )
    parser.add_argument("--json", action="store_true", help="emit one JSON document on stdout")
    parser.add_argument("--root", type=Path, help="repository root (default: parent of university-gpu)")
    parser.add_argument("--rating-package", type=Path, help="external friday-rating-package directory")
    parser.add_argument("--gate-calibration", type=Path, help="frozen Gate calibration JSON")
    parser.add_argument("--gate-scores", type=Path, help="directory containing scores.csv and labels.csv")
    parser.add_argument("--filler", type=Path, help="approved neutral filler record JSON")
    parser.add_argument("--tokenizer-record", type=Path, help="hash-pinned tokenizer record JSON")
    parser.add_argument("--compatibility-record", type=Path, help="historical compatibility review record")
    args = parser.parse_args(argv)

    root = args.root.resolve() if args.root is not None else Path(__file__).resolve().parents[1]
    report = check_all(
        root,
        rating_package=args.rating_package,
        gate_calibration=args.gate_calibration,
        gate_scores=args.gate_scores,
        filler=args.filler,
        tokenizer_record=args.tokenizer_record,
        compatibility_record=args.compatibility_record,
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=False))
    else:
        counts = report["counts"]
        print(
            f"Scientific prerequisites: {report['status']} "
            f"({counts[STATUS_PASS]} PASS, {counts[STATUS_BLOCKED]} BLOCKED, "
            f"{counts[STATUS_NOT_IMPLEMENTED]} NOT_IMPLEMENTED)"
        )
        for key in REQUIREMENT_KEYS:
            finding = report["requirements"][key]
            print(f"  {finding['status']:<16} {key}: {finding['reason']}")
            print(f"      examined: {finding['path']}")
            if finding.get("context"):
                print(f"      context: {finding['context']}")
        if report["status"] != STATUS_PASS:
            print("Exit 1: at least one requirement is not PASS.")
    return 0 if report["status"] == STATUS_PASS else 1


if __name__ == "__main__":
    sys.exit(main())
