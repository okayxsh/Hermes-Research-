"""Outcome-blind, valid-seen relevance threshold calibration for the Gate arm."""
from __future__ import annotations

import csv
import io
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from rq1.university_gpu.manifest import SBERT_MODEL, SBERT_REVISION
from rq1.university_gpu.models import atomic_write_json, content_hash, sha256_file

SCHEMA = "rq1-university-gate-calibration-v1"
METHOD = "maximize_balanced_accuracy_over_valid_seen_task_skill_scores_tie_break_higher_threshold-v1"


class CalibrationError(ValueError):
    pass


def propose_gate_calibration(scores_path: Path, labels_path: Path) -> dict[str, Any]:
    try:
        score_text = scores_path.read_bytes().decode("utf-8-sig")
        label_text = labels_path.read_bytes().decode("utf-8-sig")
    except OSError as exc:
        raise CalibrationError(f"cannot read calibration input: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise CalibrationError("score and relevance files must be UTF-8 CSV") from exc
    score_reader = csv.DictReader(io.StringIO(score_text))
    expected_columns = {"task_id", "skill_id", "cosine_similarity"}
    if not score_reader.fieldnames or set(score_reader.fieldnames) != expected_columns:
        raise CalibrationError(f"similarity CSV columns must be exactly {sorted(expected_columns)}")
    scores: dict[tuple[str, str], float] = {}
    for line, row in enumerate(score_reader, 2):
        task_id, skill_id = str(row["task_id"]).strip(), str(row["skill_id"]).strip()
        if not task_id.startswith("valid_seen:") or not skill_id or (task_id, skill_id) in scores:
            raise CalibrationError(f"score row {line} has a non-valid_seen task or duplicate/empty identity")
        try:
            score = float(row["cosine_similarity"])
        except (TypeError, ValueError) as exc:
            raise CalibrationError(f"score row {line} has an invalid cosine similarity") from exc
        if not -1 <= score <= 1:
            raise CalibrationError(f"score row {line} similarity is outside [-1, 1]")
        scores[(task_id, skill_id)] = score

    label_reader = csv.DictReader(io.StringIO(label_text))
    if not label_reader.fieldnames or set(label_reader.fieldnames) != {"task_id", "skill_id", "adjudicated"}:
        raise CalibrationError("adjudicated CSV columns must be exactly task_id, skill_id, adjudicated")
    labels: dict[tuple[str, str], bool] = {}
    per_task: dict[str, set[str]] = defaultdict(set)
    for line, row in enumerate(label_reader, 2):
        task_id, skill_id = str(row["task_id"]).strip(), str(row["skill_id"]).strip()
        label = str(row["adjudicated"]).strip().upper()
        key = (task_id, skill_id)
        if not task_id.startswith("valid_seen:") or key in labels or label not in {"RELEVANT", "IRRELEVANT"}:
            raise CalibrationError(f"label row {line} has a non-valid_seen task, duplicate, or invalid rating")
        labels[key] = label == "RELEVANT"
        per_task[task_id].add(skill_id)
    if not labels or set(scores) != set(labels):
        raise CalibrationError("similarity scores and adjudicated relevance labels must cover identical pairs")
    if any(len(skill_ids) != 18 for skill_ids in per_task.values()):
        raise CalibrationError("every valid_seen calibration task must cover exactly 18 skills")
    if {True, False} - set(labels.values()):
        raise CalibrationError("calibration labels must include both relevant and irrelevant examples")

    observed_scores = sorted({scores[key] for key in scores})
    candidates = { -1.0, 1.0 }
    candidates.update(observed_scores)
    candidates.update((left + right) / 2 for left, right in zip(observed_scores, observed_scores[1:]))
    rows = []
    positives = sum(labels.values())
    negatives = len(labels) - positives
    for threshold in sorted(candidates):
        tp = sum(labels[key] and scores[key] >= threshold for key in labels)
        tn = sum(not labels[key] and scores[key] < threshold for key in labels)
        sensitivity = tp / positives
        specificity = tn / negatives
        rows.append({"threshold": threshold, "balanced_accuracy": (sensitivity + specificity) / 2,
                     "sensitivity": sensitivity, "specificity": specificity, "true_positive": tp,
                     "true_negative": tn, "false_positive": negatives - tn, "false_negative": positives - tp})
    best_score = max(row["balanced_accuracy"] for row in rows)
    best = max((row for row in rows if row["balanced_accuracy"] == best_score), key=lambda row: row["threshold"])
    payload = {
        "schema": SCHEMA, "status": "proposed", "split": "valid_seen", "method": METHOD,
        "predeclared_rule": True, "threshold": best["threshold"],
        "calibration_metrics": best, "candidate_count": len(candidates),
        "task_count": len(per_task), "task_ids": sorted(per_task), "pair_count": len(labels),
        "library_skill_count": 18, "retrieval_model": SBERT_MODEL, "retrieval_revision": SBERT_REVISION,
        "recovery_outcomes_used": False, "confirmatory_accessed": False,
        "similarity_scores_sha256": sha256_file(scores_path), "adjudicated_labels_sha256": sha256_file(labels_path),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    payload["calibration_sha256"] = content_hash({key: value for key, value in payload.items()
                                                   if key not in {"calibration_sha256", "status", "approval"}})
    return payload


def validate_gate_calibration(value: Mapping[str, Any], *, require_frozen: bool = False) -> list[str]:
    issues = []
    if value.get("schema") != SCHEMA or value.get("split") != "valid_seen" or value.get("method") != METHOD:
        issues.append("unsupported gate calibration schema, split, or selection rule")
    if value.get("predeclared_rule") is not True or value.get("recovery_outcomes_used") is not False:
        issues.append("gate calibration must use the predeclared outcome-blind rule")
    if value.get("confirmatory_accessed") is not False:
        issues.append("gate calibration must not access confirmatory tasks")
    if value.get("retrieval_model") != SBERT_MODEL or value.get("retrieval_revision") != SBERT_REVISION:
        issues.append("gate calibration model/revision differs from the frozen retriever")
    try:
        threshold = float(value.get("threshold"))
        if not -1 <= threshold <= 1:
            issues.append("gate threshold is outside cosine range [-1, 1]")
    except (TypeError, ValueError):
        issues.append("gate threshold is not numeric")
    task_ids = value.get("task_ids")
    task_count = value.get("task_count")
    pair_count = value.get("pair_count")
    if (not isinstance(task_ids, list) or len(task_ids) != len(set(task_ids))
            or not isinstance(task_count, int) or task_count < 1 or len(task_ids) != task_count
            or not isinstance(pair_count, int) or pair_count != task_count * 18
            or value.get("library_skill_count") != 18):
        issues.append("gate calibration task/skill coverage metadata is invalid")
    for field in ("similarity_scores_sha256", "adjudicated_labels_sha256"):
        digest = value.get(field)
        if not isinstance(digest, str) or len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest.lower()):
            issues.append(f"{field} is not a SHA-256 digest")
    metrics = value.get("calibration_metrics")
    if (not isinstance(metrics, Mapping) or metrics.get("threshold") != value.get("threshold")
            or metrics.get("balanced_accuracy") is None):
        issues.append("gate calibration metrics are missing or inconsistent")
    payload = {key: item for key, item in value.items() if key not in {"calibration_sha256", "status", "approval"}}
    if value.get("calibration_sha256") != content_hash(payload):
        issues.append("gate calibration hash does not match its proposed contents")
    if require_frozen and value.get("status") != "frozen":
        issues.append("gate calibration is not frozen")
    if require_frozen:
        approval = value.get("approval")
        if not isinstance(approval, Mapping) or approval.get("approved") is not True or not approval.get("approved_by") or not approval.get("approval_reference"):
            issues.append("frozen Gate calibration lacks its human approval record")
    return issues


def freeze_gate_calibration(proposal: Mapping[str, Any], approval: Mapping[str, Any], destination: Path) -> dict[str, Any]:
    issues = validate_gate_calibration(proposal)
    if issues:
        raise CalibrationError("cannot freeze invalid gate calibration: " + "; ".join(issues))
    if approval.get("approved") is not True or not approval.get("approved_by") or not approval.get("approval_reference"):
        raise CalibrationError("explicit human approval, approver, and reference are required")
    if approval.get("calibration_sha256") != proposal.get("calibration_sha256"):
        raise CalibrationError("approval is not bound to this exact calibration proposal")
    frozen = dict(proposal)
    frozen["status"] = "frozen"
    frozen["approval"] = {"approved": True, "approved_by": str(approval["approved_by"]),
                          "approval_reference": str(approval["approval_reference"])}
    atomic_write_json(destination, frozen, overwrite=False)
    return frozen
