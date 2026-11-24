"""Independent human relevance-annotation workflow (tooling only; this code never produces a label).

Raters see: anonymous item ID, task goal, permitted recovery context, skill text, rubric -> rating + optional rationale.
Raters never see: retrieval rank, similarity score, arm, model outcome, historical performance, other raters' answers,
expert continuation.  Item order is shuffled per rater (rank cannot be inferred from order).
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
from collections import Counter
from typing import Any, Mapping, Sequence

from rq1.five_arm.protocol import PROTOCOL_VERSION
from rq1.five_arm.selection import ADJUDICATED_LABELS
from rq1.five_arm.stats import cohen_kappa

RUBRIC_VERSION = "rq1-five-arm-relevance-rubric-v1-draft"
RUBRIC = {
    "RELEVANT": "The skill gives procedural guidance that directly applies to completing THIS task from THIS recovery "
                "context (correct kind of task and steps that would actually help now).",
    "PARTIAL": "Some guidance applies, but only conditionally or in part (e.g. right sub-procedure, wrong object or "
               "receptacle; or helpful only after other steps).",
    "IRRELEVANT": "The guidance does not help complete this task from this context, even if it is about a similar topic "
                  "or shares words with the task (topical similarity is NOT relevance).",
    "UNCERTAIN": "You cannot decide from the information shown.  Use sparingly; every UNCERTAIN item is adjudicated.",
}
BINARY_FOR_INJECTION = {"RELEVANT": "relevant", "IRRELEVANT": "irrelevant", "PARTIAL": None, "UNCERTAIN": None}
EXPORT_COLUMNS = ("item_id", "rater_id", "task_goal", "recovery_context", "skill_text", "rating", "rationale")
DESIGNS = ("full_library", "candidate_set")


class RatingError(ValueError):
    pass


def item_id(task_id: str, skill_id: str, salt: str) -> str:
    if not salt or len(salt) < 16:
        raise RatingError("a coordinator-held salt of at least 16 characters is required")
    return "itm-" + hashlib.sha256(f"{PROTOCOL_VERSION}|{salt}|{task_id}|{skill_id}".encode()).hexdigest()[:20]


def build_items(tasks: Sequence[Mapping[str, str]], skills: Mapping[str, str], *, design: str, salt: str,
                candidate_sets: Mapping[str, Sequence[str]] | None = None,
                forbidden_strings: Mapping[str, Sequence[str]] | None = None) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Return (rater-facing items, coordinator key).  ``forbidden_strings[task_id]`` (e.g. oracle continuation actions,
    expected next action) must not occur in any rater-facing text."""
    if design not in DESIGNS:
        raise RatingError(f"design must be one of {DESIGNS}")
    if design == "candidate_set" and candidate_sets is None:
        raise RatingError("candidate_set design needs the frozen per-task candidate sets")
    items, key = [], {}
    for task in tasks:
        task_id = task["task_id"]
        chosen = sorted(skills) if design == "full_library" else sorted(candidate_sets.get(task_id, ()))
        if not chosen:
            raise RatingError(f"{task_id} has no skills to annotate")
        unknown = sorted(set(chosen) - set(skills))
        if unknown:
            raise RatingError(f"{task_id}: candidate skills outside the frozen library: {unknown}")
        for skill_id in chosen:
            row = {"item_id": item_id(task_id, skill_id, salt), "task_goal": task["task_goal"],
                   "recovery_context": task["recovery_context"], "skill_text": skills[skill_id]}
            for text in (forbidden_strings or {}).get(task_id, ()):
                if text and any(text in row[f] for f in ("task_goal", "recovery_context", "skill_text")):
                    raise RatingError(f"{task_id}: rater-facing text contains forbidden content {text[:30]!r}")
            items.append(row)
            key[row["item_id"]] = {"task_id": task_id, "skill_id": skill_id}
    if len(key) != len(items):
        raise RatingError("item ID collision")
    return items, {"protocol_version": PROTOCOL_VERSION, "rubric_version": RUBRIC_VERSION, "design": design,
                   "items": key, "salt_sha256": hashlib.sha256(salt.encode()).hexdigest()}


def export_for_rater(items: Sequence[Mapping[str, str]], rater_id: str) -> str:
    if not rater_id.strip():
        raise RatingError("rater_id is required")
    ordered = sorted(items, key=lambda it: hashlib.sha256(f"{rater_id}|{it['item_id']}".encode()).hexdigest())
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=EXPORT_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for it in ordered:
        writer.writerow({**{k: it[k] for k in ("item_id", "task_goal", "recovery_context", "skill_text")},
                         "rater_id": rater_id, "rating": "", "rationale": ""})
    return buf.getvalue()


def parse_submission(data: bytes, *, expected_item_ids: set[str], rater_id: str, declaration: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    if declaration.get("label_source") != "human" or declaration.get("rater_id") != rater_id:
        raise RatingError("submission must be declared as an independent HUMAN rating by this rater (AI labels refused)")
    if declaration.get("saw_other_ratings") is not False:
        raise RatingError("the rater must declare that no other rater's answers were seen")
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise RatingError("submission must be UTF-8") from exc
    reader = csv.DictReader(io.StringIO(text))
    if tuple(reader.fieldnames or ()) != EXPORT_COLUMNS:
        raise RatingError(f"submission columns must be exactly {EXPORT_COLUMNS}")
    out: dict[str, dict[str, str]] = {}
    for line, row in enumerate(reader, 2):
        iid = (row["item_id"] or "").strip()
        if iid not in expected_item_ids:
            raise RatingError(f"line {line}: unknown item {iid!r}")
        if iid in out:
            raise RatingError(f"line {line}: duplicate item {iid}")
        if (row["rater_id"] or "").strip() != rater_id:
            raise RatingError(f"line {line}: rater_id differs from the submitting rater")
        label = (row["rating"] or "").strip().upper()
        if label not in ADJUDICATED_LABELS:
            raise RatingError(f"line {line}: invalid or missing rating {row['rating']!r}")
        out[iid] = {"label": label, "rationale": (row["rationale"] or "").strip()}
    missing = expected_item_ids - set(out)
    if missing:
        raise RatingError(f"{len(missing)} item(s) unrated, e.g. {sorted(missing)[:3]}")
    return out


def agreement(a: Mapping[str, Mapping[str, str]], b: Mapping[str, Mapping[str, str]]) -> dict[str, Any]:
    if set(a) != set(b):
        raise RatingError("the two submissions cover different items")
    ids = sorted(a)
    la, lb = [a[i]["label"] for i in ids], [b[i]["label"] for i in ids]
    binary = lambda l: "RELEVANT" if l == "RELEVANT" else "NOT_RELEVANT"
    conflicts = [i for i in ids if a[i]["label"] != b[i]["label"] or a[i]["label"] == "UNCERTAIN"]
    return {"items": len(ids), "percent_agreement": sum(x == y for x, y in zip(la, lb)) / len(ids),
            "kappa_four_category": cohen_kappa(la, lb),
            "kappa_binary_relevant_vs_not": cohen_kappa([binary(x) for x in la], [binary(y) for y in lb]),
            "confusion": dict(sorted(Counter(f"{x}|{y}" for x, y in zip(la, lb)).items())),
            "requires_adjudication": conflicts}


def adjudicate(a: Mapping[str, Mapping[str, str]], b: Mapping[str, Mapping[str, str]], adjudication: Mapping[str, Mapping[str, str]],
               *, rater_ids: tuple[str, str]) -> dict[str, str]:
    needed = set(agreement(a, b)["requires_adjudication"])
    missing = needed - set(adjudication)
    if missing:
        raise RatingError(f"{len(missing)} conflict(s) lack an adjudication record")
    extra = set(adjudication) - needed
    if extra:
        raise RatingError("adjudication records for items that were not in conflict")
    final: dict[str, str] = {}
    for iid in sorted(a):
        if iid in needed:
            record = adjudication[iid]
            if record.get("adjudicator_id") in (None, "", *rater_ids):
                raise RatingError(f"{iid}: the adjudicator must be a named person distinct from both raters")
            label = str(record.get("label", "")).upper()
            if label not in ("RELEVANT", "PARTIAL", "IRRELEVANT"):
                raise RatingError(f"{iid}: adjudicated label must resolve to RELEVANT, PARTIAL or IRRELEVANT")
            final[iid] = label
        else:
            final[iid] = a[iid]["label"]
    return final


def freeze_labels(final: Mapping[str, str], key: Mapping[str, Any], *, provenance: Mapping[str, str]) -> dict[str, Any]:
    for name in ("export_sha256", "rater_a_submission_sha256", "rater_b_submission_sha256", "adjudication_sha256"):
        value = provenance.get(name, "")
        if len(value) != 64:
            raise RatingError(f"provenance {name} must be a SHA-256 digest")
    if set(final) != set(key["items"]):
        raise RatingError("frozen labels must cover every exported item exactly once")
    labels = sorted(({**key["items"][iid], "item_id": iid, "label": label} for iid, label in final.items()),
                    key=lambda r: (r["task_id"], r["skill_id"]))
    body = {"protocol_version": PROTOCOL_VERSION, "rubric_version": key["rubric_version"], "design": key["design"],
            "labels": labels, "provenance": dict(provenance), "label_source": "human_two_rater_adjudicated"}
    body["labels_sha256"] = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    body["status"] = "frozen_pending_human_approval"
    return body


def label_map(frozen: Mapping[str, Any]) -> dict[tuple[str, str], str]:
    return {(r["task_id"], r["skill_id"]): r["label"] for r in frozen["labels"]}


def workload(*, tasks: int, library_skills: int, raters: int = 2, candidate_per_task: int | None = None,
             pilot_tasks: int = 0) -> dict[str, Any]:
    full_pairs = tasks * library_skills
    out = {"full_library": {"pairs": full_pairs, "judgments": full_pairs * raters,
                            "pilot_pairs": pilot_tasks * library_skills, "pilot_judgments": pilot_tasks * library_skills * raters}}
    if candidate_per_task is not None:
        pairs = tasks * candidate_per_task
        out["candidate_set"] = {"pairs": pairs, "judgments": pairs * raters, "pilot_pairs": pilot_tasks * candidate_per_task,
                                "pilot_judgments": pilot_tasks * candidate_per_task * raters}
    out["note"] = "before adjudication; adjudication adds one judgment per conflict or UNCERTAIN item"
    return out
