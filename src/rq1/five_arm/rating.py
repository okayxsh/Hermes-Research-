"""Independent human relevance-annotation workflow (tooling only; this code never produces a label).

Raters see: anonymous item ID, task goal, permitted recovery context, skill text, rubric -> rating + optional rationale.
Raters never see: retrieval rank, similarity score, arm, model outcome, historical performance, other raters' answers,
expert continuation.  Item order is shuffled per rater (rank cannot be inferred from order).

Provenance (Phase 2B): a submission is accepted only with a DECLARATION from a pseudonymous rater code (no names,
e-mails or other personal data in research files; the code-to-person key stays with the coordinator) plus a coordinator
RECEIPT recording how the file was received.  A declaration is a traceable attestation, NOT proof that a human worked
independently; ``independence_screen`` adds checks that flag suspicious submissions for coordinator review.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import re
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
            leaked = _export_leaks(row, skill_ids=tuple(skills))
            if leaked:
                raise RatingError(f"{task_id}: rater-facing text leaks protected information: {leaked}")
            for text in (forbidden_strings or {}).get(task_id, ()):
                if text and any(text in row[f] for f in ("task_goal", "recovery_context", "skill_text")):
                    raise RatingError(f"{task_id}: rater-facing text contains forbidden content {text[:30]!r}")
            items.append(row)
            key[row["item_id"]] = {"task_id": task_id, "skill_id": skill_id}
    if len(key) != len(items):
        raise RatingError("item ID collision")
    return items, {"protocol_version": PROTOCOL_VERSION, "rubric_version": RUBRIC_VERSION, "design": design,
                   "items": key, "salt_sha256": hashlib.sha256(salt.encode()).hexdigest()}


_LEAK_TERMS = re.compile(r"\b(rank|ranked|ranking|score|scores|similarity|cosine|retrieved|retrieval|arm|condition|"
                         r"treatment|control|baseline|D0|D18|Rel|Irr|Fill|Gate|success|succeeded|failed|outcome|oracle|"
                         r"expert)\b", re.IGNORECASE)
_DECIMAL = re.compile(r"(?<![\d.])\d\.\d{2,}(?![\d])")


def _export_leaks(row: Mapping[str, str], *, skill_ids: Sequence[str]) -> list[str]:
    """Goal and recovery context may not mention ranks, scores, arms, outcomes or skill identifiers.  The skill text is
    the frozen library text raters must judge and is exempt from the term scan (identifiers are still refused)."""
    found = []
    for field in ("task_goal", "recovery_context"):
        text = row[field]
        hits = sorted({m.group(0) for m in _LEAK_TERMS.finditer(text)})
        if hits:
            found.append(f"{field}: {hits}")
        if _DECIMAL.search(text):
            found.append(f"{field}: decimal value (possible score)")
    for skill_id in skill_ids:
        if any(skill_id in row[f] for f in ("task_goal", "recovery_context", "skill_text")):
            found.append(f"skill identifier {skill_id}")
    return found


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


DECLARATION_VERSION = "rq1-five-arm-rater-declaration-v1"
DECLARATION_STATEMENTS = {
    "produced_personally": True,              # every judgment was made by the declaring person
    "no_ai_or_automated_assistance": True,    # no LLM, chatbot, script or other automated tool produced or suggested labels
    "worked_independently": True,             # no discussion of items with the other rater or the adjudicator
    "saw_other_ratings": False,
    "saw_model_outcomes": False,
    "saw_retrieval_ranks_or_scores": False,
    "saw_treatment_assignments": False,
    "saw_oracle_or_expert_continuations": False,
}
# The adjudicator necessarily sees both raters' labels (that is what adjudication is), so the rater statement
# "saw_other_ratings: false" cannot apply.  Phase 2E fix: before this, one rater-only statement set was enforced for every
# role, so a truthful adjudicator declaration could never validate while adjudicate() never checked one at all.
ADJUDICATOR_STATEMENTS = {
    **DECLARATION_STATEMENTS,
    "worked_independently": True,             # no discussion of items with either rater
    "saw_other_ratings": True,                # both raters' labels for the conflicting items are shown -- expected
    "is_one_of_the_raters": False,            # the adjudicator never rated these items
}
ROLE_STATEMENTS = {"rater": DECLARATION_STATEMENTS, "adjudicator": ADJUDICATOR_STATEMENTS}
RECEIPT_FIELDS = ("received_by_code", "received_at", "channel", "export_sha256")


def validate_declaration(declaration: Mapping[str, Any], *, rater_id: str, role: str) -> list[str]:
    if role not in ROLE_STATEMENTS:
        return [f"unknown declaration role {role!r}; expected one of {sorted(ROLE_STATEMENTS)}"]
    problems = []
    if declaration.get("declaration_version") != DECLARATION_VERSION:
        problems.append(f"declaration_version must be {DECLARATION_VERSION}")
    if declaration.get("label_source") != "human":
        problems.append("label_source must be 'human' (AI or automated labels are refused)")
    if declaration.get("rater_id") != rater_id or not str(rater_id).strip():
        problems.append("declaration rater_id differs from the submitting rater code")
    if declaration.get("role") != role:
        problems.append(f"declaration role must be {role!r}")
    for key, expected in ROLE_STATEMENTS[role].items():
        if declaration.get(key) is not expected:
            problems.append(f"declaration statement {key} must be {expected}")
    if not str(declaration.get("declared_at", "")).strip():
        problems.append("declared_at timestamp is required")
    if "@" in str(declaration.get("rater_id", "")) or declaration.get("name") or declaration.get("email"):
        problems.append("use a pseudonymous rater code only; names and e-mail addresses do not belong in research files")
    receipt = declaration.get("coordinator_receipt")
    if not isinstance(receipt, Mapping) or any(not str(receipt.get(k, "")).strip() for k in RECEIPT_FIELDS):
        problems.append(f"coordinator_receipt with {RECEIPT_FIELDS} is required")
    elif receipt.get("received_by_code") == rater_id:
        problems.append("the receipt must be recorded by the coordinator, not by the rater")
    return problems


def parse_submission(data: bytes, *, expected_item_ids: set[str], rater_id: str, declaration: Mapping[str, Any],
                     role: str = "rater") -> dict[str, dict[str, str]]:
    problems = validate_declaration(declaration, rater_id=rater_id, role=role)
    if problems:
        raise RatingError("submission declaration refused: " + "; ".join(problems))
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


def independence_screen(a: Mapping[str, Mapping[str, str]], b: Mapping[str, Mapping[str, str]], *,
                        max_identical_rationales: int = 3) -> dict[str, Any]:
    """Flags (for coordinator review; never automatic acceptance or rejection) patterns unlikely under independent
    work: identical label vectors over many items, many identical non-empty rationales."""
    ids = sorted(set(a) & set(b))
    identical_labels = bool(ids) and all(a[i]["label"] == b[i]["label"] for i in ids)
    same_rationales = sum(1 for i in ids if a[i]["rationale"] and a[i]["rationale"] == b[i]["rationale"])
    flags = []
    if identical_labels and len(ids) >= 20:
        flags.append("identical label vectors across all items")
    if same_rationales > max_identical_rationales:
        flags.append(f"{same_rationales} identical non-empty rationales")
    return {"items_compared": len(ids), "identical_label_vectors": identical_labels,
            "identical_rationales": same_rationales, "flags": flags,
            "note": "screen only; a declaration plus a clean screen is still an attestation, not proof of independence"}


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
            "requires_adjudication": conflicts, "independence_screen": independence_screen(a, b)}


def adjudicate(a: Mapping[str, Mapping[str, str]], b: Mapping[str, Mapping[str, str]], adjudication: Mapping[str, Mapping[str, str]],
               *, rater_ids: tuple[str, str], adjudicator_declaration: Mapping[str, Any]) -> dict[str, str]:
    adjudicator = str((adjudicator_declaration or {}).get("rater_id", ""))
    if not adjudicator.strip() or adjudicator in rater_ids:
        raise RatingError("the adjudicator declaration must name a pseudonymous code distinct from both raters")
    problems = validate_declaration(adjudicator_declaration, rater_id=adjudicator, role="adjudicator")
    if problems:
        raise RatingError("adjudicator declaration refused: " + "; ".join(problems))
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
            if record.get("adjudicator_id") != adjudicator:
                raise RatingError(f"{iid}: adjudicated by {record.get('adjudicator_id')!r}, not the declared adjudicator")
            label = str(record.get("label", "")).upper()
            if label not in ("RELEVANT", "PARTIAL", "IRRELEVANT"):
                raise RatingError(f"{iid}: adjudicated label must resolve to RELEVANT, PARTIAL or IRRELEVANT")
            final[iid] = label
        else:
            final[iid] = a[iid]["label"]
    return final


def freeze_labels(final: Mapping[str, str], key: Mapping[str, Any], *, provenance: Mapping[str, str]) -> dict[str, Any]:
    for name in ("export_sha256", "rater_a_submission_sha256", "rater_b_submission_sha256", "adjudication_sha256",
                 "rater_a_declaration_sha256", "rater_b_declaration_sha256", "adjudicator_declaration_sha256"):
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


def workload_from_rankings(rankings: Mapping[str, Sequence[str]], *, rel_scope: str, irr_scope: str, raters: int = 2,
                           pilot_tasks: Sequence[str] = ()) -> dict[str, Any]:
    """Exact annotation workload from FROZEN rankings (task -> skill IDs in natural rank order): the distinct pairs are
    every skill that can be retrieved (natural top 3) or injected under the frozen scopes."""
    from rq1.five_arm.selection import Ranked, required_annotation_candidates
    pairs = {}
    for task_id, ordered in rankings.items():
        ranking = [Ranked(skill_id, 0.0, rank) for rank, skill_id in enumerate(ordered, 1)]
        pairs[task_id] = required_annotation_candidates(ranking, rel_scope=rel_scope, irr_scope=irr_scope)
    total = sum(len(v) for v in pairs.values())
    pilot = sum(len(pairs[t]) for t in pilot_tasks if t in pairs)
    return {"design": "candidate_set" if rel_scope == irr_scope == "natural_top3" else "full_library_scope",
            "tasks": len(pairs), "pairs": total, "judgments_before_adjudication": total * raters,
            "pilot_pairs": pilot, "pilot_judgments_before_adjudication": pilot * raters,
            "note": "adjudication adds one judgment per conflict or agreed UNCERTAIN item"}


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
