"""Deterministic Rel / Irr selection from FROZEN adjudicated labels and the FROZEN natural ranking (pre-outcome).

Rule (pre-specified; scope is an open scientific decision, see ``protocol.SCIENTIFIC_DECISIONS``):
* natural ranking = Sentence-BERT cosine, sorted by (-score, skill_id) -- identical to ``rq1.retrieval.retriever``;
* Rel = the highest-ranked skill in the Rel scope whose adjudicated label is RELEVANT;
* Irr = the highest-ranked skill in the Irr scope whose adjudicated label is IRRELEVANT (a plausible false positive);
* PARTIAL and UNCERTAIN are never injected; low similarity is never treated as evidence of irrelevance;
* every skill in a scope must carry a frozen adjudicated label, otherwise selection FAILS (no fallback label);
* a task without a qualifying skill is simply ineligible for that mechanism arm (reason recorded).  It stays in the
  natural D0 / D18 / Fill analysis.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from rq1.five_arm.protocol import SCIENTIFIC_DECISIONS

ADJUDICATED_LABELS = ("RELEVANT", "PARTIAL", "IRRELEVANT", "UNCERTAIN")
SCOPES = tuple(SCIENTIFIC_DECISIONS["rel_candidate_scope"]["options"])


class SelectionError(ValueError):
    pass


@dataclass(frozen=True)
class Ranked:
    skill_id: str
    score: float
    rank: int


def natural_ranking(scores: Mapping[str, float]) -> list[Ranked]:
    if not scores:
        raise SelectionError("an empty library has no natural ranking")
    ordered = sorted(((str(k), float(v)) for k, v in scores.items()), key=lambda item: (-item[1], item[0]))
    return [Ranked(skill_id, score, rank) for rank, (skill_id, score) in enumerate(ordered, 1)]


def scope_candidates(ranking: Sequence[Ranked], scope: str) -> list[Ranked]:
    if scope == "natural_top3":
        return list(ranking[:3])
    if scope == "full_ranking":
        return list(ranking)
    raise SelectionError(f"unknown candidate scope {scope!r}; choose one of {SCOPES}")


def _pick(candidates: Sequence[Ranked], labels: Mapping[str, str], wanted: str, task_id: str) -> tuple[Ranked | None, str]:
    for candidate in candidates:
        label = labels.get(candidate.skill_id)
        if label is None:
            raise SelectionError(f"{task_id}: skill {candidate.skill_id} in the selection scope has no frozen adjudicated label")
        if label not in ADJUDICATED_LABELS:
            raise SelectionError(f"{task_id}: invalid adjudicated label {label!r} for {candidate.skill_id}")
    for candidate in candidates:
        if labels[candidate.skill_id] == wanted:
            return candidate, f"highest-ranked {wanted} skill in scope (rank {candidate.rank})"
    return None, f"no adjudicated {wanted} skill within the scope; task ineligible for this mechanism arm"


def select_interventions(task_id: str, ranking: Sequence[Ranked], labels: Mapping[str, str], *,
                         rel_scope: str, irr_scope: str) -> dict[str, Any]:
    rel, rel_reason = _pick(scope_candidates(ranking, rel_scope), labels, "RELEVANT", task_id)
    irr, irr_reason = _pick(scope_candidates(ranking, irr_scope), labels, "IRRELEVANT", task_id)
    as_dict = lambda r: None if r is None else {"skill_id": r.skill_id, "rank": r.rank, "score": r.score}
    return {"task_id": task_id, "rel_scope": rel_scope, "irr_scope": irr_scope,
            "natural_top3": [as_dict(r) for r in ranking[:3]],
            "rel": as_dict(rel), "rel_reason": rel_reason, "irr": as_dict(irr), "irr_reason": irr_reason}


def mechanism_subsets(selections: Sequence[Mapping[str, Any]], *, labels_sha256: str, ranking_sha256: str) -> dict[str, Any]:
    """Frozen BEFORE any model outcome: which tasks enter each mechanism contrast, bound to label/ranking hashes."""
    rel = sorted(s["task_id"] for s in selections if s["rel"] is not None)
    irr = sorted(s["task_id"] for s in selections if s["irr"] is not None)
    body = {"natural_all": sorted(s["task_id"] for s in selections), "Rel": rel, "Irr": irr,
            "Rel_and_Irr": sorted(set(rel) & set(irr)), "labels_sha256": labels_sha256, "ranking_sha256": ranking_sha256,
            "selections_sha256": hashlib.sha256(json.dumps(list(selections), sort_keys=True).encode()).hexdigest()}
    body["subset_sha256"] = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    return body


def required_annotation_candidates(ranking: Sequence[Ranked], *, rel_scope: str, irr_scope: str) -> list[str]:
    """Every skill that can be retrieved (natural top 3, for Precision@3) or injected under the frozen rules."""
    needed = {r.skill_id for r in ranking[:3]}
    needed |= {r.skill_id for r in scope_candidates(ranking, rel_scope)}
    needed |= {r.skill_id for r in scope_candidates(ranking, irr_scope)}
    return sorted(needed)
