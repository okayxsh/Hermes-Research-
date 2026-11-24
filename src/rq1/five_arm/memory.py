"""Shared model-visible memory format for the memory-bearing arms (D18, Rel, Irr, Fill) and the D0 baseline.

Model-visible block (inside the existing ``RECOVERY MEMORY:`` prompt section, serialised exactly as
``rq1.hermes.episode_driver.render_action_prompt`` does: ``json.dumps(..., ensure_ascii=False, sort_keys=True)``)::

    {"memory_entries": [{"text": "<slot 1>"}, {"text": "<slot 2>"}, {"text": "<slot 3>"}]}

* identical key names, slot count, serialisation and placement for every memory-bearing arm;
* no skill IDs, ranks, scores, relevance labels, arm names or availability flags are model-visible;
* D18: the three natural top-3 skill texts in rank order;
* Rel / Irr: the single selected skill in slot 1, reviewed neutral content in slots 2-3;
* Fill: reviewed neutral content in all three slots;
* D0: no memory section ("absent_memory_section") or the historical NoLib marker -- a documented structural difference.

Neutral content is assembled from WHOLE reviewed sentences (never truncated) to approach the task's D18 memory-block
token count (the anchor).  The residual is recorded; cells outside the approved tolerance are blocked, never run.
Procedural skill text is never truncated to satisfy a budget.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from rq1.five_arm.protocol import ACTIVE_SKILL_COUNT, CONDITIONS, SCIENTIFIC_DECISIONS

SLOT_COUNT = 3
MEMORY_KEY = "memory_entries"
ENTRY_KEYS = ("text",)
D0_REPRESENTATIONS = tuple(SCIENTIFIC_DECISIONS["d0_representation"]["options"])
HISTORICAL_NOLIB_BLOCK = {"retrieved_skills": [], "no_retrieved_skills_available": True}


class FiveArmMemoryError(ValueError):
    pass


class Tokenizer(Protocol):
    def encode(self, text: str) -> Sequence[int]: ...


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def serialize_block(inner: Mapping[str, Any]) -> str:
    """Exactly the string rendered after ``RECOVERY MEMORY:`` by the frozen prompt builder."""
    return json.dumps(inner, ensure_ascii=False, sort_keys=True)


def block_tokens(tokenizer: Tokenizer, inner: Mapping[str, Any] | None) -> int:
    return 0 if inner is None else len(tokenizer.encode(serialize_block(inner)))


def build_block(entries: Sequence[str]) -> dict[str, Any]:
    if len(entries) != SLOT_COUNT:
        raise FiveArmMemoryError(f"memory-bearing arms use exactly {SLOT_COUNT} slots")
    return {MEMORY_KEY: [{"text": str(text)} for text in entries]}


# --- neutral filler --------------------------------------------------------------------------------------------------
_ACTION_VERBS = re.compile(r"\b(go|goes|going|went|take|takes|took|put|puts|place|places|placed|open|opens|close|closes|"
                           r"toggle|turn|heat|heats|heated|cool|cools|cooled|clean|cleans|cleaned|wash|rinse|slice|sliced|"
                           r"examine|look|inventory|use|pick|picks|grab|move|moves|navigate|walk|carry|drop|find|search|"
                           r"locate|return|retrieve|fetch|bring|get)\b", re.IGNORECASE)
_DIRECTIVES = re.compile(r"\b(you|your|ignore|instruction|instructions|system|prompt|must|should|always|never|step|steps|"
                         r"first|then|next|recover|recovery|strategy|plan|solution|goal|task|action|actions|index|answer|"
                         r"respond|return|output|memory|skill|skills|relevant|irrelevant|filler|neutral|condition)\b",
                         re.IGNORECASE)
_MARKUP = re.compile(r"[\d{}\[\]<>:`|#*_=]")
DEFAULT_ALFWORLD_VOCABULARY = frozenset("""
alarmclock apple applesliced armchair baseballbat basketball bathtub bathtubbasin bed book bowl box bread breadsliced
butterknife cabinet candle cart cd cellphone cloth coffeemachine coffeetable countertop creditcard cup desk desklamp
dishsponge diningtable drawer dresser egg floorlamp fork fridge garbagecan glassbottle handtowel handtowelholder kettle
keychain knife ladle laptop laundryhamper lettuce lettucesliced lightswitch microwave mug newspaper ottoman pan
papertowelroll pen pencil peppershaker pillow plate plunger pot potato potatosliced remotecontrol safe saltshaker shelf
sidetable sinkbasin soapbar soapbottle sofa spatula spoon spraybottle statue stoveburner teddybear television tennisracket
tissuebox toaster toilet toiletpaper toiletpaperhanger tomato tomatosliced towel towelholder vase watch wateringcan
winebottle lamp sink stove table counter room floor kitchen bathroom bedroom
""".split())


def lint_filler_sentence(sentence: str, vocabulary: frozenset[str] = DEFAULT_ALFWORLD_VOCABULARY) -> list[str]:
    """Automatic screen only; passing it NEVER approves filler (separate human scientific review is required)."""
    problems = []
    if not sentence.strip():
        problems.append("empty sentence")
    for name, pattern in (("action verb", _ACTION_VERBS), ("directive or meta term", _DIRECTIVES), ("digit or markup", _MARKUP)):
        hits = sorted({m.group(0).lower() for m in pattern.finditer(sentence)})
        if hits:
            problems.append(f"{name}: {', '.join(hits)}")
    words = {w.lower() for w in re.findall(r"[A-Za-z]+", sentence)}
    objects = sorted(words & vocabulary)
    if objects:
        problems.append(f"ALFWorld object/receptacle vocabulary: {', '.join(objects)}")
    return problems


@dataclass(frozen=True)
class FillerCorpus:
    corpus_id: str
    sentences: tuple[str, ...]
    review: Mapping[str, Any] = field(default_factory=dict)

    @property
    def sha256(self) -> str:
        return sha256_text(json.dumps({"corpus_id": self.corpus_id, "sentences": list(self.sentences)}, sort_keys=True))

    @property
    def approved(self) -> bool:
        r = self.review or {}
        return (r.get("approved") is True and bool(r.get("approved_by")) and bool(r.get("approval_reference"))
                and r.get("corpus_sha256") == self.sha256)

    def lint(self, vocabulary: frozenset[str] = DEFAULT_ALFWORLD_VOCABULARY) -> dict[int, list[str]]:
        return {i: p for i, s in enumerate(self.sentences) if (p := lint_filler_sentence(s, vocabulary))}


def load_filler_corpus(path: Path) -> FillerCorpus:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    sentences = data.get("sentences")
    if not isinstance(sentences, list) or not sentences or not all(isinstance(s, str) and s.strip() for s in sentences):
        raise FiveArmMemoryError("filler corpus needs a non-empty list of sentences")
    if len(set(sentences)) != len(sentences):
        raise FiveArmMemoryError("filler corpus sentences must be unique")
    return FillerCorpus(str(data.get("corpus_id") or ""), tuple(sentences), dict(data.get("review") or {}))


def _assemble_slot(entries: list[str], slot: int, slot_target: int, corpus: FillerCorpus, tokenizer: Tokenizer,
                   salt: str, used: set[int]) -> list[int]:
    """Greedily append whole unused sentences to ``entries[slot]`` while the WHOLE block stays <= slot_target tokens."""
    order = sorted(range(len(corpus.sentences)),
                   key=lambda i: hashlib.sha256(f"{salt}|{slot}|{corpus.sentences[i]}".encode()).hexdigest())
    chosen: list[int] = []
    for index in order:
        if index in used:
            continue
        trial = list(entries)
        trial[slot] = (trial[slot] + " " + corpus.sentences[index]).strip()
        if block_tokens(tokenizer, build_block(trial)) <= slot_target:
            entries[slot] = trial[slot]
            chosen.append(index)
            used.add(index)
    return chosen


def neutral_fill(fixed: Sequence[str | None], target_tokens: int, corpus: FillerCorpus, tokenizer: Tokenizer,
                 salt: str) -> tuple[list[str], list[int]]:
    """Fill the ``None`` slots with whole neutral sentences so the block approaches ``target_tokens`` from below."""
    entries = [text if text is not None else "" for text in fixed]
    neutral_slots = [i for i, text in enumerate(fixed) if text is None]
    used: set[int] = set()
    chosen: list[int] = []
    for position, slot in enumerate(neutral_slots):
        current = block_tokens(tokenizer, build_block(entries))
        remaining_slots = len(neutral_slots) - position
        slot_target = current + max(0, target_tokens - current) // remaining_slots
        if position == len(neutral_slots) - 1:
            slot_target = target_tokens
        chosen += _assemble_slot(entries, slot, slot_target, corpus, tokenizer, salt, used)
    return entries, chosen


def tolerance_limit(anchor: int, tolerance: Mapping[str, Any]) -> int:
    return max(int(tolerance["absolute_tokens"]), int(round(float(tolerance["relative"]) * anchor)))


@dataclass(frozen=True)
class Skill:
    skill_id: str
    text: str
    score: float | None = None
    rank: int | None = None


def construct_arm_memory(arm: str, *, natural_top3: Sequence[Skill], active_skill: Skill | None, corpus: FillerCorpus | None,
                         tokenizer: Tokenizer, tolerance: Mapping[str, Any], d0_representation: str,
                         salt: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (executor memory message ``{"recovery_memory": inner-or-None}``, complete audit).

    The audit has the SAME keys for every arm; inapplicable values are explicit ``None`` / ``[]`` with
    ``not_applicable`` listing them (D0 is never under-documented).
    """
    if arm not in CONDITIONS:
        raise FiveArmMemoryError(f"unsupported arm {arm!r}")
    if d0_representation not in D0_REPRESENTATIONS:
        raise FiveArmMemoryError(f"d0_representation must be one of {D0_REPRESENTATIONS}")
    if len(natural_top3) != SLOT_COUNT:
        raise FiveArmMemoryError("the natural top-3 retrieval is required for every arm (it is the D18 token anchor)")
    anchor = block_tokens(tokenizer, build_block([s.text for s in natural_top3]))
    not_applicable: list[str] = []
    neutral_ids: list[int] = []
    if arm in ("Rel", "Irr"):
        if active_skill is None:
            raise FiveArmMemoryError(f"{arm} requires a frozen selected skill; no fallback skill is ever substituted")
    elif active_skill is not None:
        raise FiveArmMemoryError(f"{arm} must not carry an active selected skill")
    if arm in ("Rel", "Irr", "Fill") and corpus is None:
        raise FiveArmMemoryError(f"{arm} requires a neutral filler corpus")

    if arm == "D0":
        inner = None if d0_representation == "absent_memory_section" else dict(HISTORICAL_NOLIB_BLOCK)
        injected: list[Skill] = []
        not_applicable += ["token_anchor_match", "neutral_content", "slot_layout"]
    elif arm == "D18":
        inner = build_block([s.text for s in natural_top3])
        injected = list(natural_top3)
        not_applicable += ["neutral_content"]
    else:
        fixed: list[str | None] = [active_skill.text, None, None] if arm in ("Rel", "Irr") else [None, None, None]
        entries, neutral_ids = neutral_fill(fixed, anchor, corpus, tokenizer, salt)
        inner = build_block(entries)
        injected = [active_skill] if active_skill is not None else []
    tokens = block_tokens(tokenizer, inner)
    matched = arm in ("Rel", "Irr", "Fill")
    residual = tokens - anchor if arm != "D0" else None
    limit = tolerance_limit(anchor, tolerance) if matched else None
    audit = {
        "arm": arm,
        "memory_representation": ("absent" if inner is None else
                                  "historical_nolib_marker" if arm == "D0" else "three_slot_entries"),
        "model_visible_keys": sorted(inner) if inner else [],
        "slot_count": 0 if (inner is None or arm == "D0") else SLOT_COUNT,
        "active_procedural_skill_count": ACTIVE_SKILL_COUNT[arm],
        "injected_skill_ids": [s.skill_id for s in injected],
        "injected_skill_ranks": [s.rank for s in injected],
        "neutral_slot_count": {"Rel": 2, "Irr": 2, "Fill": 3}.get(arm, 0),
        "filler_corpus_id": corpus.corpus_id if matched and corpus else None,
        "filler_corpus_sha256": corpus.sha256 if matched and corpus else None,
        "filler_corpus_approved": corpus.approved if matched and corpus else None,
        "filler_sentence_indices": neutral_ids if matched else [],
        "memory_block_sha256": None if inner is None else sha256_text(serialize_block(inner)),
        "memory_block_tokens": tokens,
        "token_anchor_d18": anchor,
        "token_residual_vs_anchor": residual,
        "token_tolerance_limit": limit,
        "within_token_tolerance": (abs(residual) <= limit) if matched else None,
        "procedural_text_truncated": False,
        "not_applicable": not_applicable,
    }
    if arm in ("Rel", "Irr") and audit["memory_block_tokens"] > anchor + (limit or 0):
        audit["note"] = "the single active skill alone exceeds the D18 anchor; it is never truncated"
    return {"recovery_memory": inner}, audit
