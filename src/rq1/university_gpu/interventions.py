"""Offline-testable selective-memory interventions for the prospective RQ1 design.

No tokenizer is guessed. Production execution requires a locally supplied and
hash-bound tokenizer adapter; test counters are not scientific evidence.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from rq1.university_gpu.models import sha256_file


class TokenBudgetError(ValueError):
    pass


class Tokenizer(Protocol):
    def encode(self, text: str) -> Sequence[int]: ...
    def decode(self, token_ids: Sequence[int]) -> str: ...


@dataclass(frozen=True)
class TokenizerIdentity:
    path: Path
    sha256: str
    revision: str


class LocalGemmaTokenizer:
    """Loads a pinned local tokenizer.json; it never accesses the network."""

    def __init__(self, identity: TokenizerIdentity) -> None:
        if not identity.path.is_file():
            raise TokenBudgetError(f"pinned tokenizer file is missing: {identity.path}")
        if not identity.revision.strip() or "unverified" in identity.revision.lower():
            raise TokenBudgetError("the exact Gemma tokenizer revision must be frozen")
        actual = sha256_file(identity.path)
        if actual != identity.sha256:
            raise TokenBudgetError("local tokenizer hash differs from the frozen protocol")
        try:
            from tokenizers import Tokenizer as HFTokenizer
        except ImportError as exc:
            raise TokenBudgetError("the tokenizers package is required for exact token accounting") from exc
        self.identity = identity
        self._tokenizer = HFTokenizer.from_file(str(identity.path))

    def encode(self, text: str) -> Sequence[int]:
        return self._tokenizer.encode(text, add_special_tokens=False).ids

    def decode(self, token_ids: Sequence[int]) -> str:
        return self._tokenizer.decode(list(token_ids), skip_special_tokens=False)


def _inner(memory: Mapping[str, Any]) -> dict[str, Any]:
    value = memory.get("recovery_memory")
    if not isinstance(value, Mapping):
        raise TokenBudgetError("recovery memory has an invalid structure")
    return dict(value)


def _serialize_block(memory: Mapping[str, Any]) -> str:
    # Match the frozen prompt builder's exact recovery-memory serialization.
    # Counting compact JSON here would undercount the actual injected text.
    return json.dumps(_inner(memory), ensure_ascii=False, sort_keys=True)


def _with_filler(memory: Mapping[str, Any], filler: str) -> dict[str, Any]:
    result = {"recovery_memory": _inner(memory)}
    result["recovery_memory"]["neutral_filler"] = filler
    return result


def token_count(tokenizer: Tokenizer, memory: Mapping[str, Any]) -> int:
    # The prompt marker and framing are identical across arms; count the exact
    # JSON string emitted by RealEpisodeDriver.build_prompt.
    return len(tokenizer.encode(_serialize_block(memory)))


def pad_memory_to_budget(
    memory: Mapping[str, Any],
    *,
    token_budget: int,
    filler_unit: str,
    tokenizer: Tokenizer,
) -> tuple[dict[str, Any], int]:
    if not filler_unit.strip():
        raise TokenBudgetError("prevalidated neutral filler text is required")
    if token_budget < 1:
        raise TokenBudgetError("token budget must be positive")
    ids = list(tokenizer.encode(filler_unit))
    if not ids:
        raise TokenBudgetError("neutral filler does not encode to any tokens")
    empty = _with_filler(memory, "")
    count = token_count(tokenizer, empty)
    if count > token_budget:
        raise TokenBudgetError(f"memory content already exceeds the frozen budget ({count}>{token_budget})")
    if count == token_budget:
        return empty, count
    for length in range(1, token_budget * 4 + 64):
        filler = tokenizer.decode((ids * ((length + len(ids) - 1) // len(ids)))[:length])
        candidate = _with_filler(memory, filler)
        actual = token_count(tokenizer, candidate)
        if actual == token_budget:
            return candidate, actual
        if actual > token_budget:
            break
    raise TokenBudgetError("neutral filler cannot exactly match this memory block's token budget")


def _memory_for_skill(skill_id: str, text: str) -> dict[str, Any]:
    return {"recovery_memory": {"retrieved_skills": [{"rank": 1, "skill_id": skill_id, "text": text}],
                                 "no_retrieved_skills_available": False, "neutral_filler": ""}}


def _memory_from_top(top: Sequence[Mapping[str, Any]], texts: Mapping[str, str]) -> dict[str, Any]:
    skills = []
    for item in top:
        skill_id = str(item["skill_id"])
        if skill_id not in texts:
            raise TokenBudgetError(f"retrieved skill text is missing: {skill_id}")
        skills.append({"rank": len(skills) + 1, "skill_id": skill_id, "text": texts[skill_id]})
    return {"recovery_memory": {"retrieved_skills": skills,
                                 "no_retrieved_skills_available": not bool(skills), "neutral_filler": ""}}


def intervene(
    *,
    arm: str,
    task_id: str,
    raw_memory: Mapping[str, Any],
    retrieval_top: Sequence[Mapping[str, Any]],
    skill_texts: Mapping[str, str],
    selected_skill_ids: Mapping[str, Mapping[str, str]],
    tokenizer: Tokenizer,
    gate_threshold: float,
    filler_unit: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the exact model-facing memory block and a machine-readable audit."""
    if arm not in {"D0", "D18", "Rel", "Irr", "Fill", "Gate"}:
        raise TokenBudgetError(f"unsupported memory arm: {arm}")
    if arm == "D0":
        memory = dict(raw_memory)
        return memory, {"arm": arm, "source_skill_ids": [], "injected_skill_ids": [],
                        "token_count": token_count(tokenizer, memory), "gate_threshold": None}

    # The D18 arm is the untouched current retriever output. This is the token
    # budget anchor for the matched selective/filler arms; never rewrite it.
    baseline = dict(raw_memory)
    budget = token_count(tokenizer, baseline)
    if arm == "D18":
        memory, count = baseline, budget
        source_ids = [str(item["skill_id"]) for item in retrieval_top]
    elif arm in {"Rel", "Irr"}:
        label = "relevant" if arm == "Rel" else "irrelevant"
        selected_id = str(selected_skill_ids.get(task_id, {}).get(label, ""))
        if not selected_id or selected_id not in skill_texts:
            raise TokenBudgetError(f"no frozen {label} skill selection for {task_id}")
        candidate = _memory_for_skill(selected_id, skill_texts[selected_id])
        memory, count = pad_memory_to_budget(candidate, token_budget=budget, filler_unit=filler_unit, tokenizer=tokenizer)
        source_ids = [selected_id]
    elif arm == "Fill":
        candidate = {"recovery_memory": {"retrieved_skills": [], "no_retrieved_skills_available": True, "neutral_filler": ""}}
        memory, count = pad_memory_to_budget(candidate, token_budget=budget, filler_unit=filler_unit, tokenizer=tokenizer)
        source_ids = []
    else:
        kept = [item for item in retrieval_top if float(item.get("score", -1)) >= gate_threshold]
        candidate = _memory_from_top(kept, skill_texts)
        memory, count = pad_memory_to_budget(candidate, token_budget=budget, filler_unit=filler_unit, tokenizer=tokenizer)
        source_ids = [str(item["skill_id"]) for item in retrieval_top]
    injected = [str(item["skill_id"]) for item in _inner(memory).get("retrieved_skills", [])]
    return memory, {
        "arm": arm,
        "source_skill_ids": source_ids,
        "injected_skill_ids": injected,
        "token_count": count,
        "token_budget": budget,
        "filler_token_count": len(tokenizer.encode(str(_inner(memory).get("neutral_filler", "")))),
        "gate_threshold": gate_threshold if arm == "Gate" else None,
        "memory_sha256": __import__("hashlib").sha256(_serialize_block(memory).encode("utf-8")).hexdigest(),
    }
