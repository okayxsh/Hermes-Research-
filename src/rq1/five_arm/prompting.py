"""Prompt construction, complete prompt accounting and paired prompt-integrity checks (no model is called).

Prompts are built with the UNCHANGED frozen builder ``rq1.hermes.episode_driver.render_action_prompt``; the provider
payload is a single user message (``ollama_chat_payload``), so there is no separate system message.  Accounting levels:

1. raw text tokens of the rendered prompt string (local tokenizer);
2. memory-block tokens (the JSON after ``RECOVERY MEMORY:``);
3. complete locally constructed chat prompt tokens -- ONLY when a verified chat-template renderer is supplied;
4. provider-side tokens (Ollama ``prompt_eval_count``) -- only from a real response, never assumed equal to 1-3.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from rq1.five_arm.memory import MEMORY_KEY, SLOT_COUNT, ENTRY_KEYS, Tokenizer, serialize_block
from rq1.five_arm.protocol import CONDITIONS, HISTORICAL_SIX_ARM_CONDITIONS, MEMORY_BEARING
from rq1.hermes.episode_driver import render_action_prompt

HEADERS = ("TASK GOAL", "INITIAL OBSERVATION", "EPISODE HISTORY", "CURRENT OBSERVATION", "CURRENT INVENTORY",
           "RECOVERY MEMORY", "ADMISSIBLE ACTIONS")
_HEADER = re.compile(r"^(" + "|".join(HEADERS) + r"):\n", re.MULTILINE)
INSTRUCTION = "INSTRUCTION"


class PromptIntegrityError(ValueError):
    pass


class ContextOverflowError(PromptIntegrityError):
    pass


@dataclass(frozen=True)
class CheckpointContext:
    """Exactly what the model may see at a post-detour decision: executed history only, never oracle/future steps."""
    task_goal: str
    initial_observation: str
    observation: str
    admissible_actions: tuple[str, ...]
    history: tuple[tuple[str, str], ...] = ()
    inventory: tuple[str, ...] = ()


def build_prompt(context: CheckpointContext, memory_message: Mapping[str, Any] | None, *, attempt: int = 1) -> str:
    inner = None if memory_message is None else memory_message.get("recovery_memory", memory_message)
    return render_action_prompt(task_goal=context.task_goal, initial_observation=context.initial_observation,
                                observation=context.observation, inventory=list(context.inventory),
                                admissible_actions=list(context.admissible_actions), recovery_memory=inner,
                                attempt=attempt, history=list(context.history))


def split_sections(prompt: str) -> dict[str, str]:
    matches = list(_HEADER.finditer(prompt))
    names = [m.group(1) for m in matches]
    if len(names) != len(set(names)):
        raise PromptIntegrityError("a section header occurs more than once (ambiguous prompt)")
    sections: dict[str, str] = {}
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(prompt)
        sections[match.group(1)] = prompt[match.end():end].rstrip("\n")
    if "ADMISSIBLE ACTIONS" in sections:
        body, sep, instruction = sections["ADMISSIBLE ACTIONS"].rpartition("\n\n")
        if sep:
            sections["ADMISSIBLE ACTIONS"], sections[INSTRUCTION] = body, instruction
    return sections


# --- accounting --------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class RuntimeLimits:
    num_ctx: int
    max_output_tokens: int
    provider_discrepancy_tolerance: int = 0


def account_prompt(prompt: str, memory_message: Mapping[str, Any] | None, tokenizer: Tokenizer, limits: RuntimeLimits, *,
                   chat_template: Callable[[list[dict[str, str]]], str] | None = None,
                   chat_template_verified: bool = False,
                   provider_prompt_tokens: int | None = None) -> dict[str, Any]:
    inner = None if memory_message is None else memory_message.get("recovery_memory", memory_message)
    raw = len(tokenizer.encode(prompt))
    memory_tokens = 0 if inner is None else len(tokenizer.encode(serialize_block(inner)))
    chat_tokens = None
    if chat_template is not None:
        if not chat_template_verified:
            raise PromptIntegrityError("a chat-template renderer may only be used once verified against the provider")
        chat_tokens = len(tokenizer.encode(chat_template([{"role": "user", "content": prompt}])))
    local = chat_tokens if chat_tokens is not None else raw
    if local + limits.max_output_tokens > limits.num_ctx:
        raise ContextOverflowError(f"prompt ({local}) + max output ({limits.max_output_tokens}) exceeds num_ctx ({limits.num_ctx})")
    result = {"raw_text_tokens": raw, "memory_block_tokens": memory_tokens, "non_memory_tokens": raw - memory_tokens,
              "chat_template_tokens": chat_tokens,
              "chat_template_status": "verified" if chat_tokens is not None else "not_verified_locally",
              "provider_prompt_tokens": provider_prompt_tokens, "provider_minus_local": None,
              "possible_provider_truncation": None, "context_headroom": limits.num_ctx - local - limits.max_output_tokens,
              "exact_provider_equality_claimed": False}
    if provider_prompt_tokens is not None:
        diff = provider_prompt_tokens - local
        result["provider_minus_local"] = diff
        result["possible_provider_truncation"] = provider_prompt_tokens >= limits.num_ctx or diff < -limits.provider_discrepancy_tolerance
    return result


# --- paired structural comparison ---------------------------------------------------------------------------------------
def compare_arm_prompts(prompts: Mapping[str, str], *, d0_representation: str) -> dict[str, Any]:
    """For one task/checkpoint/decision: treatment-independent sections must be byte-identical across arms and the
    memory section must follow the shared schema (D0 documented separately)."""
    problems: list[str] = []
    parsed = {arm: split_sections(text) for arm, text in prompts.items()}
    unknown = sorted(set(prompts) - set(CONDITIONS))
    if unknown:
        problems.append(f"unknown arms {unknown}")
    shared = [h for h in (*HEADERS, INSTRUCTION) if h != "RECOVERY MEMORY"]
    for header in shared:
        values = {arm: sections.get(header) for arm, sections in parsed.items()}
        if len(set(values.values())) != 1:
            problems.append(f"treatment-independent section {header!r} differs across arms: {sorted(values)}")
    for arm, sections in parsed.items():
        has_memory = "RECOVERY MEMORY" in sections
        if arm == "D0":
            if d0_representation == "absent_memory_section" and has_memory:
                problems.append("D0 must not contain a RECOVERY MEMORY section under absent_memory_section")
            if d0_representation == "historical_nolib_marker" and not has_memory:
                problems.append("D0 must contain the historical NoLib marker block")
            continue
        if not has_memory:
            problems.append(f"{arm} lacks the RECOVERY MEMORY section")
            continue
        try:
            block = json.loads(sections["RECOVERY MEMORY"])
        except ValueError:
            problems.append(f"{arm} memory section is not the shared JSON block")
            continue
        if set(block) != {MEMORY_KEY}:
            problems.append(f"{arm} memory keys {sorted(block)} differ from the shared schema [{MEMORY_KEY!r}] "
                            "(unexpected filler-specific or condition-specific key)")
            continue
        entries = block[MEMORY_KEY]
        if not isinstance(entries, list) or len(entries) != SLOT_COUNT or any(
                not isinstance(e, dict) or tuple(sorted(e)) != ENTRY_KEYS for e in entries):
            problems.append(f"{arm} memory entries do not follow the {SLOT_COUNT}-slot schema")
    return {"ok": not problems, "problems": problems, "arms": sorted(parsed)}


# --- leakage ------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class LeakageProbe:
    oracle_continuation: tuple[str, ...] = ()
    expected_next_action: str | None = None
    future_observations: tuple[str, ...] = ()
    skill_ids: tuple[str, ...] = ()
    extra_forbidden: tuple[str, ...] = ()
    # frozen library skill texts: legitimately model-visible content, excluded from the arm/label TERM scans only
    library_texts: tuple[str, ...] = ()
    arm_names: tuple[str, ...] = tuple(sorted(set(CONDITIONS) | set(HISTORICAL_SIX_ARM_CONDITIONS)))
    label_terms: tuple[str, ...] = ("relevant", "irrelevant", "relevance", "adjudicat", "neutral filler", "filler",
                                    "synthetic", "experimental condition", "control condition", "treatment",
                                    "similarity", "cosine", "retrieval score", "rank")


def scan_leakage(prompt: str, probe: LeakageProbe) -> list[str]:
    problems: list[str] = []
    sections = split_sections(prompt)
    scanned = prompt
    for text in probe.library_texts:
        for form in (text, json.dumps(text, ensure_ascii=False)[1:-1]):
            scanned = scanned.replace(form, " ")
    for name in probe.arm_names:
        if re.search(rf"(?<![A-Za-z0-9]){re.escape(name)}(?![A-Za-z0-9])", scanned):
            problems.append(f"arm name {name!r} is model-visible")
    lowered = scanned.lower()
    for term in probe.label_terms:
        if re.search(rf"(?<![a-z]){re.escape(term)}", lowered):
            problems.append(f"relevance/label/meta term {term!r} is model-visible")
    memory = sections.get("RECOVERY MEMORY", "")
    for action in (*probe.oracle_continuation, *( [probe.expected_next_action] if probe.expected_next_action else [])):
        if action and action in memory:
            problems.append(f"oracle/expert action {action!r} appears inside the memory block")
    history_actions = re.findall(r"^Action: (.*)$", sections.get("EPISODE HISTORY", ""), re.MULTILINE)
    cont = list(probe.oracle_continuation)
    if len(cont) >= 2:
        for start in range(0, max(0, len(history_actions) - len(cont) + 1)):
            if history_actions[start:start + len(cont)] == cont:
                problems.append("the oracle continuation appears as executed history (future steps leaked)")
    for text in probe.future_observations:
        if text and text in prompt:
            problems.append("a future observation is present in the prompt")
    for skill_id in probe.skill_ids:
        if skill_id and skill_id in prompt:
            problems.append(f"skill identifier {skill_id!r} is model-visible")
    if re.search(r"(?<![\d.])-?\d\.\d{3,}(?![\d])", memory):
        problems.append("a floating-point value (possible retrieval score) appears in the memory block")
    for text in probe.extra_forbidden:
        if text and text in prompt:
            problems.append(f"forbidden text present: {text[:40]!r}")
    return problems
