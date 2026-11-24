"""NON-TASK provider compatibility probe (implemented, gated, NOT executed locally; requires separate authorisation).

Purpose: establish, before any task episode, what can only be learned from real inference:

* provider ``prompt_eval_count`` versus the local tokenizer (raw text) and the proposed chat-template counter -- the
  counter spec is accepted only if it predicts the provider count EXACTLY for every probe prompt;
* that ``prompt_eval_count`` / ``eval_count`` / ``done_reason`` are actually returned and parseable;
* repeat-call behaviour at temperature 0 with a fixed seed (identical-output rate), as non-task evidence for SD-08.

The probe prompts contain no ALFWorld content, no task, no skill and no experimental memory.  The plan is hashed; the
authorisation must be bound to that hash (readiness level ``provider_compatibility_probe``).
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Callable, Mapping, Sequence

from rq1.five_arm import readiness
from rq1.hermes.episode_driver import ollama_chat_payload

PROBE_PROMPTS = (
    "Reply with the single word: ready.",
    "Summarise in one short sentence: clouds drift across a pale sky while a quiet river bends through the valley.",
    "Count the vowels in the word harmony and answer with one number.",
    "Write one sentence about the colour of the evening sky over the sea, without any lists or headings.",
    " ".join(["The meadow was calm and the air was mild."] * 40),
)
REPEATS = 2


def probe_plan(*, model_tag: str, seed: int, counter_spec: Mapping[str, Any]) -> dict[str, Any]:
    plan = {"schema": "rq1-five-arm-provider-probe-plan-v1", "model_tag": model_tag, "seed": int(seed),
            "options": ollama_chat_payload(model_tag, "", int(seed))["options"], "think": False,
            "prompts": list(PROBE_PROMPTS), "repeats": REPEATS,
            "counter_spec": {k: counter_spec[k] for k in ("prefix", "suffix", "add_bos")},
            "content_policy": "non-task prompts only; no ALFWorld content, skills, memory or roster information"}
    plan["plan_sha256"] = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    return plan


def run_probe(plan: Mapping[str, Any], *, readiness_state: Mapping[str, Any], send: Callable[[Mapping[str, Any]], Mapping[str, Any]],
              tokenizer: Any, counter: Callable[[str], int], real: bool = True) -> dict[str, Any]:
    """``send(payload) -> Ollama /api/chat JSON``.  Gate first; nothing is sent unless the probe is authorised."""
    readiness.require("provider_compatibility_probe", readiness_state, {"provider_probe_plan": plan["plan_sha256"]}, real=real)
    rows = []
    for index, prompt in enumerate(plan["prompts"]):
        outputs = []
        for repeat in range(int(plan["repeats"])):
            payload = ollama_chat_payload(plan["model_tag"], prompt, int(plan["seed"]))
            response = send(payload)
            outputs.append({"content_sha256": hashlib.sha256(str((response.get("message") or {}).get("content", "")).encode()).hexdigest(),
                            "prompt_eval_count": response.get("prompt_eval_count"), "eval_count": response.get("eval_count"),
                            "done_reason": response.get("done_reason"), "model": response.get("model")})
        provider = outputs[0]["prompt_eval_count"]
        rows.append({"prompt_index": index, "local_raw_tokens": len(tokenizer.encode(prompt)), "local_counter_tokens": counter(prompt),
                     "provider_prompt_eval_count": provider,
                     "provider_counts_stable": len({o["prompt_eval_count"] for o in outputs}) == 1,
                     "identical_outputs": len({o["content_sha256"] for o in outputs}) == 1, "calls": outputs})
    return summarise(plan, rows)


def summarise(plan: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    present = all(isinstance(r["provider_prompt_eval_count"], int) for r in rows)
    exact = present and all(r["provider_prompt_eval_count"] == r["local_counter_tokens"] for r in rows)
    offsets = sorted({r["provider_prompt_eval_count"] - r["local_raw_tokens"] for r in rows}) if present else []
    result = {"schema": "rq1-five-arm-provider-probe-results-v1", "plan_sha256": plan["plan_sha256"], "rows": list(rows),
              "prompt_eval_count_returned": present, "counter_exact_for_all_prompts": exact,
              "raw_token_offsets": offsets, "constant_raw_offset": len(offsets) == 1,
              "identical_output_rate": sum(r["identical_outputs"] for r in rows) / len(rows) if rows else None,
              "verdict": "COMPATIBLE" if exact else "INCOMPATIBLE_OR_UNVERIFIED",
              "note": "COMPATIBLE is evidence for review; it is not an approval of the runtime or of any decision"}
    result["results_sha256"] = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()
    return result
