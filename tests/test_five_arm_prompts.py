"""Paired prompt integrity -- built with the frozen prompt builder from SYNTHETIC fixtures; no model is called."""
from __future__ import annotations

import json
import unittest

from five_arm_fixtures import (CONTEXT, FUTURE_OBSERVATION, ORACLE_CONTINUATION, SKILL_TEXTS, TOLERANCE, SyntheticTokenizer,
                               corpus, natural_top3)
from rq1.five_arm.memory import Skill, construct_arm_memory
from rq1.five_arm.prompting import (ContextOverflowError, LeakageProbe, PromptIntegrityError, RuntimeLimits, account_prompt,
                                    build_prompt, compare_arm_prompts, scan_leakage, split_sections)
from rq1.five_arm.protocol import DETOURED_CONDITIONS as CONDITIONS

TOK = SyntheticTokenizer()
ACTIVE = Skill("skill_05", SKILL_TEXTS["skill_05"], 0.75, 6)


def memories(d0="absent_memory_section"):
    out = {}
    for arm in CONDITIONS:
        out[arm] = construct_arm_memory(arm, natural_top3=natural_top3(), active_skill=ACTIVE if arm in ("Rel", "Irr") else None,
                                        corpus=corpus() if arm in ("Rel", "Irr", "Fill") else None, tokenizer=TOK,
                                        tolerance=TOLERANCE, d0_representation=d0, salt="valid_seen:task-1")[0]
    return out


def probe(**kw):
    base = dict(oracle_continuation=ORACLE_CONTINUATION, expected_next_action=ORACLE_CONTINUATION[0],
                future_observations=(FUTURE_OBSERVATION,), skill_ids=tuple(SKILL_TEXTS), library_texts=tuple(SKILL_TEXTS.values()))
    base.update(kw)
    return LeakageProbe(**base)


class PairedPromptTests(unittest.TestCase):
    def setUp(self):
        self.mem = memories()
        self.prompts = {arm: build_prompt(CONTEXT, m) for arm, m in self.mem.items()}

    def test_treatment_independent_sections_identical_and_shared_schema(self):
        report = compare_arm_prompts(self.prompts, d0_representation="absent_memory_section")
        self.assertTrue(report["ok"], report["problems"])
        self.assertNotIn("RECOVERY MEMORY", split_sections(self.prompts["D0"]))
        for arm in ("D18", "Rel", "Irr", "Fill"):
            sections = split_sections(self.prompts[arm])
            self.assertEqual(list(json.loads(sections["RECOVERY MEMORY"])), ["memory_entries"])
        historical = {**self.prompts, "D0": build_prompt(CONTEXT, memories("historical_nolib_marker")["D0"])}
        self.assertTrue(compare_arm_prompts(historical, d0_representation="historical_nolib_marker")["ok"])

    def test_differences_confined_to_memory_section(self):
        for arm, prompt in self.prompts.items():
            stripped = {k: v for k, v in split_sections(prompt).items() if k != "RECOVERY MEMORY"}
            self.assertEqual(stripped, {k: v for k, v in split_sections(self.prompts["D0"]).items()}, arm)

    def test_no_leakage_in_clean_prompts(self):
        for arm, prompt in self.prompts.items():
            self.assertEqual(scan_leakage(prompt, probe()), [], arm)

    def test_leakage_detectors(self):
        d18 = self.prompts["D18"]
        cases = {
            "arm name": d18.replace("TASK GOAL:\n", "TASK GOAL:\nCondition Irr. "),
            "label": d18.replace("TASK GOAL:\n", "TASK GOAL:\nThis memory is irrelevant. "),
            "skill id": d18.replace("TASK GOAL:\n", "TASK GOAL:\nskill_03 "),
            "oracle in memory": build_prompt(CONTEXT, {"recovery_memory": {"memory_entries": [{"text": "go to sinkbasin 1"}, {"text": "a"}, {"text": "b"}]}}),
            "future observation": d18.replace("CURRENT OBSERVATION:\n", "CURRENT OBSERVATION:\n" + FUTURE_OBSERVATION + " "),
            "score": build_prompt(CONTEXT, {"recovery_memory": {"memory_entries": [{"text": "similar 0.8731"}, {"text": "a"}, {"text": "b"}]}}),
        }
        for name, prompt in cases.items():
            self.assertTrue(scan_leakage(prompt, probe()), name)
        leaked_history = type(CONTEXT)(**{**CONTEXT.__dict__, "history": CONTEXT.history + tuple((a, "ok") for a in ORACLE_CONTINUATION)})
        self.assertTrue(any("continuation" in p for p in scan_leakage(build_prompt(leaked_history, self.mem["D18"]), probe())))

    def test_unexpected_keys_and_condition_flags_detected(self):
        legacy = {"recovery_memory": {"retrieved_skills": [], "no_retrieved_skills_available": True, "neutral_filler": "x"}}
        bad = {**self.prompts, "Fill": build_prompt(CONTEXT, legacy)}
        report = compare_arm_prompts(bad, d0_representation="absent_memory_section")
        self.assertFalse(report["ok"])
        self.assertTrue(any("Fill memory keys" in p for p in report["problems"]))
        drift = {**self.prompts, "Irr": self.prompts["Irr"].replace("You arrive at cabinet 1.", "You arrive at cabinet 2.", 1)}
        self.assertFalse(compare_arm_prompts(drift, d0_representation="absent_memory_section")["ok"])
        d0_with_memory = {**self.prompts, "D0": self.prompts["Fill"]}
        self.assertFalse(compare_arm_prompts(d0_with_memory, d0_representation="absent_memory_section")["ok"])


class AccountingTests(unittest.TestCase):
    def setUp(self):
        self.mem = memories()
        self.limits = RuntimeLimits(num_ctx=4096, max_output_tokens=256, provider_discrepancy_tolerance=4)

    def test_full_prompt_accounting(self):
        for arm, memory in self.mem.items():
            acc = account_prompt(build_prompt(CONTEXT, memory), memory, TOK, self.limits)
            self.assertEqual(acc["raw_text_tokens"], acc["memory_block_tokens"] + acc["non_memory_tokens"])
            self.assertEqual(acc["chat_template_status"], "not_verified_locally")
            self.assertFalse(acc["exact_provider_equality_claimed"])
        self.assertEqual(account_prompt(build_prompt(CONTEXT, self.mem["D0"]), self.mem["D0"], TOK, self.limits)["memory_block_tokens"], 0)

    def test_overflow_truncation_and_unverified_template(self):
        prompt = build_prompt(CONTEXT, self.mem["D18"])
        with self.assertRaises(ContextOverflowError):
            account_prompt(prompt, self.mem["D18"], TOK, RuntimeLimits(num_ctx=200, max_output_tokens=150))
        with self.assertRaises(PromptIntegrityError):
            account_prompt(prompt, self.mem["D18"], TOK, self.limits, chat_template=lambda m: m[0]["content"])
        acc = account_prompt(prompt, self.mem["D18"], TOK, self.limits, chat_template=lambda m: "<start>" + m[0]["content"] + "<end>",
                             chat_template_verified=True, provider_prompt_tokens=50)
        self.assertTrue(acc["possible_provider_truncation"])
        self.assertGreater(acc["chat_template_tokens"], acc["raw_text_tokens"])


if __name__ == "__main__":
    unittest.main()
