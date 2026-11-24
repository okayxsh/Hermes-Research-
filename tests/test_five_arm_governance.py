"""Exposure-risk acceptance, the non-task provider probe, the candidate filler corpus and the draft documents.
SYNTHETIC inputs only; nothing here is an approval, a label or a scientific result."""
from __future__ import annotations

import hashlib
import itertools
import json
import unittest
from pathlib import Path

from five_arm_fixtures import SKILL_TEXTS, TOLERANCE, SyntheticTokenizer
from rq1.five_arm import exposure_acceptance as X
from rq1.five_arm import provider_probe, readiness
from rq1.five_arm.memory import MEMORY_KEY, Skill, audit_corpus, construct_arm_memory, load_filler_corpus
from rq1.five_arm.protocol import SCIENTIFIC_DECISIONS

REPO = Path(__file__).resolve().parents[1]
CANDIDATE_V1 = REPO / "configs" / "five_arm" / "neutral_filler_candidate_v1.json"
CANDIDATE_V1_SHA256 = "cd9f3e07f848c0f289310a431bf1adb483b4a710cbf50a521d9c04ba60c2ca31"
CANDIDATE = REPO / "configs" / "five_arm" / "neutral_filler_candidate_v2.json"
CANDIDATE_SHA256 = "561fb4e8562c082b1e89c1e2877ad15fa5f760059b766011b23d896a002356b5"
TOK = SyntheticTokenizer()


def person(sha):
    return {"approved": True, "approved_by": "Reviewer Q", "approval_reference": "REF-1", "artifact_sha256": sha}


def ledger(task_id, klass, kinds=(), complete=False):
    return {"task_id": task_id, "exposure_class": klass, "exposure_kinds": list(kinds), "evidence_complete": complete,
            "provisional_gaps": ["primary records 2026-09-15..2026-10-29 incomplete"]}


class ExposureAcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.rows = [ledger("valid_unseen:a", "NO_MODEL_OUTCOME_FOUND", ("procedural_template_overlap",)),
                     ledger("valid_unseen:b", "UNKNOWN", ("unknown_evidence",)),
                     ledger("valid_unseen:c", "EXPOSED", ("model_outcome_exposure",)),
                     ledger("valid_unseen:d", "POSSIBLY_EXPOSED", ("model_outcome_exposure",))]
        self.kw = dict(ledger_rows=self.rows, coverage=X.UNIVERSITY_INVESTIGATION_COVERAGE, roster_sha256="1" * 64,
                       protocol_sha256="2" * 64, purpose="confirmatory")

    def test_statuses_are_preserved_never_upgraded(self):
        st = X.build_statement(roster=["valid_unseen:a", "valid_unseen:b"], **self.kw)
        self.assertEqual({r["task_id"]: r["residual_status"] for r in st["tasks"]},
                         {"valid_unseen:a": "NO_LOCATED_MODEL_OUTCOME", "valid_unseen:b": "UNKNOWN"})
        self.assertNotIn("VERIFIED_UNEXPOSED", json.dumps(st))
        self.assertEqual(st["acceptance_status"], "AWAITING_AUTHOR_ACCEPTANCE")
        self.assertEqual(X.verify_statement(st), [])
        self.assertEqual(st["coverage"]["files_examined"], 904)
        text = X.manuscript_text(st)
        self.assertIn("does not prove", text)
        self.assertIn("15 September to 29 October 2026", text)

    def test_known_exposure_unknown_tasks_and_verified_claims_are_refused(self):
        for roster in (["valid_unseen:a", "valid_unseen:c"], ["valid_unseen:d"], ["valid_unseen:zzz"]):
            with self.assertRaises(X.ExposureAcceptanceError):
                X.build_statement(roster=roster, **self.kw)
        with self.assertRaises(X.ExposureAcceptanceError):
            X.build_statement(roster=["valid_unseen:v"], **{**self.kw, "ledger_rows": [ledger("valid_unseen:v", "VERIFIED_UNEXPOSED")]})
        with self.assertRaises(X.ExposureAcceptanceError):
            X.build_statement(roster=["valid_unseen:a"], **{**self.kw, "protocol_sha256": "not-frozen"})

    def test_acceptance_must_be_human_and_bound(self):
        st = X.build_statement(roster=["valid_unseen:a"], **self.kw)
        bound = person(st["statement_sha256"])
        self.assertEqual(X.acceptance_valid(st, bound, roster_sha256="1" * 64, protocol_sha256="2" * 64), [])
        self.assertTrue(X.acceptance_valid(st, {**bound, "approved_by": "FIXTURE"}, roster_sha256="1" * 64, protocol_sha256="2" * 64))
        self.assertTrue(X.acceptance_valid(st, bound, roster_sha256="9" * 64, protocol_sha256="2" * 64))
        self.assertTrue(X.acceptance_valid(st, person("0" * 64), roster_sha256="1" * 64, protocol_sha256="2" * 64))
        tampered = {**st, "tasks": [{**st["tasks"][0], "residual_status": "VERIFIED_UNEXPOSED"}]}
        self.assertTrue(X.verify_statement(tampered))


class ProviderProbeTests(unittest.TestCase):
    SPEC = {"prefix": "<u> ", "suffix": " </u>", "add_bos": True}

    def counter(self, prompt):
        return len(TOK.encode(f"<u> {prompt} </u>")) + 1

    def test_probe_prompts_are_non_task_and_plan_is_hashed(self):
        plan = provider_probe.probe_plan(model_tag="gemma4:12b", seed=11, counter_spec=self.SPEC)
        text = " ".join(plan["prompts"]).lower()
        for term in ("go to", "receptacle", "alfworld", "action_index", "memory", "skill", "inventory"):
            self.assertNotIn(term, text)
        self.assertEqual(plan["options"]["temperature"], 0)
        self.assertEqual(len(plan["plan_sha256"]), 64)

    def test_probe_is_gated_and_compares_provider_with_local_counts(self):
        plan = provider_probe.probe_plan(model_tag="gemma4:12b", seed=11, counter_spec=self.SPEC)
        sent = []

        def send(payload):
            sent.append(payload)
            return {"message": {"content": "ready"}, "prompt_eval_count": self.counter(payload["messages"][0]["content"]),
                    "eval_count": 2, "done_reason": "stop", "model": "gemma4:12b"}
        with self.assertRaises(readiness.ExecutionBlocked):
            provider_probe.run_probe(plan, readiness_state={"local_tests_passed": True}, send=send, tokenizer=TOK, counter=self.counter)
        self.assertEqual(sent, [])                                                        # nothing sent before the gate
        state = {"local_tests_passed": True, "code_commit": "a" * 40, "observed_runtime": {"model_tag": "gemma4:12b"},
                 "provider_probe_authorization": {"approved": True, "approved_by": "FIXTURE", "approval_reference": "unit-test",
                                                  "artifact_sha256": plan["plan_sha256"]}}
        with self.assertRaises(readiness.ExecutionBlocked):                               # fixture authorisation, real semantics
            provider_probe.run_probe(plan, readiness_state=state, send=send, tokenizer=TOK, counter=self.counter)
        result = provider_probe.run_probe(plan, readiness_state=state, send=send, tokenizer=TOK, counter=self.counter, real=False)
        self.assertEqual(result["verdict"], "COMPATIBLE")
        self.assertEqual(result["identical_output_rate"], 1.0)
        off = provider_probe.run_probe(plan, readiness_state=state, send=lambda p: {**send(p), "prompt_eval_count": 3},
                                       tokenizer=TOK, counter=self.counter, real=False)
        self.assertEqual(off["verdict"], "INCOMPATIBLE_OR_UNVERIFIED")


class CandidateFillerTests(unittest.TestCase):
    def test_candidate_corpus_identity_lint_and_status(self):
        corpus = load_filler_corpus(CANDIDATE)
        data = json.loads(CANDIDATE.read_text(encoding="utf-8"))
        self.assertEqual(corpus.sha256, CANDIDATE_SHA256)
        self.assertEqual(data["status"], "PROPOSED_CANDIDATE_NOT_APPROVED")
        self.assertEqual(data["review"], {})
        self.assertFalse(corpus.approved)
        audit = audit_corpus(corpus)
        self.assertTrue(audit["lint_passed"], audit["lint_failures"])
        self.assertEqual((audit["sentences"], audit["short_sentences_le_4_words"]), (135, 0))
        self.assertEqual(data["supersedes"].split()[0], "rq1-five-arm-neutral-filler-candidate-v1")

    def test_superseded_v1_is_preserved_and_fails_the_phase_2e_screen(self):
        v1 = load_filler_corpus(CANDIDATE_V1)
        self.assertEqual(v1.sha256, CANDIDATE_V1_SHA256)                                  # preserved unchanged
        audit = audit_corpus(v1)
        self.assertFalse(audit["lint_passed"])
        self.assertGreaterEqual(audit["sentences_with_task_associated_cues"], 45)
        self.assertEqual(audit["short_sentences_le_4_words"], 30)

    def test_feasibility_with_synthetic_tokenizer(self):
        """Matching is deterministic, within tolerance, never truncates skills, and Rel/Irr share one slot structure.
        Uses the SYNTHETIC tokenizer: the real-tokenizer audit is a university CPU step."""
        corpus = load_filler_corpus(CANDIDATE)
        ids = sorted(SKILL_TEXTS)
        combos = list(itertools.combinations(ids, 3))[::23][:24]
        residuals = []
        for index, combo in enumerate(combos):
            natural = [Skill(s, SKILL_TEXTS[s], 0.9 - r / 10, r + 1) for r, s in enumerate(combo)]
            salt = f"valid_unseen:synthetic-{index}"
            built = {}
            for arm, active in (("Fill", None), ("Rel", natural[0]), ("Irr", natural[2])):
                memory, audit = construct_arm_memory(arm, natural_top3=natural, active_skill=active, corpus=corpus, tokenizer=TOK,
                                                     tolerance=TOLERANCE, d0_representation="absent_memory_section", salt=salt)
                again, _ = construct_arm_memory(arm, natural_top3=natural, active_skill=active, corpus=corpus, tokenizer=TOK,
                                                tolerance=TOLERANCE, d0_representation="absent_memory_section", salt=salt)
                self.assertEqual(memory, again)                                            # deterministic
                self.assertTrue(audit["within_token_tolerance"], (combo, arm, audit["token_residual_vs_anchor"]))
                self.assertFalse(audit["procedural_text_truncated"])
                entries = memory["recovery_memory"][MEMORY_KEY]
                if active is not None:
                    self.assertEqual(entries[0]["text"], active.text)                      # whole skill, slot 1
                residuals.append(audit["token_residual_vs_anchor"])
                built[arm] = (memory, audit)
            rel, irr = built["Rel"], built["Irr"]
            self.assertEqual(rel[1]["slot_count"], irr[1]["slot_count"])
            self.assertEqual(sorted(rel[1]), sorted(irr[1]))                                 # identical audit metadata keys
            self.assertEqual([sorted(e) for e in rel[0]["recovery_memory"][MEMORY_KEY]],
                             [sorted(e) for e in irr[0]["recovery_memory"][MEMORY_KEY]])
        self.assertTrue(all(r <= 0 for r in residuals))                                    # approaches the anchor from below


class DocumentTests(unittest.TestCase):
    def test_drafts_are_marked_and_register_is_complete(self):
        protocol = (REPO / "docs" / "PHASE2_FIVE_ARM_PROTOCOL_DRAFT.md").read_text(encoding="utf-8")
        sap = (REPO / "docs" / "PHASE2_STATISTICAL_ANALYSIS_PLAN_DRAFT.md").read_text(encoding="utf-8")
        register = (REPO / "docs" / "PHASE2_DECISION_REGISTER_DRAFT.md").read_text(encoding="utf-8")
        for text in (protocol, sap, register):
            self.assertIn("DRAFT", text)
        for decision in SCIENTIFIC_DECISIONS.values():
            self.assertIn(decision["id"], register)
        self.assertIn("NOT APPROVED", register.upper())


if __name__ == "__main__":
    unittest.main()
