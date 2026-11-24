"""Runtime identity, readiness gates and the gated runner -- SYNTHETIC; no provider is contacted."""
from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

from five_arm_fixtures import LIBRARY_SCORES, SKILL_TEXTS, TOLERANCE, SyntheticTokenizer, approval, corpus, natural_top3
from rq1.five_arm import readiness, runner, runtime
from rq1.five_arm.memory import Skill, construct_arm_memory
from rq1.five_arm.protocol import SCIENTIFIC_DECISIONS
from rq1.five_arm.selection import natural_ranking, select_interventions

TOK = SyntheticTokenizer()


def complete_config(**override):
    cfg = runtime.proposed_runtime_config()
    cfg.update(weight_blob_digests=["b" * 64], tokenizer_sha256="c" * 64, tokenizer_revision="rev-1", chat_template_sha256="d" * 64,
               provider_version="0.0.0-fixture", hermes_version="0.0.0-fixture", top_p=1.0, top_k=0)
    cfg.update(override)
    cfg["approval"] = {"approved": True, "approved_by": "FIXTURE", "approval_reference": "unit-test", "config_sha256": runtime.config_sha256(cfg)}
    return cfg


def observed_from(cfg, **override):
    return {**{k: cfg[k] for k in runtime.REQUIRED_FIELDS}, **override}


class RuntimeTests(unittest.TestCase):
    def test_historical_pin_preserved_and_installed_digest_only_proposed(self):
        cfg = runtime.proposed_runtime_config()
        self.assertEqual(cfg["historical_pin"]["model_manifest_digest"], "4eb23ef187e2c5462566d6a1d3bbbc2f1346d0b4327cbb66d58fffbcc9b2b05c")
        self.assertEqual(cfg["model_manifest_digest"], runtime.INSTALLED_ENGINEERING_DIGEST)
        self.assertEqual(cfg["status"], "PROPOSED_UNVERIFIED")
        problems = runtime.verify_runtime(None, cfg)
        self.assertTrue(any("not approved" in p for p in problems))
        self.assertTrue(any("tokenizer_sha256" in p for p in problems))

    def test_matching_runtime_passes_and_mismatches_fail(self):
        cfg = complete_config()
        self.assertEqual(runtime.verify_runtime(observed_from(cfg), cfg), [])
        for field, value in (("model_manifest_digest", "4eb23ef187e2c5462566d6a1d3bbbc2f1346d0b4327cbb66d58fffbcc9b2b05c"),
                             ("tokenizer_sha256", "e" * 64), ("sbert_snapshot_sha256", "f" * 64), ("temperature", 0.7),
                             ("think", True), ("chat_template_sha256", "0" * 64)):
            self.assertTrue(runtime.verify_runtime(observed_from(cfg, **{field: value}), cfg), field)
        with self.assertRaises(runtime.RuntimeIdentityError):
            runtime.require_runtime(None, cfg)

    def test_unsupported_settings_and_unbound_approval(self):
        self.assertTrue(any("thinking" in p for p in runtime.verify_runtime(None, complete_config(think=True))))
        self.assertTrue(any("temperature" in p for p in runtime.verify_runtime(None, complete_config(temperature=0.2))))
        cfg = complete_config()
        cfg["num_ctx"] = 8192                                          # edited after approval
        self.assertTrue(any("not bound" in p for p in runtime.verify_runtime(observed_from(cfg), cfg)))
        self.assertTrue(any("neither" in p for p in runtime.verify_runtime(None, complete_config(model_manifest_digest="9" * 64))))


def full_state(level="confirmatory"):
    state = {"local_tests_passed": True, "observed_runtime": {"fixture": True}, "oracle_validation_evidence_sha256": "a" * 64}
    for key in ("pilot_roster", "runtime_verification", "tokenizer_provider_linkage", "filler_corpus", "pilot_relevance_labels",
                "pilot_protocol", "exact_cap_oracle_replay"):
        state[key] = approval()
    state["scientific_decisions"] = {name: approval(choice="x") for name in SCIENTIFIC_DECISIONS}
    if level == "confirmatory":
        for key in ("exposure_provenance", "confirmatory_roster", "confirmatory_relevance_labels", "checkpoint_and_budget_policy",
                    "failure_manipulation_validation", "statistical_analysis_plan", "sample_size_justification", "pilot_go_decision"):
            state[key] = approval()
    return state


class ReadinessTests(unittest.TestCase):
    def test_engineering_tests_never_approve_science(self):
        report = readiness.evaluate({"local_tests_passed": True})["levels"]
        self.assertEqual(report["local_development"]["status"], "READY")
        self.assertEqual(report["scientific_pilot"]["status"], "BLOCKED")
        blockers = " | ".join(report["confirmatory"]["blockers"])
        for text in ("incomplete exposure provenance", "unapproved pilot roster", "unapproved confirmatory roster",
                     "unverified model/runtime pin", "unverified tokenizer/provider linkage", "unapproved neutral filler",
                     "missing pilot relevance judgments", "missing confirmatory relevance judgments",
                     "unapproved checkpoint and action-budget policy", "unvalidated failure manipulation",
                     "unapproved statistical plan", "insufficient sample justification"):
            self.assertIn(text, blockers)
        with self.assertRaises(readiness.ExecutionBlocked):
            readiness.require("scientific_pilot", {"local_tests_passed": True})
        with self.assertRaises(readiness.ExecutionBlocked):
            readiness.require("local_development", full_state())

    def test_levels_open_only_with_complete_valid_approvals(self):
        pilot = readiness.evaluate(full_state("pilot"))["levels"]
        self.assertEqual((pilot["scientific_pilot"]["status"], pilot["confirmatory"]["status"]), ("READY", "BLOCKED"))
        self.assertEqual(readiness.evaluate(full_state())["levels"]["confirmatory"]["status"], "READY")
        for broken in ({"approved": True, "approved_by": "x", "approval_reference": "y", "artifact_sha256": "not-a-hash"},
                       {"approved": "yes", "approved_by": "x", "approval_reference": "y", "artifact_sha256": "a" * 64},
                       {"approved": True, "approved_by": "", "approval_reference": "y", "artifact_sha256": "a" * 64}):
            state = full_state(); state["filler_corpus"] = broken
            self.assertEqual(readiness.evaluate(state)["levels"]["scientific_pilot"]["status"], "BLOCKED")
        state = full_state(); state["scientific_decisions"]["action_cap"] = approval()     # approved but no choice recorded
        self.assertEqual(readiness.evaluate(state)["levels"]["scientific_pilot"]["status"], "BLOCKED")


class RunnerGateTests(unittest.TestCase):
    def test_worker_refuses_before_touching_provider_or_environment(self):
        manifest = {"stage": "pilot", "status": "proposed", "input_hashes": {}, "units": []}
        with mock.patch("rq1.university_gpu.ollama.probe_endpoint") as probe, \
                mock.patch("rq1.hermes.episode_driver.RealEpisodeDriver") as driver:
            with self.assertRaises(readiness.ExecutionBlocked):
                runner.FiveArmWorker({"manifest": manifest}, readiness_state={"local_tests_passed": True},
                                     observed_runtime=None, approved_runtime=runtime.proposed_runtime_config())
            probe.assert_not_called(); driver.assert_not_called()
        cfg = complete_config()
        with self.assertRaises(runtime.RuntimeIdentityError):           # readiness passes, runtime not observed
            runner.assert_execution_permitted(manifest=manifest, readiness_state=full_state("pilot"), observed_runtime=None, approved_runtime=cfg)
        with self.assertRaises(readiness.ExecutionBlocked):              # manifest not frozen
            runner.assert_execution_permitted(manifest=manifest, readiness_state=full_state("pilot"),
                                              observed_runtime=observed_from(cfg), approved_runtime=cfg)


class PolicyAndMeasurementTests(unittest.TestCase):
    def setUp(self):
        ranking = natural_ranking(LIBRARY_SCORES)
        top = [r.skill_id for r in ranking[:3]]
        labels = {s: "IRRELEVANT" for s in SKILL_TEXTS}; labels[top[1]] = "RELEVANT"
        self.task_id = "valid_seen:task-0/trial_T1"
        self.selection = select_interventions(self.task_id, ranking, labels, rel_scope="natural_top3", irr_scope="natural_top3")
        self.hashes = {}
        for arm in ("D0", "D18", "Rel", "Irr", "Fill"):
            sel = self.selection.get(arm.lower()) if arm in ("Rel", "Irr") else None
            active = Skill(sel["skill_id"], SKILL_TEXTS[sel["skill_id"]], sel["score"], sel["rank"]) if sel else None
            _, audit = construct_arm_memory(arm, natural_top3=natural_top3(), active_skill=active,
                                            corpus=corpus() if arm in ("Rel", "Irr", "Fill") else None, tokenizer=TOK,
                                            tolerance=TOLERANCE, d0_representation="absent_memory_section", salt=self.task_id)
            self.hashes[arm] = audit["memory_block_sha256"]
        self.manifest = {"selections": {self.task_id: self.selection},
                         "units": [{"task_id": self.task_id, "condition": a, "memory_block_sha256": h} for a, h in self.hashes.items()]}
        self.outcome = SimpleNamespace(result=SimpleNamespace(top=[SimpleNamespace(skill_id=s.skill_id, score=s.score) for s in natural_top3()]))

    def policy(self, corp=None):
        return runner.FiveArmMemoryPolicy(manifest=self.manifest, skill_texts=SKILL_TEXTS, corpus=corp or corpus(), tokenizer=TOK,
                                          tolerance=TOLERANCE, d0_representation="absent_memory_section")

    def call(self, arm, policy=None, outcome=None):
        return (policy or self.policy())(SimpleNamespace(condition=arm), SimpleNamespace(task_id=self.task_id), outcome or self.outcome, {})

    def test_policy_verifies_frozen_ranking_filler_and_memory_hash(self):
        for arm in ("D0", "D18", "Rel", "Irr", "Fill"):
            memory, audit = self.call(arm)
            self.assertEqual(audit["memory_block_sha256"], self.hashes[arm])
        swapped = SimpleNamespace(result=SimpleNamespace(top=list(reversed(self.outcome.result.top))))
        with self.assertRaises(runner.InterventionIntegrityError):
            self.call("D18", outcome=swapped)
        with self.assertRaises(runner.InterventionIntegrityError):
            self.call("Fill", policy=self.policy(corpus(approved=False)))
        self.manifest["units"][1]["memory_block_sha256"] = "0" * 64
        with self.assertRaises(runner.InterventionIntegrityError):
            self.call("D18")

    def test_measurement_mapping(self):
        task = {"goal_template": "tpl", "scene": "10", "family": "clean_and_place", "oracle_sha256": "c" * 64,
                "checkpoint_digest": "d" * 64, "post_detour_digest": "e" * 64, "prefix_actions": 3, "detour_actions": 1}
        unit = {"unit_key": "u1", "task_id": self.task_id, "seed": 11, "recovery_action_budget": 46}
        for arm in ("D0", "D18", "Rel", "Irr", "Fill"):
            _, audit = self.call(arm)
            rec = runner.to_measurement(unit={**unit, "condition": arm}, task=task,
                                        outcome={"termination_reason": "recovery_budget_exhausted", "post_failure_actions": 46,
                                                 "recovery_success": False, "invalid_action_selections": 2, "runtime_seconds": 9.5},
                                        failure=None, audit=audit, provider_calls=[{"prompt_eval_count": 900, "seconds": 1.25}],
                                        runtime_identity_sha256="f" * 64, first_prompt_sha256="1" * 64,
                                        local_tokens={"raw_text_tokens": 880}, analysis_labels={}, synthetic=True, attempt=1,
                                        exposure_policy="historical_template_screened")
            self.assertEqual((rec["outcome_status"], rec["termination_reason"], rec["success"]), ("VALID", "budget_exhausted", False))
            self.assertEqual(rec["tokens"]["provider_prompt_tokens"], 900)
        infra = runner.to_measurement(unit={**unit, "condition": "D0"}, task=task, outcome=None,
                                      failure={"stage": "episode", "failure_class": "infrastructure"}, audit=self.call("D0")[1],
                                      provider_calls=[], runtime_identity_sha256="f" * 64, first_prompt_sha256=None,
                                      local_tokens={}, analysis_labels={}, synthetic=True, attempt=1, exposure_policy="x")
        self.assertEqual((infra["outcome_status"], infra["success"]), ("MISSING_INFRASTRUCTURE", None))
        odd = runner.to_measurement(unit={**unit, "condition": "D0"}, task=task, outcome={"termination_reason": "mystery"}, failure=None,
                                    audit=self.call("D0")[1], provider_calls=[], runtime_identity_sha256="f" * 64,
                                    first_prompt_sha256=None, local_tokens={}, analysis_labels={}, synthetic=True, attempt=1,
                                    exposure_policy="x")
        self.assertEqual(odd["outcome_status"], "INVALID_MEASUREMENT")
        truncated = runner.to_measurement(unit={**unit, "condition": "D0"}, task=task,
                                          outcome={"termination_reason": "success", "post_failure_actions": 5, "recovery_success": True,
                                                   "invalid_action_selections": 0, "runtime_seconds": 3.0},
                                          failure=None, audit=self.call("D0")[1],
                                          provider_calls=[{"prompt_eval_count": 32768, "seconds": 1.0}],
                                          runtime_identity_sha256="f" * 64, first_prompt_sha256=None, local_tokens={},
                                          analysis_labels={}, synthetic=True, attempt=1, exposure_policy="x", provider_context_limit=32768)
        self.assertEqual((truncated["outcome_status"], truncated["success"]), ("INVALID_MEASUREMENT", None))

    def test_provider_token_recorder(self):
        driver = SimpleNamespace(_model_response=lambda prompt: SimpleNamespace(prompt_eval_count=len(prompt), eval_count=3))
        recorder = runner.ProviderTokenRecorder(driver, TOK)
        driver._model_response("hello world")
        calls = recorder.take()
        self.assertEqual((calls[0]["prompt_eval_count"], calls[0]["local_raw_tokens"]), (11, 2))
        self.assertEqual(recorder.take(), [])


if __name__ == "__main__":
    unittest.main()
