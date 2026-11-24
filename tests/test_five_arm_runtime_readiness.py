"""Runtime identity, the six readiness levels, approval binding and the memory policy -- SYNTHETIC; no provider contact."""
from __future__ import annotations

import unittest
from types import SimpleNamespace

from five_arm_fixtures import LIBRARY_SCORES, SKILL_TEXTS, TOLERANCE, SyntheticTokenizer, corpus, natural_top3
from rq1.five_arm import readiness, runner, runtime
from rq1.five_arm.memory import Skill, construct_arm_memory
from rq1.five_arm.protocol import SCIENTIFIC_DECISIONS, decisions_blocking
from rq1.five_arm.selection import natural_ranking, select_interventions

TOK = SyntheticTokenizer()


def complete_config(**override):
    cfg = runtime.proposed_runtime_config()
    cfg.update(weight_blob_digests=["b" * 64], tokenizer_sha256="c" * 64, tokenizer_revision="rev-1", chat_template_sha256="d" * 64,
               provider_version="0.0.0-fixture", hermes_version="0.0.0-fixture", top_p=1.0, top_k=0,
               chat_template_counter_sha256="e" * 64, provider_probe_results_sha256="9" * 64)
    cfg.update(override)
    cfg["approval"] = {"approved": True, "approved_by": "FIXTURE", "approval_reference": "unit-test", "config_sha256": runtime.config_sha256(cfg)}
    return cfg


def observed_from(cfg, **override):
    return {**{k: cfg[k] for k in runtime.REQUIRED_FIELDS}, **override}


def person(sha, **extra):
    """An approval-SHAPED record from a non-synthetic-looking approver (still a test object, never a real approval)."""
    return {"approved": True, "approved_by": "Reviewer Q", "approval_reference": "REF-1", "artifact_sha256": sha, **extra}


ARTIFACTS = {k: f"{i:064x}" for i, k in enumerate((
    "provider_probe_plan", "provider_probe_results", "runtime_config", "pilot_roster", "pilot_exposure_statement",
    "diagnostic_protocol", "pilot_oracle_replay_diagnostic_cap", "diagnostic_report", "budget_decision", "filler_corpus",
    "pilot_relevance_labels", "pilot_oracle_replay_chosen_cap", "pilot_report", "final_protocol", "statistical_analysis_plan",
    "sample_size_justification", "confirmatory_roster", "confirmatory_exposure_statement", "confirmatory_relevance_labels",
    "confirmatory_oracle_replay_chosen_cap"), start=1)}


def state_for(level):
    state = {"local_tests_passed": True, "code_commit": "a" * 40, "observed_runtime": {"model_tag": "gemma4:12b"}}
    levels = readiness.LEVELS[: readiness.LEVELS.index(level) + 1]
    for lvl in levels:
        for key, artifact, _ in readiness._RULES.get(lvl, []):
            state[key] = person(ARTIFACTS[artifact])
    decisions = {}
    for lvl in ("d0_failure_manipulation_pilot", "five_arm_scientific_pilot", "confirmatory"):
        if lvl in levels:
            for name in decisions_blocking(lvl):
                decisions[name] = person(readiness.decision_sha256(name, "chosen"), choice="chosen")
    state["scientific_decisions"] = decisions
    return state


class RuntimeTests(unittest.TestCase):
    def test_historical_pin_preserved_and_installed_digest_only_proposed(self):
        cfg = runtime.proposed_runtime_config()
        self.assertEqual(cfg["historical_pin"]["model_manifest_digest"], "4eb23ef187e2c5462566d6a1d3bbbc2f1346d0b4327cbb66d58fffbcc9b2b05c")
        self.assertEqual(cfg["model_manifest_digest"], runtime.INSTALLED_ENGINEERING_DIGEST)
        self.assertEqual(cfg["status"], "PROPOSED_UNVERIFIED")
        problems = runtime.verify_runtime(None, cfg)
        self.assertTrue(any("not approved" in p for p in problems))
        self.assertTrue(any("chat_template_counter_sha256" in p for p in problems))

    def test_matching_runtime_passes_and_mismatches_fail(self):
        cfg = complete_config()
        self.assertEqual(runtime.verify_runtime(observed_from(cfg), cfg, real=False), [])
        self.assertTrue(any("synthetic/fixture" in p for p in runtime.verify_runtime(observed_from(cfg), cfg)))   # FIXTURE approver
        for field, value in (("model_manifest_digest", "4eb23ef187e2c5462566d6a1d3bbbc2f1346d0b4327cbb66d58fffbcc9b2b05c"),
                             ("tokenizer_sha256", "e" * 64), ("sbert_snapshot_sha256", "f" * 64), ("temperature", 0.7),
                             ("think", True), ("chat_template_sha256", "0" * 64)):
            self.assertTrue(runtime.verify_runtime(observed_from(cfg, **{field: value}), cfg, real=False), field)
        with self.assertRaises(runtime.RuntimeIdentityError):
            runtime.require_runtime(None, cfg)

    def test_unbound_approval_and_metadata_only_observation(self):
        cfg = complete_config()
        cfg["num_ctx"] = 8192
        self.assertTrue(any("not bound" in p for p in runtime.verify_runtime(observed_from(cfg), cfg, real=False)))
        calls = []

        def request(endpoint, path, payload=None):
            calls.append(path)
            return {"/api/version": {"version": "0.9.9"},
                    "/api/tags": {"models": [{"name": "gemma4:12b", "digest": runtime.INSTALLED_ENGINEERING_DIGEST,
                                              "details": {"quantization_level": "Q4_K_M"}}]},
                    "/api/show": {"modelfile": "FROM /m/blobs/sha256-" + "a" * 64, "template": "{{ .Prompt }}",
                                  "parameters": "top_k 64\ntop_p 0.95\nstop \"<end>\"", "details": {"quantization_level": "Q4_K_M"}}}[path]
        observed = runtime.observe_runtime(request, endpoint="http://x", model_tag="gemma4:12b", local={"tokenizer_sha256": "c" * 64})
        self.assertEqual(calls, ["/api/version", "/api/tags", "/api/show"])                 # metadata only, never /api/chat
        self.assertEqual((observed["top_k"], observed["top_p"], observed["weight_blob_digests"]), (64, 0.95, ["a" * 64]))
        self.assertFalse(observed["inference_run"])


class ReadinessTests(unittest.TestCase):
    def test_levels_have_no_circular_dependencies(self):
        report = readiness.evaluate({"local_tests_passed": True, "code_commit": "a" * 40}, {})["levels"]
        self.assertEqual(report["local_synthetic_development"]["status"], "READY")
        self.assertEqual(report["university_cpu_integration"]["status"], "READY")       # no runtime record, labels or filler needed
        self.assertFalse(report["university_cpu_integration"]["authorises_model_inference"])
        self.assertEqual(report["provider_compatibility_probe"]["status"], "BLOCKED")
        d0 = readiness.evaluate(state_for("d0_failure_manipulation_pilot"), ARTIFACTS)["levels"]
        self.assertEqual(d0["d0_failure_manipulation_pilot"]["status"], "READY")         # filler and labels NOT required
        self.assertEqual(d0["five_arm_scientific_pilot"]["status"], "BLOCKED")
        blockers = " | ".join(d0["confirmatory"]["blockers"])
        for text in ("unapproved neutral filler corpus", "missing frozen pilot relevance judgments", "no pilot GO decision",
                     "final protocol not approved", "no exposure-risk acceptance for the confirmatory roster"):
            self.assertIn(text, blockers)
        full = readiness.evaluate(state_for("confirmatory"), ARTIFACTS)["levels"]
        self.assertEqual(full["confirmatory"]["status"], "READY")

    def test_approvals_must_be_bound_to_the_artifacts_actually_used(self):
        state = state_for("d0_failure_manipulation_pilot")
        changed = {**ARTIFACTS, "pilot_roster": "f" * 64}                                  # roster edited after approval
        self.assertEqual(readiness.evaluate(state, changed)["levels"]["d0_failure_manipulation_pilot"]["status"], "BLOCKED")
        missing = {k: v for k, v in ARTIFACTS.items() if k != "diagnostic_protocol"}
        self.assertTrue(any("not loaded" in b for b in readiness.evaluate(state, missing)["levels"]["d0_failure_manipulation_pilot"]["blockers"]))
        state["scientific_decisions"]["d0_representation"]["choice"] = "something else"     # decision approval bound to its choice
        self.assertEqual(readiness.evaluate(state, ARTIFACTS)["levels"]["d0_failure_manipulation_pilot"]["status"], "BLOCKED")

    def test_synthetic_or_fixture_approvals_never_authorise_real_execution(self):
        state = state_for("confirmatory")
        state["filler_corpus"] = {**state["filler_corpus"], "approved_by": "FIXTURE-NOT-A-PERSON"}
        self.assertEqual(readiness.evaluate(state, ARTIFACTS)["levels"]["five_arm_scientific_pilot"]["status"], "BLOCKED")
        self.assertEqual(readiness.evaluate(state, ARTIFACTS, real=False)["levels"]["five_arm_scientific_pilot"]["status"], "READY")
        marked = {**state_for("confirmatory"), "synthetic": True}
        self.assertEqual(readiness.evaluate(marked, ARTIFACTS)["levels"]["confirmatory"]["status"], "BLOCKED")
        for level in ("local_synthetic_development", "university_cpu_integration"):
            with self.assertRaises(readiness.ExecutionBlocked):
                readiness.require(level, state_for("confirmatory"), ARTIFACTS)
        with self.assertRaises(readiness.ExecutionBlocked):
            readiness.require("d0_failure_manipulation_pilot", {"local_tests_passed": True}, {})

    def test_every_decision_awaits_approval(self):
        self.assertEqual(len(SCIENTIFIC_DECISIONS), 23)
        ids = [d["id"] for d in SCIENTIFIC_DECISIONS.values()]
        self.assertEqual(ids, [f"SD-{i:02d}" for i in range(1, 24)])
        for d in SCIENTIFIC_DECISIONS.values():
            for key in ("question", "consequence", "recommendation", "alternatives", "required_evidence", "blocks"):
                self.assertIn(key, d)
        self.assertNotIn("filler_corpus", decisions_blocking("d0_failure_manipulation_pilot"))
        self.assertIn("filler_corpus", decisions_blocking("five_arm_scientific_pilot"))
        self.assertEqual(decisions_blocking("university_cpu_integration"), [])


class RunnerGateTests(unittest.TestCase):
    def test_permission_requires_frozen_bound_manifest_and_commit(self):
        cfg = complete_config()
        state = state_for("five_arm_scientific_pilot")
        manifest = {"schema": "rq1-five-arm-manifest-v2", "stage": "pilot", "status": "proposed", "input_hashes": {}, "units": [],
                    "synthetic": False}
        artifacts = {**ARTIFACTS, "runtime_config": runtime.config_sha256(cfg)}
        state["runtime_verification"] = person(runtime.config_sha256(cfg))
        with self.assertRaises(runtime.RuntimeIdentityError):
            runner.assert_execution_permitted(manifest=manifest, readiness_state=state, artifacts=artifacts, observed_runtime=None,
                                              approved_runtime=cfg)
        cfg = {**cfg, "approval": {**cfg["approval"], "approved_by": "Reviewer Q"}}
        with self.assertRaises(readiness.ExecutionBlocked):
            runner.assert_execution_permitted(manifest=manifest, readiness_state=state, artifacts=artifacts,
                                              observed_runtime=observed_from(cfg), approved_runtime=cfg)


class PolicyTests(unittest.TestCase):
    def setUp(self):
        ranking = natural_ranking(LIBRARY_SCORES)
        top = [r.skill_id for r in ranking[:3]]
        labels = {s: "IRRELEVANT" for s in SKILL_TEXTS}
        labels[top[1]] = "RELEVANT"
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
        return (policy or self.policy())({"condition": arm}, SimpleNamespace(task_id=self.task_id), outcome or self.outcome, {})

    def test_policy_verifies_frozen_ranking_filler_and_memory_hash(self):
        for arm in ("D0", "D18", "Rel", "Irr", "Fill"):
            _, audit = self.call(arm)
            self.assertEqual(audit["memory_block_sha256"], self.hashes[arm])
        swapped = SimpleNamespace(result=SimpleNamespace(top=list(reversed(self.outcome.result.top))))
        with self.assertRaises(runner.InterventionIntegrityError):
            self.call("D18", outcome=swapped)
        with self.assertRaises(runner.InterventionIntegrityError):
            self.call("Fill", policy=self.policy(corpus(approved=False)))
        self.manifest["units"][1]["memory_block_sha256"] = "0" * 64
        with self.assertRaises(runner.InterventionIntegrityError):
            self.call("D18")

    def test_chat_template_counter_is_hash_bound(self):
        spec = {"prefix": "<u>", "suffix": "</u>", "add_bos": True}
        counter = runner.ChatTemplateCounter(TOK, spec)
        self.assertEqual(counter("a b"), len(TOK.encode("<u>a b</u>")) + 1)
        self.assertNotEqual(counter.sha256, runner.ChatTemplateCounter(TOK, {**spec, "add_bos": False}).sha256)


if __name__ == "__main__":
    unittest.main()
