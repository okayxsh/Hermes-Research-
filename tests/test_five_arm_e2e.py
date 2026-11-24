"""End-to-end SYNTHETIC execution through the actual five-arm worker path (executor + queue + frozen harness + real
prompt builder + single-retrieval boundary) with a scripted world and a scripted provider.  Not scientific evidence."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import five_arm_e2e_fixtures as E
from five_arm_fixtures import NEUTRAL_SENTENCES, SKILL_TEXTS, TOLERANCE, corpus
from rq1.five_arm import analysis, pilot, readiness, runner, runtime
from rq1.five_arm.cpu_checks import cap_invariance_replay, checkpoint_ranking, exact_cap_oracle_replay
from rq1.five_arm.executor import FiveArmEpisodeExecutor, ProtocolViolation
from rq1.five_arm.measurements import truncate_to_cap, validate_record
from rq1.five_arm.memory import FillerCorpus
from rq1.five_arm.prompting import compare_arm_prompts, split_sections
from rq1.five_arm.protocol import ProtocolError, assert_five_arm_records
from rq1.five_arm.queue import QueueError, QueueHalted, RunQueue


def events(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


class _Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def arm_executor(self, pilot_spec, driver, **kw):
        policy = runner.FiveArmMemoryPolicy(manifest=pilot_spec["manifest"], skill_texts=SKILL_TEXTS, corpus=pilot_spec["corpus"],
                                            tokenizer=E.TOK, tolerance=TOLERANCE, d0_representation="absent_memory_section")
        return FiveArmEpisodeExecutor(driver, mode="arm", tasks=pilot_spec["tasks"], task_meta=pilot_spec["metas"], split="valid_seen",
                                      synthetic=True, tokenizer=E.TOK, runtime_identity_sha256="f" * 64,
                                      exposure_policy="synthetic", d0_representation="absent_memory_section",
                                      library=pilot_spec["library"], embedder=pilot_spec["embedder"], memory_policy=policy,
                                      leakage_probes=pilot_spec["probes"], chat_template_tokens=E.counter,
                                      provider_token_tolerance=0, **kw)

    def queue(self, spec, executor, kind="five_arm", retries=2, name="out"):
        return RunQueue(output_root=self.root / name, manifest=spec["manifest"], executor=executor, kind=kind,
                        max_infrastructure_retries=retries, synthetic=True,
                        interruption_record=executor.interruption_record if hasattr(executor, "interruption_record") else None)


class FiveArmEndToEndTests(_Base):
    def test_all_five_arms_run_through_the_actual_worker_path(self):
        spec = E.build_pilot(self.root)
        driver = E.ScriptedDriver(E.memory_sensitive)
        q = self.queue(spec, self.arm_executor(spec, driver))
        summary = q.run()
        self.assertEqual((summary["pending_units"], summary["final_status_counts"]), (0, {"VALID": 10}))
        self.assertTrue(q.validate()["ok"])
        records = q.records()
        self.assertEqual(sorted({r["condition"] for r in records}), ["D0", "D18", "Fill", "Irr", "Rel"])
        for r in records:
            self.assertEqual(validate_record(r), [])
            self.assertTrue(r["synthetic"])
            self.assertTrue(r["provider_calls"])
            self.assertTrue(all(c["prompt_eval_count"] == c["local_chat_template_tokens"] for c in r["provider_calls"]))
            self.assertEqual(r["natural_top3_skill_ids"], [x["skill_id"] for x in spec["selections"][r["task_id"]]["natural_top3"]])
        by = {(r["task_id"], r["condition"]): r for r in records}
        task = sorted(spec["tasks"])[0]
        self.assertFalse(by[(task, "D0")]["success"])                                  # no memory -> scripted model wanders
        self.assertEqual(by[(task, "D0")]["termination_reason"], "budget_exhausted")
        self.assertEqual(by[(task, "D18")]["injected_skill_ids"], by[(task, "D18")]["natural_top3_skill_ids"])
        self.assertEqual(by[(task, "Rel")]["injected_skill_ids"], [spec["selections"][task]["rel"]["skill_id"]])
        # the first recovery prompt of each arm, from the persisted episode events
        first = {}
        for arm in ("D0", "D18", "Rel", "Irr", "Fill"):
            unit = next(u for u in spec["manifest"]["units"] if u["task_id"] == task and u["condition"] == arm)
            ev = events(q.root / "attempts" / unit["unit_key"][:24] / "a1" / "episode" / "episode-events.jsonl")
            prompts = [e["payload"]["prompt"] for e in ev if e["event"] == "model_selection"]
            first[arm] = prompts[0]
            memories = {split_sections(p).get("RECOVERY MEMORY") for p in prompts}
            self.assertEqual(len(memories), 1)                                         # static memory after ONE retrieval
            retrievals = (q.root / "attempts" / unit["unit_key"][:24] / "a1" / "result" / "retrieval.jsonl").read_text().splitlines()
            self.assertEqual(len(retrievals), 1)                                       # retrieval exactly once, every arm
            self.assertNotIn(task, prompts[0])
        self.assertTrue(compare_arm_prompts(first, d0_representation="absent_memory_section")["ok"])
        self.assertNotIn("RECOVERY MEMORY", first["D0"])
        self.assertIn(SKILL_TEXTS[spec["selections"][task]["rel"]["skill_id"]], first["Rel"])
        # analysis runs from saved records with the frozen arm subsets
        design = {"tasks": spec["metas"], "arm_subsets": spec["manifest"]["arm_subsets"], "seeds": [11], "max_infrastructure_retries": 2}
        out = analysis.analyze(records, design, bootstrap_replicates=200, permutation_draws=200)
        self.assertEqual(out["primary"]["D18-D0"]["estimate"], 1.0)
        self.assertTrue(out["synthetic_input"])

    def test_missing_rel_candidate_keeps_natural_pair(self):
        labels = lambda task_id, top: ({s: "IRRELEVANT" for s in top} if task_id.endswith("T2")
                                       else {top[0]: "RELEVANT", top[1]: "IRRELEVANT", top[2]: "PARTIAL"})
        spec = E.build_pilot(self.root, labels=labels)
        man = spec["manifest"]
        t2 = "valid_seen:beta/trial_T2"
        arms = sorted(u["condition"] for u in man["units"] if u["task_id"] == t2)
        self.assertEqual(arms, ["D0", "D18", "Fill", "Irr"])
        self.assertNotIn(t2, man["arm_subsets"]["Rel"])
        q = self.queue(spec, self.arm_executor(spec, E.ScriptedDriver(E.solver)))
        q.run()
        self.assertTrue(q.validate()["complete"])

    def test_blocked_filler_cell_never_removes_the_primary_pair(self):
        unapproved = FillerCorpus("synthetic-neutral-v0", NEUTRAL_SENTENCES, {})
        spec = E.build_pilot(self.root, corp=unapproved)
        man = spec["manifest"]
        self.assertEqual(sorted({u["condition"] for u in man["units"]}), ["D0", "D18"])
        self.assertEqual({c["condition"] for c in man["blocked_cells"]}, {"Rel", "Irr", "Fill"})
        self.assertEqual(man["arm_subsets"], {"Rel": [], "Irr": [], "Fill": []})

    def test_budget_exhaustion_trace_and_truncation(self):
        spec = E.build_pilot(self.root, cap=12)
        q = self.queue(spec, self.arm_executor(spec, E.ScriptedDriver(E.looker)))
        q.run()
        rec = q.records()[0]
        self.assertEqual((rec["outcome_status"], rec["termination_reason"], rec["recovery_actions"]), ("VALID", "budget_exhausted", 10))
        self.assertEqual(len(rec["recovery_trace"]), 10)
        self.assertEqual(truncate_to_cap(rec, 8)["termination_reason"], "budget_exhausted")
        solved = self.queue(spec, self.arm_executor(spec, E.ScriptedDriver(E.solver)), name="solved")
        solved.run()
        win = solved.records()[0]
        self.assertEqual((win["success"], win["recovery_actions"]), (True, 2))
        self.assertTrue(truncate_to_cap(win, 4)["success"])
        self.assertFalse(truncate_to_cap(win, 3)["success"])                          # allowance 1 < 2 needed actions

    def test_invalid_selection_is_an_outcome_not_retried(self):
        spec = E.build_pilot(self.root)
        q = self.queue(spec, self.arm_executor(spec, E.ScriptedDriver(E.garbage)))
        q.run()
        recs = q.records()
        self.assertTrue(all(r["termination_reason"] == "invalid_action_limit" and r["outcome_status"] == "VALID" for r in recs))
        self.assertTrue(all(r["attempt"] == 1 and r["invalid_actions"] == 3 for r in recs))

    def test_provider_failure_retry_limit_and_transient_recovery(self):
        spec = E.build_pilot(self.root, task_ids=("valid_seen:alpha/trial_T1",))
        calls = {"n": 0}

        def flaky(prompt):
            calls["n"] += 1
            return E.provider_down(prompt) if calls["n"] == 1 else E.solver(prompt)
        q = self.queue(spec, self.arm_executor(spec, E.ScriptedDriver(flaky)))
        q.run()
        first_unit = spec["manifest"]["units"][0]["unit_key"]
        rows = [r for r in q.records() if r["unit_key"] == first_unit]
        self.assertEqual([(r["attempt"], r["outcome_status"], r["termination_reason"]) for r in rows],
                         [(1, "MISSING_INFRASTRUCTURE", "provider_error"), (2, "VALID", "success")])
        down = self.queue(spec, self.arm_executor(spec, E.ScriptedDriver(E.provider_down)), name="down")
        down.run()
        attempts = {}
        for r in down.records():
            attempts.setdefault(r["unit_key"], []).append(r["outcome_status"])
        self.assertTrue(all(v == ["MISSING_INFRASTRUCTURE"] * 3 for v in attempts.values()))   # 1 + 2 retries, then stop
        self.assertTrue(down.validate()["ok"])
        self.assertEqual(down.summary(stopped=False)["pending_units"], 0)

    def test_environment_failure_is_infrastructure(self):
        spec = E.build_pilot(self.root, task_ids=("valid_seen:alpha/trial_T1",))
        driver = E.ScriptedDriver(E.solver, env_fail=lambda _t: 3)                         # fails at the 3rd bridge step
        q = self.queue(spec, self.arm_executor(spec, driver), retries=0)
        q.run()
        self.assertTrue(all(r["termination_reason"] == "environment_error" and r["success"] is None for r in q.records()))

    def test_interruption_resume_never_duplicates_scientific_units(self):
        spec = E.build_pilot(self.root, task_ids=("valid_seen:alpha/trial_T1",))
        executor = self.arm_executor(spec, E.ScriptedDriver(E.solver))
        count = {"n": 0}

        def dying(unit, **kw):
            count["n"] += 1
            if count["n"] == 3:
                raise KeyboardInterrupt
            return executor(unit, **kw)
        dying.interruption_record = executor.interruption_record
        with self.assertRaises(KeyboardInterrupt):
            self.queue(spec, dying).run()
        resumed_calls = {"n": 0}

        def counting(unit, **kw):
            resumed_calls["n"] += 1
            return executor(unit, **kw)
        counting.interruption_record = executor.interruption_record
        q = self.queue(spec, counting)
        q.run()
        self.assertEqual(resumed_calls["n"], 3)                                            # 2 done units never re-run
        report = q.validate()
        self.assertTrue(report["ok"] and report["complete"])
        rows = [r for r in q.records() if r["termination_reason"] == "infrastructure_interruption"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(sum(r["outcome_status"] == "VALID" for r in q.records()), 5)
        q.run()                                                                            # idempotent
        self.assertEqual(resumed_calls["n"], 3)

    def test_leakage_and_ranking_drift_halt_before_the_provider(self):
        sentences = tuple(x.replace(".", " skill_03.") for x in NEUTRAL_SENTENCES)
        base = FillerCorpus("leaky", sentences)
        leaky = FillerCorpus("leaky", sentences, {"approved": True, "approved_by": "FIXTURE", "approval_reference": "unit-test",
                                                  "corpus_sha256": base.sha256})
        spec = E.build_pilot(self.root, task_ids=("valid_seen:alpha/trial_T1",), corp=leaky)
        driver = E.ScriptedDriver(E.solver)
        q = self.queue(spec, self.arm_executor(spec, driver))
        with self.assertRaises(QueueHalted):
            q.run()
        journal = events(q.root / "journal.jsonl")
        violation = [e for e in journal if e["event"] == "protocol_violation"]
        self.assertTrue(violation and "skill_03" in violation[0]["detail"])
        leaked_prompts = [p for p in driver.prompts if "skill_03" in p]
        self.assertEqual(leaked_prompts, [])                                               # never sent to the provider
        spec2 = E.build_pilot(self.root / "drift", task_ids=("valid_seen:alpha/trial_T1",))
        sel = spec2["manifest"]["selections"]["valid_seen:alpha/trial_T1"]
        sel["natural_top3"] = list(reversed(sel["natural_top3"]))
        q2 = self.queue(spec2, self.arm_executor(spec2, E.ScriptedDriver(E.solver)), name="drift")
        with self.assertRaises(QueueHalted):
            q2.run()

    def test_cpu_ranking_matches_live_retrieval_and_cap_tools(self):
        spec = E.build_pilot(self.root, task_ids=("valid_seen:alpha/trial_T1",))
        task = spec["tasks"]["valid_seen:alpha/trial_T1"]
        q = self.queue(spec, self.arm_executor(spec, E.ScriptedDriver(E.solver)))
        q.run()
        live = next(r for r in q.records() if r["condition"] == "D18")["natural_top3_skill_ids"]
        self.assertEqual(spec["rankings"][task.task_id]["natural_top3"], live)
        self.assertEqual(len(spec["rankings"][task.task_id]["ranking"]), 18)
        free = E.ScriptedDriver(E.solver)
        replay = exact_cap_oracle_replay(free, task, cap=12, output_dir=self.root / "oracle", split="valid_seen")
        self.assertTrue(replay["validated"])
        self.assertFalse(exact_cap_oracle_replay(free, task, cap=4, output_dir=self.root / "oracle4", split="valid_seen")["validated"])
        inv = cap_invariance_replay(free, task, caps=(8, 12), output_dir=self.root / "inv", split="valid_seen")
        self.assertEqual(inv["status"], "PASS")
        self.assertTrue(inv["truncation_environment_assumptions_hold"])
        self.assertIn("UNVERIFIED", inv["model_trajectory_equivalence"])

    def test_cap_invariance_never_counts_an_untested_comparison_as_pass(self):
        """Regression (Phase 2E): a route whose prefix + detour cannot run below the smaller cap compared only pre-detour
        (or zero) states and was reported as holding."""
        spec = E.build_pilot(self.root, task_ids=("valid_seen:alpha/trial_T1",))
        task = spec["tasks"]["valid_seen:alpha/trial_T1"]
        free = E.ScriptedDriver(E.solver)
        untested = cap_invariance_replay(free, task, caps=(2, 12), output_dir=self.root / "inv2", split="valid_seen")
        self.assertEqual(untested["status"], "UNTESTED")
        self.assertFalse(untested["truncation_environment_assumptions_hold"])
        self.assertEqual(untested["post_detour_states_compared"], 0)
        from dataclasses import replace
        tampered = replace(task, post_detour_digest="0" * 64)
        failed = cap_invariance_replay(free, tampered, caps=(8, 12), output_dir=self.root / "inv3", split="valid_seen")
        self.assertEqual(failed["status"], "FAIL")
        self.assertFalse(failed["truncation_environment_assumptions_hold"])
        self.assertEqual(free.prompts, [])                                                 # no model calls from CPU tools
        with self.assertRaises(ProtocolViolation):
            harness_driver = E.ScriptedDriver(E.solver)
            from rq1.five_arm.cpu_checks import _NoModelDriver
            session = _NoModelDriver(harness_driver).session(output_dir=self.root / "s", run_id="r", attempt_id="a")
            session._model_response("prompt")


class DiagnosticEndToEndTests(_Base):
    def diag_queue(self, spec, driver, name="diag"):
        common = dict(tasks=spec["tasks"], task_meta=spec["metas"], split="valid_seen", synthetic=True, tokenizer=E.TOK,
                      runtime_identity_sha256="f" * 64, exposure_policy="synthetic", d0_representation="absent_memory_section",
                      chat_template_tokens=E.counter, provider_token_tolerance=0)
        ex = {"UNDISTURBED": FiveArmEpisodeExecutor(driver, mode="diagnostic_undisturbed", **common),
              "DETOURED": FiveArmEpisodeExecutor(driver, mode="diagnostic_detoured", **common)}
        dispatch = lambda unit, **kw: ex[unit["condition"]](unit, **kw)
        return RunQueue(output_root=self.root / name, manifest=spec["manifest"], executor=dispatch, kind="diagnostic",
                        max_infrastructure_retries=2, synthetic=True,
                        interruption_record=lambda u, a, d: ex[u["condition"]].interruption_record(u, a, d))

    def test_undisturbed_and_detoured_conditions(self):
        spec = E.build_diagnostic(self.root)
        q = self.diag_queue(spec, E.ScriptedDriver(E.solver))
        q.run()
        self.assertTrue(q.validate()["ok"])
        recs = q.records()
        self.assertEqual(len(recs), 8)
        for r in recs:
            self.assertEqual((r["injected_skill_ids"], r["natural_top3_skill_ids"], r["filler_corpus_sha256"]), ([], [], None))
            self.assertEqual(r["recovery_action_allowance"], 12 - 1 - 1)
            self.assertEqual(r["detour_actions"], 0 if r["condition"] == "UNDISTURBED" else 1)
        und = next(r for r in recs if r["condition"] == "UNDISTURBED")
        det = next(r for r in recs if r["condition"] == "DETOURED" and r["task_id"] == und["task_id"] and r["seed"] == und["seed"])
        self.assertNotEqual(und["first_recovery_prompt_sha256"], det["first_recovery_prompt_sha256"])   # documented difference
        with self.assertRaises(ProtocolError):                                                         # never merged
            assert_five_arm_records(recs)
        th = dict(pilot.PROPOSED_DIAGNOSTIC_THRESHOLDS)
        report = pilot.disruption_report(recs, th, replicates=200)
        self.assertEqual(report["tasks"], 2)
        self.assertIn(report["disruption_class"], tuple(pilot.RECOMMENDATIONS))
        self.assertIn("CEILING", report["methodological_findings"])                                   # scripted solver always succeeds
        variability = pilot.seed_variability(recs)
        self.assertEqual(variability["identical_trajectory_rate"], 1.0)                               # scripted = deterministic
        evidence = [cap_invariance_replay(E.ScriptedDriver(E.solver), t, caps=(8, 12), output_dir=self.root / f"inv-{i}",
                                          split="valid_seen") for i, t in enumerate(spec["tasks"].values())]
        decision = pilot.budget_decision(recs, candidate_caps=(8, 12), late_success_threshold=0.2, floor=0.1,
                                         cap_invariance_evidence=evidence)
        self.assertEqual(decision["chosen_cap"], 8)
        self.assertEqual(pilot.budget_decision(recs, candidate_caps=(8, 12), late_success_threshold=0.2, floor=0.1,
                                               cap_invariance_evidence=[])["decision"], "TRUNCATION_NOT_SUPPORTED")

    def test_diagnostic_executor_refuses_arm_conditions_and_retrieval(self):
        spec = E.build_diagnostic(self.root)
        with self.assertRaises(ProtocolViolation):
            FiveArmEpisodeExecutor(E.ScriptedDriver(E.solver), mode="diagnostic_detoured", tasks=spec["tasks"], task_meta=spec["metas"],
                                   split="valid_seen", synthetic=True, tokenizer=E.TOK, runtime_identity_sha256="f" * 64,
                                   exposure_policy="x", d0_representation="absent_memory_section", library=E.library())


class RealExecutionGateTests(_Base):
    def test_synthetic_configuration_can_never_drive_real_execution(self):
        from rq1.hermes.episode_driver import RealEpisodeDriver
        spec = E.build_pilot(self.root, task_ids=("valid_seen:alpha/trial_T1",))
        real_driver = RealEpisodeDriver(self.root, data_dir=self.root)
        common = dict(tasks=spec["tasks"], task_meta=spec["metas"], split="valid_seen", tokenizer=E.TOK,
                      runtime_identity_sha256="f" * 64, exposure_policy="x", d0_representation="absent_memory_section")
        with self.assertRaises(ProtocolViolation):
            FiveArmEpisodeExecutor(real_driver, mode="diagnostic_detoured", synthetic=True, **common)
        with self.assertRaises(ProtocolViolation):
            FiveArmEpisodeExecutor(E.ScriptedDriver(E.solver), mode="diagnostic_detoured", synthetic=False, **common)
        with self.assertRaises(QueueError):
            RunQueue(output_root=self.root, manifest=spec["manifest"], executor=lambda *a, **k: None, kind="five_arm",
                     max_infrastructure_retries=2, synthetic=False, interruption_record=None)

    def test_gate_runs_before_any_real_component(self):
        spec = E.build_pilot(self.root, task_ids=("valid_seen:alpha/trial_T1",))
        fixture_approval = {"approved": True, "approved_by": "FIXTURE", "approval_reference": "unit-test",
                            "artifact_sha256": spec["manifest"]["manifest_sha256"]}
        frozen = {**spec["manifest"], "synthetic": False, "status": "frozen", "approval": fixture_approval}
        config = {"manifest": frozen}
        with mock.patch("rq1.university_gpu.ollama.probe_endpoint") as probe, \
                mock.patch("rq1.hermes.episode_driver.RealEpisodeDriver") as driver:
            for state in ({"local_tests_passed": True}, {"local_tests_passed": True, "synthetic": True}):
                with self.assertRaises(readiness.ExecutionBlocked):
                    runner.build_university_queue(config, readiness_state=state, observed_runtime=None,
                                                  approved_runtime=runtime.proposed_runtime_config())
            with self.assertRaises(readiness.ExecutionBlocked):
                runner.build_university_queue({"manifest": spec["manifest"]}, readiness_state={}, observed_runtime=None,
                                              approved_runtime=runtime.proposed_runtime_config())
            probe.assert_not_called()
            driver.assert_not_called()

    def test_old_six_arm_worker_cannot_launch_the_five_arm_protocol(self):
        from rq1.university_gpu.manifest import validate_manifest as six_arm_validate
        spec = E.build_pilot(self.root, task_ids=("valid_seen:alpha/trial_T1",))
        self.assertTrue(any("unsupported manifest schema" in p for p in six_arm_validate(spec["manifest"])))
        six_arm_like = {"schema": "rq1-university-gpu-manifest-v1", "stage": "pilot", "units": []}
        with self.assertRaises(readiness.ExecutionBlocked):
            runner.assert_execution_permitted(manifest=six_arm_like, readiness_state={}, artifacts={}, observed_runtime=None,
                                              approved_runtime=runtime.proposed_runtime_config())


if __name__ == "__main__":
    unittest.main()
