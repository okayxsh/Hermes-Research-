"""Statistics, SAP implementation and power simulation -- SYNTHETIC data, explicitly labelled; not scientific results."""
from __future__ import annotations

import unittest

from rq1.five_arm import analysis, power, stats
from rq1.five_arm.measurements import MeasurementError, injected_relevance, make_record, retrieved_precision_at_3
from rq1.five_arm.protocol import ProtocolError

TOP3 = ["s1", "s2", "s3"]
INJECTED = {"D0": [], "D18": TOP3, "Rel": ["s1"], "Irr": ["s3"], "Fill": []}


def rec(task, arm, seed, success, *, status="VALID", attempt=1, actions=5, invalid=0, template=None, scene="10", reason=None):
    valid = status == "VALID"
    reason = reason or (("success" if success else "budget_exhausted") if valid else "provider_error")
    return make_record(
        synthetic=True, unit_key=f"{task}|{arm}|{seed}", attempt=attempt, task_id=task, goal_template=template or f"tpl-{task}",
        scene=scene, family="clean_and_place", exposure_policy="historical_template_screened", oracle_id="o", oracle_sha256="c" * 64,
        checkpoint_digest="d" * 64, post_detour_digest="e" * 64, condition=arm, seed=seed, runtime_identity_sha256="f" * 64,
        natural_top3_skill_ids=TOP3, natural_top3_scores=[0.9, 0.8, 0.7], injected_skill_ids=INJECTED[arm],
        injected_skill_count=len(INJECTED[arm]), analysis_only_injected_labels=[], filler_corpus_sha256="a" * 64 if arm in ("Rel", "Irr", "Fill") else None,
        filler_sentence_indices=[], memory_block_sha256=None if arm == "D0" else "b" * 64, first_recovery_prompt_sha256=None,
        tokens={"raw_text_tokens": 1, "memory_block_tokens": 0, "chat_template_tokens": None, "provider_prompt_tokens": None},
        total_action_cap=50, prefix_actions=3, detour_actions=1, recovery_actions=(actions if success else 46) if valid else 0,
        success=(bool(success) if valid else None), termination_reason=reason, invalid_actions=invalid, wall_clock_seconds=10.0,
        inference_latency_seconds=2.0, outcome_status=status, not_applicable=["memory"] if arm == "D0" else [])


def design(tasks, seeds=(11,), rel=None, irr=None, scenes=None, templates=None, overlap=()):
    return {"tasks": {t: {"goal_template": (templates or {}).get(t, f"tpl-{t}"), "scene": (scenes or {}).get(t, "10"),
                          "acquisition_template_overlap": t in overlap} for t in tasks},
            "mechanism_subsets": {"Rel": list(rel if rel is not None else tasks), "Irr": list(irr if irr is not None else tasks)},
            "seeds": list(seeds), "max_infrastructure_retries": 2}


def full_records(tasks, seeds, outcome):
    return [rec(t, a, s, outcome(t, a, s), template=None) for t in tasks for a in ("D0", "D18", "Rel", "Irr", "Fill") for s in seeds]


FAST = dict(bootstrap_replicates=1000, permutation_draws=2000)


class StatsTests(unittest.TestCase):
    def test_known_values(self):
        self.assertEqual(stats.holm({"a": 0.01, "b": 0.04, "c": 0.03}), {"a": 0.03, "c": 0.06, "b": 0.06})
        self.assertAlmostEqual(stats.sign_flip_test([1.0] * 6)["p_value"], 2 / 64)
        self.assertEqual(stats.sign_flip_test([0, 0])["p_value"], 1.0)
        self.assertAlmostEqual(stats.mcnemar_exact(6, 0), 2 / 64)
        self.assertAlmostEqual(stats.mcnemar_exact(3, 3), 1.0)
        self.assertAlmostEqual(stats.t_sf_two_sided(2.0, 10), 0.07339, places=4)
        self.assertAlmostEqual(stats.t_quantile(0.975, 10), 2.2281, places=3)
        self.assertAlmostEqual(stats.cohen_kappa(["a", "a", "b", "b"], ["a", "a", "b", "a"]), 0.5)
        mc = stats.sign_flip_test([1.0] * 25 + [-1.0] * 5, draws=5000, seed=1)
        self.assertEqual(mc["method"], "monte_carlo_sign_flip")
        self.assertEqual(mc, stats.sign_flip_test([1.0] * 25 + [-1.0] * 5, draws=5000, seed=1))   # reproducible


class AnalysisTests(unittest.TestCase):
    def test_known_paired_single_seed(self):
        tasks = [f"t{i}" for i in range(10)]
        records = full_records(tasks, [11], lambda t, a, s: a != "D0" or int(t[1:]) >= 6)    # D0 fails on t0..t5
        out = analysis.analyze(records, design(tasks), **FAST)
        d18_d0 = out["primary"]["D18-D0"]
        self.assertAlmostEqual(d18_d0["estimate"], 0.6)
        self.assertAlmostEqual(d18_d0["p_value"], 2 / 64)
        self.assertEqual(d18_d0["mcnemar_exact"]["discordant_treatment_better"], 6)
        self.assertGreaterEqual(d18_d0["p_value_holm"], d18_d0["p_value"])
        self.assertEqual(out["primary"]["D18-Fill"]["estimate"], 0.0)
        self.assertTrue(out["synthetic_input"])

    def test_multiple_correlated_seeds_and_templates(self):
        tasks = [f"t{i}" for i in range(6)]
        templates = {t: f"tpl{int(t[1:]) // 2}" for t in tasks}
        records = [rec(t, a, s, (a == "D18") or (s == 11), template=templates[t]) for t in tasks
                   for a in ("D0", "D18", "Rel", "Irr", "Fill") for s in (11, 29, 47)]
        out = analysis.analyze(records, design(tasks, seeds=(11, 29, 47), templates=templates), **FAST)
        c = out["primary"]["D18-D0"]
        self.assertAlmostEqual(c["estimate"], 2 / 3)                      # task-level mean over seeds
        self.assertIn("not applicable", c["mcnemar_exact"])
        self.assertEqual(c["template_sensitivity"]["templates"], 3)
        self.assertEqual(c["paired_seed_observations"], 18)

    def test_missing_cells_infrastructure_retries_and_bounds(self):
        tasks = [f"t{i}" for i in range(4)]
        records = [r for r in full_records(tasks, [11], lambda t, a, s: a == "D18")
                   if (r["task_id"], r["condition"]) not in (("t0", "D0"), ("t1", "Fill"))]          # t0/D0 never reported
        records.append(rec("t1", "Fill", 11, None, status="MISSING_INFRASTRUCTURE", attempt=1))    # infrastructure failure ...
        records.append(rec("t1", "Fill", 11, False, attempt=2))                                     # ... retried once, valid
        out = analysis.analyze(records, design(tasks), **FAST)
        self.assertEqual(out["missingness_by_arm"]["D0"]["by_status"].get("MISSING_CELL"), 1)
        self.assertEqual(out["primary"]["D18-D0"]["tasks_analysed"], 3)
        self.assertIn("t0", out["primary"]["D18-D0"]["excluded_tasks"])
        bounds = out["primary"]["D18-D0"]["missing_cell_bounds"]
        self.assertLessEqual(bounds["worst_case_estimate"], bounds["best_case_estimate"])
        self.assertEqual(out["missingness_by_arm"]["Fill"]["valid"], 4)        # retried infrastructure unit counted once

    def test_retry_rule_violations(self):
        tasks = ["t0"]
        base = [r for r in full_records(tasks, [11], lambda *_: True) if r["condition"] != "D0"]
        for bad in ([rec("t0", "D0", 11, False, attempt=1), rec("t0", "D0", 11, True, attempt=2)],               # retried a model failure
                    [rec("t0", "D0", 11, None, status="MISSING_INFRASTRUCTURE", attempt=a) for a in (1, 2, 3)] + [rec("t0", "D0", 11, True, attempt=4)],
                    [rec("t0", "D0", 11, True, attempt=2)]):                                                       # attempt gap
            with self.assertRaises(analysis.AnalysisError):
                analysis.analyze(base + bad, design(tasks), **FAST)

    def test_mechanism_subsets_and_ineligible_cells(self):
        tasks = ["t0", "t1", "t2"]
        records = [r for r in full_records(tasks, [11], lambda t, a, s: a in ("Irr", "D18")) if not (r["condition"] == "Rel" and r["task_id"] != "t0")]
        out = analysis.analyze(records, design(tasks, rel=["t0"]), **FAST)
        self.assertEqual(out["secondary"]["Rel-Irr"]["tasks_in_universe"], 1)
        self.assertEqual(out["primary"]["Irr-Fill"]["tasks_in_universe"], 3)
        with self.assertRaises(analysis.AnalysisError):                    # Rel record for a task outside the frozen subset
            analysis.analyze(full_records(tasks, [11], lambda *_: True), design(tasks, rel=["t0"]), **FAST)

    def test_censored_recovery_and_scene_and_overlap_sensitivity(self):
        tasks = [f"t{i}" for i in range(6)]
        scenes = {t: str(10 + int(t[1:]) % 3) for t in tasks}
        records = [rec(t, a, 11, a != "D0", scene=scenes[t], actions=4) for t in tasks for a in ("D0", "D18", "Rel", "Irr", "Fill")]
        out = analysis.analyze(records, design(tasks, scenes=scenes, overlap=("t0",)), **FAST)
        effort = out["secondary_effort"]["D18-D0:capped_recovery_actions"]
        self.assertAlmostEqual(effort["mean_task_difference"], 4 - 46)        # failures held at the recovery budget
        self.assertEqual(len(out["primary"]["D18-D0"]["scene_descriptive"]["leave_one_scene_out"]), 3)
        self.assertEqual(out["acquisition_overlap_sensitivity"]["excluded_tasks"], ["t0"])
        self.assertEqual(out["cumulative_success_by_recovery_actions"]["D0"][-1], 0.0)

    def test_retrieved_versus_injected_relevance(self):
        labels = {("t0", "s1"): "RELEVANT", ("t0", "s2"): "PARTIAL", ("t0", "s3"): "IRRELEVANT"}
        d18, rel, irr, fill = (rec("t0", a, 11, True) for a in ("D18", "Rel", "Irr", "Fill"))
        self.assertAlmostEqual(retrieved_precision_at_3(rel, labels), 1 / 3)               # natural top-3, any arm
        self.assertAlmostEqual(retrieved_precision_at_3(rel, labels, "lenient"), 2 / 3)
        self.assertEqual(injected_relevance(rel, labels)["injected_relevant_fraction"], 1.0)
        self.assertEqual(injected_relevance(irr, labels)["injected_irrelevant_count"], 1)
        self.assertEqual(injected_relevance(fill, labels)["status"], "not_applicable_zero_injected")
        self.assertAlmostEqual(injected_relevance(d18, labels)["injected_relevant_fraction"], 1 / 3)
        with self.assertRaises(MeasurementError):
            retrieved_precision_at_3(d18, {**labels, ("t0", "s2"): "UNCERTAIN"})
        out = analysis.analyze([rec("t0", a, 11, True) for a in ("D0", "D18", "Rel", "Irr", "Fill")], design(["t0"]), labels=labels, **FAST)
        self.assertEqual(out["relevance"]["injected_relevance_by_arm"]["Fill"]["not_applicable_zero_injected"], 1)

    def test_invalid_inputs(self):
        with self.assertRaises(ProtocolError):
            analysis.analyze([{**rec("t0", "D0", 11, True), "protocol_version": "rq1-selective-memory-six-arm-v1"}], design(["t0"]), **FAST)
        fields = {k: v for k, v in rec("t0", "Rel", 11, True).items() if k not in ("schema", "protocol_version")}
        with self.assertRaises(MeasurementError):                         # Rel must inject exactly one skill
            make_record(**{**fields, "injected_skill_ids": [], "injected_skill_count": 0})
        with self.assertRaises(MeasurementError):                         # missing cells are never coerced into failures
            make_record(**{**fields, "outcome_status": "MISSING_INFRASTRUCTURE", "termination_reason": "provider_error", "success": False})


class PilotDiagnosticTests(unittest.TestCase):
    TH = dict(min_completion_drop=0.10, min_extra_actions=3, ceiling=0.95, floor=0.05)

    def pair(self, task, u, d, ua=6, da=6, ui=0, di=0):
        return {"task_id": task, "undisturbed_success": u, "detour_success": d, "undisturbed_actions": ua, "detour_actions": da,
                "undisturbed_invalid": ui, "detour_invalid": di}

    def test_disruption_ceiling_and_null(self):
        from rq1.five_arm.pilot import disruption_report
        disrupted = disruption_report([self.pair(f"t{i}", True, i % 3 != 0, 6, 11) for i in range(9)], **self.TH)
        self.assertTrue(disrupted["meaningful_disruption"])
        ceiling = disruption_report([self.pair(f"t{i}", True, True, 6, 7) for i in range(9)], **self.TH)
        self.assertFalse(ceiling["meaningful_disruption"])
        self.assertTrue(any("ceiling" in i for i in ceiling["methodological_issues"]))
        self.assertTrue(any("no meaningful disruption" in i for i in ceiling["methodological_issues"]))

    def test_budget_rule_is_pooled_and_pre_specified(self):
        from rq1.five_arm.pilot import PilotError, budget_decision
        row = lambda arm, success, actions: {"condition": arm, "outcome_status": "VALID", "total_action_cap": 100, "success": success,
                                             "recovery_actions": actions, "prefix_actions": 3, "detour_actions": 1}
        early = [row(a, True, 10) for a in ("D0", "D18", "Fill")] * 5 + [row("D0", False, 96)] * 3
        self.assertEqual(budget_decision(early, candidate_caps=(50, 100), late_success_threshold=0.10, floor=0.05)["chosen_cap"], 50)
        late = [row(a, True, 60) for a in ("D0", "D18")] * 4 + [row("Fill", True, 10)] * 4
        self.assertEqual(budget_decision(late, candidate_caps=(50, 100), late_success_threshold=0.10, floor=0.05)["chosen_cap"], 100)
        swapped = [{**r, "condition": {"D0": "D18", "D18": "D0"}.get(r["condition"], r["condition"])} for r in late]
        self.assertEqual(budget_decision(swapped, candidate_caps=(50, 100), late_success_threshold=0.10, floor=0.05)["chosen_cap"], 100)
        with self.assertRaises(PilotError):
            budget_decision([{**early[0], "total_action_cap": 50}], candidate_caps=(50, 100), late_success_threshold=0.1, floor=0.05)


class PowerTests(unittest.TestCase):
    def test_reproducible_and_monotone(self):
        small = power.Scenario(n_tasks=20, n_templates=7, seeds=1, p_ref=0.5, delta=0.2)
        a, b = power.simulate(small, reps=60, seed=3), power.simulate(small, reps=60, seed=3)
        self.assertEqual(a, b)
        big = power.simulate(power.Scenario(n_tasks=79, n_templates=27, seeds=3, p_ref=0.5, delta=0.2), reps=60, seed=3)
        self.assertGreaterEqual(big["power_unadjusted"], a["power_unadjusted"])
        null = power.simulate(power.Scenario(n_tasks=40, n_templates=14, seeds=1, p_ref=0.5, delta=0.0), reps=200, seed=5)
        self.assertLess(null["power_unadjusted"], 0.15)
        self.assertLessEqual(null["power_bonferroni_lower_bound_for_holm"], null["power_unadjusted"])
        grid = power.run_grid([small], reps=20, seed=1)
        self.assertTrue(grid["synthetic"])
        self.assertTrue(any("+33.2" in note for note in grid["notes"]))


if __name__ == "__main__":
    unittest.main()
