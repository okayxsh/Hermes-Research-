"""Statistics, SAP implementation, pilot diagnostics and power simulation -- SYNTHETIC data; not scientific results."""
from __future__ import annotations

import unittest

from rq1.five_arm import analysis, pilot, power, stats
from rq1.five_arm.measurements import (MeasurementError, injected_relevance, make_diagnostic_record, make_record,
                                       retrieved_precision_at_3, truncate_to_cap)
from rq1.five_arm.protocol import ProtocolError

TOP3 = ["s1", "s2", "s3"]
INJECTED = {"D0": [], "D18": TOP3, "Rel": ["s1"], "Irr": ["s3"], "Fill": []}
ALLOWANCE = 46


def trace(actions, success, *, rejections=0, exhausted=False):
    steps = [{"decision": i, "action": "look", "rejections": [], "env_action_valid": True, "done": False, "success": False,
              "selection_exhausted": False} for i in range(1, actions + 1)]
    if success and steps:
        steps[-1].update(done=True, success=True)
    if rejections and steps:
        steps[0]["rejections"] = ["no_action_index"] * rejections
    if exhausted:
        steps.append({"decision": len(steps) + 1, "action": None, "rejections": ["no_action_index"] * 3, "env_action_valid": None,
                      "done": True, "success": None, "selection_exhausted": True})
    return steps


def common(task, arm, seed, *, attempt, template, scene):
    return dict(synthetic=True, unit_key=f"{task}|{arm}|{seed}", attempt=attempt, task_id=task, goal_template=template or f"tpl-{task}",
                scene=scene, family="clean_and_place", exposure_policy="synthetic", oracle_id="o", oracle_sha256="c" * 64,
                checkpoint_digest="d" * 64, post_detour_digest="e" * 64, condition=arm, seed=seed, runtime_identity_sha256="f" * 64,
                first_recovery_prompt_sha256=None, prompt_sequence_sha256=None, provider_calls=[], requested_provider_options={},
                tokens={"raw_text_tokens": 1, "memory_block_tokens": 0, "chat_template_tokens": None, "provider_prompt_tokens": None},
                total_action_cap=50, prefix_actions=3, recovery_action_allowance=ALLOWANCE, wall_clock_seconds=10.0,
                inference_latency_seconds=2.0)


def rec(task, arm, seed, success, *, status="VALID", attempt=1, actions=5, invalid=0, template=None, scene="10", reason=None):
    valid = status == "VALID"
    reason = reason or (("success" if success else "budget_exhausted") if valid else "provider_error")
    n = (actions if success else ALLOWANCE) if valid else 0
    return make_record(
        **common(task, arm, seed, attempt=attempt, template=template, scene=scene), executor_mode="arm",
        natural_top3_skill_ids=TOP3, natural_top3_scores=[0.9, 0.8, 0.7], injected_skill_ids=INJECTED[arm],
        injected_skill_count=len(INJECTED[arm]), analysis_only_injected_labels=[],
        filler_corpus_sha256="a" * 64 if arm in ("Rel", "Irr", "Fill") else None, filler_sentence_indices=[],
        memory_block_sha256=None if arm == "D0" else "b" * 64, detour_actions=1, recovery_actions=n,
        recovery_trace=trace(n, bool(success) and valid, rejections=invalid) if valid else [],
        success=(bool(success) if valid else None), termination_reason=reason, invalid_actions=invalid if valid else 0,
        outcome_status=status, failure_detail=None if valid else "synthetic provider fault",
        not_applicable=["memory"] if arm == "D0" else [])


def diag(task, condition, seed, success, *, actions=4, invalid=0, status="VALID"):
    valid = status == "VALID"
    n = (actions if success else ALLOWANCE) if valid else 0
    return make_diagnostic_record(
        **common(task, condition, seed, attempt=1, template=None, scene="10"),
        executor_mode={"UNDISTURBED": "diagnostic_undisturbed", "DETOURED": "diagnostic_detoured"}[condition],
        natural_top3_skill_ids=[], natural_top3_scores=[], injected_skill_ids=[], injected_skill_count=0,
        analysis_only_injected_labels=[], filler_corpus_sha256=None, filler_sentence_indices=[], memory_block_sha256=None,
        detour_actions=0 if condition == "UNDISTURBED" else 1, recovery_actions=n,
        recovery_trace=trace(n, bool(success) and valid, rejections=invalid) if valid else [],
        success=bool(success) if valid else None, termination_reason=("success" if success else "budget_exhausted") if valid else "provider_error",
        invalid_actions=invalid if valid else 0, outcome_status=status, failure_detail=None if valid else "x",
        not_applicable=["memory", "retrieval"])


def design(tasks, seeds=(11,), rel=None, irr=None, fill=None, scenes=None, templates=None, overlap=()):
    return {"tasks": {t: {"goal_template": (templates or {}).get(t, f"tpl-{t}"), "scene": (scenes or {}).get(t, "10"),
                          "family": "clean_and_place", "acquisition_template_overlap": t in overlap} for t in tasks},
            "arm_subsets": {"Rel": list(rel if rel is not None else tasks), "Irr": list(irr if irr is not None else tasks),
                            "Fill": list(fill if fill is not None else tasks)},
            "seeds": list(seeds), "max_infrastructure_retries": 2}


def full_records(tasks, seeds, outcome):
    return [rec(t, a, s, outcome(t, a, s)) for t in tasks for a in ("D0", "D18", "Rel", "Irr", "Fill") for s in seeds]


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
        self.assertEqual(mc, stats.sign_flip_test([1.0] * 25 + [-1.0] * 5, draws=5000, seed=1))

    def test_cluster_bootstrap_and_normal_approximation(self):
        clusters = {"a": [1.0, 1.0, 1.0], "b": [0.0], "c": [0.0, 0.0]}
        ci = stats.cluster_bootstrap_ci(clusters, seed=1, replicates=2000)
        self.assertTrue(0.0 <= ci[0] <= 0.5 <= ci[1] <= 1.0)
        self.assertIsNone(stats.cluster_bootstrap_ci({"a": [1.0]}, seed=1))
        exact = stats.sign_flip_test([1.0] * 12 + [-1.0] * 3, exact_max=20)["p_value"]
        self.assertAlmostEqual(stats.normal_sign_flip_p([1.0] * 12 + [-1.0] * 3), exact, delta=0.03)
        self.assertEqual(stats.normal_sign_flip_p([0.0, 0.0]), 1.0)


class AnalysisTests(unittest.TestCase):
    def test_single_primary_with_template_cluster_interval(self):
        tasks = [f"t{i}" for i in range(10)]
        records = full_records(tasks, [11], lambda t, a, s: a != "D0" or int(t[1:]) >= 6)    # D0 fails on t0..t5
        out = analysis.analyze(records, design(tasks), **FAST)
        self.assertEqual(list(out["primary"]), ["D18-D0"])
        p = out["primary"]["D18-D0"]
        self.assertAlmostEqual(p["estimate"], 0.6)
        self.assertAlmostEqual(p["estimate_percentage_points"], 60.0)
        self.assertAlmostEqual(p["p_value"], 2 / 64)
        self.assertEqual(p["mcnemar_exact"]["discordant_treatment_better"], 6)
        self.assertIsNotNone(p["ci_95_template_cluster_bootstrap"])
        self.assertIn("sharp null", p["null_hypothesis"])
        self.assertEqual(p["multiplicity"], "single primary comparison; no adjustment")
        self.assertEqual(sorted(out["secondary_exploratory"]), ["D18-Fill", "Irr-Fill", "Rel-Fill", "Rel-Irr"])
        self.assertTrue(all(v["role"] == "secondary_exploratory" for v in out["secondary_exploratory"].values()))
        self.assertEqual(out["secondary_exploratory"]["D18-Fill"]["estimate"], 0.0)
        self.assertTrue(out["synthetic_input"])

    def test_correlated_seeds_templates_and_family_scene_table(self):
        tasks = [f"t{i}" for i in range(6)]
        templates = {t: f"tpl{int(t[1:]) // 2}" for t in tasks}
        records = [rec(t, a, s, (a == "D18") or (s == 11), template=templates[t]) for t in tasks
                   for a in ("D0", "D18", "Rel", "Irr", "Fill") for s in (11, 29, 47)]
        out = analysis.analyze(records, design(tasks, seeds=(11, 29, 47), templates=templates), **FAST)
        c = out["primary"]["D18-D0"]
        self.assertAlmostEqual(c["estimate"], 2 / 3)
        self.assertIn("not applicable", c["mcnemar_exact"])
        self.assertEqual(c["template_sensitivity"]["templates"], 3)
        self.assertEqual(c["paired_seed_observations"], 18)
        self.assertEqual(c["scene_descriptive"]["family_by_scene"], {"clean_and_place": {"10": 6}})

    def test_missing_cells_infrastructure_retries_and_bounds(self):
        tasks = [f"t{i}" for i in range(4)]
        records = [r for r in full_records(tasks, [11], lambda t, a, s: a == "D18")
                   if (r["task_id"], r["condition"]) not in (("t0", "D0"), ("t1", "Fill"))]
        records.append(rec("t1", "Fill", 11, None, status="MISSING_INFRASTRUCTURE", attempt=1))
        records.append(rec("t1", "Fill", 11, False, attempt=2))
        out = analysis.analyze(records, design(tasks), **FAST)
        self.assertEqual(out["missingness_by_arm"]["D0"]["by_status"].get("MISSING_CELL"), 1)
        self.assertEqual(out["primary"]["D18-D0"]["tasks_analysed"], 3)
        self.assertIn("t0", out["primary"]["D18-D0"]["excluded_tasks"])
        bounds = out["primary"]["D18-D0"]["missing_cell_bounds"]
        self.assertLessEqual(bounds["worst_case_estimate"], bounds["best_case_estimate"])
        self.assertEqual(out["missingness_by_arm"]["Fill"]["valid"], 4)

    def test_retry_rule_violations(self):
        base = [r for r in full_records(["t0"], [11], lambda *_: True) if r["condition"] != "D0"]
        for bad in ([rec("t0", "D0", 11, False, attempt=1), rec("t0", "D0", 11, True, attempt=2)],
                    [rec("t0", "D0", 11, None, status="MISSING_INFRASTRUCTURE", attempt=a) for a in (1, 2, 3)] + [rec("t0", "D0", 11, True, attempt=4)],
                    [rec("t0", "D0", 11, True, attempt=2)]):
            with self.assertRaises(analysis.AnalysisError):
                analysis.analyze(base + bad, design(["t0"]), **FAST)

    def test_arm_subsets_including_blocked_fill(self):
        tasks = ["t0", "t1", "t2"]
        keep = lambda r: not (r["condition"] == "Rel" and r["task_id"] != "t0") and not (r["condition"] == "Fill" and r["task_id"] == "t2")
        records = [r for r in full_records(tasks, [11], lambda t, a, s: a in ("Irr", "D18")) if keep(r)]
        out = analysis.analyze(records, design(tasks, rel=["t0"], fill=["t0", "t1"]), **FAST)
        self.assertEqual(out["secondary_exploratory"]["Rel-Irr"]["tasks_in_universe"], 1)
        self.assertEqual(out["secondary_exploratory"]["Irr-Fill"]["tasks_in_universe"], 2)
        self.assertEqual(out["primary"]["D18-D0"]["tasks_in_universe"], 3)              # blocked Fill never removes D18-D0
        with self.assertRaises(analysis.AnalysisError):
            analysis.analyze(full_records(tasks, [11], lambda *_: True), design(tasks, rel=["t0"]), **FAST)

    def test_censored_effort_scene_and_overlap_sensitivity(self):
        tasks = [f"t{i}" for i in range(6)]
        scenes = {t: str(10 + int(t[1:]) % 3) for t in tasks}
        records = [rec(t, a, 11, a != "D0", scene=scenes[t], actions=4) for t in tasks for a in ("D0", "D18", "Rel", "Irr", "Fill")]
        out = analysis.analyze(records, design(tasks, scenes=scenes, overlap=("t0",)), **FAST)
        effort = out["secondary_effort"]["D18-D0:capped_recovery_actions"]
        self.assertAlmostEqual(effort["mean_task_difference"], 4 - ALLOWANCE)
        self.assertEqual(len(out["primary"]["D18-D0"]["scene_descriptive"]["leave_one_scene_out"]), 3)
        self.assertEqual(out["acquisition_overlap_sensitivity"]["excluded_tasks"], ["t0"])
        self.assertEqual(out["cumulative_success_by_recovery_actions"]["D0"][-1], 0.0)

    def test_retrieved_versus_injected_relevance(self):
        labels = {("t0", "s1"): "RELEVANT", ("t0", "s2"): "PARTIAL", ("t0", "s3"): "IRRELEVANT"}
        d18, rel, irr, fill = (rec("t0", a, 11, True) for a in ("D18", "Rel", "Irr", "Fill"))
        self.assertAlmostEqual(retrieved_precision_at_3(rel, labels), 1 / 3)
        self.assertAlmostEqual(retrieved_precision_at_3(rel, labels, "lenient"), 2 / 3)
        self.assertEqual(injected_relevance(rel, labels)["injected_relevant_fraction"], 1.0)
        self.assertEqual(injected_relevance(irr, labels)["injected_irrelevant_count"], 1)
        self.assertEqual(injected_relevance(fill, labels)["status"], "not_applicable_zero_injected")
        with self.assertRaises(MeasurementError):
            retrieved_precision_at_3(d18, {**labels, ("t0", "s2"): "UNCERTAIN"})
        out = analysis.analyze([rec("t0", a, 11, True) for a in ("D0", "D18", "Rel", "Irr", "Fill")], design(["t0"]), labels=labels, **FAST)
        self.assertEqual(out["relevance"]["injected_relevance_by_arm"]["Fill"]["not_applicable_zero_injected"], 1)

    def test_invalid_inputs(self):
        with self.assertRaises(ProtocolError):
            analysis.analyze([{**rec("t0", "D0", 11, True), "protocol_version": "rq1-selective-memory-six-arm-v1"}], design(["t0"]), **FAST)
        with self.assertRaises(ProtocolError):                            # diagnostic records are never five-arm data
            analysis.analyze([diag("t0", "DETOURED", 11, True)], design(["t0"]), **FAST)
        fields = {k: v for k, v in rec("t0", "Rel", 11, True).items() if k not in ("schema", "protocol_version")}
        with self.assertRaises(MeasurementError):
            make_record(**{**fields, "injected_skill_ids": [], "injected_skill_count": 0})
        with self.assertRaises(MeasurementError):
            make_record(**{**fields, "outcome_status": "MISSING_INFRASTRUCTURE", "termination_reason": "provider_error", "success": False})
        with self.assertRaises(MeasurementError):                         # trace must agree with the outcome
            make_record(**{**fields, "recovery_trace": trace(5, False)})

    def test_truncation(self):
        r = rec("t0", "D0", 11, True, actions=30)
        self.assertTrue(truncate_to_cap(r, 50)["success"])
        self.assertEqual(truncate_to_cap(r, 30)["termination_reason"], "budget_exhausted")     # allowance 26 < 30
        self.assertFalse(truncate_to_cap(r, 4)["feasible"])
        with self.assertRaises(MeasurementError):
            truncate_to_cap(r, 60)


class PilotDiagnosticTests(unittest.TestCase):
    TH = pilot.PROPOSED_DIAGNOSTIC_THRESHOLDS

    def records(self, n_tasks, und, det, und_actions=4, det_actions=4):
        out = []
        for i in range(n_tasks):
            out.append(diag(f"t{i}", "UNDISTURBED", 11, und(i), actions=und_actions))
            out.append(diag(f"t{i}", "DETOURED", 11, det(i), actions=det_actions))
        return out

    def test_classifications(self):
        meaningful = pilot.disruption_report(self.records(12, lambda i: True, lambda i: i % 2 == 0, det_actions=8), self.TH, replicates=500)
        self.assertEqual(meaningful["disruption_class"], "MEANINGFUL_DISRUPTION")
        self.assertEqual(meaningful["criteria"]["extra_effort"], "meets")
        ceiling = pilot.disruption_report(self.records(12, lambda i: True, lambda i: True, det_actions=5), self.TH, replicates=500)
        self.assertIn("CEILING", ceiling["methodological_findings"])
        self.assertEqual(ceiling["disruption_class"], "INSUFFICIENT_DISRUPTION")
        floor = pilot.disruption_report(self.records(12, lambda i: i < 2, lambda i: False), self.TH, replicates=500)
        self.assertIn("FLOOR", floor["methodological_findings"])
        self.assertIn("TASK_ABILITY_FLOOR", floor["methodological_findings"])
        uncertain = pilot.disruption_report(self.records(12, lambda i: i < 8, lambda i: i < 6), self.TH, replicates=500)
        self.assertIn(uncertain["criteria"]["completion_drop"], ("indeterminate", "below"))
        self.assertNotEqual(uncertain["recommendation"], "proceed to five-arm pilot review")

    def test_interface_and_infrastructure_findings(self):
        recs = self.records(4, lambda i: True, lambda i: True)
        recs = [{**r} for r in recs]
        noisy = [diag(f"t{i}", c, 11, True, invalid=3) for i in range(4) for c in ("UNDISTURBED", "DETOURED")]
        self.assertIn("INTERFACE_PROBLEM", pilot.disruption_report(noisy, self.TH, replicates=200)["methodological_findings"])
        broken = recs + [diag("t9", "DETOURED", 11, None, status="MISSING_INFRASTRUCTURE"),
                         diag("t9", "UNDISTURBED", 11, True)]
        self.assertIn("INFRASTRUCTURE_UNSTABLE", pilot.disruption_report(broken, self.TH, replicates=200)["methodological_findings"])
        with self.assertRaises(pilot.PilotError):
            pilot.disruption_report([rec("t0", "D0", 11, True)], self.TH)          # five-arm records are refused

    def test_budget_rule_uses_detoured_baseline_only(self):
        held = [{"task_id": f"t{i}", "truncation_environment_assumptions_hold": True} for i in range(12)]
        at100 = lambda rows: [{**r, "total_action_cap": 100, "recovery_action_allowance": 96} for r in rows]
        early = at100(self.records(12, lambda i: True, lambda i: i < 8, det_actions=10))
        self.assertEqual(pilot.budget_decision(early, candidate_caps=(50, 100), late_success_threshold=0.2, floor=0.1,
                                               cap_invariance_evidence=held)["chosen_cap"], 50)
        late = at100([diag(f"t{i}", "DETOURED", 11, True, actions=40) for i in range(12)])
        late = [{**r, "recovery_actions": 60, "recovery_trace": trace(60, True)} for r in late]
        self.assertEqual(pilot.budget_decision(late, candidate_caps=(50, 100), late_success_threshold=0.2, floor=0.1,
                                               cap_invariance_evidence=held)["chosen_cap"], 100)
        self.assertEqual(pilot.budget_decision(late, candidate_caps=(50, 100), late_success_threshold=0.2, floor=0.1,
                                               cap_invariance_evidence=held[:3])["decision"], "TRUNCATION_NOT_SUPPORTED")


class PowerTests(unittest.TestCase):
    def test_reproducible_monotone_and_null_calibrated(self):
        small = power.Scenario(n_tasks=20, seeds=1, p_ref=0.5, delta=0.2)
        a, b = power.simulate(small, reps=60, seed=3), power.simulate(small, reps=60, seed=3)
        self.assertEqual(a, b)
        big = power.simulate(power.Scenario(n_tasks=79, seeds=3, p_ref=0.5, delta=0.2), reps=60, seed=3)
        self.assertGreaterEqual(big["power_task_sign_flip"], a["power_task_sign_flip"])
        null = power.simulate(power.Scenario(n_tasks=40, seeds=1, p_ref=0.5, delta=0.0), reps=300, seed=5)
        self.assertLess(null["power_task_sign_flip"], 0.10)
        self.assertGreater(null["mc_se_power"], 0.0)
        self.assertEqual(sorted(set(power.allocation(79, 27, "balanced"))), list(range(27)))
        skewed = power.allocation(79, 27, "skewed")
        self.assertEqual((len(skewed), len(set(skewed))), (79, 27))
        grid = power.run_grid([small], reps=20, seed=1)
        self.assertTrue(grid["synthetic"])
        self.assertTrue(any("+33.2" in note for note in grid["notes"]))
        counts = power.episode_counts(n_tasks=79, rel_tasks=40, irr_tasks=50, fill_tasks=79, seeds=1, reproducibility_tasks=12,
                                      reproducibility_extra_seeds=2)
        self.assertEqual(counts, {"per_seed": 327, "main": 327, "reproducibility_subset": 48, "total": 375})


if __name__ == "__main__":
    unittest.main()
