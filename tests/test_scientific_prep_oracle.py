from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rq1.recovery.reference_route import ReferenceRouteError
from rq1.scientific_prep.horizons import horizon_rows
from rq1.scientific_prep.inventory import build_inventory
from rq1.scientific_prep.oracle import validate_candidate
from scientific_prep_fakes import ROUTE, FakeDriver, write_task

TASK = "pick_and_place_simple-Apple-None-Cabinet-21/trial_T20190906_000001_000001"
TID = "valid_seen:" + TASK


class HorizonTests(unittest.TestCase):
    def test_total_action_accounting_and_cap_boundaries(self):
        rows = {r["total_action_cap"]: r for r in horizon_rows(48, 1, 30, caps=(50, 100))}
        self.assertEqual(rows[50]["remaining_recovery_actions"], 1)        # 50 - 48 - 1: total actions != recovery actions
        self.assertEqual((rows[50]["status"], rows[50]["oracle_fits"]), ("FAILED", False))
        self.assertEqual(rows[100]["remaining_recovery_actions"], 51)
        self.assertEqual(rows[100]["status"], "ARITHMETICALLY_FEASIBLE")     # arithmetic never claims environment validation
        self.assertEqual(rows[100]["total_oracle_actions"], 79)
        infeasible = horizon_rows(74, 1, 10, caps=(50,))[0]
        self.assertFalse(infeasible["checkpoint_feasible"])
        self.assertEqual(infeasible["status"], "FAILED")
        validated = horizon_rows(3, 1, 2, caps=(5, 6), environment_validated=True)
        self.assertEqual([r["status"] for r in validated], ["FAILED", "ENVIRONMENT_VALIDATED"])   # fits one cap, not the other
        with self.assertRaises(ValueError):
            horizon_rows(-1, 1, 2)


class OracleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        write_task(self.root / "data", "valid_seen", TASK, "pick_and_place_simple")
        self.record = [e for e in build_inventory(self.root / "data", ["valid_seen"])["entries"] if e["task_id"] == TID][0]
        self.count = 0

    def tearDown(self):
        self.tmp.cleanup()

    def run_oracle(self, *, route=ROUTE, record=None, **kwargs):
        self.count += 1
        driver = FakeDriver(**kwargs.pop("world", {}))
        provider = route if callable(route) else (lambda t, s: route)
        return validate_candidate(record or self.record, driver=driver, route_provider=provider, data_root=self.root / "data",
                                  output_dir=self.root / f"out-{self.count}", **kwargs)

    def test_valid_checkpoint_detour_and_environment_validated_oracle(self):
        r = self.run_oracle(caps=(5, 6, 50))
        self.assertEqual(r["status"], "ENVIRONMENT_VALIDATED", r)
        o = r["oracle"]
        self.assertEqual((o["checkpoint_index"], o["detour_action"], o["expected_next_action"]), (3, "go to drawer 1", "go to cabinet 1"))
        self.assertEqual((o["prefix_length"], o["detour_length"], o["continuation_length"], o["total_oracle_actions"]), (3, 1, 2, 6))
        self.assertTrue(o["digests_reproduced_in_independent_session"])
        self.assertEqual([h["status"] for h in o["horizons"]], ["FAILED", "ENVIRONMENT_VALIDATED", "ENVIRONMENT_VALIDATED"])
        self.assertEqual(r["checkpoint_attempts"][0]["code"], "checkpoint_not_navigation_eligible")   # midpoint first, as historically
        self.assertEqual(r["model_calls"], 0)
        self.assertFalse(r["model_visible"])
        self.assertTrue(o["evidence_files"]["session-b-oracle"])
        self.assertEqual((o["environment_log_confirmation"]["logged_steps"], o["environment_log_confirmation"]["logged_final_success"]),
                         (6, True))
        self.assertEqual(r["stages"]["route_replay"]["environment_log"]["logged_steps"], len(ROUTE))

    def test_in_memory_success_without_persisted_environment_evidence_is_not_validated(self):
        missing = self.run_oracle(world={"log_fault": "omit"})
        self.assertEqual((missing["status"], missing["code"]), ("FAILED", "environment_log_missing"))
        contradicted = self.run_oracle(world={"log_fault": "unsuccessful"})
        self.assertEqual((contradicted["status"], contradicted["code"]), ("FAILED", "environment_log_does_not_confirm_success"))
        self.assertNotIn("oracle", contradicted)

    def test_missing_route_is_blocked_not_fabricated(self):
        def no_route(t, s):
            raise ReferenceRouteError("hand-coded expert did not expose a next action")
        r = self.run_oracle(route=no_route)
        self.assertEqual((r["status"], r["code"]), ("BLOCKED", "no_expert_route"))
        self.assertNotIn("oracle", r)
        self.assertEqual(self.run_oracle(route=())["status"], "BLOCKED")

    def test_route_that_does_not_apply_fails(self):
        r = self.run_oracle(route=("go to drawer 1", "take apple 1 from countertop 1", "go to cabinet 1"))
        self.assertEqual((r["status"], r["code"]), ("FAILED", "route_replay_action_not_admissible"))
        r = self.run_oracle(route=ROUTE[:-1])                                           # never completes the task
        self.assertEqual((r["status"], r["code"]), ("FAILED", "route_not_completing"))

    def test_action_budget_exhaustion(self):
        r = self.run_oracle(max_total_actions=4)
        self.assertEqual(r["status"], "FAILED")
        self.assertIn(r["code"], ("route_replay_terminal_too_early", "route_not_completing"))

    def test_invalid_checkpoint_and_invalid_detours(self):
        short = ("go to countertop 1", "take apple 1 from countertop 1", "go to cabinet 1", "put apple 1 in/on cabinet 1")
        r = self.run_oracle(route=short, world={"locations": ("cabinet 1", "countertop 1")})   # no alternative navigation
        self.assertEqual((r["status"], r["code"]), ("FAILED", "no_checkpoint_admits_validated_controlled_failure"))
        self.assertIn("no_reversible_navigation_detour", {a.get("code") for a in r["checkpoint_attempts"]})
        r = self.run_oracle(world={"detour_noop": True}, max_checkpoints=2)
        self.assertEqual(r["status"], "FAILED")
        self.assertIn("detour_no_state_change", {a.get("code") for a in r["checkpoint_attempts"]})

    def test_broken_oracle_continuation(self):
        r = self.run_oracle(world={"oracle_breaks": True}, max_checkpoints=2)
        self.assertEqual(r["status"], "FAILED")
        self.assertIn("oracle_continuation_action_not_admissible", {a.get("code") for a in r["checkpoint_attempts"]})

    def test_rejected_checkpoint_falls_back_in_historical_midpoint_order(self):
        r = self.run_oracle(world={"oracle_breaks": True})
        self.assertEqual(r["status"], "ENVIRONMENT_VALIDATED")
        self.assertEqual([a["checkpoint_index"] for a in r["checkpoint_attempts"]], [2, 3, 1])
        self.assertEqual(r["oracle"]["checkpoint_index"], 1)
        self.assertEqual(r["oracle"]["detour_action"], "go to cabinet 1")

    def test_state_hash_mismatch_across_independent_sessions(self):
        r = self.run_oracle(world={"nondeterministic": True})
        self.assertEqual((r["status"], r["code"]), ("FAILED", "checkpoint_state_hash_mismatch"))

    def test_selection_cap_and_source_identity(self):
        r = self.run_oracle(selection_cap=5)
        self.assertEqual(r["status"], "FAILED")
        self.assertIn("reference_rejoin_exceeds_recovery_budget", {a.get("code") for a in r["checkpoint_attempts"]})
        (self.root / "data" / "json_2.1.1" / "valid_seen" / TASK / "game.tw-pddl").write_text("tampered", encoding="utf-8")
        r = self.run_oracle()
        self.assertEqual((r["status"], r["code"]), ("BLOCKED", "source_identity_mismatch"))
        r = self.run_oracle(record={**self.record, "installed": False})
        self.assertEqual((r["status"], r["code"]), ("BLOCKED", "task_not_installed"))

    def test_arithmetic_rows_before_validation_never_claim_environment(self):
        r = self.run_oracle(world={"oracle_breaks": True}, max_checkpoints=2)
        statuses = {h["status"] for a in r["checkpoint_attempts"] for h in a["horizons"]}
        self.assertNotIn("ENVIRONMENT_VALIDATED", statuses)


if __name__ == "__main__":
    unittest.main()
