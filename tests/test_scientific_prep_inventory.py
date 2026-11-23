from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rq1.scientific_prep.identity import TaskIdentityError, normalize_task_reference, parse_task_id, template_key
from rq1.scientific_prep.inventory import InventoryError, build_inventory
from rq1.tasks.discovery import discover_tasks
from scientific_prep_fakes import write_task

T1 = "pick_heat_then_place_in_recep-Apple-None-Fridge-10/trial_T20190908_000001_000001"
T2 = "pick_heat_then_place_in_recep-Apple-None-Fridge-10/trial_T20190908_000002_000002"   # same template, other trial
T3 = "look_at_obj_in_light-Mug-None-DeskLamp-308/trial_T20190908_000003_000003"
T4 = "pick_and_place_simple-Mug-None-Desk-219/trial_T20190908_000004_000004"
S1 = "pick_cool_then_place_in_recep-Cup-None-SideTable-21/trial_T20190906_000005_000005"
S2 = "pick_two_obj_and_place-Book-None-Desk-302/trial_T20190906_000006_000006"


def fixture(root: Path) -> None:
    write_task(root, "valid_unseen", T1, "pick_heat_then_place_in_recep", game_text="IDENTICAL-GAME")
    write_task(root, "valid_unseen", T2, "pick_heat_then_place_in_recep", game_text="IDENTICAL-GAME")
    write_task(root, "valid_unseen", T3, "look_at_obj_in_light")
    write_task(root, "valid_unseen", T4, "pick_and_place_simple", game=False)          # indexed, not installed
    write_task(root, "valid_seen", S1, "pick_cool_then_place_in_recep")
    write_task(root, "valid_seen", S2, "pick_two_obj_and_place")
    write_task(root, "train", "pick_and_place_simple-Pen-None-Desk-301/trial_T1_1_1", "pick_and_place_simple")


class IdentityTests(unittest.TestCase):
    def test_parse_and_templates(self):
        i = parse_task_id("valid_unseen:" + T1)
        self.assertEqual((i.family, i.object, i.recep, i.scene), ("heat_and_place", "Apple", "Fridge", "10"))
        j = parse_task_id("valid_unseen:" + T2)
        self.assertNotEqual(i.task_id, j.task_id)
        self.assertEqual(template_key(i, "goal_template"), template_key(j, "goal_template"))   # distinct trials, one template
        self.assertEqual(template_key(i, "scene_template"), template_key(j, "scene_template"))
        for bad in ("valid_unseen:nonsense", "test:" + T1, T1, "valid_seen:pick-A-None-B-1/not_a_trial"):
            with self.assertRaises(TaskIdentityError):
                parse_task_id(bad)
        alfred_only = parse_task_id("valid_unseen:pick_and_place_with_movable_recep-ButterKnife-Cup-CounterTop-10/trial_T1_2_3")
        self.assertIsNone(alfred_only.family)          # stable identity, never a runnable ALFWorld family

    def test_reference_normalization_is_cross_format(self):
        expected = "valid_unseen:" + T1
        for ref in (expected, f"json_2.1.1/valid_unseen/{T1}/traj_data.json", f"$ALFWORLD_DATA/json_2.1.1/valid_unseen/{T1}/game.tw-pddl",
                    f"C:\\data\\json_2.1.1\\valid_unseen\\{T1.replace('/', chr(92))}\\traj_data.json"):
            self.assertEqual(expected, normalize_task_reference(ref), ref)
        self.assertIsNone(normalize_task_reference("some free text"))
        self.assertIsNone(normalize_task_reference(""))


class InventoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        fixture(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_split_enumeration_missing_games_and_family_counts(self):
        inv = build_inventory(self.root, ["valid_seen", "valid_unseen", "train"])
        u = inv["splits"]["valid_unseen"]
        self.assertEqual((u["indexed_entries"], u["installed_runnable_games"], u["missing_game_files"]), (4, 3, 1))
        self.assertEqual(u["installed_by_family"], {"heat_and_place": 2, "look_at_object": 1})
        self.assertEqual(u["installed_templates"]["goal_template"], 2)              # two trials share one template
        self.assertEqual(inv["splits"]["valid_seen"]["installed_runnable_games"], 2)
        self.assertEqual(inv["splits"]["train"]["installed_runnable_games"], 1)
        missing = [e for e in inv["entries"] if e["task_id"] == "valid_unseen:" + T4][0]
        self.assertFalse(missing["installed"])
        self.assertIn("missing_game_file", missing["problems"])

    def test_duplicate_file_hashes_are_grouped(self):
        inv = build_inventory(self.root, ["valid_unseen"])
        groups = inv["splits"]["valid_unseen"]["duplicate_game_hashes"]
        self.assertEqual(len(groups), 1)
        self.assertEqual(sorted(groups[0]["task_ids"]), ["valid_unseen:" + T1, "valid_unseen:" + T2])

    def test_stable_hashes_and_deterministic_ordering(self):
        a = build_inventory(self.root, ["valid_unseen", "valid_seen"])
        b = build_inventory(self.root, ["valid_seen", "valid_unseen"])
        self.assertEqual(a, b)
        ids = [e["task_id"] for e in a["entries"] if e["split"] == "valid_unseen"]
        self.assertEqual(ids, sorted(ids))
        (self.root / "json_2.1.1" / "valid_seen" / S1 / "game.tw-pddl").write_text("changed", encoding="utf-8")
        c = build_inventory(self.root, ["valid_seen", "valid_unseen"])
        self.assertNotEqual(a["splits"]["valid_seen"]["discovery_identity"], c["splits"]["valid_seen"]["discovery_identity"])

    def test_identity_matches_existing_discovery_function(self):
        inv = build_inventory(self.root, ["valid_seen"])
        self.assertEqual(inv["splits"]["valid_seen"]["discovery_identity"],
                         discover_tasks(self.root, "valid_seen").data_root_identity)

    def test_problems_are_reported_not_hidden(self):
        bad = self.root / "json_2.1.1" / "valid_seen" / "pick_and_place_simple-X-None-Y-1/trial_bad_1"
        bad.mkdir(parents=True)
        (bad / "traj_data.json").write_text("not json", encoding="utf-8")
        (bad / "game.tw-pddl").write_text("{}", encoding="utf-8")
        orphan = self.root / "json_2.1.1" / "valid_seen" / "look_at_obj_in_light-A-None-B-2/trial_orphan_1"
        orphan.mkdir(parents=True)
        (orphan / "game.tw-pddl").write_text("{}", encoding="utf-8")
        inv = build_inventory(self.root, ["valid_seen"])
        problems = {e["task_id"]: e["problems"] for e in inv["entries"] if e["problems"]}
        self.assertTrue(any("malformed_trajectory_metadata" in p for p in problems.values()))
        self.assertTrue(any("orphan_game_without_trajectory" in p for p in problems.values()))
        self.assertEqual(inv["splits"]["valid_seen"]["orphan_games"], 1)

    def test_unsupported_alfred_type_without_game_is_expected_not_anomalous(self):
        before = build_inventory(self.root, ["valid_unseen"])["splits"]["valid_unseen"]["entries_with_problems"]
        write_task(self.root, "valid_unseen", "pick_and_place_with_movable_recep-ButterKnife-Cup-CounterTop-10/trial_T1_2_3",
                   "pick_and_place_with_movable_recep", game=False)
        inv = build_inventory(self.root, ["valid_unseen"])
        self.assertEqual(inv["splits"]["valid_unseen"]["entries_with_problems"], before)
        self.assertEqual(inv["splits"]["valid_unseen"]["missing_game_files"], 2)
        row = [e for e in inv["entries"] if "movable_recep" in e["task_id"]][0]
        self.assertIsNone(row["family"])
        self.assertIn("unsupported_task_type", row["problems"])

    def test_missing_inputs_fail_clearly(self):
        with self.assertRaises(InventoryError):
            build_inventory(self.root / "nowhere", ["valid_seen"])
        with self.assertRaises(InventoryError):
            build_inventory(self.root, ["test"])


if __name__ == "__main__":
    unittest.main()
