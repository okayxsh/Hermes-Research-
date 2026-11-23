from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from rq1.scientific_prep import exposure as EXP
from rq1.scientific_prep.inventory import build_inventory
from test_scientific_prep_inventory import S1, S2, T1, T2, T3, T4, fixture

REPLACED = "replaced_before_freeze_no_oracle_validated_controlled_failure"


def eval_rules():
    return {REPLACED: {"class": "POSSIBLY_EXPOSED", "role": "evaluation_preparation", "reason": "rejected oracle preparation"},
            "not_selected": {"class": None, "note": "historically route-screened; no model outcome located"},
            "missing_game_file": {"class": None, "note": "not installed"}}


POLICIES = [{"name": "historical_template_screened", "exclude_template_roles": ["evaluation", "evaluation_preparation"]},
            {"name": "acquisition_engineering_template_screened",
             "exclude_template_roles": ["evaluation", "evaluation_preparation", "acquisition", "engineering"]}]


class ExposureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        fixture(self.root / "data")
        self.inventory = build_inventory(self.root / "data", ["valid_seen", "valid_unseen"])
        ev = self.root / "evidence"
        ev.mkdir()
        (ev / "evaluation.json").write_text(json.dumps({
            "tasks": [{"task_id": "valid_unseen:" + T3}],
            "exclusions": [{"task_id": "valid_unseen:" + T1, "reason": REPLACED},
                           {"task_id": "valid_unseen:" + T2, "reason": "not_selected"},
                           {"source": f"json_2.1.1/valid_unseen/{T4}/traj_data.json", "reason": "missing_game_file"}]}), encoding="utf-8")
        (ev / "engineering.json").write_text(json.dumps({"tasks": [{"task_id": "valid_seen:" + S1}]}), encoding="utf-8")
        (ev / "acquisition.json").write_text(json.dumps({"tasks": [
            {"task_id": "train:pick_two_obj_and_place-Book-None-Desk-7/trial_T1_1_1"}], "exclusions": []}), encoding="utf-8")
        self.evidence = ev

    def tearDown(self):
        self.tmp.cleanup()

    def sources(self, extra=(), **override):
        base = [{"id": "evaluation", "kind": "task_manifest", "path": "evaluation.json", "role": "evaluation", "required": True,
                 "scope_splits": ["valid_unseen"], "selected": {"class": "EXPOSED", "reason": "360-episode evaluation"},
                 "exclusions": eval_rules()},
                {"id": "engineering", "kind": "task_list", "path": "engineering.json", "role": "engineering", "required": True,
                 "scope_splits": ["valid_seen"], "class": "EXPOSED", "reason": "500-episode D0/D18 engineering"},
                {"id": "acquisition", "kind": "task_manifest", "path": "acquisition.json", "role": "acquisition", "required": True,
                 "scope_splits": ["train"], "selected": {"class": "EXPOSED"}, "exclusions": {}},
                {"id": "demo", "kind": "inline_task_ids", "role": "demonstration", "scope_splits": ["valid_unseen"],
                 "class": "POSSIBLY_EXPOSED", "provenance": "unit-test fixture", "required": False,
                 "task_ids": [f"json_2.1.1/valid_unseen/{T3}/traj_data.json"]}]
        config = {"schema": EXP.SOURCES_SCHEMA, "template_definition": "goal_template",
                  "sources": base + list(extra), "screening_policies": POLICIES}
        config.update(override)
        path = self.evidence / f"sources-{len(list(self.evidence.iterdir()))}.json"
        path.write_text(json.dumps(config), encoding="utf-8")
        return EXP.load_sources(path)

    def ledger(self, config):
        return EXP.build_ledger(self.inventory, config, config_base=self.evidence)

    def test_classes_provenance_and_cross_source_matching(self):
        result = self.ledger(self.sources())
        self.assertEqual(result["blocked"], [])
        rows = {r["task_id"]: r for r in result["ledger"]}
        self.assertEqual(rows["valid_unseen:" + T3]["screening_class"], "EXPOSED")
        self.assertEqual({e["source"] for e in rows["valid_unseen:" + T3]["evidence"]}, {"evaluation", "demo"})
        self.assertEqual(rows["valid_unseen:" + T1]["screening_class"], "POSSIBLY_EXPOSED")
        self.assertEqual(rows["valid_unseen:" + T2]["screening_class"], "NO_MODEL_OUTCOME_FOUND")
        self.assertTrue(rows["valid_unseen:" + T2]["notes"])                        # route-screened note kept
        self.assertEqual(rows["valid_unseen:" + T4]["screening_class"], "INELIGIBLE")
        self.assertEqual(rows["valid_seen:" + S1]["screening_class"], "EXPOSED")
        for record in result["sources"]:
            if record["kind"] != "inline_task_ids":
                self.assertEqual(record["sha256"], hashlib.sha256(Path(record["path"]).read_bytes()).hexdigest())
        for row in result["ledger"]:
            for e in row["evidence"]:
                self.assertTrue(e["reason"])

    def test_template_overlap_policies_and_separate_pools(self):
        config = self.sources()
        pools = EXP.screen_pools(self.ledger(config), config)
        conf, pilot = pools["confirmatory_valid_unseen"], pools["pilot_valid_seen"]
        self.assertEqual(conf["outcome_screened"]["task_ids"], ["valid_unseen:" + T2])
        self.assertEqual(conf["policies"]["historical_template_screened"]["tasks"], 0)       # T2 shares T1's template
        self.assertEqual(conf["policies"]["historical_template_screened"]["removed_by_template_overlap"], 1)
        self.assertEqual(pilot["outcome_screened"]["task_ids"], ["valid_seen:" + S2])
        self.assertEqual(pilot["policies"]["historical_template_screened"]["tasks"], 1)
        self.assertEqual(pilot["policies"]["acquisition_engineering_template_screened"]["tasks"], 0)  # acquisition template
        self.assertEqual(pools["pilot_vs_confirmatory_and_exposure"]["pilot_confirmatory_shared_task_ids"], [])

    def test_no_promotion_from_unknown(self):
        missing = {"id": "debug-log", "kind": "task_list", "path": "absent.json", "role": "debugging", "required": False,
                   "scope_splits": ["valid_seen"], "class": "POSSIBLY_EXPOSED"}
        rows = {r["task_id"]: r for r in self.ledger(self.sources([missing]))["ledger"]}
        self.assertEqual(rows["valid_seen:" + S2]["exposure_class"], "UNKNOWN")       # not NO_MODEL_OUTCOME_FOUND
        (self.evidence / "garbled.json").write_text(json.dumps({"tasks": [{"task_id": "?? unreadable ref"}]}), encoding="utf-8")
        garbled = {"id": "garbled", "kind": "task_list", "path": "garbled.json", "role": "debugging", "required": True,
                   "scope_splits": ["valid_unseen"], "class": "EXPOSED"}
        result = self.ledger(self.sources([garbled]))
        rows = {r["task_id"]: r for r in result["ledger"]}
        self.assertEqual(rows["valid_unseen:" + T2]["exposure_class"], "UNKNOWN")
        self.assertTrue(result["uncertainty"])
        rules = eval_rules()
        rules.pop("not_selected")                                                       # undeclared reason -> UNKNOWN
        config = self.sources()
        config["sources"][0]["exclusions"] = rules
        rows = {r["task_id"]: r for r in self.ledger(config)["ledger"]}
        self.assertEqual(rows["valid_unseen:" + T2]["exposure_class"], "UNKNOWN")

    def test_required_sources_and_hashes_block(self):
        required_missing = {"id": "pilot-records", "kind": "task_list", "path": "absent.json", "role": "pilot", "required": True,
                            "scope_splits": ["valid_seen"], "class": "EXPOSED"}
        self.assertTrue(self.ledger(self.sources([required_missing]))["blocked"])
        config = self.sources()
        config["sources"][1]["expected_sha256"] = "0" * 64
        self.assertTrue(self.ledger(config)["blocked"])

    def test_expected_hash_tolerates_only_crlf_checkout_conversion(self):
        lf = json.dumps({"tasks": [{"task_id": "valid_seen:" + S1}]}, indent=1).encode()
        canonical = hashlib.sha256(lf).hexdigest()
        (self.evidence / "engineering.json").write_bytes(lf.replace(b"\n", b"\r\n"))      # Windows autocrlf checkout
        config = self.sources()
        config["sources"][1]["expected_sha256"] = canonical
        result = self.ledger(config)
        self.assertEqual(result["blocked"], [])
        self.assertEqual(next(s for s in result["sources"] if s["id"] == "engineering")["sha256_matched"], "lf_normalized")
        (self.evidence / "engineering.json").write_bytes(lf.replace(b"\n", b"\r\n").replace(b"valid_seen", b"valid_seeN"))
        self.assertTrue(self.ledger(config)["blocked"])                                     # any other change blocks

    def test_config_validation(self):
        bad = [{"id": "x", "kind": "inline_task_ids", "role": "demo", "scope_splits": ["valid_seen"], "class": "EXPOSED", "task_ids": []},
               {"id": "x", "kind": "task_list", "path": "p", "role": "r", "scope_splits": ["valid_seen"], "class": "INELIGIBLE"},
               {"id": "x", "kind": "task_manifest", "path": "p", "role": "r", "scope_splits": ["valid_seen"],
                "exclusions": {"why": {"class": "NO_MODEL_OUTCOME_FOUND"}}},
               {"id": "x", "kind": "mystery", "path": "p", "role": "r", "scope_splits": ["valid_seen"]}]
        for source in bad:
            path = self.evidence / "bad.json"
            path.write_text(json.dumps({"schema": EXP.SOURCES_SCHEMA, "sources": [source]}), encoding="utf-8")
            with self.assertRaises(EXP.ExposureConfigError):
                EXP.load_sources(path)


U = {  # valid_unseen fixture; each task is built to be removed by exactly one KIND of evidence
    "U1": "pick_heat_then_place_in_recep-Apple-None-Fridge-10/trial_T20190901_000001_000001",   # evaluated (EXPOSED)
    "U2": "pick_heat_then_place_in_recep-Apple-None-Fridge-10/trial_T20190901_000002_000002",   # goal template of U1
    "U3": "pick_clean_then_place_in_recep-Pan-None-CounterTop-10/trial_T20190901_000003_000003",  # goal template of acquisition
    "U4": "pick_heat_then_place_in_recep-Potato-None-GarbageCan-10/trial_T20190901_000004_000004",  # explicit known template
    "U5": "pick_cool_then_place_in_recep-Tomato-None-Microwave-10/trial_T20190901_000005_000005",  # type|object of engineering
    "U6": "look_at_obj_in_light-Mug-None-DeskLamp-10/trial_T20190901_000006_000006",            # trial ID reused in acquisition
    "U7": "pick_and_place_simple-Mug-None-Desk-10/trial_T20190901_000007_000007",               # no relation to any source
}
POLICY_SET = [
    {"name": "historical_template_screened", "exclude_template_roles": ["evaluation", "evaluation_preparation"]},
    {"name": "known_template_screened", "exclude_template_roles": ["evaluation", "evaluation_preparation", "known_template_match"]},
    {"name": "acquisition_template_screened", "exclude_template_roles": ["evaluation", "evaluation_preparation", "acquisition"]},
    {"name": "engineering_template_screened", "exclude_template_roles": ["evaluation", "evaluation_preparation", "engineering"]},
    {"name": "semantic_type_object_screened", "template_definition": "type_object",
     "exclude_template_roles": ["evaluation", "evaluation_preparation", "acquisition", "engineering"]},
    {"name": "exact_trial_reuse_screened", "exclude_template_roles": ["evaluation", "evaluation_preparation"],
     "exclude_trial_reuse_roles": ["acquisition", "engineering"]},
]


class PolicyTests(unittest.TestCase):
    """Each named policy must produce exactly the classification implied by its supplied sources."""

    def setUp(self):
        from scientific_prep_fakes import write_task
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        for key, rel in U.items():
            write_task(self.root / "data", "valid_unseen", rel, rel.split("-", 1)[0])
        self.inventory = build_inventory(self.root / "data", ["valid_unseen"])
        ev = self.root / "ev"
        ev.mkdir()
        (ev / "evaluation.json").write_text(json.dumps({"tasks": [{"task_id": "valid_unseen:" + U["U1"]}], "exclusions": [
            {"task_id": "valid_unseen:" + U[k], "reason": "not_selected"} for k in ("U2", "U3", "U4", "U5", "U6", "U7")]}), encoding="utf-8")
        (ev / "acquisition.json").write_text(json.dumps({"tasks": [
            {"task_id": "train:pick_clean_then_place_in_recep-Pan-None-CounterTop-22/trial_T20190801_000001_000001"},
            {"task_id": "train:look_at_obj_in_light-Bowl-None-DeskLamp-304/trial_T20190901_000006_000006"}],  # U6's trial ID
            "exclusions": []}), encoding="utf-8")
        (ev / "engineering.json").write_text(json.dumps({"tasks": [
            {"task_id": "valid_seen:pick_cool_then_place_in_recep-Tomato-None-Fridge-23/trial_T20190801_000002_000002"}]}),
            encoding="utf-8")
        (ev / "prelaunch.json").write_text(json.dumps({"valid_seen_task_id": "valid_seen:pick_cool_then_place_in_recep-Tomato-None-Fridge-23/trial_T20190801_000002_000002",
                                                       "train_task_id": "train:pick_and_place_simple-Pen-None-Desk-301/trial_T20190801_000009_000009"}), encoding="utf-8")
        self.ev = ev

    def tearDown(self):
        self.tmp.cleanup()

    def config(self, extra=(), policies=POLICY_SET):
        return {"schema": EXP.SOURCES_SCHEMA, "template_definition": "goal_template", "screening_policies": policies, "sources": [
            {"id": "evaluation", "kind": "task_manifest", "path": "evaluation.json", "role": "evaluation", "required": True,
             "scope_splits": ["valid_unseen"], "selected": {"class": "EXPOSED"}, "exclusions": eval_rules()},
            {"id": "acquisition", "kind": "task_manifest", "path": "acquisition.json", "role": "acquisition", "required": True,
             "scope_splits": ["train"], "selected": {"class": "EXPOSED"}, "exclusions": {}},
            {"id": "engineering", "kind": "task_list", "path": "engineering.json", "role": "engineering", "required": True,
             "scope_splits": ["valid_seen"], "class": "EXPOSED"},
            {"id": "known", "kind": "template_list", "role": "known_template_match", "scope_splits": ["valid_unseen"],
             "template_definition": "goal_template", "templates": ["pick_heat_then_place_in_recep|Potato|None|GarbageCan"],
             "provenance": "unit-test fixture"}, *extra]}

    def run_policies(self, config, **kw):
        EXP.validate_sources(config)
        result = EXP.build_ledger(self.inventory, config, config_base=self.ev, **kw)
        return result, EXP.screen_pools(result, config)

    def test_each_policy_removes_exactly_its_evidence(self):
        result, pools = self.run_policies(self.config())
        conf = pools["confirmatory_valid_unseen"]
        ids = lambda *keys: sorted("valid_unseen:" + U[k] for k in keys)
        self.assertEqual(sorted(conf["outcome_screened"]["task_ids"]), ids("U2", "U3", "U4", "U5", "U6", "U7"))
        expected = {"historical_template_screened": ids("U3", "U4", "U5", "U6", "U7"),
                    "known_template_screened": ids("U3", "U5", "U6", "U7"),
                    "acquisition_template_screened": ids("U4", "U5", "U6", "U7"),
                    "engineering_template_screened": ids("U3", "U4", "U5", "U6", "U7"),   # goal templates differ: none removed
                    "semantic_type_object_screened": ids("U4", "U6", "U7"),
                    "exact_trial_reuse_screened": ids("U3", "U4", "U5", "U7")}
        for name, kept in expected.items():
            self.assertEqual(sorted(conf["policies"][name]["task_ids"]), kept, name)
        removed = {r["task_id"]: r for r in conf["policies"]["exact_trial_reuse_screened"]["removed"]}
        self.assertEqual(removed["valid_unseen:" + U["U6"]]["matched_trial_reuse_roles"], ["acquisition"])
        self.assertEqual(removed["valid_unseen:" + U["U2"]]["matched_template_roles"], ["evaluation"])
        diff = {tuple(d["policies"]): d for d in conf["policy_set_differences"]}
        pair = diff[("known_template_screened", "acquisition_template_screened")]
        self.assertEqual((pair["only_in_first"], pair["only_in_second"]), (ids("U3"), ids("U4")))

    def test_overlap_is_recorded_but_is_not_exposure(self):
        result, _ = self.run_policies(self.config())
        rows = {r["task_id"].split(":", 1)[1]: r for r in result["ledger"]}
        self.assertEqual(rows[U["U1"]]["exposure_kinds"][0], "model_outcome_exposure")
        u3, u5, u6, u7 = (rows[U[k]] for k in ("U3", "U5", "U6", "U7"))
        self.assertEqual((u3["exposure_class"], u3["exposure_kinds"]), ("NO_MODEL_OUTCOME_FOUND", ["procedural_template_overlap"]))
        self.assertEqual(u3["template_overlap_roles"]["goal_template"], ["acquisition"])
        self.assertEqual(u5["template_overlap_roles"], {"goal_template": [], "scene_template": [], "type_object": ["engineering"]})
        self.assertEqual((u6["exposure_class"], u6["exact_trial_reuse_roles"]), ("NO_MODEL_OUTCOME_FOUND", ["acquisition"]))
        self.assertIn("exact_task_reuse", u6["exposure_kinds"])
        self.assertEqual((u7["exposure_kinds"], u7["evidence_complete"]), ([], True))

    def test_unresolved_sources_fail_closed_unless_explicitly_provisional(self):
        unresolved = {"id": "pilot-records", "kind": "task_list", "role": "pilot", "status": "unresolved", "required": True,
                      "scope_splits": ["valid_unseen"], "unresolved_reason": "university smoke-test task IDs not recorded"}
        result, pools = self.run_policies(self.config([unresolved]))
        self.assertEqual(result["blocked"], [])
        self.assertEqual({r["exposure_class"] for r in result["ledger"]}, {"UNKNOWN", "EXPOSED"})          # fail closed
        self.assertEqual(pools["confirmatory_valid_unseen"]["outcome_screened"]["tasks"], 0)
        self.assertFalse(pools["evidence_complete"])
        result, pools = self.run_policies(self.config([unresolved]), allow_provisional=True)
        conf = pools["confirmatory_valid_unseen"]["outcome_screened"]
        self.assertEqual(conf["tasks"], 6)
        self.assertEqual(sorted(conf["provisional_task_ids"]), sorted(conf["task_ids"]))   # every kept task carries the gap
        self.assertTrue(pools["provisional"])
        self.assertTrue(any("provisional exposure gap" in b for b in pools["roster_approval_blockers"]))
        row = next(r for r in result["ledger"] if r["task_id"].endswith(U["U7"]))
        self.assertEqual((row["exposure_class"], row["evidence_complete"]), ("NO_MODEL_OUTCOME_FOUND", False))
        self.assertEqual(row["provisional_gaps"], ["exposure source pilot-records is unresolved: university smoke-test task IDs not recorded"])

    def test_json_task_fields_and_template_list_sources(self):
        source = {"id": "gemma-demo", "kind": "json_task_fields", "path": "prelaunch.json", "role": "demonstration",
                  "pointers": ["/valid_seen_task_id", "/train_task_id"], "scope_splits": ["valid_seen", "train"],
                  "class": "EXPOSED", "required": True}
        result, pools = self.run_policies(self.config([source]))
        record = next(s for s in result["sources"] if s["id"] == "gemma-demo")
        self.assertEqual(record["sha256"], hashlib.sha256((self.ev / "prelaunch.json").read_bytes()).hexdigest())
        self.assertIn("pick_and_place_simple|Pen|None|Desk", result["referenced_templates_by_definition"]["goal_template"]["demonstration"])
        broken = dict(source, pointers=["/no_such_field"])
        result, _ = self.run_policies(self.config([broken]))
        self.assertTrue(result["uncertainty"])                                         # missing pointer is uncertainty
        self.assertNotIn("demonstration", result["referenced_templates"])
        for bad in ({"id": "t", "kind": "template_list", "role": "known", "scope_splits": ["valid_unseen"], "templates": [],
                     "template_definition": "goal_template"},                                       # no provenance
                    {"id": "t", "kind": "template_list", "role": "known", "scope_splits": ["valid_unseen"], "templates": [],
                     "template_definition": "made_up", "provenance": "x"},
                    {"id": "t", "kind": "json_task_fields", "path": "p", "role": "r", "scope_splits": ["train"],
                     "class": "EXPOSED", "pointers": []},
                    {"id": "t", "kind": "task_list", "role": "r", "scope_splits": ["train"], "status": "unresolved"}):
            with self.assertRaises(EXP.ExposureConfigError):
                EXP.validate_sources({"schema": EXP.SOURCES_SCHEMA, "sources": [bad]})
        with self.assertRaises(EXP.ExposureConfigError):
            EXP.validate_sources(self.config(policies=[*POLICY_SET, {"name": "x", "exclude_template_roles": [],
                                                                     "template_definition": "made_up"}]))

    def test_policy_naming_a_role_without_evidence_is_reported(self):
        policies = [{"name": "pilot_screened", "exclude_template_roles": ["evaluation", "pilot"]}]
        _, pools = self.run_policies(self.config(policies=policies))
        self.assertEqual(pools["confirmatory_valid_unseen"]["policies"]["pilot_screened"]["roles_without_any_evidence"], ["pilot"])
        self.assertTrue(any("pilot_screened" in b for b in pools["roster_approval_blockers"]))


@unittest.skipUnless(os.environ.get("RQ1_SCI_PREP_HISTORICAL_EVIDENCE"), "set RQ1_SCI_PREP_HISTORICAL_EVIDENCE to the "
                     "directory holding the frozen historical manifests to reproduce the published screening counts")
class HistoricalScreeningReproduction(unittest.TestCase):
    """Reproduces the historical valid_unseen screening bounds from the REAL frozen manifests (no ALFWorld data needed:
    the evaluation manifest lists every valid_unseen index entry as a selected task or an exclusion with a reason)."""

    @staticmethod
    def historical_inventory(base):
        evaluation = json.loads((base / "evaluation-amended-b83a2ecacba19992.json").read_text(encoding="utf-8"))
        from rq1.scientific_prep.identity import TEMPLATE_DEFINITIONS, normalize_task_reference, parse_task_id, template_key
        entries = []
        for item in [*evaluation["tasks"], *evaluation["exclusions"]]:
            task_id = normalize_task_reference(item.get("task_id") or item.get("source"))
            ident = parse_task_id(task_id)
            installed = item.get("reason") != "missing_game_file"
            entries.append({"task_id": task_id, "split": "valid_unseen", "relative_dir": task_id.split(":", 1)[1],
                            "installed": installed, "problems": [] if installed else ["missing_game_file"],
                            **{k: v for k, v in ident.to_dict().items() if k not in ("task_id", "split")},
                            "templates": {n: template_key(ident, n) for n in TEMPLATE_DEFINITIONS}})
        return {"splits": {"valid_unseen": {}}, "entries": entries}

    def test_shipped_template_verifies_preserved_evidence_and_fails_closed(self):
        import shutil
        repo = Path(__file__).resolve().parents[1]
        base = Path(os.environ["RQ1_SCI_PREP_HISTORICAL_EVIDENCE"])
        template = repo / "configs" / "scientific_prep" / "exposure-sources.template.json"
        self.assertNotIn("<FILL", template.read_text(encoding="utf-8"))
        config = EXP.load_sources(template)
        with tempfile.TemporaryDirectory() as tmp:
            hist = Path(tmp)
            for name, sub in (("acquisition-b64c383e0f335b90.json", "frozen"), ("acquisition-extension-4b9e3c8cda4dd809.json", "frozen_extension"),
                              ("evaluation-amended-b83a2ecacba19992.json", "frozen_evaluation")):
                (hist / "artifacts" / "task_manifests" / sub).mkdir(parents=True)
                shutil.copyfile(base / name, hist / "artifacts" / "task_manifests" / sub / name)
            for rel in ("university-gpu/prospective/feasibility-200-candidate.json",
                        "artifacts/prelaunch/prelaunch-a718b4e4d9644b01aa4861e6905cd587/pilot-report.json"):
                (hist / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(repo / rel, hist / rel)          # this checkout's bytes (CRLF on a Windows autocrlf clone)
            inventory = self.historical_inventory(base)
            closed = EXP.build_ledger(inventory, config, config_base=hist)
            self.assertEqual(closed["blocked"], [])
            for record in closed["sources"]:
                if record["status"] == "resolved":
                    self.assertIn(record["sha256_matched"], ("raw", "lf_normalized"), record["id"])
            self.assertEqual({r["exposure_class"] for r in closed["ledger"] if r["installed"] and
                              r["exposure_class"] not in ("EXPOSED", "POSSIBLY_EXPOSED")}, {"UNKNOWN"})   # fail closed
            provisional = EXP.build_ledger(inventory, config, config_base=hist, allow_provisional=True)
            pools = EXP.screen_pools(provisional, config)
            hist_policy = pools["confirmatory_valid_unseen"]["policies"]["historical_template_screened"]
            self.assertGreater(hist_policy["tasks"], 0)
            self.assertTrue(pools["provisional"] and pools["roster_approval_blockers"])
            self.assertEqual(pools["confirmatory_valid_unseen"]["policies"]["known_template_screened"]["roles_without_any_evidence"],
                             ["known_template_match"])

    def test_reproduces_94_and_79(self):
        base = Path(os.environ["RQ1_SCI_PREP_HISTORICAL_EVIDENCE"])
        inventory = self.historical_inventory(base)
        from rq1.scientific_prep.identity import normalize_task_reference, parse_task_id, template_key
        policies = [*POLICIES,
                    {"name": "known_template_screened",
                     "exclude_template_roles": ["evaluation", "evaluation_preparation", "known_template_match"]},
                    {"name": "historical_type_object_screened", "template_definition": "type_object",
                     "exclude_template_roles": ["evaluation", "evaluation_preparation"]}]
        config = {"schema": EXP.SOURCES_SCHEMA, "template_definition": "goal_template", "screening_policies": policies, "sources": [
            {"id": "hypothesis-known-template", "kind": "template_list", "role": "known_template_match",
             "scope_splits": ["valid_unseen"], "template_definition": "goal_template",
             "templates": ["pick_heat_then_place_in_recep|Potato|None|GarbageCan"],
             "provenance": "HYPOTHESIS under test: reconstructs the university 77; not an authoritative source"},
            {"id": "evaluation", "kind": "task_manifest", "path": "evaluation-amended-b83a2ecacba19992.json", "role": "evaluation",
             "required": True, "scope_splits": ["valid_unseen"], "selected": {"class": "EXPOSED"}, "exclusions": eval_rules()},
            {"id": "acquisition-180", "kind": "task_manifest", "path": "acquisition-b64c383e0f335b90.json", "role": "acquisition",
             "required": True, "scope_splits": ["train"], "selected": {"class": "EXPOSED"}, "exclusions": {}},
            {"id": "acquisition-ext-60", "kind": "task_manifest", "path": "acquisition-extension-4b9e3c8cda4dd809.json",
             "role": "acquisition", "required": True, "scope_splits": ["train"], "selected": {"class": "EXPOSED"}, "exclusions": {}},
            {"id": "engineering-six", "kind": "task_list", "path": "feasibility-200-candidate.json", "role": "engineering",
             "required": True, "scope_splits": ["valid_seen"], "class": "EXPOSED"}]}
        cfg_path = base / "__never_written__.json"
        self.assertFalse(cfg_path.exists())
        result = EXP.build_ledger(inventory, config, config_base=base)
        for unexpected in [u for u in result["uncertainty"] if "not in this inventory" not in u["problem"]]:
            self.fail(f"unexpected uncertainty: {unexpected}")
        counts = Counter(r["screening_class"] for r in result["ledger"])
        self.assertEqual(counts, Counter({"INELIGIBLE": 121, "NO_MODEL_OUTCOME_FOUND": 94, "EXPOSED": 30, "POSSIBLY_EXPOSED": 10}))
        pools = EXP.screen_pools(result, config)["confirmatory_valid_unseen"]
        self.assertEqual(pools["outcome_screened"]["tasks"], 94)
        hist = pools["policies"]["historical_template_screened"]
        self.assertEqual(hist["tasks"], 79)
        self.assertEqual(hist["by_family"], {"clean_and_place": 23, "cool_and_place": 15, "heat_and_place": 17,
                                             "look_at_object": 9, "pick_and_place": 15})
        # Acquisition/engineering goal-template policy: the removed set must equal an INDEPENDENT recomputation (any of
        # the historically screened tasks whose goal template occurs among the 240 acquisition / 6 engineering tasks).
        acq_eng = {template_key(parse_task_id(normalize_task_reference(t["task_id"])), "goal_template")
                   for name in ("acquisition-b64c383e0f335b90.json", "acquisition-extension-4b9e3c8cda4dd809.json", "feasibility-200-candidate.json")
                   for t in json.loads((base / name).read_text(encoding="utf-8"))["tasks"]}
        rows = {r["task_id"]: r for r in result["ledger"]}
        expected_kept = sorted(t for t in hist["task_ids"] if rows[t]["templates"]["goal_template"] not in acq_eng)
        broad = pools["policies"]["acquisition_engineering_template_screened"]
        self.assertEqual(sorted(broad["task_ids"]), expected_kept)
        # Hypothesis check (NOT eligibility): the university 77 per-family distribution equals the historical screen
        # minus the single goal template heat|Potato|None|GarbageCan.  A count-only match is insufficient: type_object
        # screening yields the same total with a different family distribution.
        reported_university_77 = {"clean_and_place": 23, "cool_and_place": 15, "heat_and_place": 15, "look_at_object": 9, "pick_and_place": 15}
        known = pools["policies"]["known_template_screened"]
        self.assertEqual(known["by_family"], reported_university_77)
        self.assertEqual({r["template"] for r in known["removed"] if r["matched_template_roles"] == ["known_template_match"]},
                         {"pick_heat_then_place_in_recep|Potato|None|GarbageCan"})
        type_object = pools["policies"]["historical_type_object_screened"]
        self.assertEqual(type_object["tasks"], known["tasks"])
        self.assertNotEqual(type_object["by_family"], reported_university_77)
        print(f"\n[historical reproduction] outcome {pools['outcome_screened']['tasks']} -> historical {hist['tasks']} -> "
              f"acquisition/engineering goal-template {broad['tasks']}; hypothesised known-template {known['tasks']}")


if __name__ == "__main__":
    unittest.main()
