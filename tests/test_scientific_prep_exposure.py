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


@unittest.skipUnless(os.environ.get("RQ1_SCI_PREP_HISTORICAL_EVIDENCE"), "set RQ1_SCI_PREP_HISTORICAL_EVIDENCE to the "
                     "directory holding the frozen historical manifests to reproduce the published screening counts")
class HistoricalScreeningReproduction(unittest.TestCase):
    """Reproduces the historical valid_unseen screening bounds from the REAL frozen manifests (no ALFWorld data needed:
    the evaluation manifest lists every valid_unseen index entry as a selected task or an exclusion with a reason)."""

    def test_reproduces_94_and_79(self):
        base = Path(os.environ["RQ1_SCI_PREP_HISTORICAL_EVIDENCE"])
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
        inventory = {"splits": {"valid_unseen": {}}, "entries": entries}
        config = {"schema": EXP.SOURCES_SCHEMA, "template_definition": "goal_template", "screening_policies": POLICIES, "sources": [
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
        broader = pools["policies"]["acquisition_engineering_template_screened"]["tasks"]
        print(f"\n[historical reproduction] 94 -> 79 reproduced; goal_template acquisition/engineering screen -> {broader} "
              "(the university-reported 77 used an undocumented narrower rule; see report)")


if __name__ == "__main__":
    unittest.main()
