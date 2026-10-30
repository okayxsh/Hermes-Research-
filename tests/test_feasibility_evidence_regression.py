"""Negative regressions for engineering-only evidence; all providers/assets are fixtures."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from tests.test_university_gpu_feasibility import candidate_manifest, make_fake_frozen_manifest, FakeWorker, ROOT
from rq1.university_gpu.scheduler import RunStore, Coordinator, SchedulerError, _validate_feasibility_result
from rq1.university_gpu.evidence import inventory_attempt
from rq1.university_gpu.snapshot import export_progress, verify_progress

class EvidenceRegressionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.manifest = make_fake_frozen_manifest(candidate_manifest())
        self.store = RunStore(self.base / "run", self.manifest)
        self.store.initialize()
        self.unit = next(u for u in self.manifest["units"] if u["condition"] == "D18")
        self.attempt = self.store.attempts_dir / self.unit["unit_key"] / "001-fixtureattempt"
        self.result = FakeWorker("fixture").execute(self.unit, self.attempt, "fixtureattempt")

    def publish(self):
        return self.store.record_result(self.unit, self.result, attempt_id="fixtureattempt", worker_id="fixture")

    def test_valid_partial_export_is_portable_and_resume_keeps_authority(self):
        self.publish()
        out = self.base / "export"
        report = export_progress(ROOT, self.store.root, self.store.manifest_path, out)
        self.assertFalse(report["complete"])
        self.assertEqual(verify_progress(out)["status"], "PASS")
        with self.assertRaisesRegex(SchedulerError, "cannot be overwritten"):
            self.publish()
        changed = copy.deepcopy(self.manifest)
        changed["source_commit"] = "f" * 40
        from rq1.university_gpu.feasibility import _manifest_digest
        changed["manifest_sha256"] = _manifest_digest(changed)
        with self.assertRaisesRegex(SchedulerError, "different frozen manifest"):
            RunStore(self.store.root, changed).initialize()

    def test_missing_or_mutated_log_blocks_completed_reads_and_export(self):
        self.publish()
        path = self.attempt / "result/retrieval.jsonl"
        original = path.read_bytes()
        for mutation in (None, original + b" "):
            with self.subTest(mutation=mutation):
                if mutation is None: path.unlink()
                else: path.write_bytes(mutation)
                with self.assertRaisesRegex(SchedulerError, "mandatory trace"):
                    self.store.completed()
                with self.assertRaises((SchedulerError, ValueError)):
                    export_progress(ROOT, self.store.root, self.store.manifest_path, self.base / "bad-export")
                path.write_bytes(original)

    def test_semantic_negative_and_boundary_counts(self):
        for updates in ({"actions": -1}, {"actions": 10000}, {"steps": -1}, {"invalid_actions": -1},
                        {"steps": 2}, {"success": True}, {"recovery_success": "yes"}):
            with self.subTest(updates=updates):
                value = copy.deepcopy(self.result); value.update(updates)
                self.assertTrue(_validate_feasibility_result(value, self.unit, self.manifest))
        self.assertEqual(_validate_feasibility_result(self.result, self.unit, self.manifest), [])
        value = copy.deepcopy(self.result)
        budget = self.unit["recovery_action_budget"]
        value.update(actions=budget, steps=budget)
        value["measurements"].update(post_failure_actions=budget, recovery_actions=["look"] * budget,
                                     termination_reason="recovery_budget_exhausted")
        self.assertEqual(_validate_feasibility_result(value, self.unit, self.manifest), [])
        value["actions"] += 1
        self.assertTrue(_validate_feasibility_result(value, self.unit, self.manifest))

    def test_invalid_retrieval_values(self):
        for ids, scores in [(["nonexistent"] * 3,[0.5] * 3),(["skill-00"] * 3,[0.5]*3),
                            (["skill-00","skill-01","skill-02"],[float("nan"),0,0]),
                            (["skill-00","skill-01","skill-02"],[float("inf"),0,0]),
                            (["skill-00","skill-01","skill-02"],[1.5,0,0]),([],[])]:
            with self.subTest(ids=ids,scores=scores):
                value=copy.deepcopy(self.result); value.update(retrieved_skill_ids=ids,retrieval_similarity_scores=scores)
                self.assertTrue(_validate_feasibility_result(value,self.unit,self.manifest))

    def test_rehashed_conflicting_raw_contents_rejected(self):
        path=self.attempt / "result/retrieval.jsonl"
        data=json.loads(path.read_text()); data["attempt_id"]="anotherattempt"
        path.write_text(json.dumps(data)+"\n")
        self.result["raw_evidence"]=inventory_attempt(self.attempt,self.result)
        with self.assertRaisesRegex(SchedulerError,"identity mismatch"):
            self.publish()

    def test_inventory_traversal_and_identity_rejected(self):
        self.result["raw_evidence"]["files"][0]["path"]="../../outside"
        with self.assertRaises(SchedulerError): self.publish()
        self.result["raw_evidence"]=inventory_attempt(self.attempt,self.result)
        self.result["raw_evidence"]["attempt_id"]="other"
        with self.assertRaises(SchedulerError): self.publish()

    def test_episode_dispatch_conflict_rejected_even_with_updated_hash(self):
        path=self.attempt / "episode/episode-events.jsonl"
        data=[json.loads(x) for x in path.read_text().splitlines()]
        data[1]["payload"]["request"]["action"]="substituted action"
        path.write_text("".join(json.dumps(e)+"\n" for e in data))
        self.result["raw_evidence"]=inventory_attempt(self.attempt,self.result)
        with self.assertRaisesRegex(SchedulerError,"dispatched actions"):
            self.publish()

    def test_raw_environment_conflict_rejected_even_with_updated_hash(self):
        path=self.attempt / "episode/episode-events.jsonl"
        data=[json.loads(x) for x in path.read_text().splitlines()]
        data[-2]["payload"]["response"]["done"]=True
        data[-2]["payload"]["response"]["success"]=True
        path.write_text("".join(json.dumps(e)+"\n" for e in data))
        self.result["raw_evidence"]=inventory_attempt(self.attempt,self.result)
        with self.assertRaisesRegex(SchedulerError,"environment response"):
            self.publish()

    def test_no_action_selection_exhaustion_is_valid_non_action_step(self):
        self.result.update(steps=1, invalid_actions=1)
        self.result["measurements"].update(termination_reason="action_selection_exhausted", selection_exhausted=True)
        (self.attempt / "result/recovery_steps.jsonl").write_text(json.dumps({
            "step": None, "phase": "recovery", "action": None, "action_valid": False,
            "done": False, "success": None, "controller_failure": "fixture exhaustion"}) + "\n")
        self.result["raw_evidence"]=inventory_attempt(self.attempt,self.result)
        self.publish()
        self.assertEqual(len(self.store.completed()),1)
