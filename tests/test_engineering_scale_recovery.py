from __future__ import annotations
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from rq1.university_gpu import engineering_scale as scale
from rq1.university_gpu.models import content_hash, atomic_write_json
from rq1.university_gpu.scheduler import Coordinator, RunStore, SchedulerError
from rq1.university_gpu.evidence import inventory_attempt
from rq1.university_gpu.snapshot import verify_unit_checkpoint, export_progress, verify_progress
from rq1.university_gpu.scale_protection import protect_segment, verify_last_protection
from tests.test_engineering_scale import frozen
from tests.test_university_gpu_feasibility import FakeWorker as BaseWorker

ROOT=Path(__file__).resolve().parents[1]

class Worker(BaseWorker):
    def execute(self, unit, attempt_dir, attempt_id):
        r=super().execute(unit,attempt_dir,attempt_id)
        r.update(profile_id=scale.PROFILE_ID,repetition_id=unit["repetition_id"],pair_id=unit["pair_id"])
        r["measurements"].update(evaluation_profile=scale.PROFILE_ID,result_schema=scale.RESULT_SCHEMA)
        path=attempt_dir/"result/retrieval.jsonl";v=json.loads(path.read_text());v["run_id"]=scale.PROFILE_ID+"-offline-test";path.write_text(json.dumps(v)+"\n")
        r["raw_evidence"]=inventory_attempt(attempt_dir,r)
        return r

class ScaleRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.base=Path(self.temp.name)
        self.manifest=frozen(acceptance=True);self.manifest["backup_destination"]=str(self.base/"independent")
        Path(self.manifest["backup_destination"]).mkdir()
        self.manifest["engineering_approval"]["backup_destination"]=self.manifest["backup_destination"]
        self.manifest["manifest_sha256"]=scale._manifest_digest(self.manifest)
        self.store=RunStore(self.base/"run",self.manifest)

    def test_checkpoint_export_anchor_protection_and_no_duplicate_on_resume(self):
        w=[Worker(str(i)) for i in range(4)]
        result=Coordinator(self.store,w).run(max_attempts=2,storage_minimum_bytes=0,minimum_start_seconds=0)
        self.assertEqual(result["completed_units"],2)
        hashes={k:v["result_sha256"] for k,v in self.store.completed().items()}
        receipt=protect_segment(ROOT,self.store.root,self.store.manifest_path)
        self.assertEqual(receipt["status"],"VERIFIED_AT_APPROVED_DESTINATION")
        self.assertEqual(len(verify_last_protection(self.store.root,self.manifest)["unit_keys"]),2)
        result=Coordinator(self.store,w).run(resume=True,storage_minimum_bytes=0,minimum_start_seconds=0)
        self.assertEqual(result["completed_units"],4)
        for k,h in hashes.items():self.assertEqual(self.store.completed()[k]["result_sha256"],h)
        protect_segment(ROOT,self.store.root,self.store.manifest_path)
        for k in self.store.completed():self.assertEqual(verify_unit_checkpoint(self.store,k)["status"],"PASS")
        self.assertTrue(Coordinator(self.store,w).validate()["passed"])
        self.assertEqual(sum(len(x.assignments) for x in w),4)

    def test_unprotected_or_corrupt_destination_blocks_resume_gate(self):
        Coordinator(self.store,[Worker("w")]).run(max_attempts=1,storage_minimum_bytes=0,minimum_start_seconds=0)
        with self.assertRaises(FileNotFoundError):verify_last_protection(self.store.root,self.manifest)
        r=protect_segment(ROOT,self.store.root,self.store.manifest_path)
        (Path(r["destination"])/"export.tar.gz").write_bytes(b"CORRUPT TEST ONLY")
        with self.assertRaises(ValueError):verify_last_protection(self.store.root,self.manifest)

    def test_copy_failure_preserves_staging_and_source(self):
        Coordinator(self.store,[Worker("w")]).run(max_attempts=1,storage_minimum_bytes=0,minimum_start_seconds=0)
        with patch("rq1.university_gpu.snapshot.shutil.copy2",side_effect=OSError("synthetic disk full")):
            with self.assertRaises(OSError):protect_segment(ROOT,self.store.root,self.store.manifest_path)
        self.assertEqual(len(self.store.completed()),1)
        self.assertTrue(list(Path(self.manifest["backup_destination"]).glob(".incomplete-*")))

    def test_500_four_worker_matrix_in_ten_segments(self):
        m=frozen();m["backup_destination"]=self.manifest["backup_destination"]
        m["engineering_approval"]["backup_destination"]=m["backup_destination"];m["manifest_sha256"]=scale._manifest_digest(m)
        store=RunStore(self.base/"full-500",m);workers=[Worker(str(i)) for i in range(4)]
        for boundary in range(50,501,50):
            result=Coordinator(store,workers).run(resume=boundary>50,max_attempts=50,storage_minimum_bytes=0,minimum_start_seconds=0)
            self.assertEqual(result["completed_units"],boundary)
            protect_segment(ROOT,store.root,store.manifest_path)
        self.assertEqual(sum(len(w.assignments) for w in workers),500)
        self.assertTrue(Coordinator(store,workers).validate()["passed"])
        Coordinator(store,workers).merge()
        self.assertEqual(len(verify_last_protection(store.root,m)["unit_keys"]),500)

    def test_monitor_failure_stops_before_new_assignments(self):
        def failed():raise OSError("synthetic monitoring failure")
        workers=[Worker("w")]
        value=Coordinator(self.store,workers).run(storage_minimum_bytes=0,minimum_start_seconds=0,monitor=failed)
        self.assertEqual(value["completed_units"],0)
        self.assertEqual(workers[0].assignments,[])
        self.assertEqual(value["failures"][0]["failure_class"],"monitor")

    def test_selective_unit_checkpoint_check_still_rejects_raw_corruption(self):
        Coordinator(self.store,[Worker("w")]).run(max_attempts=1,storage_minimum_bytes=0,minimum_start_seconds=0)
        key=next(iter(self.store.completed()))
        path=next((self.store.root/"attempts"/key).glob("*/result/retrieval.jsonl"))
        path.write_text("SYNTHETIC CORRUPTION")
        with self.assertRaises(SchedulerError):self.store.completed(only_unit=key)

    def test_unknown_mutation_blocks_supervisor_before_worker_start(self):
        output=self.base/"supervised";target=output/self.manifest["run_id"]
        store=RunStore(target,self.manifest);store.initialize()
        unit=self.manifest["units"][0]
        store.append_event({"event":"attempt_started","unit_key":unit["unit_key"],"attempt_id":"unknown123","worker_id":"w"})
        with patch.object(scale,"_validate_engineering_approval"), patch.object(scale,"_run_supervised") as inner:
            with self.assertRaises(scale.FeasibilityError):
                scale.run_supervised(ROOT,{"run_output_dir":str(output)},self.manifest,approval=self.manifest["engineering_approval"],yes=True,resume=True)
            inner.assert_not_called()

    def test_separate_acceptance_export_is_bound_to_source_assets_and_four_workers(self):
        from rq1.university_gpu.models import load_json, sha256_file
        Coordinator(self.store,[Worker(str(i)) for i in range(4)]).run(storage_minimum_bytes=0,minimum_start_seconds=0)
        atomic_write_json(self.store.root/"launches"/"synthetic-test-only.json",{
            "manifest_sha256":self.manifest["manifest_sha256"],"gpu_placement":{"passed":True,"assignments":{str(i):"SYNTHETIC-GPU-"+str(i) for i in range(4)}}},overwrite=False)
        destination=self.base/"acceptance-export"
        export_progress(ROOT,self.store.root,self.store.manifest_path,destination)
        metadata=load_json(destination/"EXPORT-MANIFEST.json")
        # Explicit synthetic provenance, used ONLY to exercise the acceptance contract.
        metadata["analysis_source"]={"clean":True,"commit":"a"*40,"matches_run_commit":True}
        atomic_write_json(destination/"EXPORT-MANIFEST.json",metadata)
        production=frozen();production["engineering_approval"]["acceptance_export_manifest_sha256"]=sha256_file(destination/"EXPORT-MANIFEST.json")
        config={"acceptance_export_path":str(destination)}
        self.assertEqual(scale.verify_acceptance_export(config,production)["status"],"PASS")
        production["source_commit"]="c"*40
        with self.assertRaises(scale.FeasibilityError):scale.verify_acceptance_export(config,production)

    def test_protected_external_anchor_detects_changed_source_optional_trace(self):
        Coordinator(self.store,[Worker("w")]).run(max_attempts=1,storage_minimum_bytes=0,minimum_start_seconds=0)
        protect_segment(ROOT,self.store.root,self.store.manifest_path)
        path=next((self.store.root/"attempts").glob("*/*/raw-attempt.jsonl"))
        path.write_text("SYNTHETIC SOURCE TAMPERING")
        self.assertEqual(len(self.store.completed()),1)
        with self.assertRaises(ValueError):verify_last_protection(self.store.root,self.manifest)

    def test_final_summary_is_descriptive_and_create_only(self):
        Coordinator(self.store,[Worker(str(i)) for i in range(4)]).run(storage_minimum_bytes=0,minimum_start_seconds=0)
        scale.finalize_run(self.store.root,self.manifest)
        summary=json.loads((self.store.root/"ENGINEERING-SUMMARY.json").read_text())
        self.assertEqual(summary["completed"],4)
        self.assertEqual(len(summary["paired_results"]),2)
        self.assertEqual(summary["conditions"]["D0"]["completed"],2)
        self.assertEqual(summary["conditions"]["D18"]["completed"],2)
        self.assertIs(summary["scientific_evidence"],False)
        scale.finalize_run(self.store.root,self.manifest)
        (self.store.root/"ENGINEERING-SUMMARY.json").write_text("{}")
        with self.assertRaises(scale.FeasibilityError):scale.finalize_run(self.store.root,self.manifest)

    def test_checkpoint_flush_failure_preserves_authority_and_resume_rebuilds(self):
        workers=[Worker("w")]
        with patch("rq1.university_gpu.snapshot._fsync_path",side_effect=OSError("synthetic flush failure")):
            result=Coordinator(self.store,workers).run(max_attempts=1,storage_minimum_bytes=0,minimum_start_seconds=0)
        self.assertEqual(result["status"],"durability_failed")
        self.assertEqual(len(self.store.completed()),1)
        original=next(iter(self.store.completed().values()))["result_sha256"]
        resumed=Coordinator(self.store,workers).run(resume=True,storage_minimum_bytes=0,minimum_start_seconds=0)
        self.assertEqual(resumed["completed_units"],4)
        self.assertEqual(len(workers[0].assignments),4)
        self.assertIn(original,[v["result_sha256"] for v in self.store.completed().values()])

    def test_failed_unknown_mutation_is_never_made_retryable(self):
        class Failed:
            worker_id="failed";quarantine_on_failure=True
            def execute(self,unit,attempt_dir,attempt_id):
                (attempt_dir/"partial-side-effect.txt").write_text("synthetic partial execution")
                raise RuntimeError("unknown state test")
        result=Coordinator(self.store,[Failed()]).run(storage_minimum_bytes=0,minimum_start_seconds=0)
        self.assertEqual(result["completed_units"],0)
        event=next(e for e in self.store.event_rows() if e["event"]=="attempt_failed")
        self.assertIsNone(event["safe_to_continue"]);self.assertIsNone(event["mutation_state_known"])
        with self.assertRaises(SchedulerError):
            Coordinator(self.store,[Worker("w")]).run(resume=True,retry_failed=True,storage_minimum_bytes=0,minimum_start_seconds=0)

    def test_duplicate_authoritative_result_and_competing_lock_are_rejected(self):
        from rq1.university_gpu.scheduler import RunLock
        Coordinator(self.store,[Worker("w")]).run(max_attempts=1,storage_minimum_bytes=0,minimum_start_seconds=0)
        key,envelope=next(iter(self.store.completed().items()));unit=next(u for u in self.manifest["units"] if u["unit_key"]==key)
        with self.assertRaises(SchedulerError):
            self.store.record_result(unit,envelope["result"],attempt_id=envelope["attempt_id"],worker_id="w")
        with patch("rq1.university_gpu.scheduler._pid_alive",return_value=True), patch("rq1.university_gpu.scheduler.require_posix_process_management"):
            with RunLock(self.base/"supervisor.lock",resume=False):
                with self.assertRaises(SchedulerError):
                    with RunLock(self.base/"supervisor.lock",resume=True):pass

    def test_supervisor_resumes_only_clean_protected_segments(self):
        import time
        calls=[]
        def segment(root,config,manifest,**kwargs):
            calls.append(("launch",kwargs.get("resume",False)))
            target=Path(config["run_output_dir"])/manifest["run_id"];target.mkdir(parents=True,exist_ok=True)
            atomic_write_json(target/"manifest.json",manifest)
            return {"run":{"status":"paused" if len([c for c in calls if c[0]=="launch"])==1 else "completed",
                           "failures":[],"cutoff_reached":False,"attempts_started":50}}
        def protect(*args):calls.append(("protect",True));return {"status":"synthetic"}
        with patch.object(scale,"_validate_engineering_approval"), patch.object(scale,"run_feasibility_launch",side_effect=segment), \
             patch("rq1.university_gpu.scale_protection.protect_segment",side_effect=protect), \
             patch("rq1.university_gpu.scale_protection.verify_last_protection"), patch.object(scale,"finalize_run"):
            value=scale.run_supervised(ROOT,{"run_output_dir":str(self.base/"supervision")},self.manifest,
                    approval=self.manifest["engineering_approval"],yes=True,cutoff_epoch=time.time()+10000)
        self.assertEqual(value["run"]["status"],"completed")
        self.assertEqual(calls,[("launch",False),("protect",True),("launch",True),("protect",True)])

    def test_supervisor_does_not_restart_worker_failure(self):
        import time
        result={"run":{"status":"incomplete","failures":[{"error":"synthetic unknown state"}]}}
        with patch.object(scale,"_validate_engineering_approval"), patch.object(scale,"run_feasibility_launch",return_value=result) as launch, \
             patch("rq1.university_gpu.scale_protection.protect_segment") as protect:
            scale.run_supervised(ROOT,{"run_output_dir":str(self.base/"failed-supervision")},self.manifest,
                    approval=self.manifest["engineering_approval"],yes=True,cutoff_epoch=time.time()+10000)
            self.assertEqual(launch.call_count,1);protect.assert_not_called()

    def test_supervisor_does_not_restart_after_backup_failure(self):
        import time
        result={"run":{"status":"paused","failures":[],"cutoff_reached":False,"attempts_started":50}}
        with patch.object(scale,"_validate_engineering_approval"), patch.object(scale,"run_feasibility_launch",return_value=result) as launch, \
             patch("rq1.university_gpu.scale_protection.protect_segment",side_effect=OSError("synthetic destination failure")):
            with self.assertRaises(OSError):
                scale.run_supervised(ROOT,{"run_output_dir":str(self.base/"backup-supervision")},self.manifest,
                        approval=self.manifest["engineering_approval"],yes=True,cutoff_epoch=time.time()+10000)
            self.assertEqual(launch.call_count,1)

    def test_published_authority_without_completion_event_is_recovered_without_reexecution(self):
        output=self.base/"published";target=output/self.manifest["run_id"]
        store=RunStore(target,self.manifest);store.initialize();unit=self.manifest["units"][0];attempt="known123"
        directory=store.root/"attempts"/unit["unit_key"]/("001-"+attempt);directory.mkdir(parents=True)
        store.append_event({"event":"attempt_started","unit_key":unit["unit_key"],"attempt_id":attempt,"worker_id":"w"})
        result=Worker("publisher").execute(unit,directory,attempt)
        store.record_result(unit,result,attempt_id=attempt,worker_id="publisher")
        with patch.object(scale,"_validate_engineering_approval"),patch.object(scale,"_run_supervised",return_value={"test":"delegated"}) as inner:
            scale.run_supervised(ROOT,{"run_output_dir":str(output)},self.manifest,approval=self.manifest["engineering_approval"],yes=True,resume=True)
            inner.assert_called_once()
        worker=Worker("resume")
        result=Coordinator(store,[worker]).run(resume=True,max_attempts=1,storage_minimum_bytes=0,minimum_start_seconds=0)
        self.assertEqual(result["completed_units"],2)
        self.assertNotIn(unit["unit_key"],worker.assignments)
        self.assertTrue(any(e["event"]=="recovered_authoritative_result" for e in store.event_rows()))
