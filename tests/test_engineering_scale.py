from __future__ import annotations
import copy
import os
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch
from rq1.university_gpu import engineering_scale as s
from rq1.university_gpu.models import load_json, content_hash
from rq1.university_gpu.scheduler import _validate_run_manifest, _validate_feasibility_result
from rq1.university_gpu.authorization import verify_engineering_authorization, AuthorizationError
from tests.test_university_gpu_feasibility import candidate_manifest, make_fake_frozen_manifest


def frozen(acceptance=False):
    candidate = s.build_scale_candidate(candidate_manifest(), acceptance=acceptance)
    value = make_fake_frozen_manifest(candidate_manifest())
    for k, v in candidate.items():
        if k not in {"model", "retrieval", "accum18_library", "tokenizer"}:
            value[k] = copy.deepcopy(v)
    value.update(status="engineering_frozen", run_id=s.PROFILE_ID+"-offline-test", source_commit="a"*40,
                 candidate_manifest=candidate, candidate_manifest_sha256=candidate["manifest_sha256"], backup_destination="/approved-independent-fixture")
    value["runtime_dependencies"] = {"python": "synthetic-test-only", "packages": {"fixture": "0"}}
    value["verified_inputs"]["runtime_dependencies_sha256"] = content_hash(value["runtime_dependencies"])
    value["engineering_approval"].update(schema="rq1-engineering-scale-approval-v1", source_commit="a"*40,
        candidate_manifest_sha256=candidate["manifest_sha256"], episode_count=value["episode_count"],
        maximum_attempts=value["maximum_attempts"], backup_destination=value["backup_destination"],
        backup_independence_confirmed=True, access_end_epoch=2000000000, acceptance_export_manifest_sha256="f"*64)
    value["units"] = s._unit_rows(value["tasks"], value["accum18_library"]["sha256"], value["episode_count"])
    value["manifest_sha256"] = s._manifest_digest(value)
    return value


class ScaleTests(unittest.TestCase):
    def test_matrix_is_exact_balanced_deterministic_and_paired(self):
        a=s.build_scale_candidate(candidate_manifest()); b=s.build_scale_candidate(candidate_manifest())
        self.assertEqual(a,b); units=a["units"]
        self.assertEqual(len({u["unit_key"] for u in units}),500)
        self.assertEqual(Counter(u["condition"] for u in units), {"D0":250,"D18":250})
        self.assertEqual(sorted(Counter(u["task_id"] for u in units).values()),[82,82,84,84,84,84])
        self.assertEqual({u["seed"] for u in units},{11})
        pairs={}
        for u in units:pairs.setdefault(u["pair_id"],[]).append(u)
        self.assertEqual(len(pairs),250)
        for pair in pairs.values():
            self.assertEqual({u["condition"] for u in pair},{"D0","D18"})
            self.assertEqual(len({u["repetition_id"] for u in pair}),1)
        self.assertEqual(s.validate_feasibility_manifest(a),[])

    def test_changed_matrix_policy_and_ancestry_fail_even_rehashed(self):
        for field,value in [("pause_every",60),("maximum_attempts",521),("cutoff_reserve_seconds",1800),("purpose","scientific")]:
            m=s.build_scale_candidate(candidate_manifest());m[field]=value;m["manifest_sha256"]=s._manifest_digest(m)
            self.assertTrue(s.validate_feasibility_manifest(m))
        m=s.build_scale_candidate(candidate_manifest());m["units"][1]=m["units"][0];m["manifest_sha256"]=s._manifest_digest(m)
        self.assertTrue(s.validate_feasibility_manifest(m))

    def test_four_episode_acceptance_is_separate_and_strict(self):
        m=s.build_scale_candidate(candidate_manifest(),acceptance=True)
        self.assertEqual(len(m["units"]),4);self.assertEqual(m["purpose"],"acceptance")
        self.assertEqual(s.validate_feasibility_manifest(m),[])
        m["episode_count"]=500;m["manifest_sha256"]=s._manifest_digest(m)
        self.assertTrue(s.validate_feasibility_manifest(m))

    def test_frozen_dispatch_and_source_approval_bindings(self):
        m=frozen();self.assertEqual(_validate_run_manifest(m),[])
        m["engineering_approval"]["episode_count"]=12;m["manifest_sha256"]=s._manifest_digest(m)
        self.assertTrue(_validate_run_manifest(m))

    def test_scale_never_uses_self_attested_authorization(self):
        approval={"test_only":False,"schema":"rq1-engineering-scale-approval-v1","authorization_type":"engineering_gpu",
        "approval_scope":"engineering_feasibility_only","gpu_inference":True,"scientific_run":False,
        "not_for_scientific_claims":True,"candidate_manifest_sha256":"b"*64,"profile_id":s.PROFILE_ID,
        "model_tag":s.MODEL_TAG,"model_digest":s.MODEL_DIGEST,"model_quantization":s.MODEL_QUANTIZATION,
        "total_action_budget":200,"run_id":"test","conditions":["D0","D18"],"prior_pick_place_exposure_acknowledged":True,
        "approver_id":"test-only","approved_by":"test-only","approval_reference":"not-real","approval_provenance":"self-attestation"}
        with patch.dict(os.environ,{"RQ1_ENGINEERING_AUTHORITY_MODE":"self_attested"}):
            with self.assertRaises(AuthorizationError):
                verify_engineering_authorization(approval,candidate_sha256="b"*64,model_tag=s.MODEL_TAG,
                    model_digest=s.MODEL_DIGEST,profile_id=s.PROFILE_ID,action_budget=200,run_id="test",model_quantization=s.MODEL_QUANTIZATION)

    def test_original_twelve_episode_candidate_still_valid(self):
        from rq1.university_gpu.feasibility import validate_feasibility_manifest
        self.assertEqual(validate_feasibility_manifest(candidate_manifest()),[])
        self.assertEqual(len(candidate_manifest()["units"]),12)

    def test_original_candidate_hash_is_pinned_even_if_modified_candidate_is_rehashed(self):
        from rq1.university_gpu import feasibility
        original=candidate_manifest();original["model"]["identity_source"]="altered test metadata"
        original["manifest_sha256"]=feasibility._manifest_digest(original)
        with self.assertRaises(s.FeasibilityError):s.build_scale_candidate(original)

    def test_attempt_allowance_is_explicit_and_bounded(self):
        value=s.build_scale_candidate(candidate_manifest(),additional_attempts=2)
        self.assertEqual(value["maximum_attempts"],502)
        self.assertEqual(len(value["units"]),500)
        self.assertEqual(s.validate_feasibility_manifest(value),[])
        for n in (-1,21,True):
            with self.assertRaises(s.FeasibilityError):s.build_scale_candidate(candidate_manifest(),additional_attempts=n)

    def test_cli_candidate_roundtrip_and_create_only_outputs(self):
        import tempfile, subprocess, sys, json
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"candidate.json"
            base=[sys.executable,"-m","rq1.cli","university-gpu","engineering-scale"]
            result=subprocess.run(base+["candidate","--baseline-candidate",str(Path(__file__).resolve().parents[1]/"university-gpu/prospective/feasibility-200-candidate.json"),"--acceptance","--output",str(path)],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertEqual(load_json(path)["episode_count"],4)
            validation=subprocess.run(base+["validate","--manifest",str(path)],capture_output=True,text=True)
            self.assertEqual(validation.returncode,0,validation.stdout)
            duplicate=subprocess.run(base+["candidate","--baseline-candidate",str(Path(__file__).resolve().parents[1]/"university-gpu/prospective/feasibility-200-candidate.json"),"--output",str(path)],capture_output=True,text=True)
            self.assertNotEqual(duplicate.returncode,0)
            self.assertEqual(load_json(path)["episode_count"],4)

    def test_scale_worker_entrypoint_keeps_json_protocol_on_failed_startup(self):
        import subprocess, sys, json, tempfile
        with tempfile.TemporaryDirectory() as tmp:
            result=subprocess.run([sys.executable,"-m","rq1.university_gpu.engineering_scale_worker","--config",str(Path(tmp)/"missing.json")],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            lines=result.stdout.splitlines();self.assertEqual(len(lines),1,result.stdout)
            self.assertIs(json.loads(lines[0])["ready"],False)

    def test_hardware_contract_is_four_distinct_rtx5000_ada_devices(self):
        report={"status":"PASS","devices":[{"uuid":f"GPU-{i}","name":"NVIDIA RTX 5000 Ada Generation","memory_total_mib":32760} for i in range(4)]}
        self.assertTrue(s.scale_machine_matches(report))
        report["devices"][0]["name"]="NVIDIA different GPU"
        self.assertFalse(s.scale_machine_matches(report))

    def test_preflight_rejects_backup_overlapping_run_storage_before_asset_access(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            location=Path(tmp)/"storage";location.mkdir()
            m=frozen(acceptance=True);m["backup_destination"]=str(location)
            m["engineering_approval"]["backup_destination"]=str(location);m["manifest_sha256"]=s._manifest_digest(m)
            with self.assertRaisesRegex(s.FeasibilityError,"backup must be disjoint"):
                s.verify_frozen_feasibility_assets(Path(__file__).resolve().parents[1],{"run_output_dir":str(location)},m)

    def test_approved_backup_path_rejects_traversal(self):
        import tempfile
        from rq1.university_gpu.scale_protection import approved_destination
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):approved_destination({"backup_destination":str(Path(tmp)/"nested"/".."/"backup")})

    def test_frozen_runtime_dependency_inventory_is_integrity_bound(self):
        m=frozen();self.assertEqual(_validate_run_manifest(m),[])
        m["runtime_dependencies"]["packages"]["fixture"]="changed"
        m["manifest_sha256"]=s._manifest_digest(m)
        self.assertTrue(_validate_run_manifest(m))

    def test_scale_acceptance_freeze_and_cpu_preflight_with_synthetic_assets(self):
        import shutil, subprocess, time
        from tests.test_feasibility_provenance_regression import ProvenanceRegressionTests, FixtureTokenizer
        from rq1.university_gpu.models import sha256_file
        fixture=ProvenanceRegressionTests("test_candidate_reference_freeze_preflight_and_resume_drift")
        fixture.setUp();self.addCleanup(fixture.doCleanups)
        repo_root=Path(__file__).resolve().parents[1]
        for name in ("pyproject.toml","uv.lock"):shutil.copy2(repo_root/name,fixture.repo/name)
        subprocess.run(["git","-C",str(fixture.repo),"add","pyproject.toml","uv.lock"],check=True,capture_output=True)
        subprocess.run(["git","-C",str(fixture.repo),"commit","-m","synthetic dependency manifests only"],check=True,capture_output=True)
        commit=subprocess.run(["git","-C",str(fixture.repo),"rev-parse","HEAD"],check=True,capture_output=True,text=True).stdout.strip()
        library_hash=sha256_file(fixture.model)
        from rq1.acquisition.environment import sbert_snapshot_sha256
        with patch.object(s,"BASELINE_CANDIDATE_SHA256",fixture.candidate["manifest_sha256"]), \
             patch.object(s,"MODEL_DIGEST",library_hash),patch.object(s,"SBERT_SNAPSHOT_SHA256",sbert_snapshot_sha256(fixture.cache)), \
             patch.object(s,"LocalGemmaTokenizer",FixtureTokenizer):
            candidate=s.build_scale_candidate(fixture.candidate,acceptance=True)
            backup=fixture.base/"independent-backup";backup.mkdir()
            output=fixture.base/"run-storage";output.mkdir()
            approval=dict(fixture.approval)
            approval.update(schema="rq1-engineering-scale-approval-v1",profile_id=s.PROFILE_ID,run_id=s.PROFILE_ID+"-freeze-fixture",
                candidate_manifest_sha256=candidate["manifest_sha256"],source_commit=commit,episode_count=4,maximum_attempts=4,
                backup_destination=str(backup),backup_independence_confirmed=True,access_end_epoch=time.time()+10000)
            # Synthetic approval pins are confined to this unit test, never production records.
            with patch.dict(os.environ,{"RQ1_ENGINEERING_AUTHORITY_MODE":"independent",
                                       "RQ1_TRUSTED_ENGINEERING_APPROVAL_SHA256":content_hash(approval)}):
                manifest=s.freeze_engineering_manifest(fixture.repo,candidate,run_id=approval["run_id"],
                    data_report=fixture.data_report,data_review=fixture.data_review,oracle_report=fixture.oracle,
                    identity_report=fixture.identity,identity_review=fixture.identity_review,
                    accum18_library_path=fixture.library,accum18_reference_path=fixture.reference,
                    sbert_cache=fixture.cache,tokenizer_path=fixture.tokenizer,engineering_approval=approval)
            self.assertEqual(s.validate_feasibility_manifest(manifest,require_frozen=True),[])
            config=dict(fixture.config,run_output_dir=str(output))
            with patch("shutil.disk_usage",return_value=type("Disk",(),{"free":100_000_000_000})()):
                self.assertEqual(s.verify_frozen_feasibility_assets(fixture.repo,config,manifest)["status"],"PASS")
                with patch.object(s,"runtime_dependencies",return_value={"python":"changed","packages":{}}):
                    with self.assertRaisesRegex(s.FeasibilityError,"package versions changed"):
                        s.verify_frozen_feasibility_assets(fixture.repo,config,manifest)

    def test_acceptance_based_storage_projection_has_floor_and_growth(self):
        self.assertEqual(s.projected_storage_bytes(4_000_000,4,500),50_000_000_000)
        self.assertEqual(s.projected_storage_bytes(400_000_000,4,500),600_000_000_000)
        with self.assertRaises(s.FeasibilityError):s.projected_storage_bytes(0,4,500)
