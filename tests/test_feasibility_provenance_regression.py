"""Offline integration fixtures; no real model, approval or ALFWorld claims."""
import copy
import io
import json
import subprocess
import shutil
import tarfile
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from rq1.university_gpu import feasibility as f
from rq1.university_gpu.models import content_hash, load_json, sha256_file
from rq1.university_gpu.prospective_identity import verify_candidate
from rq1.acquisition.environment import sbert_snapshot_sha256
from tests.test_university_gpu_feasibility import candidate_manifest, FakeWorker


class FixtureTokenizer:
    def __init__(self, identity):
        if sha256_file(identity.path) != identity.sha256:
            raise ValueError("fixture tokenizer hash mismatch")
    def encode(self, text):
        return list(text.encode())


class ProvenanceRegressionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.repo = self.base / "repo"
        self.repo.mkdir()
        source_root = Path(__file__).resolve().parents[1]
        shutil.copytree(source_root / "src", self.repo / "src", ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copy(source_root / "pyproject.toml", self.repo / "pyproject.toml")
        for args in (["init"], ["add", "src", "pyproject.toml"], ["-c", "user.name=Offline fixture", "-c", "user.email=fixture@example.test", "commit", "--allow-empty", "-m", "test-only source"]):
            subprocess.run(["git", "-C", str(self.repo), *args], check=True, capture_output=True)
        self.library = self.base / "library.json"
        self.write(self.library, {"libraries": {"Accum-18": {"skills": [
            {"skill_id": f"fixture-{i:02}", "text": f"test-only skill {i}"} for i in range(18)]}}})
        self.archive = self.base / "historical-fixture.tar.gz"
        data = self.library.read_bytes()
        with tarfile.open(self.archive, "w:gz") as a:
            member = tarfile.TarInfo(f.HISTORICAL_LIBRARY_MEMBER)
            member.size = len(data)
            a.addfile(member, io.BytesIO(data))
        self.ledger = self.base / "member-ledger.json"
        self.write(self.ledger, {"archive_sha256": sha256_file(self.archive), "members": [
            {"path": f.HISTORICAL_LIBRARY_MEMBER, "sha256": sha256_file(self.library), "size": len(data)}]})
        self.stack.enter_context(patch.object(f, "HISTORICAL_EVALUATION_ARCHIVE_SHA256", sha256_file(self.archive)))
        self.reference = self.base / "reference.json"
        self.write(self.reference, f.build_accum18_reference(self.library, archive_path=self.archive, archive_manifest_path=self.ledger))
        self.tokenizer = self.base / "tokenizer.json"
        self.tokenizer.write_text("FAKE TOKENIZER: TEST PROVIDER ONLY")
        self.link = self.base / "link.json"
        self.link.write_text("TEST LINK ONLY: NOT SCIENTIFIC EVIDENCE")
        blob = self.base / "blob"
        blob.write_bytes(b"SYNTHETIC MODEL ONLY")
        self.blobs = self.base / "blobs"
        self.blobs.mkdir()
        self.blob = self.blobs / ("sha256-" + sha256_file(blob))
        blob.rename(self.blob)
        self.model = self.base / "model-manifest.json"
        ref = {"digest": "sha256:" + sha256_file(self.blob), "size": self.blob.stat().st_size}
        self.write(self.model, {"config": ref, "layers": [ref]})
        self.stack.enter_context(patch.object(f, "MODEL_DIGEST", sha256_file(self.model)))
        self.cache = self.base / "cache"
        snap = self.cache / "models--sentence-transformers--all-mpnet-base-v2" / "snapshots" / f.SBERT_REVISION
        snap.mkdir(parents=True)
        (snap / "config.json").write_text("TEST SNAPSHOT ONLY")
        self.stack.enter_context(patch.object(f, "SBERT_SNAPSHOT_SHA256", sbert_snapshot_sha256(self.cache)))
        self.stack.enter_context(patch.object(f, "LocalGemmaTokenizer", FixtureTokenizer))
        self.candidate = candidate_manifest()
        self.candidate["model"]["digest"] = sha256_file(self.model)
        self.candidate["retrieval"]["snapshot_sha256"] = f.SBERT_SNAPSHOT_SHA256
        self.data = self.base / "data"
        (self.data / "json_2.1.1/train").mkdir(parents=True)
        from rq1.tasks.discovery import CANONICAL_FAMILIES
        for task in self.candidate["tasks"]:
            folder = self.data / "json_2.1.1/valid_seen" / task["source_task"]["source_identifier"]
            folder.mkdir(parents=True)
            native = next(k for k, v in CANONICAL_FAMILIES.items() if v == task["task_family"])
            self.write(folder / "traj_data.json", {"task_type": native})
            self.write(folder / "game.tw-pddl", {"solvable": True})
            task["source_task"]["source_sha256"] = sha256_file(folder / "traj_data.json")
            task["source_task"]["game_sha256"] = sha256_file(folder / "game.tw-pddl")
        self.candidate["data_identity_expectations"] = {
            "discovery_valid_seen_sha256": f.discover_tasks(self.data, "valid_seen").data_root_identity,
            "adapter_default_train_valid_seen_sha256": f.build_task_index(self.data).identity}
        self.candidate["units"] = f._unit_rows(self.candidate["tasks"], None)
        self.candidate["manifest_sha256"] = f._manifest_digest(self.candidate)
        self.data_report = f.reconcile_data_identity(self.data, self.candidate)
        self.data_review = {"schema": "rq1-data-identity-review-v1", "reviewed_report_sha256": self.data_report["report_sha256"],
                            "decision": "ACCEPT_SCOPED_DIFFERENCES", "reviewed_by": "TEST ONLY", "review_reference": "fixture"}
        self.provenance = {"model_manifest_sha256": sha256_file(self.model), "tokenizer_sha256": sha256_file(self.tokenizer),
            "revision": "e" * 40, "source_repository": "https://example.test/fixture", "model_tokenizer_link_evidence_sha256": sha256_file(self.link),
            "samples": [{"text": t, "token_ids": list(t.encode()), "reference_source": "TEST ONLY"} for t in ["", "abc", "ascii-reference", "{}", "task"]]}
        self.identity = verify_candidate(self.model, self.blobs, self.tokenizer, self.provenance,
                                         lambda t: list(t.encode()), self.link, expected_model_digest=f.MODEL_DIGEST)
        self.identity_review = {"schema": "rq1-prospective-identity-review-v1", "reviewed_report_sha256": self.identity["content_sha256"],
            "model_tag": f.MODEL_TAG, "model_digest": f.MODEL_DIGEST, "quantization": f.MODEL_QUANTIZATION,
            "tokenizer_sha256": sha256_file(self.tokenizer), "tokenizer_revision": "e" * 40,
            "model_tokenizer_link_confirmed": True, "reviewed_by": "TEST ONLY", "review_reference": "not human approval"}
        self.approval = {"approval_scope": "engineering_feasibility_only", "gpu_inference": True, "scientific_run": False,
            "not_for_scientific_claims": True, "candidate_manifest_sha256": self.candidate["manifest_sha256"],
            "prior_pick_place_exposure_acknowledged": True, "approved_by": "TEST ONLY", "approval_reference": "not real authorization"}
        rows = []
        for task in self.candidate["tasks"]:
            actions = task["prefix_actions"] + [task["detour_action"]] + task["reference_actions"][len(task["prefix_actions"]):]
            rows.append({"task_id": task["task_id"], "validated": True, "total_action_limit": 200,
                "checkpoint_digest": task["checkpoint_digest"], "post_detour_digest": task["post_detour_digest"],
                "oracle_checkpoint_digest": task["checkpoint_digest"], "oracle_post_detour_digest": task["post_detour_digest"],
                "detour_action": task["detour_action"], "oracle_success": True, "action_sequence_sha256": content_hash(actions),
                "oracle_event_log_sha256": content_hash({"test_only": actions}), "split": "valid_seen", "seed": 11, "total_actions": len(actions)})
        self.oracle = {"schema": "rq1-feasibility-oracle-replay-v1", "manifest_sha256": self.candidate["manifest_sha256"],
            "tasks": rows, "status": "PASS", "model_calls": 0, "gpu_inference": False, "scientific_evidence": False}
        self.oracle["report_sha256"] = content_hash(self.oracle)
        self.config = {"alfworld_data_dir": str(self.data), "accum18_library_path": str(self.library),
            "accum18_reference_manifest_path": str(self.reference), "sbert_cache": str(self.cache),
            "tokenizer_path": str(self.tokenizer), "tokenizer_revision": "e" * 40,
            "ollama_manifest_path": str(self.model), "ollama_blobs_dir": str(self.blobs), "model_tokenizer_link_evidence_path": str(self.link)}
        for key, value in [("data_reconciliation_report_path", self.data_report), ("data_identity_review_path", self.data_review),
                           ("oracle_replay_report_path", self.oracle), ("model_identity_report_path", self.identity),
                           ("model_identity_review_path", self.identity_review), ("tokenizer_provenance_path", self.provenance)]:
            path = self.base / (key + ".json")
            self.write(path, value)
            self.config[key] = str(path)

    def write(self, path, value):
        path.write_text(json.dumps(value), encoding="utf-8")

    def freeze(self):
        return f.freeze_engineering_manifest(self.repo, self.candidate, run_id=f.PROFILE_ID + "-test-fixture",
            data_report=self.data_report, data_review=self.data_review, oracle_report=self.oracle,
            identity_report=self.identity, identity_review=self.identity_review, accum18_library_path=self.library,
            accum18_reference_path=self.reference, sbert_cache=self.cache, tokenizer_path=self.tokenizer,
            engineering_approval=self.approval)

    def test_candidate_reference_freeze_preflight_and_resume_drift(self):
        self.assertEqual(f.validate_feasibility_manifest(self.candidate), [])
        frozen = self.freeze()
        self.assertNotEqual(frozen["manifest_sha256"], self.oracle["manifest_sha256"])
        self.assertEqual(f.validate_oracle_report(self.oracle, frozen), [])
        self.assertEqual(f.verify_frozen_feasibility_assets(self.repo, self.config, frozen)["status"], "PASS")
        from rq1.university_gpu.scheduler import Coordinator, RunStore
        from rq1.university_gpu.snapshot import export_progress, verify_progress
        from rq1.university_gpu.evidence import inventory_attempt
        fixture_digest = f.MODEL_DIGEST
        fixture_ids = frozen["accum18_library"]["skill_ids"][:3]
        class BoundFixtureWorker(FakeWorker):
            def execute(self, unit, attempt_dir, attempt_id):
                result = super().execute(unit, attempt_dir, attempt_id)
                result["model_digest"] = fixture_digest
                result["measurements"]["model_digest"] = fixture_digest
                ids = [] if unit["condition"] == "D0" else fixture_ids
                result["retrieved_skill_ids"] = ids
                log = attempt_dir / "result/retrieval.jsonl"
                event = json.loads(log.read_text())
                event["run_id"] = frozen["run_id"]
                event["episode_id"] = unit["unit_key"]
                event["top"] = [{"skill_id": i, "score": 0.5} for i in ids]
                log.write_text(json.dumps(event) + "\n", encoding="utf-8")
                result["raw_evidence"] = inventory_attempt(attempt_dir, result)
                return result
        run_root = self.base / "runs" / frozen["run_id"]
        store = RunStore(run_root, frozen)
        coordinator = Coordinator(store, [BoundFixtureWorker("test-provider-only")])
        partial = coordinator.run(max_attempts=4)
        self.assertEqual(partial["completed_units"], 4)
        self.assertFalse(coordinator.validate()["passed"])
        partial_dest = self.base / "partial-export"
        self.assertFalse(export_progress(self.repo, run_root, store.manifest_path, partial_dest)["complete"])
        self.assertEqual(verify_progress(partial_dest)["status"], "PASS")
        authority_before = {p.name: p.read_bytes() for p in store.results_dir.glob("*.json")}
        self.assertEqual(f.verify_frozen_feasibility_assets(self.repo, self.config, frozen)["status"], "PASS")
        self.assertEqual(coordinator.run(resume=True)["completed_units"], 12)
        self.assertTrue(coordinator.validate()["passed"])
        for name, data in authority_before.items():
            self.assertEqual((store.results_dir / name).read_bytes(), data)
        coordinator.merge()
        full_dest = self.base / "full-export"
        self.assertTrue(export_progress(self.repo, run_root, store.manifest_path, full_dest)["complete"])
        self.assertEqual(verify_progress(full_dest)["status"], "PASS")
        self.blob.write_bytes(b"MUTATION")
        with self.assertRaisesRegex(ValueError, "blob missing or corrupt"):
            f.verify_frozen_feasibility_assets(self.repo, self.config, frozen)

    def test_wrong_candidate_ancestry_and_records_fail(self):
        frozen = self.freeze()
        frozen["tasks"][0]["checkpoint_digest"] = "f" * 64
        frozen["units"] = f._unit_rows(frozen["tasks"], frozen["accum18_library"]["sha256"])
        frozen["manifest_sha256"] = f._manifest_digest(frozen)
        self.assertIn("frozen candidate ancestry or original task records differ", f.validate_feasibility_manifest(frozen, require_frozen=True))
        self.oracle["manifest_sha256"] = "f" * 64
        self.assertTrue(f.validate_oracle_report(self.oracle, self.candidate))

    def test_historical_wrong_library_and_missing_or_wrong_ledger_fail(self):
        self.library.write_text("{}")
        with self.assertRaisesRegex(f.FeasibilityError, "library bytes differ"):
            f.build_accum18_reference(self.library, archive_path=self.archive, archive_manifest_path=self.ledger)
        ledger = load_json(self.ledger)
        ledger["members"][0]["sha256"] = "f" * 64
        self.write(self.ledger, ledger)
        # Restore genuine fixture library to isolate the ledger mismatch.
        with tarfile.open(self.archive, "r:gz") as archive:
            self.library.write_bytes(archive.extractfile(f.HISTORICAL_LIBRARY_MEMBER).read())
        with self.assertRaisesRegex(f.FeasibilityError, "library bytes differ"):
            f.build_accum18_reference(self.library, archive_path=self.archive, archive_manifest_path=self.ledger)
        self.ledger.unlink()
        with self.assertRaisesRegex(f.FeasibilityError, "existing regular"):
            f.build_accum18_reference(self.library, archive_path=self.archive, archive_manifest_path=self.ledger)

    def test_self_attested_frozen_manifest_cannot_anchor_library_reference(self):
        fabricated_manifest = self.base / "fabricated-frozen-manifest.json"
        self.write(fabricated_manifest, {"status": "engineering_frozen", "inputs": {
            "library_sha256": sha256_file(self.library)}, "library_skill_ids": [
                f"fixture-{i:02}" for i in range(18)]})
        with self.assertRaisesRegex(f.FeasibilityError, "pinned historical archive"):
            f.build_accum18_reference(self.library, fabricated_manifest)
        fabricated_reference = {
            "schema": "rq1-accum18-frozen-reference-v1", "status": "verified_frozen",
            "source_kind": "university_manifest", "source_frozen_manifest_path": str(fabricated_manifest),
            "library_file_sha256": sha256_file(self.library), "skill_count": 18,
            "skill_ids": [f"fixture-{i:02}" for i in range(18)],
        }
        fabricated_reference["reference_sha256"] = content_hash(fabricated_reference)
        with self.assertRaisesRegex(f.FeasibilityError, "lacks authenticated source ancestry"):
            f.verify_accum18_reference(self.library, fabricated_reference)

    def test_substituted_archive_and_invalid_skill_sets_fail(self):
        original = load_json(self.library)
        with patch.object(f, "HISTORICAL_EVALUATION_ARCHIVE_SHA256", ""):
            with self.assertRaisesRegex(f.FeasibilityError, "archive hash differs"):
                f.build_accum18_reference(self.library, archive_path=self.archive, archive_manifest_path=self.ledger)
        for mutation in ("text", "duplicate", "missing"):
            with self.subTest(mutation=mutation):
                payload = copy.deepcopy(original)
                skills = payload["libraries"]["Accum-18"]["skills"]
                if mutation == "text":
                    skills[0]["text"] = "altered text"
                elif mutation == "duplicate":
                    skills[1]["skill_id"] = skills[0]["skill_id"]
                else:
                    skills.pop()
                self.write(self.library, payload)
                data = self.library.read_bytes()
                with tarfile.open(self.archive, "w:gz") as archive:
                    member = tarfile.TarInfo(f.HISTORICAL_LIBRARY_MEMBER)
                    member.size = len(data)
                    archive.addfile(member, io.BytesIO(data))
                self.write(self.ledger, {"archive_sha256": sha256_file(self.archive), "members": [
                    {"path": f.HISTORICAL_LIBRARY_MEMBER, "sha256": sha256_file(self.library), "size": len(data)}]})
                with self.assertRaisesRegex(f.FeasibilityError, "archive hash differs"):
                    f.build_accum18_reference(self.library, archive_path=self.archive, archive_manifest_path=self.ledger)
                if mutation != "text":
                    with patch.object(f, "HISTORICAL_EVALUATION_ARCHIVE_SHA256", sha256_file(self.archive)):
                        with self.assertRaisesRegex(f.FeasibilityError, "18 distinct IDs"):
                            f.build_accum18_reference(self.library, archive_path=self.archive, archive_manifest_path=self.ledger)

    def test_resume_asset_validation_rejects_altered_reference(self):
        frozen = self.freeze()
        reference = load_json(self.reference)
        reference["source_kind"] = "university_manifest"
        reference["reference_sha256"] = content_hash({k: v for k, v in reference.items() if k != "reference_sha256"})
        self.write(self.reference, reference)
        with self.assertRaises(f.FeasibilityError):
            f.verify_frozen_feasibility_assets(self.repo, self.config, frozen)

    def test_tokenizer_revision_mismatch_fails_freeze_and_preflight(self):
        self.identity_review["tokenizer_revision"] = "a" * 40
        with self.assertRaisesRegex(f.FeasibilityError, "review does not bind"):
            self.freeze()
        self.identity_review["tokenizer_revision"] = "e" * 40
        frozen = self.freeze()
        self.config["tokenizer_revision"] = "a" * 40
        with self.assertRaisesRegex(f.FeasibilityError, "tokenizer revision differs"):
            f.verify_frozen_feasibility_assets(self.repo, self.config, frozen)
        self.config["tokenizer_revision"] = "e" * 40
        mutated = dict(self.provenance, revision="a" * 40)
        self.write(Path(self.config["tokenizer_provenance_path"]), mutated)
        with self.assertRaisesRegex(f.FeasibilityError, "provenance revision differs"):
            f.verify_frozen_feasibility_assets(self.repo, self.config, frozen)

    def test_run_id_consumption_and_containment(self):
        frozen = self.freeze()
        frozen["run_id"] = "../../outside"
        frozen["manifest_sha256"] = f._manifest_digest(frozen)
        self.assertTrue(f.validate_feasibility_manifest(frozen, require_frozen=True))
        with self.assertRaises(f.FeasibilityError):
            f.bounded_run_root(self.base, "../../outside")

    def test_nonfinite_and_expired_cutoffs_fail_before_assets(self):
        frozen = self.freeze()
        for cutoff in [float("nan"), float("inf"), -float("inf"), 0, -10]:
            with self.subTest(cutoff=cutoff), self.assertRaisesRegex(f.FeasibilityError, "real university access cutoff"):
                f.run_feasibility_launch(self.repo, {}, frozen, approval=self.approval, yes=True, cutoff_epoch=cutoff)
