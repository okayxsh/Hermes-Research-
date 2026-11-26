"""Parallel four-lane execution: assignment, coordinator, claims, restart, late/duplicate results and halts.
SYNTHETIC: scripted drivers stand in for the four GPU provider endpoints; no model, GPU or provider is used."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import five_arm_e2e_fixtures as E
from five_arm_fixtures import NEUTRAL_SENTENCES, SKILL_TEXTS, TOLERANCE
from rq1.five_arm import parallel, runner
from rq1.five_arm.analysis import resolve_units
from rq1.five_arm.executor import FiveArmEpisodeExecutor, InterruptionRecordFactory, ProtocolViolation
from rq1.five_arm.measurements import validate_record
from rq1.five_arm.memory import FillerCorpus

DIGEST = "6" * 64
RUNTIME = "f" * 64
LANES = [{"lane_id": f"L{i}", "endpoint": f"http://127.0.0.1:{11435 + i}", "gpu_uuid": f"GPU-synthetic-{i}"} for i in range(4)]
TASKS = tuple(f"valid_seen:task{i}/trial_T{i}" for i in range(4))
SALT = "lane-salt-0123456789"


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


class _Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp_class = tempfile.TemporaryDirectory()
        cls.spec = E.build_pilot(Path(cls.tmp_class.name), task_ids=TASKS)
        cls.assignment = parallel.assign_lanes(cls.spec["manifest"], LANES, salt=SALT)

    @classmethod
    def tearDownClass(cls):
        cls.tmp_class.cleanup()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.clock = Clock()

    def tearDown(self):
        self.tmp.cleanup()

    def executor(self, respond=E.solver, corp=None, counter=E.counter):
        spec = self.spec
        policy = runner.FiveArmMemoryPolicy(manifest=spec["manifest"], skill_texts=SKILL_TEXTS, corpus=corp or spec["corpus"],
                                            tokenizer=E.TOK, tolerance=TOLERANCE, d0_representation="absent_memory_section")
        return FiveArmEpisodeExecutor(E.ScriptedDriver(respond), mode="arm", tasks=spec["tasks"], task_meta=spec["metas"],
                                      split="valid_seen", synthetic=True, tokenizer=E.TOK, runtime_identity_sha256=RUNTIME,
                                      exposure_policy="synthetic", d0_representation="absent_memory_section",
                                      library=spec["library"], embedder=spec["embedder"], memory_policy=policy,
                                      leakage_probes=spec["probes"], chat_template_tokens=counter, provider_token_tolerance=0)

    def coordinator(self, root=None, lease=100.0):
        factory = InterruptionRecordFactory(tasks=self.spec["tasks"], task_meta=self.spec["metas"], synthetic=True,
                                            runtime_identity_sha256=RUNTIME, exposure_policy="synthetic", model_name="gemma4:12b")
        return parallel.Coordinator(root or self.root / "run", manifest=self.spec["manifest"], assignment=self.assignment,
                                    kind_validator=validate_record, interruption_record=factory.interruption_record,
                                    max_infrastructure_retries=2, expected_runtime_identity_sha256=RUNTIME,
                                    expected_model_digest=DIGEST, synthetic=True, lease_seconds=lease, clock=self.clock).start()

    def worker(self, client, lane, executor=None, snapshot=None):
        gpu = next(l["gpu_uuid"] for l in LANES if l["lane_id"] == lane)
        snap = snapshot or (lambda: {"provider_version": "0.0-synthetic", "model_digest": DIGEST, "processor": "gpu_full",
                                     "cuda_visible_devices": gpu})
        return parallel.LaneWorker(client, lane_id=lane, worker_id=f"{lane}-w", executor=executor or self.executor(),
                                   root=self.root / "run", snapshot=snap, heartbeat_seconds=3600)

    def drain(self, coord, workers):
        progressed = True
        while progressed:
            progressed = any([w.run_once() for w in workers])


class AssignmentTests(_Base):
    def test_balanced_frozen_and_cell_preserving(self):
        a = self.assignment
        self.assertEqual(parallel.validate_assignment(a, self.spec["manifest"]), [])
        self.assertEqual(sorted(a["lane_unit_counts"].values()), [7, 7, 7, 7])                 # 4 cells x 7 conditions
        cells = {}
        for unit in self.spec["manifest"]["units"]:
            cells.setdefault((unit["task_id"], unit["seed"]), set()).add(a["unit_lane"][unit["unit_key"]])
        self.assertTrue(all(len(lanes) == 1 for lanes in cells.values()))                     # all arms of a cell on one lane
        self.assertEqual(a, parallel.assign_lanes(self.spec["manifest"], LANES, salt=SALT))   # deterministic
        tampered = json.loads(json.dumps(a))
        key = next(iter(tampered["unit_lane"]))
        tampered["unit_lane"][key] = "L9"
        self.assertTrue(parallel.validate_assignment(tampered, self.spec["manifest"]))
        with self.assertRaises(ValueError):
            parallel.assign_lanes(self.spec["manifest"], [LANES[0], {**LANES[1], "gpu_uuid": LANES[0]["gpu_uuid"]}], salt=SALT)


class ParallelRunTests(_Base):
    def test_four_lanes_complete_every_unit_exactly_once(self):
        coord = self.coordinator()
        self.drain(coord, [self.worker(coord, l["lane_id"]) for l in LANES])
        summary = coord.summary()
        self.assertEqual((summary["pending_units"], summary["final_status_counts"], summary["halted"]), (0, {"VALID": 28}, None))
        self.assertTrue(coord.validate()["ok"])
        for record in coord.records():
            prov = record["execution_provenance"]
            self.assertEqual(prov["lane_id"], self.assignment["unit_lane"][record["unit_key"]])
            self.assertEqual((prov["mode"], prov["processor_before"]), ("parallel", "gpu_full"))
            self.assertTrue(prov["hermes_workspace"].startswith(str(self.root / "run" / "lanes" / prov["lane_id"])))
        self.assertEqual(len(resolve_units(coord.records(), max_infrastructure_retries=2)), 28)
        coord.stop()

    def test_worker_loss_lease_expiry_and_late_result(self):
        coord = self.coordinator(lease=100.0)
        lost = coord.claim("L0", "L0-dead")                                   # worker claims, then dies
        self.clock.now += 101
        replacement = coord.claim("L0", "L0-new")                             # lease expiry closes the lost attempt
        self.assertEqual((replacement["unit"]["unit_key"], replacement["attempt"]), (lost["unit"]["unit_key"], 2))
        late = self.executor()(lost["unit"], attempt=1, attempt_dir=self.root / "late")
        with self.assertRaises(parallel.ClaimRejected) as caught:
            coord.submit(lost["token"], late)
        self.assertIn("late", str(caught.exception))
        statuses = [r["outcome_status"] for r in coord.records() if r["unit_key"] == lost["unit"]["unit_key"]]
        self.assertEqual(statuses, ["MISSING_INFRASTRUCTURE"])
        coord.stop()

    def test_coordinator_restart_revokes_open_claims_and_rejects_their_results(self):
        coord = self.coordinator()
        w = self.worker(coord, "L1")
        self.assertTrue(w.run_once())                                         # one completed unit survives the restart
        open_claim = coord.claim("L1", "L1-w")
        coord.stop()
        restarted = self.coordinator()
        late = self.executor()(open_claim["unit"], attempt=open_claim["attempt"], attempt_dir=self.root / "late")
        with self.assertRaises(parallel.ClaimRejected):
            restarted.submit(open_claim["token"], late)
        self.drain(restarted, [self.worker(restarted, l["lane_id"]) for l in LANES])
        self.assertTrue(restarted.validate()["ok"])
        attempts = [r for r in restarted.records() if r["unit_key"] == open_claim["unit"]["unit_key"]]
        self.assertEqual([r["outcome_status"] for r in attempts], ["MISSING_INFRASTRUCTURE", "VALID"])
        self.assertEqual(restarted.summary()["final_status_counts"]["VALID"], 28)
        restarted.stop()

    def test_duplicate_results_are_rejected(self):
        coord = self.coordinator()
        claim = coord.claim("L2", "L2-w")
        w = self.worker(coord, "L2")
        record = w.executor(claim["unit"], attempt=claim["attempt"], attempt_dir=self.root / "dup")
        record["execution_provenance"].update(mode="parallel", lane_id="L2", worker_id="L2-w", endpoint=LANES[2]["endpoint"],
                                              gpu_uuid=LANES[2]["gpu_uuid"], cuda_visible_devices=LANES[2]["gpu_uuid"],
                                              provider_version="x", model_digest=DIGEST, processor_before="gpu_full",
                                              processor_after="gpu_full", host="h", pid=1, hermes_workspace="w",
                                              claim_token_sha256=__import__("hashlib").sha256(claim["token"].encode()).hexdigest())
        coord.submit(claim["token"], record)
        with self.assertRaises(parallel.ClaimRejected) as caught:
            coord.submit(claim["token"], record)
        self.assertIn("duplicate", str(caught.exception))
        with self.assertRaises(parallel.ClaimRejected):
            coord.submit("0" * 32, record)                                    # unknown token
        self.assertEqual(sum(r["unit_key"] == claim["unit"]["unit_key"] for r in coord.records()), 1)
        coord.stop()

    def test_infrastructure_retry_limit_is_unchanged(self):
        coord = self.coordinator()
        self.drain(coord, [self.worker(coord, "L3", executor=self.executor(respond=E.provider_down))])
        rows = [r for r in coord.records() if coord.assignment["unit_lane"][r["unit_key"]] == "L3"]
        per_unit = {}
        for r in rows:
            per_unit.setdefault(r["unit_key"], []).append(r["outcome_status"])
        self.assertTrue(all(v == ["MISSING_INFRASTRUCTURE"] * 3 for v in per_unit.values()))
        self.assertTrue(coord.validate()["ok"])
        coord.stop()


class HaltTests(_Base):
    def run_one(self, coord, lane="L0", **kw):
        w = self.worker(coord, lane, **kw)
        return w.run_once()

    def gpu(self, lane="L0"):
        return next(l["gpu_uuid"] for l in LANES if l["lane_id"] == lane)

    def test_cpu_fallback_halts_and_quarantines(self):
        coord = self.coordinator()
        snap = lambda: {"provider_version": "x", "model_digest": DIGEST, "processor": "gpu_partial", "cuda_visible_devices": self.gpu()}
        with self.assertRaises(parallel.CoordinatorHalted):
            self.run_one(coord, snapshot=snap)
        self.assertIn("validation or lane provenance", coord.halted()["reason"])
        self.assertEqual(len(list((self.root / "run" / "quarantine").glob("*.json"))), 1)
        self.assertEqual(coord.records(), [])
        with self.assertRaises(parallel.CoordinatorHalted):
            coord.claim("L1", "L1-w")                                         # no lane may continue
        coord.clear_halt(reviewed_by="operator", note="synthetic test")
        self.assertIsNotNone(coord.claim("L1", "L1-w"))
        coord.stop()

    def test_runtime_drift_and_lane_mismatch_halt(self):
        coord = self.coordinator()
        drift = lambda: {"provider_version": "x", "model_digest": "9" * 64, "processor": "gpu_full", "cuda_visible_devices": self.gpu()}
        with self.assertRaises(parallel.CoordinatorHalted):
            self.run_one(coord, snapshot=drift)
        coord.stop()
        other = self.coordinator(root=self.root / "run2")
        wrong_gpu = lambda: {"provider_version": "x", "model_digest": DIGEST, "processor": "gpu_full", "cuda_visible_devices": "GPU-other"}
        w = parallel.LaneWorker(other, lane_id="L0", worker_id="L0-w", executor=self.executor(), root=self.root / "run2",
                                snapshot=wrong_gpu, heartbeat_seconds=3600)
        with self.assertRaises(parallel.CoordinatorHalted):
            w.run_once()
        other.stop()

    def test_token_discrepancy_is_recorded_then_halts(self):
        coord = self.coordinator()
        executor = self.executor(counter=lambda prompt: E.counter(prompt) + 1)    # local count disagrees with provider
        self.run_one(coord, executor=executor)
        self.assertTrue(coord.halted()["reason"].startswith("provider token discrepancy"))
        record = coord.records()[0]
        self.assertEqual(record["outcome_status"], "INVALID_MEASUREMENT")          # preserved as evidence
        coord.stop()

    def test_protocol_violation_from_a_worker_halts(self):
        sentences = tuple(x.replace(".", " skill_03.") for x in NEUTRAL_SENTENCES)
        base = FillerCorpus("leaky", sentences)
        leaky = FillerCorpus("leaky", sentences, {"approved": True, "approved_by": "FIXTURE", "approval_reference": "unit-test",
                                                  "corpus_sha256": base.sha256})
        coord = self.coordinator()
        lane = next(l for l in LANES if any(u["condition"] == "Rel" for u in coord.lane_units[l["lane_id"]][:3]))["lane_id"]
        w = self.worker(coord, lane, executor=self.executor(corp=leaky))
        with self.assertRaises(ProtocolViolation):
            while w.run_once():
                pass
        self.assertIn("protocol violation", coord.halted()["reason"])
        coord.stop()

    def test_evidence_corruption_halts_restart(self):
        coord = self.coordinator()
        self.run_one(coord)
        coord.stop()
        path = self.root / "run" / "records.jsonl"
        lines = path.read_text(encoding="utf-8").splitlines()
        lines[0] = lines[0].replace('"success":true', '"success":false')
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        with self.assertRaises(parallel.CoordinatorHalted):
            self.coordinator()
        self.assertIn("evidence corruption", json.loads((self.root / "run" / "HALTED.json").read_text())["reason"])


class TransportTests(_Base):
    def test_file_transport_between_worker_and_coordinator(self):
        coord = self.coordinator()
        run = self.root / "run"
        client = parallel.FileCoordinatorClient(run, pump=lambda: parallel.serve_once(coord, run))
        w = self.worker(client, "L0")
        self.drain(coord, [w])
        lane_units = [u for u in self.spec["manifest"]["units"] if self.assignment["unit_lane"][u["unit_key"]] == "L0"]
        self.assertEqual(sum(r["outcome_status"] == "VALID" for r in coord.records()), len(lane_units))
        with self.assertRaises(parallel.ClaimRejected):
            client.submit("0" * 32, coord.records()[0])
        self.assertEqual(list((run / "inbox").glob("*.json")), [])
        coord.stop()


if __name__ == "__main__":
    unittest.main()
