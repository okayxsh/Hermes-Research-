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


class Processes:
    """Synthetic process table for the fencing probe: identities are 'running' unless marked stopped or unknown."""

    def __init__(self):
        self.stopped, self.unknown = set(), set()

    @staticmethod
    def key(identity):
        return (identity.get("host"), identity.get("pid"), identity.get("start_time"))

    def __call__(self, identity):
        if any(identity.get(k) in (None, "") for k in parallel.IDENTITY_KEYS) or self.key(identity) in self.unknown:
            return "unknown"
        return "stopped" if self.key(identity) in self.stopped else "running"

    def stop(self, identity, *roles):
        for role in roles or ("worker", "provider"):
            self.stopped.add(self.key(identity[role]))


def ident(lane, gen=0):
    i = int(lane[1:])
    return {"worker": {"host": "lab", "pid": 1000 + 10 * i + gen, "start_time": f"w{gen}"},
            "provider": {"host": "lab", "pid": 5000 + 10 * i + gen, "start_time": f"p{gen}"}}


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
        self.procs = Processes()

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
                                    expected_model_digest=DIGEST, synthetic=True, lease_seconds=lease, clock=self.clock,
                                    process_probe=self.procs).start()

    def worker(self, client, lane, executor=None, snapshot=None, gen=0):
        gpu = next(l["gpu_uuid"] for l in LANES if l["lane_id"] == lane)
        snap = snapshot or (lambda: {"provider_version": "0.0-synthetic", "model_digest": DIGEST, "processor": "gpu_full",
                                     "placement": "lane_gpu_exclusive", "cuda_visible_devices": gpu})
        return parallel.LaneWorker(client, lane_id=lane, worker_id=f"{lane}-w", executor=executor or self.executor(),
                                   root=self.root / "run", snapshot=snap, identity=lambda: ident(lane, gen), heartbeat_seconds=3600)

    def provenance(self, record, claim, lane):
        l = next(x for x in LANES if x["lane_id"] == lane)
        record["execution_provenance"].update(
            mode="parallel", lane_id=lane, worker_id=f"{lane}-w", endpoint=l["endpoint"], gpu_uuid=l["gpu_uuid"],
            cuda_visible_devices=l["gpu_uuid"], provider_version="x", model_digest=DIGEST, processor_before="gpu_full",
            processor_after="gpu_full", placement_before="lane_gpu_exclusive", placement_after="lane_gpu_exclusive",
            host="h", pid=1, hermes_workspace="w", claim_token_sha256=__import__("hashlib").sha256(claim["token"].encode()).hexdigest())
        return record

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
        lost = coord.claim("L0", "L0-dead", ident("L0"))                      # worker claims, then dies
        self.clock.now += 101
        self.procs.stop(ident("L0"))                                          # worker and provider demonstrably gone
        replacement = coord.claim("L0", "L0-w", ident("L0", 1))               # lease expiry closes the lost attempt
        self.assertEqual((replacement["unit"]["unit_key"], replacement["attempt"]), (lost["unit"]["unit_key"], 2))
        coord.submit(replacement["token"], self.provenance(self.executor()(replacement["unit"], attempt=2,
                                                                          attempt_dir=self.root / "r"), replacement, "L0"))
        statuses = [r["outcome_status"] for r in coord.records() if r["unit_key"] == lost["unit"]["unit_key"]]
        self.assertEqual(statuses, ["MISSING_INFRASTRUCTURE", "VALID"])
        coord.stop()

    def test_coordinator_restart_revokes_open_claims_and_rejects_their_results(self):
        coord = self.coordinator()
        w = self.worker(coord, "L1")
        self.assertTrue(w.run_once())                                         # one completed unit survives the restart
        open_claim = coord.claim("L1", "L1-w", ident("L1"))
        coord.stop()
        restarted = self.coordinator()
        late = self.executor()(open_claim["unit"], attempt=open_claim["attempt"], attempt_dir=self.root / "late")
        with self.assertRaises(parallel.ClaimRejected):
            restarted.submit(open_claim["token"], late)                       # fenced, not yet verified: rejected as late
        with self.assertRaises(parallel.LaneSuspended):
            restarted.claim("L1", "L1-w", ident("L1", 1))                     # old worker/provider still running
        self.procs.stop(ident("L1"))
        self.drain(restarted, [self.worker(restarted, l["lane_id"], gen=1 if l["lane_id"] == "L1" else 0) for l in LANES])
        self.assertTrue(restarted.validate()["ok"])
        attempts = [r for r in restarted.records() if r["unit_key"] == open_claim["unit"]["unit_key"]]
        self.assertEqual([r["outcome_status"] for r in attempts], ["MISSING_INFRASTRUCTURE", "VALID"])
        self.assertEqual(restarted.summary()["final_status_counts"]["VALID"], 28)
        restarted.stop()

    def test_duplicate_results_are_rejected(self):
        coord = self.coordinator()
        claim = coord.claim("L2", "L2-w", ident("L2"))
        w = self.worker(coord, "L2")
        record = self.provenance(w.executor(claim["unit"], attempt=claim["attempt"], attempt_dir=self.root / "dup"), claim, "L2")
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
        snap = lambda: {"provider_version": "x", "model_digest": DIGEST, "processor": "gpu_partial",
                        "placement": "lane_gpu_exclusive", "cuda_visible_devices": self.gpu()}
        with self.assertRaises(parallel.CoordinatorHalted):
            self.run_one(coord, snapshot=snap)
        self.assertIn("validation or lane provenance", coord.halted()["reason"])
        self.assertEqual(len(list((self.root / "run" / "quarantine").glob("*.json"))), 1)
        self.assertEqual(coord.records(), [])
        with self.assertRaises(parallel.CoordinatorHalted):
            coord.claim("L1", "L1-w", ident("L1"))                            # no lane may continue
        coord.clear_halt(reviewed_by="operator", note="synthetic test")
        self.assertIsNotNone(coord.claim("L1", "L1-w", ident("L1")))
        coord.stop()

    def test_runtime_drift_and_lane_mismatch_halt(self):
        coord = self.coordinator()
        drift = lambda: {"provider_version": "x", "model_digest": "9" * 64, "processor": "gpu_full",
                         "placement": "lane_gpu_exclusive", "cuda_visible_devices": self.gpu()}
        with self.assertRaises(parallel.CoordinatorHalted):
            self.run_one(coord, snapshot=drift)
        coord.stop()
        other = self.coordinator(root=self.root / "run2")
        wrong_gpu = lambda: {"provider_version": "x", "model_digest": DIGEST, "processor": "gpu_full",
                             "placement": "lane_gpu_exclusive", "cuda_visible_devices": "GPU-other"}
        w = parallel.LaneWorker(other, lane_id="L0", worker_id="L0-w", executor=self.executor(), root=self.root / "run2",
                                snapshot=wrong_gpu, identity=lambda: ident("L0"), heartbeat_seconds=3600)
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


class FencingTests(_Base):
    """Phase 3D: a timed-out or abandoned unit is not reassigned until the previous worker AND its provider server are
    verified stopped; unknown termination halts the lane; late results after verified termination halt the lane."""

    def expired(self, coord, lane):
        lost = coord.claim(lane, f"{lane}-old", ident(lane))
        self.clock.now += 101
        return lost

    def test_expired_worker_that_remains_alive_blocks_reassignment(self):
        coord = self.coordinator(lease=100.0)
        lost = self.expired(coord, "L0")
        with self.assertRaises(parallel.LaneSuspended):
            coord.claim("L0", "L0-new", ident("L0", 1))                        # old worker is still alive
        self.assertEqual([r["outcome_status"] for r in coord.records()], ["MISSING_INFRASTRUCTURE"])
        self.assertEqual(coord.summary()["fenced_lanes"], ["L0"])
        with self.assertRaises(parallel.ClaimRejected):
            coord.heartbeat(lost["token"])                                     # its heartbeats no longer extend anything
        late = self.provenance(self.executor()(lost["unit"], attempt=1, attempt_dir=self.root / "late"), lost, "L0")
        with self.assertRaises(parallel.ClaimRejected) as caught:
            coord.submit(lost["token"], late)                                  # alive worker's result: rejected as late
        self.assertIn("late", str(caught.exception))
        self.procs.stop(ident("L0"), "worker")
        with self.assertRaises(parallel.LaneSuspended):
            coord.claim("L0", "L0-new", ident("L0", 1))                        # worker gone, provider still running
        self.procs.stop(ident("L0"), "provider")
        with self.assertRaises(parallel.ClaimRejected):
            coord.claim("L0", "L0-old", ident("L0"))                           # fenced processes may never claim again
        replacement = coord.claim("L0", "L0-new", ident("L0", 1))
        self.assertEqual((replacement["unit"]["unit_key"], replacement["attempt"]), (lost["unit"]["unit_key"], 2))
        self.assertEqual(coord.summary()["fenced_lanes"], [])
        events = [json.loads(x)["event"] for x in (self.root / "run" / "journal.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertLess(events.index("termination_verified"), len(events) - 1 - events[::-1].index("claimed"))
        coord.stop()

    def test_outstanding_provider_request_during_coordinator_restart(self):
        coord = self.coordinator()
        open_claim = coord.claim("L2", "L2-w", ident("L2"))                    # request in flight at the provider
        coord.stop()
        restarted = self.coordinator()
        self.assertEqual(restarted.summary()["fenced_lanes"], ["L2"])
        self.assertEqual([r["outcome_status"] for r in restarted.records()], ["MISSING_INFRASTRUCTURE"])
        self.procs.stop(ident("L2"), "worker")                                  # client gone; server-side inference may not be
        with self.assertRaises(parallel.LaneSuspended):
            restarted.claim("L2", "L2-w", ident("L2", 1))
        self.assertIsNotNone(restarted.claim("L0", "L0-w", ident("L0")))       # other lanes are unaffected
        self.procs.stop(ident("L2"), "provider")
        again = restarted.claim("L2", "L2-w", ident("L2", 1))
        self.assertEqual((again["unit"]["unit_key"], again["attempt"]), (open_claim["unit"]["unit_key"], 2))
        restarted.stop()
        replayed = self.coordinator()                                           # verified state survives another restart
        self.assertEqual(replayed.claims[open_claim["token"]]["status"], "terminated_verified")
        replayed.stop()

    def test_no_reassignment_before_verified_termination_and_one_claim_per_lane(self):
        coord = self.coordinator(lease=100.0)
        with self.assertRaises(parallel.ClaimRejected):
            coord.claim("L1", "L1-w", {"worker": {"host": "lab", "pid": 1}, "provider": ident("L1")["provider"]})
        first = coord.claim("L1", "L1-w", ident("L1"))
        with self.assertRaises(parallel.ClaimRejected):
            coord.claim("L1", "L1-x", ident("L1", 1))                          # one in-flight unit per lane
        self.clock.now += 101
        for _ in range(3):
            with self.assertRaises(parallel.LaneSuspended):
                coord.claim("L1", "L1-x", ident("L1", 1))
        attempts = [r for r in coord.records() if r["unit_key"] == first["unit"]["unit_key"]]
        self.assertEqual(len(attempts), 1)
        self.assertFalse(any(c["status"] == "active" and c["lane_id"] == "L1" for c in coord.claims.values()))
        coord.stop()

    def test_unknown_termination_halts_only_the_affected_lane(self):
        coord = self.coordinator(lease=100.0)
        self.expired(coord, "L3")
        self.procs.stop(ident("L3"), "worker")
        self.procs.unknown.add(Processes.key(ident("L3")["provider"]))          # e.g. other host, no /proc evidence
        with self.assertRaises(parallel.LaneHalted):
            coord.claim("L3", "L3-new", ident("L3", 1))
        self.assertTrue((self.root / "run" / "LANE_HALTED_L3.json").is_file())
        self.assertIsNone(coord.halted())
        self.assertIsNotNone(coord.claim("L0", "L0-w", ident("L0")))
        self.assertEqual(coord.summary()["halted_lanes"], ["L3"])
        lane_units = len(coord.lane_units["L3"])
        self.assertEqual(coord.summary()["pending_units"] - coord.summary()["pending_on_live_lanes"], lane_units)
        with self.assertRaises(parallel.LaneHalted):
            coord.claim("L3", "L3-new", ident("L3", 1))
        with self.assertRaises(ValueError):
            coord.clear_lane_halt("L3", reviewed_by="", note="x")
        coord.clear_lane_halt("L3", reviewed_by="operator", note="verified on the host that both processes are gone")
        self.assertIsNotNone(coord.claim("L3", "L3-new", ident("L3", 1)))
        coord.stop()

    def test_late_result_after_termination_and_reassignment_halts_the_lane(self):
        coord = self.coordinator(lease=100.0)
        lost = self.expired(coord, "L0")
        self.procs.stop(ident("L0"))                                            # probe says both stopped...
        replacement = coord.claim("L0", "L0-new", ident("L0", 1))
        late = self.provenance(self.executor()(lost["unit"], attempt=1, attempt_dir=self.root / "late"), lost, "L0")
        with self.assertRaises(parallel.LaneHalted):
            coord.submit(lost["token"], late)                                   # ...yet a result arrives: overlap evidence
        self.assertIn("overlap", coord.lane_halted("L0")["reason"])
        self.assertEqual(len(list((self.root / "run" / "quarantine").glob("*.json"))), 1)
        current = self.provenance(self.executor()(replacement["unit"], attempt=2, attempt_dir=self.root / "r"), replacement, "L0")
        with self.assertRaises(parallel.LaneHalted):
            coord.submit(replacement["token"], current)                         # the overlapping replacement is not accepted
        self.assertEqual([r["outcome_status"] for r in coord.records()], ["MISSING_INFRASTRUCTURE"])
        with self.assertRaises(parallel.LaneHalted):
            coord.claim("L0", "L0-new", ident("L0", 2))
        self.assertIsNone(coord.halted())
        coord.stop()


class PlacementTests(_Base):
    """Phase 3D: GPU execution needs /api/ps full offload AND process-bound CUDA evidence on the lane's GPU UUID."""
    GPU = "GPU-synthetic-0"

    def test_compute_apps_and_process_tree(self):
        rows = parallel.parse_compute_apps("5001, GPU-synthetic-0, 9000\n7777, GPU-synthetic-1, 100\n[N/A]\n")
        self.assertEqual([r["pid"] for r in rows], [5001, 7777])
        tree = parallel.process_tree(5000, children_of=lambda pid: {5000: [5001], 5001: [5002]}.get(pid, []))
        self.assertEqual(tree, {5000, 5001, 5002})
        place = lambda text: parallel.gpu_placement(parallel.parse_compute_apps(text), self.GPU, tree)
        self.assertEqual(place("5001, GPU-synthetic-0, 9000"), "lane_gpu_exclusive")
        self.assertEqual(place(""), "no_provider_cuda_context_on_lane_gpu")              # CUDA_VISIBLE_DEVICES set, nothing on GPU
        self.assertEqual(place("5001, GPU-synthetic-1, 9000"), "provider_on_other_gpu")
        self.assertEqual(place("5001, GPU-synthetic-0, 9000\n5002, GPU-synthetic-2, 10"), "provider_on_other_gpu")
        self.assertEqual(place("5001, GPU-synthetic-0, 9000\n6000, GPU-synthetic-0, 10"), "lane_gpu_shared_with_foreign_process")

    def test_ollama_status_fields(self):
        ps = {"models": [{"name": "gemma4:12b", "model": "gemma4:12b", "size": 100, "size_vram": 100, "digest": DIGEST,
                          "expires_at": "2026-11-26T00:00:00Z"}]}
        request = lambda e, path: {"/api/version": {"version": "0.0"}, "/api/ps": ps}[path]
        kw = dict(compute_apps=[{"pid": 5001, "gpu_uuid": self.GPU, "used_memory_mib": 1}], provider_pids={5000, 5001},
                  lane_gpu_uuid=self.GPU)
        snap = parallel.ollama_lane_provenance(request, "e", "gemma4:12b", **kw)
        self.assertEqual((snap["processor"], snap["placement"], snap["model_digest"]), ("gpu_full", "lane_gpu_exclusive", DIGEST))
        for vram, expected in ((60, "gpu_partial"), (0, "cpu")):
            ps["models"][0]["size_vram"] = vram
            self.assertEqual(parallel.ollama_lane_provenance(request, "e", "gemma4:12b", **kw)["processor"], expected)
        del ps["models"][0]["size_vram"]
        self.assertEqual(parallel.ollama_lane_provenance(request, "e", "gemma4:12b", **kw)["processor"], "unknown")
        self.assertEqual(parallel.ollama_lane_provenance(request, "e", "other:1b", **kw)["processor"], "not_loaded")
        ps["models"][0]["size_vram"] = 100
        cpu_only = dict(kw, compute_apps=[])                                    # full offload claimed, no CUDA process
        self.assertEqual(parallel.ollama_lane_provenance(request, "e", "gemma4:12b", **cpu_only)["placement"],
                         "no_provider_cuda_context_on_lane_gpu")

    def test_unverified_placement_halts_and_quarantines(self):
        for n, placement in enumerate(("no_provider_cuda_context_on_lane_gpu", "lane_gpu_shared_with_foreign_process", None)):
            run = self.root / f"run-{n}"
            coord = self.coordinator(root=run)
            snap = lambda: {"provider_version": "x", "model_digest": DIGEST, "processor": "gpu_full", "placement": placement,
                            "cuda_visible_devices": self.GPU}
            w = parallel.LaneWorker(coord, lane_id="L0", worker_id="L0-w", executor=self.executor(), root=run,
                                    snapshot=snap, identity=lambda: ident("L0"), heartbeat_seconds=3600)
            with self.assertRaises(parallel.CoordinatorHalted):
                w.run_once()
            self.assertIn("validation or lane provenance", coord.halted()["reason"])
            self.assertEqual(coord.records(), [])
            coord.stop()

    def test_process_state_is_unknown_without_local_evidence(self):
        self.assertEqual(parallel.linux_process_state({"host": "elsewhere", "pid": 1, "start_time": "1"}), "unknown")
        self.assertEqual(parallel.linux_process_state({"host": "lab", "pid": 1, "start_time": None}), "unknown")


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
