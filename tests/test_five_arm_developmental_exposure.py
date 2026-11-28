"""Phase 3O: developmental-pilot exposure eligibility at manifest freeze and execution preflight.

SYNTHETIC evidence is generated here (no private task identities enter Git).  ``RealApprovedEvidenceTests`` checks the
exact approved bytes from the private coordinator evidence when ``RQ1_PHASE3O_EVIDENCE_ROOT`` points at the extracted
``rq1-phase3o-coordinator`` directory; it is skipped otherwise.  No provider, model or GPU is used."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from rq1.five_arm import exposure_acceptance as X
from rq1.five_arm import manifest as M
from rq1.five_arm import readiness, runner, runtime
from rq1.five_arm.measurements import DIAGNOSTIC_PROTOCOL_VERSION
from rq1.five_arm.protocol import PROTOCOL_VERSION, SCIENTIFIC_DECISIONS
from test_five_arm_runtime_readiness import ARTIFACTS, complete_config, observed_from, person, state_for

sha = lambda data: hashlib.sha256(data).hexdigest()
dump = lambda value: json.dumps(value, indent=1, sort_keys=True).encode("utf-8")
TASKS = [f"valid_seen:synthetic_family_{i % 5}-Obj-None-Rec-{i}/trial_T2019010{i:02d}" for i in range(12)]
SD19_CHOICE = "Reuse the exact 12 synthetic valid_seen tasks for a developmental pilot only (test fixture choice)."


def build_evidence(tasks=TASKS):
    roster = dump({"approved": False, "preparation_only": True, "tasks": [{"task_id": t} for t in tasks]})
    amendment = b"# synthetic CPU protocol and SAP amendment\n"
    units, records, journal = [], [], []
    for t in tasks:
        for seed in (11, 29, 47):
            for cond in ("UNDISTURBED", "DETOURED"):
                key = sha(f"{t}|{seed}|{cond}".encode())
                units.append({"unit_key": key, "task_id": t, "seed": seed, "condition": cond})
                records.append({"schema": "rq1-five-arm-diagnostic-v1", "protocol_version": DIAGNOSTIC_PROTOCOL_VERSION,
                                "unit_key": key, "task_id": t, "seed": seed, "condition": cond, "attempt": 1,
                                "outcome_status": "VALID", "synthetic": False})
                journal += [{"event": "attempt_started", "unit_key": key, "attempt": 1},
                            {"event": "attempt_finished", "unit_key": key, "attempt": 1, "outcome_status": "VALID"}]
    dman = {"schema": M.DIAGNOSTIC_SCHEMA, "stage": "pilot_diagnostic", "split": "valid_seen",
            "run_id": "rq1-phase2g-d0-diagnostic", "synthetic": False, "tasks": [{"task_id": t} for t in tasks], "units": units}
    dman["manifest_sha256"] = M.manifest_hash(dman)
    dman["status"] = "frozen"
    records_b = b"".join(json.dumps(r, sort_keys=True).encode() + b"\n" for r in records)
    journal_b = b"".join(json.dumps(e, sort_keys=True).encode() + b"\n" for e in journal)
    receipt = dump({"summary": {"run_id": "rq1-phase2g-d0-diagnostic", "units": 72, "final_status_counts": {"VALID": 72},
                                "pending_units": 0, "synthetic": False},
                    "validation": {"ok": True, "complete": True, "problems": []}})
    statement = dump({
        "schema": X.DEVELOPMENTAL_SCHEMA, "purpose": X.DEVELOPMENTAL_PURPOSE, "roster_sha256": sha(roster),
        "protocol_amendment_sha256": sha(amendment), "roster_size": len(tasks), "status": "REVIEW_ONLY_PENDING_SEPARATE_APPROVAL",
        "historical_status_counts": {"NO_LOCATED_MODEL_OUTCOME": len(tasks), "UNKNOWN": 0},
        "known_pilot_roster_model_outcome_exposure": {
            "run_id": "rq1-phase2g-d0-diagnostic", "records_sha256": sha(records_b), "completion_receipt_sha256": sha(receipt),
            "per_task": 6, "task_count": 12, "valid_d0_units": 72, "seeds": [11, 29, 47], "conditions": ["UNDISTURBED", "DETOURED"]},
        "retrospective_coverage_as_of_original_audit": {"known_gaps": ["records for part of the period are incomplete"],
                                                        "investigation_closed": True},
        "tasks": [{"task_id": t, "current_known_exposure": X.DEVELOPMENTAL_EXPOSURE_CLASS, "valid_diagnostic_units": 6,
                   "historical_ledger": {"ledger_class": "NO_MODEL_OUTCOME_FOUND", "ledger_evidence_complete": False}}
                  for t in tasks]})
    pins = {**X.APPROVED_DEVELOPMENTAL, "exposure_statement_sha256": sha(statement), "roster_sha256": sha(roster),
            "cpu_amendment_sha256": sha(amendment), "diagnostic_manifest_sha256": dman["manifest_sha256"],
            "sd19_decision_sha256": readiness.decision_sha256(X.SD19_KEY, SD19_CHOICE)}
    evidence = {"kind": "developmental", "statement": statement, "roster": roster, "cpu_amendment": amendment,
                "diagnostic_records": records_b, "diagnostic_receipt": receipt, "diagnostic_manifest": dump(dman),
                "diagnostic_journal": journal_b,
                "exposure_acceptance": person(sha(statement), stage_scope=X.DEVELOPMENTAL_STAGE_SCOPE, roster_sha256=sha(roster),
                                              protocol_amendment_sha256=sha(amendment)),
                "cpu_amendment_approval": person(sha(amendment))}
    return evidence, pins


def sd19_state(pins, choice=SD19_CHOICE, state=None):
    state = copy.deepcopy(state) if state is not None else {"scientific_decisions": {}}
    state.setdefault("scientific_decisions", {})[X.SD19_KEY] = person(readiness.decision_sha256(X.SD19_KEY, choice), choice=choice)
    return state


def seven_condition_manifest(pins, *, stage="pilot", split="valid_seen", tasks=TASKS, exposure=None, roster=None, extra=None):
    m = {"schema": M.SCHEMA, "protocol_version": PROTOCOL_VERSION, "stage": stage, "split": split, "run_id": "dev-pilot-test",
         "synthetic": False, "total_action_cap": 50, "units": [], "tasks": [{"task_id": t} for t in tasks],
         "source_commit": "a" * 40,
         "input_hashes": {"roster_sha256": roster or pins["roster_sha256"],
                          "exposure_statement_sha256": exposure or pins["exposure_statement_sha256"],
                          "runtime_config_sha256": "0" * 64, **(extra or {})}}
    m["manifest_sha256"] = M.manifest_hash(m)
    return m


def dev_artifacts(pins):
    return {**ARTIFACTS, "pilot_roster": pins["roster_sha256"], "pilot_exposure_statement": pins["exposure_statement_sha256"]}


def edit_json_bytes(data: bytes, change):
    value = json.loads(data.decode("utf-8"))
    change(value)
    return dump(value)


class DevelopmentalValidatorTests(unittest.TestCase):
    def setUp(self):
        self.ev, self.pins = build_evidence()
        self.state = sd19_state(self.pins)

    def check(self, ev=None, state=None):
        return X.developmental_eligibility(ev or self.ev, readiness_state=self.state if state is None else state, approved=self.pins)

    def test_valid_developmental_eligibility_acknowledges_both_exposures(self):
        result = self.check()
        self.assertEqual(result["problems"], [])
        ack = result["acknowledgement"]
        self.assertEqual(ack["authorized_diagnostic_exposure"]["valid_units"], 72)
        self.assertEqual(ack["authorized_diagnostic_exposure"]["status"], X.DEVELOPMENTAL_EXPOSURE_CLASS)
        self.assertTrue(ack["unresolved_historical_exposure"]["known_gaps"])
        self.assertFalse(ack["unresolved_historical_exposure"]["status_upgraded"])
        self.assertFalse(ack["confirmatory_eligibility"])
        self.assertEqual(result["task_ids"], sorted(TASKS))
        gate = X.exposure_eligibility_problems(seven_condition_manifest(self.pins), self.ev, readiness_state=self.state,
                                               artifacts=dev_artifacts(self.pins), approved=self.pins)
        self.assertEqual(gate, [])

    def test_altered_exposure_or_diagnostic_evidence_is_refused(self):
        cases = {
            "statement": lambda ev: ev["statement"] + b" ",
            "diagnostic_records": lambda ev: ev["diagnostic_records"].replace(b'"VALID"', b'"INVALID_MEASUREMENT"', 1),
            "diagnostic_receipt": lambda ev: ev["diagnostic_receipt"] + b"\n",
            "diagnostic_manifest": lambda ev: edit_json_bytes(ev["diagnostic_manifest"], lambda m: m["units"].pop()),
            "diagnostic_journal": lambda ev: ev["diagnostic_journal"].replace(b'"attempt_finished"', b'"attempt_started"', 1),
            "roster": lambda ev: ev["roster"] + b" ",
            "cpu_amendment": lambda ev: ev["cpu_amendment"] + b"edited\n",
        }
        for name, change in cases.items():
            with self.subTest(name=name):
                ev = dict(self.ev); ev[name] = change(self.ev)
                self.assertTrue(self.check(ev)["problems"], name)
        ev = dict(self.ev); del ev["diagnostic_journal"]
        self.assertIn("lacks the exact bytes", self.check(ev)["problems"][0])

    def test_status_upgrade_or_missing_acknowledgement_is_refused_even_with_matching_pins(self):
        for change in (lambda s: s["tasks"][0]["historical_ledger"].update(ledger_class="VERIFIED_UNEXPOSED"),
                       lambda s: s["tasks"][0].update(current_known_exposure="NO_LOCATED_MODEL_OUTCOME"),
                       lambda s: s["retrospective_coverage_as_of_original_audit"].update(known_gaps=[]),
                       lambda s: s["known_pilot_roster_model_outcome_exposure"].update(valid_d0_units=60)):
            ev = dict(self.ev); ev["statement"] = edit_json_bytes(self.ev["statement"], change)
            pins = {**self.pins, "exposure_statement_sha256": sha(ev["statement"])}
            ev["exposure_acceptance"] = {**self.ev["exposure_acceptance"], "artifact_sha256": pins["exposure_statement_sha256"]}
            self.assertTrue(X.developmental_eligibility(ev, readiness_state=self.state, approved=pins)["problems"])

    def test_missing_or_mismatched_approvals_are_refused(self):
        variants = {
            "no exposure approval": dict(self.ev, exposure_acceptance=None),
            "approval for other bytes": dict(self.ev, exposure_acceptance={**self.ev["exposure_acceptance"], "artifact_sha256": "1" * 64}),
            "approval not scoped": dict(self.ev, exposure_acceptance={**self.ev["exposure_acceptance"], "stage_scope": "pilot"}),
            "fixture approver": dict(self.ev, exposure_acceptance={**self.ev["exposure_acceptance"], "approved_by": "FIXTURE"}),
            "no CPU amendment approval": dict(self.ev, cpu_amendment_approval=None),
        }
        for name, ev in variants.items():
            with self.subTest(name=name):
                self.assertTrue(self.check(ev)["problems"])
        original = SCIENTIFIC_DECISIONS[X.SD19_KEY]["recommendation"]
        for state in ({"scientific_decisions": {}}, sd19_state(self.pins, choice=original)):
            self.assertTrue(any("SD-19" in p for p in self.check(state=state)["problems"]))


class ManifestGateTests(unittest.TestCase):
    def setUp(self):
        self.ev, self.pins = build_evidence()
        self.state = sd19_state(self.pins)

    def gate(self, manifest, ev=None, artifacts=None):
        return X.exposure_eligibility_problems(manifest, self.ev if ev is None else ev, readiness_state=self.state,
                                               artifacts=dev_artifacts(self.pins) if artifacts is None else artifacts,
                                               approved=self.pins)

    def test_incorrect_stage_or_split_is_refused(self):
        self.assertTrue(self.gate(seven_condition_manifest(self.pins, split="valid_unseen")))
        self.assertTrue(self.gate(seven_condition_manifest(self.pins, stage="pilot_rehearsal")))
        bad = seven_condition_manifest(self.pins); bad["protocol_version"] = "rq1-five-arm-v1"
        self.assertTrue(self.gate(bad))
        self.assertTrue(self.gate(seven_condition_manifest(self.pins), ev=None or {"kind": "other"}))
        self.assertTrue(self.gate(seven_condition_manifest(self.pins), ev={}))

    def test_changed_or_missing_task_identities_are_refused(self):
        self.assertTrue(self.gate(seven_condition_manifest(self.pins, tasks=TASKS[:11])))
        swapped = TASKS[:11] + ["valid_seen:synthetic_family_9-Other-None-Rec-99/trial_T20190199"]
        self.assertTrue(any("task identities" in p for p in self.gate(seven_condition_manifest(self.pins, tasks=swapped))))
        ev, pins = build_evidence(swapped)                                  # self-consistent evidence for other tasks
        self.assertTrue(self.gate(seven_condition_manifest(self.pins, tasks=swapped), ev=ev))

    def test_exposure_digest_must_match_approval_manifest_and_readiness(self):
        self.assertTrue(self.gate(seven_condition_manifest(self.pins, exposure="2" * 64)))
        self.assertTrue(self.gate(seven_condition_manifest(self.pins), artifacts={**dev_artifacts(self.pins),
                                                                                  "pilot_exposure_statement": "3" * 64}))
        self.assertTrue(self.gate(seven_condition_manifest(self.pins, roster="4" * 64)))
        self.assertTrue(self.gate(seven_condition_manifest(self.pins), artifacts={**dev_artifacts(self.pins), "pilot_roster": "5" * 64}))

    def test_diagnostic_manifests_keep_their_existing_requirements(self):
        self.assertEqual(X.exposure_eligibility_problems({"schema": M.DIAGNOSTIC_SCHEMA, "stage": "pilot_diagnostic"}, None,
                                                         readiness_state={}, artifacts={}), [])


class ConfirmatoryIsolationTests(unittest.TestCase):
    def setUp(self):
        self.ev, self.pins = build_evidence()
        self.state = sd19_state(self.pins)
        self.unseen = [f"valid_unseen:synthetic_family_{i % 5}-Obj-None-Rec-{100 + i}/trial_T2019020{i:02d}" for i in range(4)]
        self.protocol = "7" * 64

    def independent(self, tasks):
        rows = [{"task_id": t, "exposure_class": "NO_MODEL_OUTCOME_FOUND", "exposure_kinds": [], "evidence_complete": False,
                 "provisional_gaps": []} for t in tasks]
        statement = X.build_statement(roster=tasks, ledger_rows=rows, coverage=X.UNIVERSITY_INVESTIGATION_COVERAGE,
                                      roster_sha256="8" * 64, protocol_sha256=self.protocol, purpose="confirmatory")
        return {"kind": "independent", "statement": statement, "acceptance": person(statement["statement_sha256"]),
                "protocol_sha256": self.protocol, "known_exposed_statements": [self.ev["statement"]]}

    def manifest(self, tasks, eligibility):
        return seven_condition_manifest(self.pins, stage="confirmatory", split="valid_unseen", tasks=tasks, roster="8" * 64,
                                        exposure=eligibility["statement"]["statement_sha256"])

    def gate(self, manifest, eligibility, artifacts=None):
        return X.exposure_eligibility_problems(manifest, eligibility, readiness_state=self.state, artifacts=artifacts or {},
                                               approved=self.pins)

    def test_independent_confirmatory_requirements_are_preserved(self):
        elig = self.independent(self.unseen)
        self.assertEqual(self.gate(self.manifest(self.unseen, elig), elig), [])
        self.assertTrue(self.gate(self.manifest(self.unseen, elig), dict(elig, acceptance=person("6" * 64))))
        self.assertTrue(self.gate(self.manifest(self.unseen, elig), dict(elig, protocol_sha256="9" * 64)))
        self.assertTrue(self.gate(self.manifest(self.unseen, elig), dict(elig, known_exposed_statements=[])))
        with self.assertRaises(X.ExposureAcceptanceError):                # known exposure still blocks statement construction
            X.build_statement(roster=self.unseen[:1], ledger_rows=[{"task_id": self.unseen[0], "exposure_class": "EXPOSED"}],
                              coverage=X.UNIVERSITY_INVESTIGATION_COVERAGE, roster_sha256="8" * 64,
                              protocol_sha256=self.protocol, purpose="confirmatory")

    def test_developmental_exposure_artifacts_never_authorise_confirmation(self):
        elig = self.independent(self.unseen)
        confirmatory = self.manifest(self.unseen, elig)
        self.assertTrue(any("never authorises confirmatory" in p for p in self.gate(confirmatory, self.ev)))
        as_independent = dict(elig, statement=json.loads(self.ev["statement"]))
        self.assertTrue(any("cannot serve as confirmatory" in p for p in self.gate(confirmatory, as_independent)))
        self.assertTrue(self.gate(confirmatory, elig, {"confirmatory_exposure_statement": self.pins["exposure_statement_sha256"]}))

    def test_known_exposed_tasks_cannot_enter_the_confirmatory_roster(self):
        mixed = self.unseen + [TASKS[0]]
        elig = self.independent(mixed)
        problems = self.gate(self.manifest(mixed, elig), elig)
        self.assertTrue(any("known model-outcome exposure" in p for p in problems))


class FreezeAndLaunchTests(unittest.TestCase):
    """Rejection at manifest freeze and at execution preflight, before any provider call."""

    def setUp(self):
        self.ev, self.pins = build_evidence()
        self.patch = mock.patch.dict(X.APPROVED_DEVELOPMENTAL, self.pins)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.artifacts = dev_artifacts(self.pins)
        state = state_for("five_arm_scientific_pilot")
        state["pilot_roster"] = person(self.pins["roster_sha256"])
        state["pilot_exposure_acceptance"] = person(self.pins["exposure_statement_sha256"])
        self.state = sd19_state(self.pins, state=state)
        self.cfg = {**complete_config(), }
        self.cfg["approval"] = {**self.cfg["approval"], "approved_by": "Reviewer Q"}
        self.state["runtime_verification"] = person(runtime.config_sha256(self.cfg))
        self.artifacts["runtime_config"] = runtime.config_sha256(self.cfg)

    def manifest(self, **kw):
        m = seven_condition_manifest(self.pins, extra={"runtime_config_sha256": runtime.config_sha256(self.cfg)}, **kw)
        m["input_hashes"]["runtime_config_sha256"] = runtime.config_sha256(self.cfg)
        m["manifest_sha256"] = M.manifest_hash(m)
        return m

    def freeze(self, manifest, eligibility):
        return M.freeze_manifest(manifest, person(manifest["manifest_sha256"]), self.state, self.artifacts, real=True,
                                 exposure_eligibility=eligibility)

    def launch(self, manifest, eligibility):
        return runner.assert_execution_permitted(manifest=manifest, readiness_state=self.state, artifacts=self.artifacts,
                                                 observed_runtime=observed_from(self.cfg), approved_runtime=self.cfg,
                                                 exposure_eligibility=eligibility)

    def test_freeze_requires_valid_developmental_eligibility(self):
        manifest = self.manifest()
        frozen = self.freeze(manifest, self.ev)
        self.assertEqual((frozen["status"], M.manifest_hash(frozen)), ("frozen", manifest["manifest_sha256"]))
        altered = dict(self.ev, diagnostic_receipt=self.ev["diagnostic_receipt"] + b" ")
        for eligibility in (None, altered, dict(self.ev, exposure_acceptance=None)):
            with self.assertRaises(M.FiveArmManifestError) as caught:
                self.freeze(manifest, eligibility)
            self.assertIn("exposure eligibility refused", str(caught.exception))
        with self.assertRaises(M.FiveArmManifestError):
            self.freeze(self.manifest(tasks=TASKS[:11]), self.ev)

    def test_launch_requires_valid_developmental_eligibility(self):
        frozen = self.freeze(self.manifest(), self.ev)
        self.assertEqual(self.launch(frozen, self.ev), "five_arm_scientific_pilot")
        for eligibility in (None, dict(self.ev, statement=self.ev["statement"] + b" "),
                            {**self.ev, "kind": "independent"}):
            with self.assertRaises(readiness.ExecutionBlocked) as caught:
                self.launch(frozen, eligibility)
            self.assertIn("exposure eligibility refused", str(caught.exception))
        wrong_sd19 = sd19_state(self.pins, choice=SCIENTIFIC_DECISIONS[X.SD19_KEY]["recommendation"], state=self.state)
        with self.assertRaises(readiness.ExecutionBlocked):
            runner.assert_execution_permitted(manifest=frozen, readiness_state=wrong_sd19, artifacts=self.artifacts,
                                              observed_runtime=observed_from(self.cfg), approved_runtime=self.cfg,
                                              exposure_eligibility=self.ev)

    def test_executor_construction_stops_before_any_provider_call(self):
        frozen = self.freeze(self.manifest(), self.ev)
        with tempfile.TemporaryDirectory() as tmp:
            spec = {"kind": "developmental"}
            for name in X.DEVELOPMENTAL_EVIDENCE_FILES:
                path = Path(tmp) / name
                path.write_bytes(self.ev[name] + (b" " if name == "diagnostic_records" else b""))   # one altered file
                spec[name] = str(path)
            approvals = Path(tmp) / "approvals.json"
            approvals.write_text(json.dumps({"developmental_pilot_exposure_acceptance": self.ev["exposure_acceptance"],
                                             "sequential_cpu_amendment": self.ev["cpu_amendment_approval"]}), encoding="utf-8")
            spec["approvals_record"] = str(approvals)
            config = {"manifest": frozen, "exposure_eligibility": spec, "ollama_endpoint": "http://127.0.0.1:9"}
            with mock.patch.object(runner, "artifact_hashes", return_value=self.artifacts), \
                    mock.patch("rq1.university_gpu.ollama.probe_endpoint") as probe:
                with self.assertRaises(readiness.ExecutionBlocked):
                    runner.build_university_executor(config, readiness_state=self.state, observed_runtime=observed_from(self.cfg),
                                                     approved_runtime=self.cfg)
                probe.assert_not_called()
            self.assertIsNone(runner.load_exposure_eligibility({}))
            with self.assertRaises(readiness.ExecutionBlocked):                # unreadable evidence fails closed
                runner.load_exposure_eligibility({"exposure_eligibility": {**spec, "diagnostic_records": str(Path(tmp) / "missing")}})


ROOT = os.environ.get("RQ1_PHASE3O_EVIDENCE_ROOT")
REAL_FILES = {"statement": "approvals/phase3l-recorded/pilot-exposure-updated-for-review.json",
              "roster": "rosters/pilot/pilot-roster-proposed.json",
              "cpu_amendment": "approvals/phase3l/CPU-PROTOCOL-SAP-AMENDMENT.md",
              "diagnostic_records": "diagnostic/phase3a/run/rq1-phase2g-d0-diagnostic/records.jsonl",
              "diagnostic_receipt": "diagnostic/phase3a/queue-completion.json",
              "diagnostic_manifest": "diagnostic/phase3a/run/rq1-phase2g-d0-diagnostic/manifest.json",
              "diagnostic_journal": "diagnostic/phase3a/run/rq1-phase2g-d0-diagnostic/journal.jsonl"}


@unittest.skipUnless(ROOT, "set RQ1_PHASE3O_EVIDENCE_ROOT to the private extracted rq1-phase3o-coordinator directory")
class RealApprovedEvidenceTests(unittest.TestCase):
    def setUp(self):
        root = Path(ROOT)
        self.ev = {"kind": "developmental", **{k: (root / v).read_bytes() for k, v in REAL_FILES.items()}}
        self.approvals = json.loads((root / "approvals/phase3n/reconciled-author-approvals.json").read_text(encoding="utf-8"))
        self.prior = json.loads((root / "approvals/phase3l-recorded/author-approvals.json").read_text(encoding="utf-8"))
        self.preimage = (root / "register/sd19-hash-input.json").read_bytes()
        self.ev["exposure_acceptance"] = self.approvals["developmental_pilot_exposure_acceptance"]
        self.ev["cpu_amendment_approval"] = self.approvals["sequential_cpu_amendment"]
        self.state = {"scientific_decisions": self.approvals["scientific_decisions"]}

    def test_exact_approved_bytes_are_eligible_for_the_developmental_pilot_only(self):
        result = X.developmental_eligibility(self.ev, readiness_state=self.state)
        self.assertEqual(result["problems"], [])
        self.assertEqual(len(result["task_ids"]), 12)
        self.assertFalse(result["acknowledgement"]["confirmatory_eligibility"])
        self.assertTrue(any("never authorises confirmatory" in p for p in X.exposure_eligibility_problems(
            {"schema": M.SCHEMA, "stage": "confirmatory", "protocol_version": PROTOCOL_VERSION, "tasks": []}, self.ev,
            readiness_state=self.state, artifacts={})))

    def test_sd19_hash_inputs_and_the_other_21_bindings_are_preserved(self):
        self.assertEqual(sha(self.preimage), X.APPROVED_DEVELOPMENTAL["sd19_decision_sha256"])
        pre = json.loads(self.preimage)
        self.assertEqual(pre["entry"], json.loads(json.dumps(SCIENTIFIC_DECISIONS[X.SD19_KEY], default=str)))   # register unchanged
        self.assertEqual(pre["protocol_version"], PROTOCOL_VERSION)
        self.assertEqual(readiness.decision_sha256(pre["decision"], pre["choice"]), X.APPROVED_DEVELOPMENTAL["sd19_decision_sha256"])
        new, old = self.approvals["scientific_decisions"], self.prior["scientific_decisions"]
        self.assertEqual(len(new), 22)
        changed = sorted(k for k in set(new) | set(old) if new.get(k) != old.get(k))
        self.assertEqual(changed, [X.SD19_KEY])
        for name, entry in new.items():
            self.assertTrue(readiness.valid_approval(entry, readiness.decision_sha256(name, entry["choice"])), name)


if __name__ == "__main__":
    unittest.main()
