from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rq1.analysis.pipeline import AnalysisInputError, compute_metrics, validate_inputs, write_analysis
from rq1.cli import build_parser


class ControlledRecoveryAnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name); self.run_id = "eval-real-1"
        (self.root / "artifacts" / "evaluation_reports" / self.run_id).mkdir(parents=True)
        (self.root / "configs").mkdir(); (self.root / "configs" / "relevance_rules.yaml").write_text('{"status":"FROZEN"}')
        self.activation = self.root / "activation.json"
        self.activation.write_text(json.dumps(self._activation()))
        self.attempts=[]
        for snapshot, success, skill_count, relevance in (("L0", False, 0, "irrelevant"), ("L1", True, 2, "relevant")):
            result = self.root / f"{snapshot}.json"
            result.write_text(json.dumps({"recovery_success":success,"post_failure_budget_complete":True,"relevant_skill_available":True,"post_failure_actions":4-skill_count,"post_failure_model_calls":1,"post_failure_tool_calls":2,"invalid_post_failure_actions":1 if snapshot=="L0" else 0,"recovery_latency_ms":1000-skill_count*100,"phase_boundary_valid":True,"reconciled":True,"profile_read_only":True,"skill_writes":False}))
            log = self.root / f"{snapshot}.jsonl"
            log.write_text(json.dumps({"phase":"post_failure","event":"skill_loaded","payload":{"skill_id":snapshot},"relevance":relevance,"timestamp":"t"})+"\n")
            self.attempts.append({"run_id":f"run-{snapshot}","attempt_id":f"attempt-{snapshot}","task_id":"valid_unseen:task","task_family":"heat","checkpoint_digest":"checkpoint","perturbation_digest":"perturbation","recovery_context_digest":"context","repetition":1,"seed":7,"snapshot_id":snapshot,"snapshot_hash":f"hash-{snapshot}","skill_count":skill_count,"cumulative_skill_operations":skill_count,"perturbation_type":"relocation","result_path":str(result),"hermes_log_path":str(log)})
        report={"schema_version":1,"mode":"real","valid":True,"status":"validated","activation_manifest_path":str(self.activation),"relevance_rules_path":str(self.root / "configs" / "relevance_rules.yaml"),"configuration_hashes":{"policy":"x"},"snapshot_hashes":{"L0":"hash-L0","L1":"hash-L1"},"queue_sha256":"queue","reconciliation_valid":True,"expected_pairs":[{"task_id":"valid_unseen:task","checkpoint_digest":"checkpoint","perturbation_digest":"perturbation","recovery_context_digest":"context","repetition":1,"seed":7,"snapshots":["L0","L1"]}],"attempts":self.attempts}
        (self.root / "artifacts" / "evaluation_reports" / self.run_id / "evaluation-report.json").write_text(json.dumps(report))

    def tearDown(self) -> None: self.temp.cleanup()

    def _activation(self) -> dict:
        return {"schema_version":1,"activation_id":"a","status":"active","activated_at":"now","repository_commit":"c","environment_freeze_sha256":"x","protocol_freeze_sha256":"x","pilot_run_id":"p","pilot_report_sha256":"x","model_digest":"m","alfworld_data_sha256":"d","hermes_version":"h","hermes_capability_sha256":"x","evaluation_task_manifest_sha256":"x","acquisition_validation_sha256":"x","snapshot_set_sha256":"x","recovery_profile_validation_hashes":[],"checkpoint_set_sha256":"x","perturbation_set_sha256":"x","recovery_context_sha256":"x","prompt_hashes":{},"relevance_rule_sha256":"x","repetition_count":1,"action_budget":1,"timeout_seconds":1,"queue_policy_version":"v","approval_reference":"a","approval_file_sha256":"x","evidence":[],"content_sha256":"x"}

    @patch("rq1.analysis.pipeline.validate_activation", return_value=[])
    def test_metrics_pairing_noise_and_reproducible_outputs(self, _validate) -> None:
        validation=validate_inputs(self.root,self.run_id)
        self.assertTrue(validation.valid, validation.errors)
        metrics=compute_metrics(validation, 7)
        self.assertEqual(2, metrics["sample_counts"]["episodes"])
        self.assertEqual(1.0, metrics["snapshot_summary"][0]["retrieval_noise_rate"])
        self.assertEqual(1, len(metrics["paired_comparisons"]))
        output=write_analysis(self.root,validation,metrics,7)
        self.assertTrue((output/"analysis_manifest.json").is_file())
        self.assertTrue((output/"relevance_audit_sample.csv").is_file())

    @patch("rq1.analysis.pipeline.validate_activation", return_value=[])
    def test_missing_pair_rejected(self, _validate) -> None:
        path=self.root/"artifacts"/"evaluation_reports"/self.run_id/"evaluation-report.json"; report=json.loads(path.read_text()); report["attempts"].pop(); path.write_text(json.dumps(report))
        self.assertFalse(validate_inputs(self.root,self.run_id).valid)

    @patch("rq1.analysis.pipeline.validate_activation", return_value=[])
    def test_no_retrieval_is_not_noise(self, _validate) -> None:
        log=Path(self.attempts[0]["hermes_log_path"]); log.write_text(json.dumps({"phase":"post_failure","event":"hermes_tool_result"})+"\n")
        metrics=compute_metrics(validate_inputs(self.root,self.run_id), 7)
        self.assertEqual(1.0, metrics["snapshot_summary"][0]["no_retrieval_rate"])
        self.assertIsNone(metrics["snapshot_summary"][0]["retrieval_noise_rate"])

    def test_cli_accepts_all_offline_analysis_commands(self) -> None:
        parser = build_parser()
        for command in ("validate-inputs", "compute", "audit-sample", "figures", "report", "run-all"):
            args = parser.parse_args(["analysis", command, "--evaluation-run", "run"])
            self.assertEqual(command, args.analysis_command)
