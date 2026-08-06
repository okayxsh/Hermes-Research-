from __future__ import annotations
import tempfile, unittest
from pathlib import Path
from unittest.mock import patch
from rq1.autopilot.executor import Autopilot
from rq1.autopilot.models import RunMode
from rq1.autopilot.packaging import package
from rq1.cli import build_parser

class AutopilotTests(unittest.TestCase):
    def setUp(self): self.temp=tempfile.TemporaryDirectory(); self.root=Path(self.temp.name)
    def tearDown(self): self.temp.cleanup()
    @patch("rq1.autopilot.executor.git_state",return_value=("commit",True,None))
    def test_bootstrap_stops_before_final_and_never_claims_go(self,_git):
        autopilot=Autopilot(self.root); plan=autopilot.plan(RunMode.BOOTSTRAP); autopilot.create(plan); data=autopilot.run(plan.run_plan_id)
        self.assertEqual("BLOCKED",data["top_status"]); self.assertEqual("passed",data["stages"]["preflight"]["status"]); self.assertNotIn("acquisition",data["stages"])
    @patch("rq1.autopilot.executor.git_state",return_value=("commit",True,None))
    def test_final_requires_gates_and_has_single_terminal_status(self,_git):
        autopilot=Autopilot(self.root); plan=autopilot.plan(RunMode.FINAL); autopilot.create(plan); data=autopilot.run(plan.run_plan_id)
        self.assertEqual("BLOCKED",data["top_status"]); self.assertEqual("blocked",data["stages"]["revalidate_gates"]["status"])
    @patch("rq1.autopilot.executor.git_state",return_value=("commit",True,None))
    def test_stop_is_cooperative_and_resume_preserves_state(self,_git):
        autopilot=Autopilot(self.root); plan=autopilot.plan(RunMode.BOOTSTRAP); autopilot.create(plan); autopilot.stop(plan.run_plan_id); data=autopilot.run(plan.run_plan_id)
        self.assertEqual("STOPPED_BY_USER",data["top_status"])
    def test_archive_requires_expected_files_and_validates_zip(self):
        root=self.root/"final"; root.mkdir()
        for name in ("FINAL_SUMMARY.md","runtime_summary.json","metrics.json"): (root/name).write_text("{}")
        archive,report=package(root); self.assertTrue(report["valid"]); self.assertTrue(archive.is_file())
    def test_cli_contract(self):
        args=build_parser().parse_args(["autopilot","plan","--mode","bootstrap"]); self.assertEqual("plan",args.autopilot_command)
