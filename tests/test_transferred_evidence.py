import importlib.util
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location("audit_transfer", Path(__file__).resolve().parents[1] / "university-gpu/audit_transferred_evidence.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)

class TransferTests(unittest.TestCase):
    def test_checksum_mismatch_creates_no_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); source=root/"bad.tar.gz";source.write_bytes(b"not evidence")
            with self.assertRaises(ValueError): audit.run(source,root/"output")
            self.assertFalse((root/"output").exists())

    def test_replayed_states_and_actual_count_are_preserved(self):
        task="valid_seen:fixture"
        def state(count,done=False):
            return {"task_id":task,"observation":str(count),"admissible_actions":[],"step_number":count,
                    "action_count":count,"action_valid":True,"done":done,"success":done}
        payloads=[{"tool":"alfworld_start","request":{"action_limit":200},"response":state(0)}]
        for i in range(1,4):
            payloads.append({"tool":"alfworld_step","request":{"action":"go to room"},"response":state(i,i==3)})
        rows=[{"event":"tool_result","payload":p} for p in payloads]
        result=audit.audit_episode(rows,1,task)
        self.assertEqual(200,result["action_limit"])
        self.assertEqual(3,result["actual_total_actions"])
        self.assertTrue(result["oracle_success"])
        self.assertEqual(audit.bridge_state_digest(state(1)),result["checkpoint_digest"])
        self.assertFalse(audit.audit_episode(rows,1,"valid_seen:other")["task_matches"])
