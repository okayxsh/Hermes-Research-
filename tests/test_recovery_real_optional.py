from __future__ import annotations
import os, sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rq1.recovery.verification import real_recovery_capabilities

@unittest.skipUnless(os.environ.get("RQ1_RUN_REAL_RECOVERY_TESTS") == "1", "set RQ1_RUN_REAL_RECOVERY_TESTS=1 after installing and capability-probing ALFWorld")
class OptionalRealRecoveryTests(unittest.TestCase):
    def test_real_recovery_capabilities_are_observed_before_execution(self):
        capabilities = real_recovery_capabilities()
        if not all(capabilities[key] for key in ("real_adapter_available", "reset_replay_supported", "state_observation_supported", "perturbation_supported")):
            self.skipTest("real recovery capabilities have not been observed")
        self.fail("A version-specific real recovery implementation must be added only after observed adapter evidence.")
