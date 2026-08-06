from __future__ import annotations
from pathlib import Path
from rq1.profiles.lifecycle import profile_plan, real_profile_lifecycle
def recovery_profile_plan(root: Path, snapshot_id: str, snapshot_hash: str):
    return profile_plan("rq1-recovery-" + snapshot_id, root, snapshot_hash=snapshot_hash)
def create_profiles(*_args, **_kwargs):
    raise RuntimeError("real recovery profile materialization requires an observed read-only Hermes snapshot mechanism")
