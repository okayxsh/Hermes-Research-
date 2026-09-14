"""Record, read-only, that the production CLI refuses scientific acquisition beyond the 240-unit hard cap.

It evaluates the hard-cap ledger directly and runs only the dry-run `plan` commands; no
run, resume, or retry command is executed.  Writes
artifacts/acquisition-closeout/rq1-acquisition-240-final/acquisition-hard-cap-verification.json
and refuses to overwrite it.
"""
import hashlib
import json
import pathlib
import subprocess
import sys
from datetime import datetime, timezone

ROOT = pathlib.Path("/workspace/persistent/rq1-protocol-migration")
sys.path.insert(0, str(ROOT / "src"))
from rq1.acquisition.extension_protocol import EXTENSION_FROZEN_DIR, EXTENSION_RUN_ID, PARENT  # noqa: E402
from rq1.acquisition.gates import FROZEN_TASK_DIR, load_task_manifest  # noqa: E402
from rq1.acquisition.launch import hard_cap_status, scientific_acquisition_allocation  # noqa: E402

OUT = ROOT / "artifacts" / "acquisition-closeout" / "rq1-acquisition-240-final" / "acquisition-hard-cap-verification.json"
if OUT.exists():
    sys.exit(f"refusing to overwrite {OUT}")

parent_families = [task.family for task in load_task_manifest(sorted((ROOT / FROZEN_TASK_DIR).glob("acquisition-*.json"))[0]).tasks]
extension_families = [task.family for task in load_task_manifest(sorted((ROOT / EXTENSION_FROZEN_DIR).glob("acquisition-extension-*.json"))[0]).tasks]
scenarios = {
    "resume_completed_parent_run": (hard_cap_status(ROOT, PARENT.run_id, parent_families), True),
    "resume_completed_extension_run": (hard_cap_status(ROOT, EXTENSION_RUN_ID, extension_families), True),
    "new_run_of_extension_queue": (hard_cap_status(ROOT, "rq1-acquisition-gemma4-12b-ext-second", extension_families), False),
    "new_run_of_initial_queue": (hard_cap_status(ROOT, "rq1-acquisition-new-run", parent_families), False),
    "single_episode_241": (hard_cap_status(ROOT, "rq1-acquisition-episode-241", ["cool_and_place"]), False),
}


def plan(*arguments: str) -> dict:
    completed = subprocess.run([sys.executable, "-m", "rq1.cli", *arguments], cwd=ROOT, capture_output=True, text=True)
    payload = json.loads(completed.stdout[completed.stdout.index("{"):])
    return {"command": "python -m rq1.cli " + " ".join(arguments), "exit_code": completed.returncode,
            "launch_permitted": payload.get("launch_permitted"), "gate_reasons": payload["gate"]["reasons"], "hard_cap": payload.get("hard_cap")}


plans = {
    "acquisition_plan_new_run": plan("acquisition", "plan"),
    "acquisition_extension_plan_new_run": plan("acquisition-extension", "plan"),
    "acquisition_extension_plan_resume_existing": plan("acquisition-extension", "plan", "--run-id", EXTENSION_RUN_ID),
}
checks = {name: status["permitted"] is expected for name, (status, expected) in scenarios.items()}
checks["acquisition_plan_new_run_refused_by_hard_cap"] = plans["acquisition_plan_new_run"]["launch_permitted"] is False and plans["acquisition_plan_new_run"]["hard_cap"]["permitted"] is False
checks["acquisition_extension_plan_new_run_refused_by_hard_cap"] = (plans["acquisition_extension_plan_new_run"]["launch_permitted"] is False
                                                                    and plans["acquisition_extension_plan_new_run"]["hard_cap"]["permitted"] is False)
checks["acquisition_extension_resume_within_hard_cap"] = plans["acquisition_extension_plan_resume_existing"]["hard_cap"]["permitted"] is True
allocation = scientific_acquisition_allocation(ROOT)
checks["ledger_is_exactly_240"] = sum(value["units"] for value in allocation["runs"].values()) == 240 and not allocation["problems"]
head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
report = {
    "schema_version": 1,
    "kind": "rq1-acquisition-hard-cap-verification",
    "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "repository_head": head,
    "hard_cap": {"total": 240, "per_family": 40},
    "allocation": allocation,
    "scenarios": {name: {"expected_permitted": expected, **status} for name, (status, expected) in scenarios.items()},
    "plan_dry_runs": plans,
    "note": ("No run, resume, or retry command was executed. Independently of the hard cap, the approved acquisition and extension freezes are "
             "bound to commits 8bd452e and 670627a, so every launch gate at a later commit is also blocked (see gate_reasons)."),
    "checks": checks,
    "passed": all(checks.values()),
    "generator": {"script": str(pathlib.Path(__file__).resolve()), "sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()},
}
OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
for name, passed in checks.items():
    print("CHECK", name, passed)
print("HARD_CAP_VERIFICATION_PASSED", report["passed"], OUT, hashlib.sha256(OUT.read_bytes()).hexdigest())
sys.exit(0 if report["passed"] else 2)
