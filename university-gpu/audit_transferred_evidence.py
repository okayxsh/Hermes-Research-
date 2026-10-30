"""Read university evidence without extraction; emit unapproved candidate records."""
import argparse
import hashlib
import json
import tarfile
from pathlib import Path

from rq1.pilot.real_runtime.harnesses import bridge_state_digest
from rq1.tasks.validation import manifest_hash

ARCHIVE_SHA = "11c8ca453ef4ceaef5dce189c976ffc09636fdc218e40e22a0803deac9e429e2"
BATCH = "rq1-checkpoint-tests/758210385a12406194886aa9df20e2a1"
FIRST = "rq1-checkpoint-tests/205f7af291c44b5b8fc4bf3652b68ab2/4"
PROPOSAL = "artifacts/task_manifests/proposals/pilot-39cce59288e59c8d.json"


def audit_episode(rows, prefix, expected_task):
    tools = [r["payload"] for r in rows if r.get("event") == "tool_result"]
    start = tools[0]
    steps = [r for r in tools if r["tool"] == "alfworld_step"]
    actions = [r["request"]["action"] for r in steps]
    if actions[:prefix] == [] or prefix >= len(steps):
        raise ValueError("missing checkpoint or detour")
    cp, post, final = steps[prefix-1]["response"], steps[prefix]["response"], steps[-1]["response"]
    return {"actions": actions, "checkpoint": cp, "post": post,
            "detour": actions[prefix], "start": start["response"],
            "task_matches": all(r["response"].get("task_id") == expected_task for r in tools),
            "all_actions_valid": all(r["response"].get("action_valid") is True for r in steps),
            "oracle_success": final.get("done") is True and final.get("success") is True,
            "actual_total_actions": final["action_count"], "action_limit": start["request"]["action_limit"],
            "checkpoint_digest": bridge_state_digest(cp), "post_detour_digest": bridge_state_digest(post)}


def run(archive, output):
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != ARCHIVE_SHA:
        raise ValueError("archive checksum mismatch")
    if output.exists():
        raise FileExistsError(output)
    with tarfile.open(archive) as tar:
        names = [m.name for m in tar.getmembers() if m.isfile()]
        if len(names) != len(set(names)):
            raise ValueError("duplicate archive member names")
        def read(name):
            member = tar.getmember(name)
            if not member.isfile() or member.size > 10_000_000:
                raise ValueError("unsafe evidence member")
            return tar.extractfile(member).read()
        proposal_bytes = read(PROPOSAL)
        proposal = json.loads(proposal_bytes)
        proposal_hash_verified = manifest_hash(proposal) == proposal["manifest_sha256"]
        if not proposal_hash_verified:
            raise ValueError("proposal manifest content hash mismatch")
        batch = json.loads(read(BATCH + "/checkpoint-results.json"))
        candidates, audits = [], []
        for task in proposal["tasks"]:
            family, task_id = task["family"], task["task_id"]
            record = batch.get(family)
            prefix = len(record["prefix_actions"]) if record else 4
            directory = BATCH + f"/{family}/{prefix}" if record else FIRST
            oracle_path = directory + "/solvability-oracle/episode-events.jsonl"
            raw = read(oracle_path)
            e = audit_episode([json.loads(x) for x in raw.splitlines()], prefix, task_id)
            route = e["actions"][:prefix] + e["actions"][prefix+1:]
            main_path = directory + "/episode-events.jsonl"
            main_raw = read(main_path)
            main_tools = [json.loads(x)["payload"] for x in main_raw.splitlines() if json.loads(x).get("event") == "tool_result"]
            main_steps = [r for r in main_tools if r["tool"] == "alfworld_step"]
            cross_state_match = (bridge_state_digest(main_steps[prefix-1]["response"]) == e["checkpoint_digest"]
                and bridge_state_digest(main_steps[prefix]["response"]) == e["post_detour_digest"])
            recorded = record is None or (record["route_actions"] == route and record["prefix_actions"] == route[:prefix]
                and record["checkpoint_digest"] == e["checkpoint_digest"]
                and record["post_detour_digest"] == e["post_detour_digest"] and record["detour"] == e["detour"])
            valid = (cross_state_match and recorded and e["task_matches"] and e["all_actions_valid"] and e["oracle_success"]
                and e["start"]["split"] == "valid_seen" and e["actual_total_actions"] == len(route)+1)
            bounded_trace = valid and e["actual_total_actions"] <= 50
            status = "BLOCKED" if not bounded_trace else "REQUIRES HUMAN REVIEW"
            item = {"task_id": task_id, "task_family": family, "split": "valid_seen", "source_task": task,
                "reference_actions": route, "reference_route_length": len(route), "prefix_actions": route[:prefix],
                "checkpoint_id": directory, "checkpoint_digest": e["checkpoint_digest"],
                "checkpoint_observation": e["checkpoint"]["observation"],
                "detour_action": e["detour"], "post_detour_digest": e["post_detour_digest"],
                "post_detour_observation": e["post"]["observation"],
                "expected_next_action": route[prefix], "total_action_budget": 50, "recovery_action_budget": 49-prefix,
                "oracle": {"validated": False, "historical_large_budget_success": e["oracle_success"],
                    "recorded_action_limit": e["action_limit"], "actual_total_actions": e["actual_total_actions"],
                    "trace_within_50": bounded_trace, "cap_50_replay_performed": False},
                "source_evidence": {"archive_sha256": digest, "member": oracle_path,
                    "member_sha256": hashlib.sha256(raw).hexdigest()},
                "budget_feasibility_status": status, "scientifically_frozen": False}
            candidates.append(item)
            audits.append({"task_id":task_id,"family":family,"route_length":len(route),"prefix_length":prefix,
                "continuation":len(route)-prefix,"recovery_budget":49-prefix,"detour":e["detour"],
                "actual_oracle_actions":e["actual_total_actions"],"oracle_action_limit":e["action_limit"],
                "oracle_success":e["oracle_success"],"summary_and_log_match":recorded,"main_oracle_state_match":cross_state_match,"state_digests_verified":valid,
                "recorded_data_identity":e["start"].get("field_sources",{}).get("data_identity"),
                "proposal_data_identity":proposal["data_root_identity"],"verdict":status,"evidence":oracle_path})
        output.mkdir(parents=True)
        (output/"original-proposal.json").write_bytes(proposal_bytes)
        (output/"candidate-failures.json").write_text(json.dumps({"schema_version":1,"status":"candidate_unapproved",
            "tasks":candidates,"model_calls":0},indent=2)+"\n")
        report={"archive_sha256":digest,"proposal_hash_verified":proposal_hash_verified,"status":"BLOCKED",
            "rows":audits,"limitations":["Four successful traces fit 50 but all were executed with a 200-action cap; cap-50 replay pending",
            "Task source hashes are recorded in proposal; original ALFWorld data absent from archive, not independently rehashed",
            "Observed adapter data identity differs from proposal discovery identity; semantics require reconciliation",
            "No bounded alternate-route search was performed; long expert route is not proof of task impossibility",
            "Pick-two route length in actual logs may contradict earlier narrative; preserve both",
            "Candidate records intentionally retain oracle.validated=false and cannot enter launch"]}
        (output/"eligibility-audit.json").write_text(json.dumps(report,indent=2)+"\n")
        return report

if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(run(args.archive,args.output),indent=2))
