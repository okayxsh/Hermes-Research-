"""Parallel-execution entry points: ``python -m rq1.five_arm.parallel_cli <command>`` (DRAFT; never run locally).

assign       CPU only: frozen, balanced lane assignment (all conditions of a task-seed cell on one lane) for a manifest
coordinator  the single authoritative coordinator: gate, start (evidence check + restart reconciliation), serve the
             file transport and expire leases until the run is complete or HALTED
worker       one lane worker process: requires CUDA_VISIBLE_DEVICES == the lane's GPU UUID; builds its own gated
             executor on the lane's provider endpoint; claim -> execute -> submit until no work remains

Model-executing commands are gated exactly like the sequential path (``runner.assert_execution_permitted``).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from rq1.five_arm import parallel, readiness, runner, runtime


def _load(path: str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _emit(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, default=str))


def _gate(args) -> tuple[dict[str, Any], dict[str, Any]]:
    config, approved = _load(args.run_config), _load(args.approved_runtime)
    artifacts = runner.artifact_hashes(config, approved)
    runner.assert_execution_permitted(manifest=config["manifest"], readiness_state=_load(args.state), artifacts=artifacts,
                                      observed_runtime=_load(args.observed_runtime), approved_runtime=approved)
    return config, approved


def _coordinator(config, approved, assignment, run_root: Path) -> parallel.Coordinator:
    from rq1.evaluation.amended_matrix import EvaluationTask
    from rq1.five_arm.executor import InterruptionRecordFactory
    from rq1.five_arm.measurements import validate_record
    manifest = config["manifest"]
    tasks = {}
    for t in manifest["tasks"]:
        spec = dict(config["evaluation_tasks"][str(t["task_id"])])
        spec["prefix_actions"], spec["reference_actions"] = tuple(spec["prefix_actions"]), tuple(spec["reference_actions"])
        tasks[str(t["task_id"])] = EvaluationTask(**spec)
    factory = InterruptionRecordFactory(tasks=tasks, task_meta={str(t["task_id"]): t for t in manifest["tasks"]}, synthetic=False,
                                        runtime_identity_sha256=runtime.config_sha256(approved),
                                        exposure_policy=str(config["exposure_policy"]), model_name=approved["model_tag"])
    return parallel.Coordinator(run_root, manifest=manifest, assignment=assignment, kind_validator=validate_record,
                                interruption_record=factory.interruption_record,
                                max_infrastructure_retries=int(config["max_infrastructure_retries"]),
                                expected_runtime_identity_sha256=runtime.config_sha256(approved),
                                expected_model_digest=approved["model_manifest_digest"], synthetic=False,
                                lease_seconds=float(config.get("lease_seconds", 900)))


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m rq1.five_arm.parallel_cli", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("assign")
    p.add_argument("--manifest", required=True); p.add_argument("--lanes", required=True)
    p.add_argument("--salt", required=True); p.add_argument("--output", required=True)
    for name in ("coordinator", "worker"):
        p = sub.add_parser(name)
        for flag in ("--run-config", "--state", "--observed-runtime", "--approved-runtime", "--assignment", "--run-root"):
            p.add_argument(flag, required=True)
        if name == "worker":
            p.add_argument("--lane", required=True)
    args = ap.parse_args(argv)
    try:
        if args.command == "assign":
            out = Path(args.output)
            if out.exists():
                raise FileExistsError(f"assignment output exists (never overwritten): {out}")
            assignment = parallel.assign_lanes(_load(args.manifest), _load(args.lanes), salt=args.salt)
            out.write_text(json.dumps(assignment, indent=2, sort_keys=True), encoding="utf-8")
            _emit({"status": "written", "output": str(out), "lane_unit_counts": assignment["lane_unit_counts"],
                   "assignment_sha256": assignment["assignment_sha256"]})
            return 0
        config, approved = _gate(args)
        assignment = _load(args.assignment)
        run_root = Path(args.run_root)
        if args.command == "coordinator":
            coordinator = _coordinator(config, approved, assignment, run_root).start()
            try:
                # a halted lane stops only that lane; the loop ends when nothing runnable remains
                while not coordinator.halted() and coordinator.summary()["pending_on_live_lanes"]:
                    parallel.serve_once(coordinator, run_root)
                    coordinator.expire_leases()
                    time.sleep(0.5)
                parallel.serve_once(coordinator, run_root)
                _emit({"summary": coordinator.summary(), "validation": coordinator.validate()})
            finally:
                coordinator.stop()
            return 0 if not coordinator.halted() and not coordinator.summary()["halted_lanes"] else 4
        lane = next(l for l in assignment["lanes"] if str(l["lane_id"]) == args.lane)
        if os.environ.get("CUDA_VISIBLE_DEVICES") != lane["gpu_uuid"]:
            raise readiness.ExecutionBlocked("CUDA_VISIBLE_DEVICES must equal the lane's GPU UUID (one GPU per lane)")
        from rq1.university_gpu.ollama import _request
        provider_pid = int(os.environ.get("RQ1_LANE_OLLAMA_PID", "0"))
        if provider_pid <= 0:
            raise readiness.ExecutionBlocked("RQ1_LANE_OLLAMA_PID must name the pid of this lane's Ollama server (fencing "
                                             "and process-bound GPU placement are bound to it)")
        identity = lambda: {"worker": parallel.process_identity(os.getpid()),
                            "provider": parallel.process_identity(provider_pid),
                            "provider_descendants": parallel.provider_descendant_identities(provider_pid)}
        compute_apps = parallel.nvidia_compute_apps
        built = runner.build_university_executor(config, readiness_state=_load(args.state), observed_runtime=_load(args.observed_runtime),
                                                 approved_runtime=approved, endpoint=lane["endpoint"],
                                                 execution_provenance={"lane_id": args.lane},
                                                 bridge_log_root=run_root / "lanes" / args.lane / "bridge")
        snapshot = lambda: {**parallel.ollama_lane_provenance(lambda e, p: _request(e, p), lane["endpoint"], approved["model_tag"],
                                                              compute_apps=compute_apps(), lane_gpu_uuid=lane["gpu_uuid"],
                                                              provider_pids=parallel.process_tree(provider_pid)),
                            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
        worker = parallel.LaneWorker(parallel.FileCoordinatorClient(run_root), lane_id=args.lane,
                                     worker_id=f"{args.lane}-{os.getpid()}", executor=built["dispatch"], root=run_root,
                                     snapshot=snapshot, identity=identity,
                                     heartbeat_seconds=float(config.get("heartbeat_seconds", 60)))
        try:
            while worker.run_once():
                pass
        except (parallel.LaneSuspended, parallel.LaneHalted) as exc:
            _emit({"lane": args.lane, "status": type(exc).__name__, "reason": str(exc),
                   "action": "stop this lane's worker AND restart its Ollama server; relaunch only after the coordinator "
                             "verifies termination (or after a recorded lane-halt review)"})
            return 3
        finally:
            built["driver"].__exit__(None, None, None)
        _emit({"lane": args.lane, "status": "no further work for this lane"})
        return 0
    except (readiness.ExecutionBlocked, runtime.RuntimeIdentityError, parallel.CoordinatorHalted, ValueError,
            FileExistsError, KeyError) as exc:
        _emit({"status": "blocked", "reason": f"{type(exc).__name__}: {exc}"})
        return 2


if __name__ == "__main__":
    sys.exit(main())
