from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rq1.doctor.report import machine_manifest
from rq1.orchestration.locks import StageLock
from rq1.orchestration.reports import completed_report
from rq1.orchestration.stages import STAGE_MAP
from rq1.orchestration.state_registry import StageRegistry, StageTransitionError
from rq1.runners.mock import run_mock_workflow
from rq1.utils.config import validate_config_tree
from rq1.utils.ids import new_attempt_id
from rq1.utils.paths import repository_root
from rq1.utils.time import utc_now


def _root() -> Path:
    return repository_root()


def _registry(root: Path) -> StageRegistry:
    return StageRegistry(root / "state" / "stage_status.json")


def command_preflight(root: Path) -> int:
    manifest = machine_manifest(root)
    path = root / "artifacts" / "manifests" / "machine_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "recorded", "manifest": str(path.relative_to(root))}))
    return 0


def command_doctor(root: Path) -> int:
    print(json.dumps(machine_manifest(root), indent=2, sort_keys=True))
    return 0


def command_status(root: Path) -> int:
    states = _registry(root).status()
    print(json.dumps({name: state.__dict__ for name, state in states.items()}, indent=2, sort_keys=True))
    return 0


def command_validate_config(root: Path) -> int:
    errors = validate_config_tree(root / "configs")
    if errors:
        print(json.dumps({"valid": False, "errors": errors}, indent=2), file=sys.stderr)
        return 1
    print(json.dumps({"valid": True, "config_root": "configs"}))
    return 0


def _run_stage(root: Path, stage: str, dry_run: bool) -> int:
    registry = _registry(root)
    attempt = new_attempt_id()
    started = utc_now()
    lock_path = root / "state" / "locks" / f"{stage}.lock"
    with StageLock(lock_path):
        registry.mark_running(stage, attempt)
        report_path = root / "artifacts" / "stage_reports" / f"{stage}-{attempt}.json"
        try:
            if stage == "preflight":
                command_preflight(root)
                warnings: list[str] = []
            elif STAGE_MAP[stage].external and not dry_run:
                raise RuntimeError(
                    f"{stage} requires unverified external integrations and is not implemented in this foundation. "
                    "Use --dry-run for orchestration validation or mock-run for local workflow tests."
                )
            else:
                warnings = ["Placeholder stage: no external integration was executed."] if stage in STAGE_MAP else []
            next_name = STAGE_MAP.get(stage)
            report = completed_report(stage, attempt, started, dry_run=dry_run, outputs=[], warnings=warnings, next_command=None, metadata={"external": next_name.external if next_name else False})
            report.write(report_path)
            next_stage = registry.finish(stage, "passed", str(report_path.relative_to(root)))
            print(json.dumps({"stage": stage, "status": "passed", "next_stage": next_stage}))
            return 0
        except Exception as exc:
            report = completed_report(stage, attempt, started, dry_run=dry_run, outputs=[], warnings=[], error=str(exc), next_command=None)
            report.status = "failed"
            report.write(report_path)
            registry.finish(stage, "failed", str(report_path.relative_to(root)))
            print(json.dumps({"stage": stage, "status": "failed", "error": str(exc)}), file=sys.stderr)
            return 1


def command_run_until(root: Path, target: str, dry_run: bool) -> int:
    if target not in STAGE_MAP:
        print(f"Unknown stage: {target}", file=sys.stderr)
        return 2
    for stage in STAGE_MAP:
        states = _registry(root).status()
        if states[stage].status == "passed":
            if stage == target:
                return 0
            continue
        result = _run_stage(root, stage, dry_run)
        if result != 0 or stage == target:
            return result
    return 0


def command_mock(root: Path) -> int:
    result = run_mock_workflow(root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def command_bridge_server(root: Path, host: str, port: int) -> int:
    """Run the Phase 2 fake bridge explicitly; this never selects real ALFWorld."""
    from rq1.bridge.app import create_bridge_server

    server = create_bridge_server(root / "runs" / "pilot" / "bridge", host=host, port=port)
    print(json.dumps({"status": "serving", "mode": "fake", "host": host, "port": server.server_port}))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        server.server_close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RQ1 experiment foundation CLI")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("preflight")
    sub.add_parser("doctor")
    sub.add_parser("stage-status")
    sub.add_parser("validate-config")
    sub.add_parser("mock-run")
    bridge = sub.add_parser("bridge-server", help="Run the fake ALFWorld bridge on localhost.")
    bridge.add_argument("--host", default="127.0.0.1")
    bridge.add_argument("--port", type=int, default=8000)
    stage = sub.add_parser("stage")
    stage.add_argument("name", choices=tuple(STAGE_MAP))
    stage.add_argument("--dry-run", action="store_true")
    run = sub.add_parser("run-until")
    run.add_argument("stage", choices=tuple(STAGE_MAP))
    run.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = _root()
    try:
        if args.command == "preflight": return command_preflight(root)
        if args.command == "doctor": return command_doctor(root)
        if args.command == "stage-status": return command_status(root)
        if args.command == "validate-config": return command_validate_config(root)
        if args.command == "mock-run": return command_mock(root)
        if args.command == "bridge-server": return command_bridge_server(root, args.host, args.port)
        if args.command == "stage": return _run_stage(root, args.name, args.dry_run)
        if args.command == "run-until": return command_run_until(root, args.stage, args.dry_run)
    except (RuntimeError, StageTransitionError, FileNotFoundError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
