from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from rq1.doctor.report import machine_manifest
from rq1.orchestration.locks import StageLock
from rq1.orchestration.reports import completed_report
from rq1.orchestration.stages import STAGE_MAP
from rq1.orchestration.state_registry import StageRegistry, StageTransitionError
from rq1.pilot.models import EvidenceLevel
from rq1.runners.mock import run_mock_workflow
from rq1.setup.models import SETUP_STAGE_MAP, SetupOptions
from rq1.setup.orchestrator import SetupError, SetupOrchestrator
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


def command_hermes_capabilities(root: Path) -> int:
    from rq1.hermes.capabilities import probe_hermes_capabilities

    report = probe_hermes_capabilities(project_root=root)
    path = root / "artifacts" / "manifests" / "hermes_integration_capabilities.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"capabilities": report.to_dict(), "report": str(path.relative_to(root))}, indent=2, sort_keys=True))
    return 0


def command_verify_hermes_integration(root: Path, mode: str) -> int:
    from rq1.hermes.verification import verify_fake_hermes_integration, verify_real_hermes_integration

    report = verify_fake_hermes_integration(root) if mode == "fake" else verify_real_hermes_integration(root)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("mock_integration") or report.get("real_plugin_loading") else 1


def _profile_manifest_from_path(path: Path):
    from rq1.profiles.models import ProfileManifest

    return ProfileManifest(**json.loads(path.read_text(encoding="utf-8")))


def command_profiles(root: Path, args: argparse.Namespace) -> int:
    from rq1.hermes.capabilities import probe_hermes_capabilities
    from rq1.profiles.lifecycle import (
        ProfileLifecycleError,
        base_profile_plans,
        profile_plan,
        real_profile_lifecycle,
        recovery_profile_template,
        verify_fake_profile_lifecycle,
        write_phase4_report,
    )

    if args.profile_command == "capabilities":
        report = probe_hermes_capabilities(project_root=root)
        path = root / "artifacts" / "manifests" / "hermes_profile_capabilities.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"capabilities": report.to_dict(), "report": str(path.relative_to(root))}, indent=2, sort_keys=True))
        return 0 if report.installed else 1
    if args.profile_command == "plan":
        plans = [plan.to_dict() for plan in (*base_profile_plans(root), recovery_profile_template(root))]
        print(json.dumps({"plans": plans, "dry_run": True}, indent=2, sort_keys=True))
        return 0
    if args.profile_command == "isolation-test":
        report = verify_fake_profile_lifecycle(root)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["mock_profile_lifecycle_passed"] and report["mock_isolation_tested"] else 1
    if args.profile_command == "create-base" and args.dry_run:
        print(json.dumps({"dry_run": True, "plans": [plan.to_dict() for plan in base_profile_plans(root)]}, indent=2, sort_keys=True))
        return 0
    if args.profile_command == "create-base" and not args.yes:
        raise ProfileLifecycleError("Real profile creation requires --yes; use `rq1 profiles plan` for a non-mutating preview.")
    lifecycle = real_profile_lifecycle(root)
    if args.profile_command == "create-base":
        manifests = []
        for plan in base_profile_plans(root):
            manifest = lifecycle.create(plan)
            lifecycle.write_manifest(manifest)
            manifests.append(manifest.to_dict())
        report = {
            "schema_version": 1,
            "generated_at": utc_now(),
            "mock_profile_lifecycle_passed": False,
            "mock_isolation_tested": False,
            "contamination_checks_passed_mock": False,
            "hermes_detected": True,
            "profile_capability_detected": True,
            "no_skills_capability_detected": True,
            "pilot_profile_actually_created": True,
            "acquisition_profile_actually_created": True,
            "real_profile_isolation_tested": False,
            "contamination_checks_passed_real": all(item["validation_result"]["valid"] for item in manifests),
            "future_recovery_profile_template_generated": True,
            "phase6_blocked": True,
            "real_compatibility": False,
        }
        report_path = write_phase4_report(root, report)
        print(json.dumps({"created": manifests, "report": str(report_path.relative_to(root))}, indent=2, sort_keys=True))
        return 0
    name = args.profile_name
    plan = profile_plan(name, root)
    if args.profile_command == "inspect":
        print(json.dumps(lifecycle.inspect(name).to_dict(), indent=2, sort_keys=True))
        return 0
    manifest_path = root / "artifacts" / "manifests" / "profiles" / f"{name}.json"
    baseline = _profile_manifest_from_path(manifest_path) if manifest_path.is_file() else None
    if args.profile_command in {"validate", "contamination-check", "manifest"}:
        manifest = lifecycle.validate(plan, baseline=baseline)
        if args.profile_command == "manifest":
            path = lifecycle.write_manifest(manifest)
            print(json.dumps({"manifest": str(path.relative_to(root)), "value": manifest.to_dict()}, indent=2, sort_keys=True))
        else:
            print(json.dumps(manifest.to_dict(), indent=2, sort_keys=True))
        return 0 if manifest.validation_result["valid"] else 1
    if args.profile_command == "archive-manifest":
        manifest = lifecycle.validate(plan, baseline=baseline)
        path = lifecycle.archive_manifest(manifest)
        print(json.dumps({"archive": str(path.relative_to(root)), "value": manifest.to_dict()}, indent=2, sort_keys=True))
        return 0 if manifest.validation_result["valid"] else 1
    if args.profile_command == "cleanup-test-profile":
        if args.dry_run:
            print(json.dumps({"dry_run": True, "would_clean": name}))
            return 0
        if not args.confirm_destructive:
            raise ProfileLifecycleError("Cleanup requires --confirm-destructive and is limited to rq1-test-* profiles.")
        lifecycle.backend.delete(name)
        print(json.dumps({"cleaned": name, "real_operation": True}))
        return 0
    raise ProfileLifecycleError("Unknown profile command")


def command_recovery(root: Path, args: argparse.Namespace) -> int:
    from rq1.recovery.fake import FakeRecoveryEnvironment
    from rq1.recovery.models import CheckpointPolicy, to_dict
    from rq1.recovery.checkpoints import create_manifest
    from rq1.recovery.validation import validate_checkpoint_payload, validate_perturbation_payload
    from rq1.recovery.verification import real_recovery_capabilities, verify_fake_recovery

    if args.recovery_command == "plan":
        print(json.dumps({"policies": ["prefix_length", "action_index", "trajectory_fraction", "frozen_prefix"], "real_status": "TO_BE_VERIFIED_BY_RECOVERY_PILOT"}, indent=2))
        return 0
    if args.recovery_command == "capabilities":
        print(json.dumps(real_recovery_capabilities(), indent=2)); return 1
    if args.recovery_command == "verify":
        if args.mode != "fake":
            print(json.dumps({"ok": False, "error": "real recovery verification requires observed installed ALFWorld capabilities"}), file=sys.stderr); return 1
        report = verify_fake_recovery(root); print(json.dumps(report, indent=2, sort_keys=True)); return 0 if report["mock_recovery"] else 1
    try:
        payload = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(json.dumps({"valid": False, "errors": [str(exc)]})); return 1
    errors = validate_checkpoint_payload(payload) if args.recovery_command == "validate-checkpoint" else validate_perturbation_payload(payload)
    print(json.dumps({"valid": not errors, "errors": errors}, indent=2)); return 0 if not errors else 1


def _pilot_selection(args: argparse.Namespace) -> dict[str, object]:
    return {
        "test_id": getattr(args, "test_id", None),
        "group": getattr(args, "group", None),
        "start": getattr(args, "start", None),
        "end": getattr(args, "end", None),
        "include_prerequisites": bool(getattr(args, "include_prerequisites", False)),
    }


def _authorize_real_pilot(args: argparse.Namespace) -> None:
    if not bool(getattr(args, "yes", False)):
        raise RuntimeError("Real pilot execution requires --yes; use `rq1 pilot plan --mode real` first")
    if os.environ.get("RQ1_RUN_REAL_PILOT_TESTS") != "1":
        raise RuntimeError("Real pilot execution requires RQ1_RUN_REAL_PILOT_TESTS=1")


def command_pilot(root: Path, args: argparse.Namespace) -> int:
    from rq1.pilot.catalog import PILOT_GROUPS, PILOT_TEST_MAP, select_tests
    from rq1.pilot.models import EvidenceLevel, PilotMode
    from rq1.pilot.registry import PilotRegistry
    from rq1.pilot.report import generate_reports
    from rq1.pilot.runner import PilotRunner, add_manual_evidence, catalog_payload

    command = args.pilot_command
    if command == "list":
        print(json.dumps(catalog_payload(), indent=2, sort_keys=True)); return 0
    if command == "prerequisites":
        spec = PILOT_TEST_MAP.get(args.test_id)
        if spec is None: raise RuntimeError(f"Unknown pilot test: {args.test_id}")
        print(json.dumps({"test": spec.to_dict(), "prerequisites": list(spec.prerequisites)}, indent=2, sort_keys=True)); return 0
    if command == "plan":
        selected = select_tests(**_pilot_selection(args))
        print(json.dumps({"mode": args.mode, "dry_run": True, "groups": PILOT_GROUPS, "tests": [item.to_dict() for item in selected], "mutations": []}, indent=2, sort_keys=True)); return 0
    registry = PilotRegistry(root)
    if command == "run":
        if args.dry_run:
            selected = select_tests(**_pilot_selection(args))
            print(json.dumps({"mode": args.mode, "dry_run": True, "tests": [item.test_id for item in selected], "mutations": []}, indent=2)); return 0
        mode = PilotMode(args.mode)
        if mode == PilotMode.REAL: _authorize_real_pilot(args)
        result = PilotRunner(root).create_and_run(mode, candidate_model=args.candidate_model, **_pilot_selection(args))
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["mock_orchestration_ready"] or result["experimental_ready"] else 1
    if command in {"resume", "retry-failed"}:
        run_id = args.run_id
        state = registry.load(run_id)
        if state["mode"] == "real": _authorize_real_pilot(args)
        result = PilotRunner(root).resume(run_id, retry_failed=command == "retry-failed")
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["mock_orchestration_ready"] or result["experimental_ready"] else 1
    run_id = getattr(args, "run_id", None) or registry.latest()
    if not run_id: raise RuntimeError("No pilot run is available")
    if command == "status":
        print(json.dumps(registry.load(run_id), indent=2, sort_keys=True)); return 0
    if command == "report":
        report = generate_reports(root, registry.load(run_id)); print(json.dumps(report, indent=2, sort_keys=True)); return 0
    if command == "evidence":
        evidence = add_manual_evidence(root, run_id, args.test_id, Path(args.path), EvidenceLevel(args.level))
        print(json.dumps(evidence, indent=2, sort_keys=True)); return 0
    raise RuntimeError(f"Unsupported pilot command: {command}")


def _setup_options(args: argparse.Namespace) -> SetupOptions:
    return SetupOptions(
        dry_run=bool(getattr(args, "dry_run", False)),
        yes=bool(getattr(args, "yes", False)),
        resume=bool(getattr(args, "resume", False)),
        skip_system_packages=bool(getattr(args, "skip_system_packages", False)),
        skip_model=bool(getattr(args, "skip_model", False)),
        skip_alfworld_data=bool(getattr(args, "skip_alfworld_data", False)),
        install_fallback_model=bool(getattr(args, "install_fallback_model", False)),
        force_stage=getattr(args, "force_stage", None),
        verbose=bool(getattr(args, "verbose", False)),
    )


def _add_setup_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-system-packages", action="store_true")
    parser.add_argument("--skip-model", action="store_true")
    parser.add_argument("--skip-alfworld-data", action="store_true")
    parser.add_argument("--install-fallback-model", action="store_true")
    parser.add_argument("--force-stage", choices=tuple(SETUP_STAGE_MAP))
    parser.add_argument("--verbose", action="store_true")


def _setup_argv(options: SetupOptions) -> list[str]:
    values = ["setup-machine", "--resume"]
    for enabled, flag in (
        (options.yes, "--yes"),
        (options.skip_system_packages, "--skip-system-packages"),
        (options.skip_model, "--skip-model"),
        (options.skip_alfworld_data, "--skip-alfworld-data"),
        (options.install_fallback_model, "--install-fallback-model"),
        (options.verbose, "--verbose"),
    ):
        if enabled:
            values.append(flag)
    if options.force_stage:
        values.extend(("--force-stage", options.force_stage))
    return values


def _project_venv_python(root: Path) -> Path:
    return root / ".venv" / "bin" / "python"


def command_setup_machine(root: Path, options: SetupOptions) -> int:
    if not options.dry_run and not options.yes:
        raise SetupError("Real setup requires --yes; use --dry-run to inspect the plan without external changes")
    venv_python = _project_venv_python(root)
    inside_venv = venv_python.exists() and Path(sys.executable).resolve() == venv_python.resolve()
    if not options.dry_run and not inside_venv and venv_python.exists():
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(root / "src")
        os.execve(str(venv_python), [str(venv_python), "-m", "rq1.cli", *_setup_argv(options)], environment)
    orchestrator = SetupOrchestrator(root, options)
    if not options.dry_run and not inside_venv:
        partial = orchestrator.run(stop_after="python-environment")
        if partial["status"] in {"failed", "blocked"} or not venv_python.exists():
            print(json.dumps(partial, indent=2, sort_keys=True))
            return 1
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(root / "src")
        os.execve(str(venv_python), [str(venv_python), "-m", "rq1.cli", *_setup_argv(options)], environment)
    result = orchestrator.run()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["installation_ready"] or options.dry_run else 1


def command_setup_stage(root: Path, stage: str, options: SetupOptions) -> int:
    if not options.dry_run and not options.yes:
        raise SetupError("Running a setup stage requires --yes; use --dry-run for a non-installing preview")
    result = SetupOrchestrator(root, options).run(only_stage=stage)
    print(json.dumps(result, indent=2, sort_keys=True))
    status = result["attempts"][-1]["status"] if result["attempts"] else "passed"
    return 0 if status in {"passed", "skipped"} else 1


def command_setup_status(root: Path) -> int:
    result = SetupOrchestrator(root, SetupOptions()).status()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


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
    sub.add_parser("hermes-capabilities", help="Probe Hermes read-only capability evidence without modifying Hermes.")
    hermes_verify = sub.add_parser("verify-hermes-integration", help="Verify the project Hermes boundary in fake or explicitly opted-in real mode.")
    hermes_verify.add_argument("--mode", choices=("fake", "real"), required=True)
    profiles = sub.add_parser("profiles", help="Plan and validate isolated Hermes research profiles.")
    profile_sub = profiles.add_subparsers(dest="profile_command", required=True)
    profile_sub.add_parser("capabilities")
    profile_sub.add_parser("plan")
    create_base = profile_sub.add_parser("create-base")
    create_base.add_argument("--yes", action="store_true")
    create_base.add_argument("--dry-run", action="store_true")
    profile_sub.add_parser("isolation-test")
    for name in ("inspect", "validate", "contamination-check", "manifest", "archive-manifest", "cleanup-test-profile"):
        command = profile_sub.add_parser(name)
        command.add_argument("profile_name")
        command.add_argument("--dry-run", action="store_true")
        if name == "cleanup-test-profile":
            command.add_argument("--confirm-destructive", action="store_true")
    recovery = sub.add_parser("recovery", help="Controlled-recovery contract and fake verification.")
    recovery_sub = recovery.add_subparsers(dest="recovery_command", required=True)
    recovery_sub.add_parser("plan"); recovery_sub.add_parser("capabilities")
    recovery_verify = recovery_sub.add_parser("verify"); recovery_verify.add_argument("--mode", choices=("fake", "real"), required=True)
    for name in ("validate-checkpoint", "validate-perturbation"):
        command = recovery_sub.add_parser(name); command.add_argument("manifest")
    pilot = sub.add_parser("pilot", help="Run the typed recovery-aware Phase 6 pilot.")
    pilot_sub = pilot.add_subparsers(dest="pilot_command", required=True)
    pilot_sub.add_parser("list")
    pilot_prerequisites = pilot_sub.add_parser("prerequisites")
    pilot_prerequisites.add_argument("--test", dest="test_id", required=True)
    def add_pilot_selection(command: argparse.ArgumentParser) -> None:
        selection = command.add_mutually_exclusive_group()
        selection.add_argument("--test", dest="test_id")
        selection.add_argument("--group")
        selection.add_argument("--from", dest="start")
        command.add_argument("--to", dest="end")
        command.add_argument("--include-prerequisites", action="store_true")
    pilot_plan = pilot_sub.add_parser("plan")
    pilot_plan.add_argument("--mode", choices=("fake", "real"), required=True)
    add_pilot_selection(pilot_plan)
    pilot_run = pilot_sub.add_parser("run")
    pilot_run.add_argument("--mode", choices=("fake", "real"), required=True)
    pilot_run.add_argument("--candidate-model", choices=("hermes3:8b", "llama3.1:8b"), default="hermes3:8b")
    pilot_run.add_argument("--dry-run", action="store_true")
    pilot_run.add_argument("--yes", action="store_true")
    pilot_run.add_argument("--confirm-destructive", action="store_true")
    add_pilot_selection(pilot_run)
    for name in ("resume", "retry-failed"):
        command = pilot_sub.add_parser(name)
        command.add_argument("--run-id", required=True)
        command.add_argument("--yes", action="store_true")
        command.add_argument("--confirm-destructive", action="store_true")
    for name in ("status", "report"):
        command = pilot_sub.add_parser(name); command.add_argument("--run-id")
    evidence = pilot_sub.add_parser("evidence")
    evidence.add_argument("action", choices=("add",))
    evidence.add_argument("--run-id", required=True); evidence.add_argument("--test", dest="test_id", required=True)
    evidence.add_argument("--path", required=True); evidence.add_argument("--level", choices=tuple(item.value for item in EvidenceLevel), required=True)
    setup = sub.add_parser("setup-machine", help="Run the resumable Ubuntu machine setup.")
    _add_setup_options(setup)
    setup_stage = sub.add_parser("setup-stage", help="Run one machine-setup stage.")
    setup_stage.add_argument("name", choices=tuple(SETUP_STAGE_MAP))
    _add_setup_options(setup_stage)
    verify = sub.add_parser("verify-installation", help="Run the final installation verification stage.")
    _add_setup_options(verify)
    sub.add_parser("setup-status", help="Show machine-setup stage state.")
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
        if args.command == "hermes-capabilities": return command_hermes_capabilities(root)
        if args.command == "verify-hermes-integration": return command_verify_hermes_integration(root, args.mode)
        if args.command == "profiles": return command_profiles(root, args)
        if args.command == "recovery": return command_recovery(root, args)
        if args.command == "pilot": return command_pilot(root, args)
        if args.command == "setup-machine": return command_setup_machine(root, _setup_options(args))
        if args.command == "setup-stage": return command_setup_stage(root, args.name, _setup_options(args))
        if args.command == "verify-installation": return command_setup_stage(root, "installation-verification", _setup_options(args))
        if args.command == "setup-status": return command_setup_status(root)
        if args.command == "stage": return _run_stage(root, args.name, args.dry_run)
        if args.command == "run-until": return command_run_until(root, args.stage, args.dry_run)
    except (RuntimeError, SetupError, StageTransitionError, FileNotFoundError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
