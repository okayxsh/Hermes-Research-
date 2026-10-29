"""CLI surface for the offline-first university GPU execution package."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def add_parser(subparsers: Any) -> None:
    command = subparsers.add_parser("university-gpu", help="Offline-first RQ1 university GPU execution controls.")
    actions = command.add_subparsers(dest="university_gpu_command", required=True)

    item = actions.add_parser("check", help="Read-only source, machine, dependency, and freeze check.")
    item.add_argument("--config", type=Path)
    item.add_argument("--json", action="store_true", help="Print the machine-readable report (the default).")

    item = actions.add_parser("setup", help="Show the project-local setup plan; apply only with approval.")
    item.add_argument("--apply", action="store_true")
    item.add_argument("--approval-file", type=Path)
    item.add_argument("--yes", action="store_true")

    manifest = actions.add_parser("manifest", help="Propose, validate, or freeze an immutable episode manifest.")
    manifest_actions = manifest.add_subparsers(dest="manifest_command", required=True)
    item = manifest_actions.add_parser("propose")
    item.add_argument("--stage", choices=("pilot", "confirmatory"), required=True)
    item.add_argument("--run-id", required=True)
    item.add_argument("--config", type=Path, required=True)
    item.add_argument("--output", type=Path, required=True)
    item = manifest_actions.add_parser("validate")
    item.add_argument("--manifest", type=Path, required=True)
    item.add_argument("--require-frozen", action="store_true")
    item = manifest_actions.add_parser("freeze")
    item.add_argument("--manifest", type=Path, required=True)
    item.add_argument("--approval-file", type=Path, required=True)
    item.add_argument("--output", type=Path, required=True)

    item = actions.add_parser("preflight", help="Validate a frozen manifest and optionally probe four live endpoints.")
    item.add_argument("--config", type=Path, required=True)
    item.add_argument("--manifest", type=Path, required=True)
    item.add_argument("--check-endpoints", action="store_true")

    calibration = actions.add_parser("gate-calibration", help="Propose or freeze an outcome-blind valid_seen gate threshold.")
    calibration_actions = calibration.add_subparsers(dest="calibration_command", required=True)
    item = calibration_actions.add_parser("propose")
    item.add_argument("--scores", type=Path, required=True)
    item.add_argument("--labels", type=Path, required=True)
    item.add_argument("--output", type=Path, required=True)
    item = calibration_actions.add_parser("validate")
    item.add_argument("--calibration", type=Path, required=True)
    item.add_argument("--require-frozen", action="store_true")
    item = calibration_actions.add_parser("freeze")
    item.add_argument("--calibration", type=Path, required=True)
    item.add_argument("--approval-file", type=Path, required=True)
    item.add_argument("--output", type=Path, required=True)

    for name in ("launch", "resume"):
        item = actions.add_parser(name, help="Run/resume episodes; requires explicit GPU and scientific approval.")
        item.add_argument("--config", type=Path, required=True)
        item.add_argument("--manifest", type=Path, required=True)
        item.add_argument("--approval-file", type=Path, required=True)
        item.add_argument("--yes", action="store_true")
        item.add_argument("--retry-failed", action="store_true")
        item.add_argument("--cutoff-epoch", type=float, required=True, help="Actual access deadline in Unix seconds; assignment stops 30 minutes early.")
        item.add_argument("--max-attempts", type=int, help="Stop assigning after this many attempts for a short approved benchmark.")

    item = actions.add_parser("status", help="Read run progress and validate stored event/result records.")
    item.add_argument("--run-root", type=Path, required=True)
    item.add_argument("--manifest", type=Path, required=True)
    item = actions.add_parser("stop", help="Request a safe stop after active episodes finish.")
    item.add_argument("--run-root", type=Path, required=True)
    item.add_argument("--reason", default="operator requested safe stop")
    item = actions.add_parser("validate", help="Validate a complete run without exporting it.")
    item.add_argument("--run-root", type=Path, required=True)
    item.add_argument("--manifest", type=Path, required=True)
    item = actions.add_parser("merge", help="Write the canonical results JSONL after validation.")
    item.add_argument("--run-root", type=Path, required=True)
    item.add_argument("--manifest", type=Path, required=True)
    item = actions.add_parser("export", help="Copy a validated run to a new export directory.")
    item.add_argument("--run-root", type=Path, required=True)
    item.add_argument("--manifest", type=Path, required=True)
    item.add_argument("--destination", type=Path, required=True)
    item = actions.add_parser("export-progress", help="Export a stopped complete or incomplete run with integrity checks.")
    item.add_argument("--run-root", type=Path, required=True)
    item.add_argument("--manifest", type=Path, required=True)
    item.add_argument("--destination", type=Path, required=True)
    item.add_argument("--relevance", type=Path)
    item = actions.add_parser("verify-export", help="Verify an exported run on this machine without inference.")
    item.add_argument("--destination", type=Path, required=True)
    item = actions.add_parser("benchmark-report", help="Summarize the first four completed pilot attempts.")
    item.add_argument("--run-root", type=Path, required=True)
    item.add_argument("--manifest", type=Path, required=True)
    item = actions.add_parser("progress-report", help="Show descriptive counts and missing units for a stopped run.")
    item.add_argument("--run-root", type=Path, required=True)
    item.add_argument("--manifest", type=Path, required=True)
    item = actions.add_parser("analysis-artifacts", help="Create descriptive CSV tables and an SVG from complete-run analysis JSON.")
    item.add_argument("--analysis", type=Path, required=True)
    item.add_argument("--output-dir", type=Path, required=True)
    item = actions.add_parser("analyze", help="Run exploratory paired/task-cluster and rater-sensitive analyses offline.")
    item.add_argument("--results", type=Path, required=True)
    item.add_argument("--manifest", type=Path, required=True)
    item.add_argument("--relevance", type=Path)
    item.add_argument("--output", type=Path, required=True)
    item.add_argument("--seed", type=int, default=20261026)
    item.add_argument("--bootstrap-replicates", type=int, default=10_000)


def _emit(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str))


def run(root: Path, args: argparse.Namespace) -> int:
    from rq1.university_gpu import ops
    from rq1.university_gpu.manifest import ManifestError, freeze_manifest, validate_manifest
    from rq1.university_gpu.models import atomic_write_json, load_json
    from rq1.university_gpu.scheduler import Coordinator, RunStore, SchedulerError

    try:
        action = args.university_gpu_command
        if action == "analysis-artifacts":
            from rq1.university_gpu.figures import write_artifacts
            _emit(write_artifacts(args.analysis, args.output_dir))
            return 0
        if action == "progress-report":
            from rq1.university_gpu.progress import report
            _emit(report(args.run_root, args.manifest))
            return 0
        if action == "benchmark-report":
            from rq1.university_gpu.benchmark import report
            _emit(report(args.run_root, args.manifest))
            return 0
        if action == "verify-export":
            from rq1.university_gpu.snapshot import verify_progress
            _emit(verify_progress(args.destination))
            return 0
        if action == "export-progress":
            from rq1.university_gpu.snapshot import export_progress
            _emit(export_progress(root, args.run_root, args.manifest, args.destination, args.relevance))
            return 0
        if action == "check":
            config = ops.load_external_config(args.config) if args.config else {}
            payload = ops.check(root, config=config)
            _emit(payload)
            return 0 if payload["status"] == "PASS" else 1
        if action == "setup":
            if not args.apply:
                _emit(ops.setup_plan(root))
                return 0
            if not args.yes or not args.approval_file:
                raise ops.OperatorError("setup apply requires --yes and an approval file")
            _emit(ops.apply_project_setup(root, args.approval_file))
            return 0
        if action == "gate-calibration":
            from rq1.university_gpu.calibration import freeze_gate_calibration, propose_gate_calibration, validate_gate_calibration
            if args.calibration_command == "propose":
                proposal = propose_gate_calibration(args.scores, args.labels)
                atomic_write_json(args.output, proposal, overwrite=False)
                _emit({"status": "PROPOSED", "path": args.output,
                       "calibration_sha256": proposal["calibration_sha256"], "threshold": proposal["threshold"],
                       "task_count": proposal["task_count"], "pair_count": proposal["pair_count"],
                       "recovery_outcomes_used": False})
                return 0
            value = load_json(args.calibration)
            if args.calibration_command == "validate":
                issues = validate_gate_calibration(value, require_frozen=args.require_frozen)
                _emit({"status": "PASS" if not issues else "FAIL", "issues": issues,
                       "calibration_sha256": value.get("calibration_sha256"), "threshold": value.get("threshold")})
                return 0 if not issues else 1
            frozen = freeze_gate_calibration(value, load_json(args.approval_file), args.output)
            _emit({"status": "FROZEN", "path": args.output,
                   "calibration_sha256": frozen["calibration_sha256"], "threshold": frozen["threshold"]})
            return 0
        if action == "analyze":
            from rq1.university_gpu.analysis import analyze_files
            payload = analyze_files(args.results, args.manifest, relevance_path=args.relevance,
                                    seed=args.seed, bootstrap_replicates=args.bootstrap_replicates)
            atomic_write_json(args.output, payload, overwrite=False)
            _emit({"status": "PASS", "path": args.output, "run_id": payload["run_id"],
                   "exploratory": True, "power_or_sufficiency_claim": False})
            return 0
        if action == "manifest":
            if args.manifest_command == "propose":
                config = ops.load_external_config(args.config)
                proposed = ops.build_proposed_manifest(root, config, stage=args.stage, run_id=args.run_id)
                atomic_write_json(args.output, proposed, overwrite=False)
                _emit({"status": "PROPOSED", "path": args.output, "manifest_sha256": proposed["manifest_sha256"],
                       "stage": args.stage, "episodes": proposed["episode_count"], "scientific_evidence": False})
                return 0
            if args.manifest_command == "validate":
                value = load_json(args.manifest)
                problems = validate_manifest(value, require_frozen=args.require_frozen)
                _emit({"status": "PASS" if not problems else "FAIL", "problems": problems,
                       "manifest_sha256": value.get("manifest_sha256"), "stage": value.get("stage"),
                       "episodes": value.get("episode_count")})
                return 0 if not problems else 1
            value = load_json(args.manifest)
            approval = load_json(args.approval_file)
            frozen = freeze_manifest(value, approval, args.output)
            _emit({"status": "FROZEN", "path": args.output, "manifest_sha256": frozen["manifest_sha256"],
                   "stage": frozen["stage"], "episodes": frozen["episode_count"]})
            return 0
        if action == "preflight":
            config = ops.load_external_config(args.config)
            payload = ops.preflight(root, config, load_json(args.manifest), check_endpoints=args.check_endpoints)
            _emit(payload)
            return 0 if payload["status"] in {"PASS", "STATIC_PASS"} else 1
        if action in {"launch", "resume"}:
            config = ops.load_external_config(args.config)
            manifest_value = load_json(args.manifest)
            approval = load_json(args.approval_file)
            payload = ops.run_design_launch(root, config, manifest_value, approval=approval, yes=args.yes,
                                            resume=action == "resume", retry_failed=args.retry_failed,
                                            cutoff_epoch=args.cutoff_epoch, max_attempts=args.max_attempts)
            _emit(payload)
            return 0 if payload["run"]["status"] == "completed" else 1
        if action == "stop":
            run_root = args.run_root.resolve()
            run_root.mkdir(parents=True, exist_ok=True)
            from rq1.university_gpu.models import atomic_write_bytes
            atomic_write_bytes(run_root / "STOP", (args.reason.strip() + "\n").encode("utf-8"))
            _emit({"status": "STOP_REQUESTED", "run_root": run_root, "reason": args.reason})
            return 0
        manifest_value = load_json(args.manifest)
        run_root = args.run_root.resolve()
        store = RunStore(run_root, manifest_value)
        coordinator = Coordinator(store, [type("OfflineWorker", (), {"worker_id": "offline-validation"})()])
        if action == "status":
            _emit(ops.status(run_root, args.manifest))
            return 0
        if action == "validate":
            payload = coordinator.validate()
            _emit(payload)
            return 0 if payload["passed"] else 1
        if action == "merge":
            _emit(coordinator.merge())
            return 0
        if action == "export":
            _emit(ops.export_run(run_root, args.manifest, args.destination))
            return 0
        raise ops.OperatorError(f"unsupported university-gpu action: {action}")
    except (ops.OperatorError, ManifestError, SchedulerError, OSError, ValueError, KeyError, TypeError) as exc:
        _emit({"status": "BLOCKED", "error_type": type(exc).__name__, "error": str(exc), "scientific_evidence": False})
        return 1
