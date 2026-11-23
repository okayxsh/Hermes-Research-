"""CLI: ``python -m rq1.scientific_prep <command>`` (run with ``PYTHONPATH=src`` from a scientific worktree).

Exit codes: 0 success | 1 partial (some candidates failed/blocked or uncertainties recorded) | 2 blocked input |
3 refused (output safety / authorization).  No command calls a language model or a GPU.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from rq1.scientific_prep import exposure as EXP
from rq1.scientific_prep.horizons import DEFAULT_CAPS, ENVIRONMENT_VALIDATED, horizon_rows
from rq1.scientific_prep.inventory import InventoryError, build_inventory, load_inventory
from rq1.scientific_prep.safety import (EXIT_BLOCKED, EXIT_OK, EXIT_PARTIAL, EXIT_REFUSED, OutputRefused,
                                        prepare_output_dir, stamp, write_csv_once, write_json_once)

REPO_ROOT = Path(__file__).resolve().parents[3]
POOLS = ("confirmatory_valid_unseen", "pilot_valid_seen")


def _data_root(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    return path.parent if path.name == "json_2.1.1" else path


def _emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


def _run_record(out: Path, command: str, args: argparse.Namespace) -> None:
    from rq1.freeze.validation import git_state
    commit, clean, error = git_state(REPO_ROOT)
    write_json_once(out / "run.json", {
        "command": command, "arguments": {k: v for k, v in vars(args).items() if k != "func"},
        "repository_commit": commit, "repository_clean": clean, "git_error": error,
        "python": sys.version.split()[0], "platform": platform.platform(),
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "note": "run metadata is kept out of the content-hashed outputs so repeated runs stay byte-stable"})


def cmd_check_runtime(args: argparse.Namespace) -> int:
    root = _data_root(args.data_root)
    report: dict[str, Any] = {"data_root": str(root), "python": sys.version.split()[0],
                              "python_supported": (3, 11) <= sys.version_info[:2] < (3, 12)}
    for package in ("alfworld", "textworld"):
        found = importlib.util.find_spec(package) is not None   # find_spec never imports or downloads
        try:
            version = importlib.metadata.version(package) if found else None
        except importlib.metadata.PackageNotFoundError:
            version = None
        report[package] = {"importable": found, "version": version}
    report["alfworld_0_4_2"] = report["alfworld"]["version"] == "0.4.2"
    report["json_2_1_1_present"] = (root / "json_2.1.1").is_dir()
    report["logic_files_present"] = all((root / "logic" / f).is_file() for f in ("alfred.pddl", "alfred.twl2"))
    report["splits_present"] = {s: (root / "json_2.1.1" / s).is_dir() for s in ("train", "valid_seen", "valid_unseen")}
    report["valid_unseen_access"] = os.environ.get("RQ1_VALID_UNSEEN_ACCESS")
    report["note"] = "metadata checks only; no model, no ALFWorld environment and no download is started"
    ok = report["python_supported"] and report["alfworld_0_4_2"] and report["json_2_1_1_present"] and report["logic_files_present"]
    report["ready_for_oracle_validation"] = ok
    _emit(report)
    return EXIT_OK if ok else EXIT_BLOCKED


def _unseen_gate(splits: Sequence[str]) -> str | None:
    if "valid_unseen" not in splits:
        return None
    from rq1.bridge.adapters.unseen_access import TASK_PREPARATION, UnseenAccessError, require_unseen_access
    try:
        require_unseen_access(TASK_PREPARATION)
    except UnseenAccessError as exc:
        return str(exc)
    return None


def cmd_inventory(args: argparse.Namespace) -> int:
    gate = _unseen_gate(args.splits)
    if gate:
        _emit({"status": "blocked", "reason": gate})
        return EXIT_BLOCKED
    try:
        inventory = build_inventory(_data_root(args.data_root), args.splits)
    except InventoryError as exc:
        _emit({"status": "blocked", "reason": str(exc)})
        return EXIT_BLOCKED
    summary = {"splits": inventory["splits"], "cross_split": inventory["cross_split"]}
    if args.dry_run:
        _emit({"status": "dry_run", "written": False, **summary})
        return EXIT_OK
    try:
        out = prepare_output_dir(Path(args.output_root), "inventory", args.label, repo_root=REPO_ROOT,
                                 extra_protected=args.protect)
    except OutputRefused as exc:
        _emit({"status": "refused", "reason": str(exc)})
        return EXIT_REFUSED
    if "valid_unseen" in args.splits:
        from rq1.bridge.adapters.unseen_access import log_unseen_access
        log_unseen_access(out / "valid-unseen-access.jsonl", operation="sci_prep_inventory_metadata",
                          detail=f"{inventory['splits']['valid_unseen']['indexed_entries']} indexed entries; task_type only")
    write_json_once(out / "task-inventory.json", stamp(inventory, "rq1-sci-prep-task-inventory-file-v1"))
    columns = ["task_id", "split", "family", "task_type", "object", "mrecep", "recep", "scene", "trial", "indexed",
               "installed", "game_present", "trajectory_path", "trajectory_sha256", "game_path", "game_sha256", "problems"]
    write_csv_once(out / "task-inventory.csv", inventory["entries"], columns)
    _run_record(out, "inventory", args)
    problems = sum(v["entries_with_problems"] for v in inventory["splits"].values())
    _emit({"status": "written", "output_dir": str(out), "entries_with_problems": problems, **summary})
    return EXIT_OK if not problems else EXIT_PARTIAL


def cmd_exposure(args: argparse.Namespace) -> int:
    try:
        inventory_file = load_inventory_file(Path(args.inventory))
        config = EXP.load_sources(Path(args.sources))
    except (InventoryError, EXP.ExposureConfigError, OSError, ValueError) as exc:
        _emit({"status": "blocked", "reason": str(exc)})
        return EXIT_BLOCKED
    result = EXP.build_ledger(inventory_file, config, config_base=Path(args.sources_base or Path(args.sources).parent))
    if result["blocked"]:
        _emit({"status": "blocked", "reason": result["blocked"]})
        return EXIT_BLOCKED
    pools = EXP.screen_pools(result, config)
    try:
        out = prepare_output_dir(Path(args.output_root), "exposure", args.label, repo_root=REPO_ROOT,
                                 extra_protected=args.protect)
    except OutputRefused as exc:
        _emit({"status": "refused", "reason": str(exc)})
        return EXIT_REFUSED
    common = {"inventory_content_sha256": inventory_file.get("content_sha256"), "template_definition": result["template_definition"],
              "sources": result["sources"]}
    write_json_once(out / "exposure-ledger.json", stamp({**common, "ledger": result["ledger"]}, "rq1-sci-prep-exposure-ledger-v1"))
    write_csv_once(out / "exposure-ledger.csv", result["ledger"],
                   ["task_id", "split", "family", "scene", "template", "installed", "exposure_class", "screening_class", "evidence", "notes"])
    overlap = {"template_definition": result["template_definition"], "referenced_templates_by_role": result["referenced_templates"]}
    write_json_once(out / "template-overlap-summary.json", stamp({**common, **overlap}, "rq1-sci-prep-template-overlap-v1"))
    write_json_once(out / "candidate-screening-summary.json", stamp({**common, **pools}, "rq1-sci-prep-candidate-screening-v1"))
    write_json_once(out / "uncertainty-ledger.json", stamp({**common, "uncertainties": result["uncertainty"],
                                                            "unknown_tasks": [r["task_id"] for r in result["ledger"] if r["exposure_class"] == "UNKNOWN"]},
                                                           "rq1-sci-prep-uncertainty-ledger-v1"))
    _run_record(out, "exposure", args)
    counts = {}
    for row in result["ledger"]:
        counts.setdefault(row["split"], {}).setdefault(row["screening_class"], 0)
        counts[row["split"]][row["screening_class"]] += 1
    brief = {name: {"outcome_screened": pools[name]["outcome_screened"]["tasks"],
                    **{p: v["tasks"] for p, v in pools[name]["policies"].items()}} for name in POOLS}
    _emit({"status": "written", "output_dir": str(out), "screening_class_counts": counts, "pool_counts": brief,
           "uncertainties": len(result["uncertainty"])})
    return EXIT_OK if not result["uncertainty"] else EXIT_PARTIAL


def load_inventory_file(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") == "rq1-sci-prep-task-inventory-file-v1":
        return value
    return load_inventory(path)


def cmd_horizons(args: argparse.Namespace) -> int:
    _emit({"rows": horizon_rows(args.prefix, args.detour, args.continuation, caps=args.caps),
           "note": "arithmetic only: never evidence of environment validation"})
    return EXIT_OK


def cmd_oracle_validate(args: argparse.Namespace) -> int:
    if not args.yes:
        _emit({"status": "refused", "reason": "oracle validation executes real ALFWorld sessions; pass --yes"})
        return EXIT_REFUSED
    from rq1.freeze.validation import git_state
    commit, clean, error = git_state(REPO_ROOT)
    if error or not clean or not commit:
        _emit({"status": "blocked", "reason": "oracle validation requires a clean committed scientific worktree"})
        return EXIT_BLOCKED
    try:
        inventory = load_inventory_file(Path(args.inventory))
        ledger = json.loads(Path(args.ledger).read_text(encoding="utf-8"))
        screening = json.loads(Path(args.screening).read_text(encoding="utf-8"))
    except (OSError, ValueError, InventoryError) as exc:
        _emit({"status": "blocked", "reason": str(exc)})
        return EXIT_BLOCKED
    pool = screening[args.pool]
    pool_ids = pool["outcome_screened"]["task_ids"] if args.policy == "outcome_screened" else \
        pool["policies"].get(args.policy, {}).get("task_ids")
    if pool_ids is None:
        _emit({"status": "blocked", "reason": f"policy {args.policy!r} not present in the screening summary"})
        return EXIT_BLOCKED
    classes = {r["task_id"]: r["screening_class"] for r in ledger["ledger"]}
    chosen = list(args.task_id) if args.task_id else sorted(pool_ids)
    outside = [t for t in chosen if t not in pool_ids or classes.get(t) != "NO_MODEL_OUTCOME_FOUND"]
    if outside:
        _emit({"status": "refused", "reason": "tasks outside the selected screened pool", "tasks": outside})
        return EXIT_REFUSED
    if args.limit:
        chosen = chosen[: args.limit]
    split = "valid_unseen" if args.pool == "confirmatory_valid_unseen" else "valid_seen"
    gate = _unseen_gate([split])
    if gate:
        _emit({"status": "blocked", "reason": gate})
        return EXIT_BLOCKED
    data_root = _data_root(args.data_root)
    from rq1.bridge.adapters.capabilities import probe_alfworld_capabilities
    capability = probe_alfworld_capabilities(data_root)
    if not capability.real_adapter_ready:
        _emit({"status": "blocked", "reason": capability.details})
        return EXIT_BLOCKED
    try:
        out = prepare_output_dir(Path(args.output_root), "oracle", args.label, repo_root=REPO_ROOT, extra_protected=args.protect)
    except OutputRefused as exc:
        _emit({"status": "refused", "reason": str(exc)})
        return EXIT_REFUSED
    from rq1.hermes.episode_driver import RealEpisodeDriver
    from rq1.recovery.reference_route import derive_handcoded_reference
    from rq1.scientific_prep.oracle import validate_candidate
    entries = {r["task_id"]: r for r in inventory["entries"]}
    results, aborted = [], None
    import hashlib
    try:
        with RealEpisodeDriver(REPO_ROOT, data_dir=data_root, bridge_log_root=out / "bridge") as driver:
            for task_id in chosen:
                if split == "valid_unseen":
                    from rq1.bridge.adapters.unseen_access import log_unseen_access
                    log_unseen_access(out / "valid-unseen-access.jsonl", operation="sci_prep_oracle_validation", task_id=task_id)
                task_dir = out / "tasks" / hashlib.sha256(task_id.encode()).hexdigest()[:16]
                try:
                    result = validate_candidate(entries[task_id], driver=driver, data_root=data_root, output_dir=task_dir,
                                                route_provider=lambda t, s: derive_handcoded_reference(data_root, t, s).actions,
                                                seed=args.seed, caps=args.caps, selection_cap=args.selection_cap,
                                                max_checkpoints=args.max_checkpoints)
                except Exception as exc:  # noqa: BLE001 - infrastructure faults stop the run; never a task outcome
                    aborted = {"task_id": task_id, "error": f"{type(exc).__name__}: {str(exc)[:500]}"}
                    results.append({"task_id": task_id, "family": entries[task_id].get("family"), "status": "BLOCKED",
                                    "code": "aborted_infrastructure_error", "detail": aborted["error"]})
                    break
                write_json_once(task_dir / "oracle-evidence.json", stamp(result, "rq1-sci-prep-oracle-evidence-v1"))
                results.append(result)
    except Exception as exc:  # noqa: BLE001 - bridge could not start
        aborted = {"task_id": None, "error": f"{type(exc).__name__}: {str(exc)[:500]}"}
    rows = [{"task_id": r["task_id"], "family": r["family"], "status": r["status"], "code": r.get("code"),
             "total_oracle_actions": (r.get("oracle") or {}).get("total_oracle_actions"),
             **{f"cap_{h['total_action_cap']}": h["status"] for h in (r.get("oracle") or {}).get("horizons", [])}} for r in results]
    write_json_once(out / "oracle-validation-summary.json", stamp({
        "repository_commit": commit, "pool": args.pool, "policy": args.policy, "seed": args.seed,
        "selection_cap": args.selection_cap, "caps": list(args.caps), "alfworld_version": capability.version,
        "inventory_content_sha256": inventory.get("content_sha256"), "ledger_content_sha256": ledger.get("content_sha256"),
        "rows": rows, "model_calls": 0, "aborted": aborted,
        "not_attempted": [t for t in chosen if t not in {r["task_id"] for r in results}]}, "rq1-sci-prep-oracle-summary-v1"))
    write_csv_once(out / "oracle-validation-summary.csv", rows,
                   ["task_id", "family", "status", "code", "total_oracle_actions", *[f"cap_{c}" for c in sorted(set(args.caps))]])
    _run_record(out, "oracle-validate", args)
    validated = sum(r["status"] == ENVIRONMENT_VALIDATED for r in results)
    _emit({"status": "aborted" if aborted else "written", "output_dir": str(out), "validated": validated,
           "attempted": len(results), "aborted": aborted})
    if aborted:
        return EXIT_BLOCKED
    return EXIT_OK if validated == len(results) else EXIT_PARTIAL


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m rq1.scientific_prep", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def outputs(p: argparse.ArgumentParser) -> None:
        p.add_argument("--output-root", required=True, help="directory OUTSIDE protected/historical areas")
        p.add_argument("--label", required=True, help="new versioned subdirectory name; never reused")
        p.add_argument("--protect", action="append", default=[], help="additional protected path (repeatable)")

    p = sub.add_parser("check-runtime", help="dependency/data checks without downloads or environment starts")
    p.add_argument("--data-root", required=True)
    p.set_defaults(func=cmd_check_runtime)
    p = sub.add_parser("inventory", help="deterministic task inventory (read-only on the data root)")
    p.add_argument("--data-root", required=True, help="ALFWorld data root or its json_2.1.1 directory")
    p.add_argument("--splits", nargs="+", default=["valid_seen", "valid_unseen"], choices=["train", "valid_seen", "valid_unseen"])
    p.add_argument("--dry-run", action="store_true", help="print the summary; write nothing")
    outputs(p)
    p.set_defaults(func=cmd_inventory)
    p = sub.add_parser("exposure", help="exposure ledger, template overlap and candidate screening")
    p.add_argument("--inventory", required=True)
    p.add_argument("--sources", required=True, help="exposure sources config (rq1-sci-prep-exposure-sources-v1)")
    p.add_argument("--sources-base", help="base directory for relative source paths (default: config directory)")
    outputs(p)
    p.set_defaults(func=cmd_exposure)
    p = sub.add_parser("horizons", help="action-horizon arithmetic only")
    p.add_argument("--prefix", type=int, required=True)
    p.add_argument("--detour", type=int, default=1)
    p.add_argument("--continuation", type=int, required=True)
    p.add_argument("--caps", type=int, nargs="+", default=list(DEFAULT_CAPS))
    p.set_defaults(func=cmd_horizons)
    p = sub.add_parser("oracle-validate", help="CPU-only real ALFWorld oracle validation (no model)")
    p.add_argument("--inventory", required=True)
    p.add_argument("--ledger", required=True, help="exposure-ledger.json")
    p.add_argument("--screening", required=True, help="candidate-screening-summary.json")
    p.add_argument("--pool", required=True, choices=POOLS)
    p.add_argument("--policy", required=True, help="outcome_screened or a screening policy name")
    p.add_argument("--data-root", required=True)
    p.add_argument("--task-id", action="append", help="restrict to these pool tasks (repeatable)")
    p.add_argument("--limit", type=int)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--caps", type=int, nargs="+", default=list(DEFAULT_CAPS))
    p.add_argument("--selection-cap", type=int, help="optional historical-style budget filter on checkpoints")
    p.add_argument("--max-checkpoints", type=int)
    p.add_argument("--yes", action="store_true")
    outputs(p)
    p.set_defaults(func=cmd_oracle_validate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))
