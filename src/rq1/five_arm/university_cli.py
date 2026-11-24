"""University workstation entry points: ``python -m rq1.five_arm.university_cli <command>``.

CPU-only commands (readiness ``university_cpu_integration``; no model inference):

observe-runtime   metadata-only runtime record (/api/version, /api/tags, /api/show) + locally observed identities
cpu-rankings      frozen natural rankings at each roster checkpoint (real bridge/ALFWorld + Sentence-BERT, no model)
oracle-replay     exact-cap oracle replay of every roster task at --cap
cap-invariance    cap-invariance replay at two caps (environment evidence for cap-100 -> cap-50 truncation)

Model-execution command (gated by readiness for the manifest's stage, runtime identity and a frozen approved manifest):

run-queue         build the real executor + resumable queue and run it; prints the post-run validation

Every output goes to a NEW versioned directory under --output-root (never overwritten).  Nothing here is executed in
local development; the commands are exercised locally only with an injected scripted driver.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from rq1.five_arm import readiness, runtime
from rq1.scientific_prep.safety import (EXIT_BLOCKED, EXIT_OK, EXIT_REFUSED, OutputRefused, prepare_output_dir, stamp,
                                        write_json_once)

REPO_ROOT = Path(__file__).resolve().parents[3]


def _emit(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, default=str))


def _load(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _real_driver(config: Mapping[str, Any]):
    from rq1.hermes.episode_driver import RealEpisodeDriver
    return RealEpisodeDriver(Path(str(config["repository_root"])).resolve(), data_dir=Path(str(config["alfworld_data_dir"])),
                             bridge_log_root=Path(str(config["bridge_log_root"])))


def _real_embedder(config: Mapping[str, Any]):
    from rq1.retrieval.embedder import SentenceBERTEmbedder
    return SentenceBERTEmbedder(config["sbert_model"], cache_folder=Path(str(config["sbert_cache"])),
                                revision=config["sbert_revision"], local_files_only=True)


DRIVER_FACTORY: Callable[[Mapping[str, Any]], Any] = _real_driver       # replaced only by local tests (scripted driver)
EMBEDDER_FACTORY: Callable[[Mapping[str, Any]], Any] = _real_embedder   # replaced only by local tests (synthetic embedder)


def _tasks(config: Mapping[str, Any]):
    from rq1.evaluation.amended_matrix import EvaluationTask
    return {tid: EvaluationTask(**{**spec, "prefix_actions": tuple(spec["prefix_actions"]),
                                   "reference_actions": tuple(spec["reference_actions"])})
            for tid, spec in sorted(config["tasks"].items())}


def _require_cpu(state_path: str) -> None:
    report = readiness.evaluate(_load(state_path), {}, real=True)["levels"]["university_cpu_integration"]
    if report["status"] != "READY":
        raise readiness.ExecutionBlocked("university_cpu_integration blocked: " + "; ".join(report["blockers"]))


def _with_driver(config: Mapping[str, Any], work: Callable[[Any], Any]) -> Any:
    driver = DRIVER_FACTORY(config)
    enter = getattr(driver, "__enter__", None)
    if callable(enter):
        enter()
    try:
        return work(driver)
    finally:
        leave = getattr(driver, "__exit__", None)
        if callable(leave):
            leave(None, None, None)


def cmd_observe_runtime(args) -> dict[str, Any]:
    from rq1.university_gpu.ollama import _request
    local = _load(args.local_json)
    observed = runtime.observe_runtime(lambda e, p, payload=None: _request(e, p, payload), endpoint=args.endpoint,
                                       model_tag=args.model_tag, local=local)
    return {"name": "observed-runtime.json", "schema": "rq1-five-arm-observed-runtime-v1", "body": observed}


def cmd_cpu_rankings(args, config) -> dict[str, Any]:
    from rq1.evaluation.amended_libraries import AmendedLibrary
    from rq1.five_arm.cpu_checks import checkpoint_ranking
    library = AmendedLibrary("Accum-18", tuple(dict(s) for s in config["library"]["skills"]))
    embedder = EMBEDDER_FACTORY(config)
    work_root = Path(args.work_dir)
    rows = _with_driver(config, lambda d: [checkpoint_ranking(d, task, library, embedder, output_dir=work_root / f"rank-{i:03d}",
                                                              split=config["split"], total_action_cap=args.cap)
                                           for i, task in enumerate(_tasks(config).values())])
    return {"name": "frozen-rankings.json", "schema": "rq1-five-arm-frozen-rankings-v1",
            "body": {"library_sha256": library.content_sha256, "tasks": rows,
                     "tie_fragile_tasks": [r["task_id"] for r in rows if r["tie_fragile"]], "model_calls": 0}}


def cmd_oracle_replay(args, config) -> dict[str, Any]:
    from rq1.five_arm.cpu_checks import exact_cap_oracle_replay
    work_root = Path(args.work_dir)
    rows = _with_driver(config, lambda d: [exact_cap_oracle_replay(d, task, cap=args.cap, output_dir=work_root / f"oracle-{i:03d}",
                                                                   split=config["split"])
                                           for i, task in enumerate(_tasks(config).values())])
    return {"name": f"oracle-replay-cap{args.cap}.json", "schema": "rq1-five-arm-exact-cap-oracle-replay-v1",
            "body": {"total_action_limit": args.cap, "tasks": rows, "all_validated": all(r["validated"] for r in rows)}}


def cmd_cap_invariance(args, config) -> dict[str, Any]:
    from rq1.five_arm.cpu_checks import cap_invariance_replay
    work_root = Path(args.work_dir)
    rows = _with_driver(config, lambda d: [cap_invariance_replay(d, task, caps=args.caps, output_dir=work_root / f"inv-{i:03d}",
                                                                 split=config["split"])
                                           for i, task in enumerate(_tasks(config).values())])
    return {"name": "cap-invariance.json", "schema": "rq1-five-arm-cap-invariance-v1",
            "body": {"caps": sorted(args.caps), "tasks": rows,
                     "all_hold": all(r["truncation_environment_assumptions_hold"] for r in rows)}}


def cmd_run_queue(args) -> dict[str, Any]:
    from rq1.five_arm.runner import build_university_queue
    queue, driver = build_university_queue(_load(args.run_config), readiness_state=_load(args.state),
                                           observed_runtime=_load(args.observed_runtime), approved_runtime=_load(args.approved_runtime))
    try:
        summary = queue.run()
    finally:
        driver.__exit__(None, None, None)
    return {"summary": summary, "validation": queue.validate()}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m rq1.five_arm.university_cli", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("observe-runtime", "cpu-rankings", "oracle-replay", "cap-invariance"):
        p = sub.add_parser(name)
        p.add_argument("--state", required=True, help="readiness state JSON (university_cpu_integration must be READY)")
        p.add_argument("--output-root", required=True); p.add_argument("--label", required=True)
        p.add_argument("--protect", action="append", default=[])
        if name == "observe-runtime":
            p.add_argument("--endpoint", required=True); p.add_argument("--model-tag", default="gemma4:12b")
            p.add_argument("--local-json", required=True, help="tokenizer/Hermes/ALFWorld/Sentence-BERT identities observed locally")
        else:
            p.add_argument("--config", required=True, help="CPU config JSON: repository_root, alfworld_data_dir, bridge_log_root, "
                                                           "split, tasks, library, sbert_*")
            p.add_argument("--work-dir", required=True, help="new scratch directory for episode logs")
            if name == "cap-invariance":
                p.add_argument("--caps", type=int, nargs=2, default=[50, 100])
            else:
                p.add_argument("--cap", type=int, required=True)
    p = sub.add_parser("run-queue")
    for flag in ("--run-config", "--state", "--observed-runtime", "--approved-runtime"):
        p.add_argument(flag, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "run-queue":
            _emit(cmd_run_queue(args)); return EXIT_OK
        _require_cpu(args.state)
        if args.command == "observe-runtime":
            result = cmd_observe_runtime(args)
        else:
            if Path(args.work_dir).exists():
                raise OutputRefused(f"work directory already exists (never reused): {args.work_dir}")
            config = _load(args.config)
            result = {"cpu-rankings": cmd_cpu_rankings, "oracle-replay": cmd_oracle_replay,
                      "cap-invariance": cmd_cap_invariance}[args.command](args, config)
        out = prepare_output_dir(Path(args.output_root), args.command, args.label, repo_root=REPO_ROOT, extra_protected=args.protect)
        write_json_once(out / result["name"], stamp(result["body"], result["schema"]))
        _emit({"status": "written", "output": str(out / result["name"]), "model_calls": 0})
        return EXIT_OK
    except OutputRefused as exc:
        _emit({"status": "refused", "reason": str(exc)}); return EXIT_REFUSED
    except (readiness.ExecutionBlocked, runtime.RuntimeIdentityError, ValueError, OSError, KeyError) as exc:
        _emit({"status": "blocked", "reason": f"{type(exc).__name__}: {exc}"}); return EXIT_BLOCKED


if __name__ == "__main__":
    sys.exit(main())
