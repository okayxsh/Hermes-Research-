"""CLI: ``python -m rq1.five_arm <command>`` -- preparation-only commands; none calls a model or a GPU.

decisions        print the open scientific-decision register (all unapproved)
readiness        evaluate readiness levels from a state file (approvals are read, never written)
workload         exact relevance-annotation workload for the supported designs
power            run the prospective power/precision grid (synthetic) into a new versioned output directory
rating-export    write per-rater CSV exports and the coordinator key from an items-input JSON
rating-agreement agreement/conflict report for two validated human submissions
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from rq1.five_arm import power, rating, readiness
from rq1.five_arm.protocol import decision_register
from rq1.scientific_prep.safety import (EXIT_BLOCKED, EXIT_OK, EXIT_REFUSED, OutputRefused, prepare_output_dir, stamp,
                                        write_json_once)

REPO_ROOT = Path(__file__).resolve().parents[3]


def _emit(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, default=str))


def _out(args, kind: str) -> Path:
    return prepare_output_dir(Path(args.output_root), kind, args.label, repo_root=REPO_ROOT, extra_protected=args.protect)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m rq1.five_arm", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("decisions")
    p = sub.add_parser("readiness"); p.add_argument("--state", required=True)
    p = sub.add_parser("workload")
    p.add_argument("--tasks", type=int, default=79); p.add_argument("--skills", type=int, default=18)
    p.add_argument("--candidates-per-task", type=int, default=3); p.add_argument("--pilot-tasks", type=int, default=0)
    for name in ("power", "rating-export", "rating-agreement"):
        p = sub.add_parser(name)
        p.add_argument("--output-root", required=True); p.add_argument("--label", required=True)
        p.add_argument("--protect", action="append", default=[])
        if name == "power":
            p.add_argument("--reps", type=int, default=400); p.add_argument("--seed", type=int, default=20261124)
        if name == "rating-export":
            p.add_argument("--items-input", required=True, help="JSON: tasks, skills, design, salt, candidate_sets?, forbidden_strings?, raters")
        if name == "rating-agreement":
            p.add_argument("--key", required=True)
            for r in ("a", "b"):
                p.add_argument(f"--rater-{r}-csv", required=True); p.add_argument(f"--rater-{r}-declaration", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "decisions":
            _emit(decision_register()); return EXIT_OK
        if args.command == "readiness":
            report = readiness.evaluate(json.loads(Path(args.state).read_text(encoding="utf-8")))
            _emit(report); return EXIT_OK
        if args.command == "workload":
            _emit(rating.workload(tasks=args.tasks, library_skills=args.skills, candidate_per_task=args.candidates_per_task,
                                  pilot_tasks=args.pilot_tasks)); return EXIT_OK
        if args.command == "power":
            result = power.run_grid(power.default_grid(), reps=args.reps, seed=args.seed)
            result["minimal_detectable_delta_80pct_bonferroni"] = power.minimal_detectable(result["rows"])
            out = _out(args, "power")
            schema = result.pop("schema")
            write_json_once(out / "power-simulation.json", stamp(result, schema))
            _emit({"status": "written", "output_dir": str(out), "minimal_detectable": result["minimal_detectable_delta_80pct_bonferroni"]})
            return EXIT_OK
        if args.command == "rating-export":
            spec = json.loads(Path(args.items_input).read_text(encoding="utf-8"))
            items, key = rating.build_items(spec["tasks"], spec["skills"], design=spec["design"], salt=spec["salt"],
                                            candidate_sets=spec.get("candidate_sets"), forbidden_strings=spec.get("forbidden_strings"))
            out = _out(args, "rating-export")
            for rater in spec["raters"]:
                with (out / f"export-{rater}.csv").open("x", encoding="utf-8", newline="") as handle:
                    handle.write(rating.export_for_rater(items, rater))
            write_json_once(out / "coordinator-key.json", stamp(key, "rq1-five-arm-rating-key-v1"))
            _emit({"status": "written", "output_dir": str(out), "items": len(items), "raters": spec["raters"]}); return EXIT_OK
        if args.command == "rating-agreement":
            key = json.loads(Path(args.key).read_text(encoding="utf-8"))
            ids = set(key["items"])
            subs = {}
            for r in ("a", "b"):
                decl = json.loads(Path(getattr(args, f"rater_{r}_declaration")).read_text(encoding="utf-8"))
                subs[r] = rating.parse_submission(Path(getattr(args, f"rater_{r}_csv")).read_bytes(), expected_item_ids=ids,
                                                  rater_id=decl.get("rater_id", ""), declaration=decl)
            report = rating.agreement(subs["a"], subs["b"])
            out = _out(args, "rating-agreement")
            write_json_once(out / "agreement.json", stamp(report, "rq1-five-arm-rating-agreement-v1"))
            _emit({"status": "written", "output_dir": str(out), **{k: report[k] for k in ("items", "percent_agreement")},
                   "conflicts": len(report["requires_adjudication"])}); return EXIT_OK
    except OutputRefused as exc:
        _emit({"status": "refused", "reason": str(exc)}); return EXIT_REFUSED
    except (rating.RatingError, ValueError, OSError, KeyError) as exc:
        _emit({"status": "blocked", "reason": f"{type(exc).__name__}: {exc}"}); return EXIT_BLOCKED
    return EXIT_BLOCKED


if __name__ == "__main__":
    sys.exit(main())
