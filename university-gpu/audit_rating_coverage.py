"""Count historical task–skill coverage without selecting or changing labels."""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--join", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tasks = json.loads(args.tasks.read_text(encoding="utf-8"))
    library = json.loads(args.library.read_text(encoding="utf-8"))
    task_ids = {row["task_id"] for row in tasks["tasks"]}
    skill_ids = {row["skill_id"] for row in library["libraries"]["Accum-18"]["skills"]}
    if len(task_ids) != 30 or len(skill_ids) != 18:
        raise SystemExit("expected 30 tasks and 18 distinct library skills")
    pairs = collections.defaultdict(list)
    with args.join.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["task_id"] in task_ids and row["skill_id"] in skill_ids:
                pairs[(row["task_id"], row["skill_id"])].append((row["rater_1_label"], row["rater_2_label"], row["final_relevance_label"]))
    report = {"historical_join_sha256": hashlib.sha256(args.join.read_bytes()).hexdigest(),
              "confirmatory_tasks": len(task_ids), "skills": len(skill_ids),
              "confirmatory_pairs_required": len(task_ids) * len(skill_ids), "historical_distinct_pairs": len(pairs),
              "confirmatory_pairs_without_historical_rating": len(task_ids) * len(skill_ids) - len(pairs),
              "pairs_with_different_rater_or_final_tuples": sum(len(set(labels)) > 1 for labels in pairs.values()),
              "pilot_pairs_if_six_tasks": 6 * len(skill_ids),
              "warning": "Coverage is not prospective eligibility, verified blinding, or authorization to reuse any rating."}
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
