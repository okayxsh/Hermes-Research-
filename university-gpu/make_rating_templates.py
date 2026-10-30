"""Create empty independent rating forms from existing frozen task and library inputs."""
import argparse
import csv
import json
from pathlib import Path

from rq1.university_gpu.pilot_readiness import budget_problems


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True, type=Path)
    parser.add_argument("--failures", required=True, type=Path)
    parser.add_argument("--library", required=True, type=Path)
    parser.add_argument("--records", type=Path, help="Optional historical records used only to obtain task instructions.")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    tasks = json.loads(args.tasks.read_text(encoding="utf-8"))
    failures = json.loads(args.failures.read_text(encoding="utf-8"))
    libraries = json.loads(args.library.read_text(encoding="utf-8"))
    if tasks.get("status") != "frozen":
        raise SystemExit("task manifest is not frozen")
    definitions = {item["task_id"]: item for item in failures["tasks"]}
    skills = libraries.get("libraries", libraries)["Accum-18"]["skills"]
    if len(skills) != 18 or len({s["skill_id"] for s in skills}) != 18:
        raise SystemExit("expected 18 distinct frozen skills")
    if args.output.exists():
        raise SystemExit("output directory already exists")
    tasks_list = tasks["tasks"]
    goals = {}
    if args.records:
        for line in args.records.read_text(encoding="utf-8").splitlines():
            row = json.loads(line).get("record", {})
            task_id = row.get("task_id")
            goal = row.get("failure_context", {}).get("task_instruction")
            if task_id and goal:
                if task_id in goals and goals[task_id] != goal:
                    raise SystemExit(f"conflicting recorded task instructions: {task_id}")
                goals[task_id] = goal
    if any(t["task_id"] not in definitions or definitions[t["task_id"]].get("oracle", {}).get("validated") is not True for t in tasks_list):
        raise SystemExit("task is missing an oracle-validated failure definition")
    for task in tasks_list:
        definition = definitions[task["task_id"]]
        issues = budget_problems(definition)
        if issues:
            raise SystemExit(f"scientific-budget ineligible context: {task['task_id']}: {issues}")
        if not goals.get(task["task_id"], "").strip() or not str(definition.get("post_detour_observation", "")).strip():
            raise SystemExit("complete task instructions and post-detour observation are required before creating rating materials")
    args.output.mkdir(parents=True)
    for name in ("rater_a.csv", "rater_b.csv"):
        with (args.output / name).open("x", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["task_id", "skill_id", "label"])
            for task in tasks_list:
                for skill in skills:
                    writer.writerow([task["task_id"], skill["skill_id"], ""])
    with (args.output / "final-labels.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["task_id", "skill_id", "adjudicated"])
        for task in tasks_list:
            for skill in skills:
                writer.writerow([task["task_id"], skill["skill_id"], ""])
    with (args.output / "worksheet.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["task_id", "task_family", "task_instruction", "post_detour_observation", "skill_id", "skill_text"])
        for task in tasks_list:
            failure = definitions[task["task_id"]]
            for skill in skills:
                writer.writerow([task["task_id"], failure["task_family"], goals.get(task["task_id"], ""), failure["post_detour_observation"],
                                 skill["skill_id"], skill["text"]])
    with (args.output / "relevance-template.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["task_id", "skill_id", "rater_a", "rater_b", "adjudicated"])
        for task in tasks_list:
            for skill in skills:
                writer.writerow([task["task_id"], skill["skill_id"], "", "", ""])
    print(json.dumps({"tasks": len(tasks_list), "skills": len(skills), "task_skill_pairs": len(tasks_list) * len(skills),
                      "missing_task_instructions": sum(task["task_id"] not in goals for task in tasks_list),
                      "ratings_filled": 0, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
