"""Combine completed independent rater and adjudicator files without choosing labels."""
import argparse
import csv
from pathlib import Path


def read(path, label_column):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if set(reader.fieldnames or []) != {"task_id", "skill_id", label_column}:
            raise ValueError(f"incorrect columns in {path}")
        values = {}
        for row in reader:
            key = (row["task_id"].strip(), row["skill_id"].strip())
            label = row[label_column].strip().upper()
            if not all(key) or key in values or label not in {"RELEVANT", "IRRELEVANT"}:
                raise ValueError(f"duplicate, missing identity, or missing/invalid human label in {path}")
            values[key] = label
        return values


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rater-a", type=Path, required=True)
    p.add_argument("--rater-b", type=Path, required=True)
    p.add_argument("--final-labels", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    a, b, final = read(args.rater_a, "label"), read(args.rater_b, "label"), read(args.final_labels, "adjudicated")
    if not a or set(a) != set(b) or set(a) != set(final):
        raise SystemExit("rater and final-label files must cover the same nonempty task–skill pairs")
    with args.output.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["task_id", "skill_id", "rater_a", "rater_b", "adjudicated"])
        for key in sorted(a):
            writer.writerow([*key, a[key], b[key], final[key]])
    print(f"Combined {len(a)} pairs; original rater labels retained.")


if __name__ == "__main__":
    main()
