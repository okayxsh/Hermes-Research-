"""Exploratory task-cluster resampling using historical paired outcomes only."""
import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean


def percentile(values, q):
    values = sorted(values)
    position = (len(values) - 1) * q
    lo = int(position)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] * (hi - position) + values[hi] * (position - lo) if hi != lo else values[lo]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--outer", type=int, default=1000)
    parser.add_argument("--inner", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261029)
    args = parser.parse_args()
    raw = args.records.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if sha != "834004f8ec6ee5e0b795cfba562c8499c12fa2fce80c120f1bcd65f9a53c1ea0":
        raise SystemExit("frozen historical result hash mismatch")
    if args.outer < 100 or args.inner < 100:
        raise SystemExit("at least 100 outer and inner replicates are required")
    pairs = defaultdict(dict)
    for line in raw.splitlines():
        record = json.loads(line)["record"]
        if record["condition"] in {"NoLib", "Accum-18"}:
            key = (record["task_id"], record["seed"])
            if record["condition"] in pairs[key] or not isinstance(record["recovery_success"], bool):
                raise SystemExit("invalid duplicate or non-binary historical pair")
            pairs[key][record["condition"]] = record["recovery_success"]
    task_values = defaultdict(list)
    transitions = Counter()
    for (task, seed), values in pairs.items():
        if set(values) != {"NoLib", "Accum-18"} or seed not in {11, 29, 47}:
            raise SystemExit("incomplete or unexpected historical pair")
        a, b = int(values["NoLib"]), int(values["Accum-18"])
        transitions[f"{a}{b}"] += 1
        task_values[task].append(b - a)
    if len(task_values) != 30 or any(len(v) != 3 for v in task_values.values()):
        raise SystemExit("expected 30 tasks with three paired seeds each")
    population = [fmean(task_values[task]) for task in sorted(task_values)]
    rng = random.Random(args.seed)
    widths = []
    excluded_zero = 0
    for _ in range(args.outer):
        sample = [rng.choice(population) for _ in population]
        draws = [fmean(rng.choice(sample) for _ in sample) for _ in range(args.inner)]
        low, high = percentile(draws, .025), percentile(draws, .975)
        widths.append(high - low)
        excluded_zero += int(low > 0 or high < 0)
    rate = excluded_zero / args.outer
    result = {"schema": "rq1-historical-cluster-resampling-v1", "source_sha256": sha, "seed": args.seed,
              "outer_replicates": args.outer, "inner_bootstrap_replicates": args.inner,
              "historical_task_clusters": 30, "seeds_per_task": 3, "paired_seed_transitions_nolib_to_accum18": dict(transitions),
              "historical_paired_difference": fmean(population), "resampled_ci_median_width": statistics_median(widths),
              "resampled_ci_width_10th": percentile(widths, .1), "resampled_ci_width_90th": percentile(widths, .9),
              "interval_excludes_zero_fraction": rate, "monte_carlo_standard_error": math.sqrt(rate * (1-rate) / args.outer),
              "limitations": "Exploratory plug-in simulation resamples the 30 observed historical task means as a population; it assumes those tasks and historical effects represent future tasks, uses one contrast and no multiplicity correction. It does not establish power for other effects or validate 576 episodes.",
              "author_approved": False}
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))


def statistics_median(values):
    return percentile(values, .5)


if __name__ == "__main__":
    main()
