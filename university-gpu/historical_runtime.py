"""Extract observed episode event times and conservative timing scenarios offline."""
import argparse
import csv
import hashlib
import json
import statistics
import tarfile
from datetime import datetime, timezone
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--records", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise SystemExit("output already exists")
    expected_archive = "d52ed82e01a073f8c3382711d03f65916cb41ca89bc0b773e6845817bd37e4fd"
    if digest(args.archive) != expected_archive:
        raise SystemExit("archive hash mismatch")
    records = [json.loads(line)["record"] for line in args.records.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(records) != 360 or digest(args.records) != "834004f8ec6ee5e0b795cfba562c8499c12fa2fce80c120f1bcd65f9a53c1ea0":
        raise SystemExit("frozen records count/hash mismatch")
    events = {}
    with tarfile.open(args.archive, "r:gz") as archive:
        for member in archive:
            if not member.isfile() or not member.name.endswith("/episode/episode-events.jsonl"):
                continue
            parts = member.name.split("/")
            idx = parts.index("evaluation")
            key = (parts[idx + 1], parts[idx + 2])
            times = [float(row["timestamp"]) for line in archive.extractfile(member)
                     if (row := json.loads(line)) and isinstance(row.get("timestamp"), (int, float))]
            if key in events or not times:
                raise SystemExit("duplicate or empty episode event log")
            events[key] = (min(times), max(times))
    rows = []
    for record in records:
        key = (record["run_key"], record["attempt_id"])
        if key not in events:
            raise SystemExit(f"missing episode event log: {key}")
        first, last = events[key]
        rows.append({"unit_key": key[0], "attempt_id": key[1], "condition": record["condition"],
                     "termination_reason": record["termination_reason"], "recovery_success": record["recovery_success"],
                     "first_event_utc": datetime.fromtimestamp(first, timezone.utc).isoformat(),
                     "last_event_utc": datetime.fromtimestamp(last, timezone.utc).isoformat(),
                     "event_span_seconds": round(last - first, 3), "recorded_runtime_seconds": record["runtime_seconds"],
                     "record_timestamp_utc": record["timestamp"]})
    durations = sorted(float(r["recorded_runtime_seconds"]) for r in rows)
    args.output.mkdir(parents=True)
    with (args.output / "episode-timings.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    mean = statistics.mean(durations)
    summary = {"archive_sha256": expected_archive, "records_sha256": digest(args.records), "episodes": len(rows),
               "event_logs_matched": len(rows), "min_seconds": durations[0], "median_seconds": statistics.median(durations),
               "mean_seconds": mean, "p90_seconds": durations[int(.90 * (len(durations) - 1))],
               "p95_seconds": durations[int(.95 * (len(durations) - 1))], "max_seconds": durations[-1],
               "observed_first_event_utc": min(r["first_event_utc"] for r in rows),
               "observed_last_event_utc": max(r["last_event_utc"] for r in rows),
               "baseline_four_worker_hours_36": 36 * mean / 4 / 3600,
               "baseline_four_worker_hours_540": 540 * mean / 4 / 3600,
               "baseline_four_worker_hours_576": 576 * mean / 4 / 3600,
               "warning": "Event times bracket instrumented activity, not necessarily total episode runtime. Historical hardware and new arm runtimes may differ; four-worker scaling is unverified."}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
