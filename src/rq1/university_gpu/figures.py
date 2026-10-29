"""Export descriptive CSV tables and a self-contained SVG from analysis JSON."""
import csv
import json
from html import escape
from pathlib import Path

from rq1.university_gpu.manifest import CONDITIONS


def write_artifacts(analysis_path: Path, destination: Path) -> dict:
    data = json.loads(analysis_path.read_text(encoding="utf-8"))
    if data.get("schema") != "rq1-university-gpu-analysis-v1" or destination.exists():
        raise ValueError("expected an analysis report and a new output directory")
    conditions = data.get("conditions", {})
    if set(conditions) != set(CONDITIONS):
        raise ValueError("analysis report must include all six conditions")
    for arm in CONDITIONS:
        item = conditions[arm]
        n, successes, rate = item["episodes"], item["successes"], item["recovery_rate"]
        if not isinstance(n, int) or n < 1 or not isinstance(successes, int) or not 0 <= successes <= n:
            raise ValueError(f"invalid outcome counts for {arm}")
        if abs(float(rate) - successes / n) > 1e-9:
            raise ValueError(f"rate does not match counts for {arm}")
    destination.mkdir(parents=True)
    with (destination / "conditions.csv").open("x", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["condition", "episodes", "successes", "recovery_rate"])
        for arm in CONDITIONS:
            item = conditions[arm]
            writer.writerow([arm, item["episodes"], item["successes"], item["recovery_rate"]])
    with (destination / "paired-contrasts.csv").open("x", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["contrast", "estimate", "ci_95_low", "ci_95_high", "task_clusters", "interpretation"])
        for name, item in sorted(data.get("paired_task_cluster_contrasts", {}).items()):
            ci = item.get("ci_95") or [None, None]
            writer.writerow([name, item.get("estimate"), *ci, item.get("task_clusters"), item.get("interpretation")])
    pieces = ['<svg xmlns="http://www.w3.org/2000/svg" width="800" height="400" viewBox="0 0 800 400">',
              '<rect width="800" height="400" fill="white"/>',
              '<text x="45" y="30" font-family="sans-serif" font-size="18">Recovery by condition (descriptive)</text>',
              '<text x="45" y="53" font-family="sans-serif" font-size="12">Counts are episodes; task-cluster contrasts are in paired-contrasts.csv.</text>']
    for index, arm in enumerate(CONDITIONS):
        item = conditions[arm]
        x, y = 45 + 120 * index, 330 - 240 * float(item["recovery_rate"])
        pieces.extend([f'<rect x="{x}" y="{y:.2f}" width="75" height="{330-y:.2f}" fill="#356c8d"/>',
                       f'<text x="{x}" y="350" font-family="sans-serif" font-size="14">{escape(arm)}</text>',
                       f'<text x="{x}" y="{y-7:.2f}" font-family="sans-serif" font-size="12">{item["successes"]}/{item["episodes"]}</text>'])
    pieces.append('</svg>')
    (destination / "conditions.svg").write_text("\n".join(pieces) + "\n", encoding="utf-8")
    return {"status": "PASS", "destination": str(destination), "files": ["conditions.csv", "paired-contrasts.csv", "conditions.svg"],
            "interpretation": "Descriptive figure and exploratory task-cluster contrasts; no causal conclusion is generated."}
