"""Publication-oriented figures generated solely from computed analysis JSON."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rq1.utils.hashing import sha256_file


class FigureDependencyError(RuntimeError):
    pass


def generate_figures(output: Path) -> list[Path]:
    """Render only already-computed values; never read experiment logs or services."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
    except ImportError as exc:
        raise FigureDependencyError("Figures require the optional analysis dependency: `uv sync --extra analysis`") from exc
    metrics: dict[str, Any] = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    rows = metrics.get("snapshot_summary", [])
    figure_dir = output / "figures"; figure_dir.mkdir(exist_ok=True)
    created: list[Path] = []
    specs = (("conditional_recovery_rate", "Conditional recovery rate"), ("retrieval_noise_rate", "Post-failure retrieval-noise rate"),
             ("relevant_skill_hit_rate", "Relevant-skill hit rate"), ("recovery_actions", "Post-failure actions"),
             ("recovery_latency_ms", "Recovery latency (ms)"))
    x = [row.get("skill_count") for row in rows]
    for key, label in specs:
        values = [row.get(key) for row in rows]
        if not any(value is not None for value in values):
            continue
        fig, axis = plt.subplots(figsize=(6.5, 4.0))
        axis.plot(x, values, marker="o")
        axis.set(xlabel="Frozen library skill count", ylabel=label, title=f"{label} versus library size")
        axis.grid(alpha=.25); fig.tight_layout()
        path = figure_dir / f"{key}_vs_library_size.png"; fig.savefig(path, dpi=180); plt.close(fig); created.append(path)
    paired = metrics.get("paired_comparisons", [])
    if paired:
        fig, axis = plt.subplots(figsize=(6.5, 4.0))
        values = [row.get("recovery_rate_difference") for row in paired]
        axis.axhline(0, color="black", linewidth=.8); axis.scatter(range(1, len(values) + 1), values)
        axis.set(xlabel="Paired recovery unit", ylabel="Recovery-rate difference", title="Paired recovery differences")
        axis.grid(alpha=.25); fig.tight_layout(); path = figure_dir / "paired_recovery_differences.png"; fig.savefig(path, dpi=180); plt.close(fig); created.append(path)
    noise = [row.get("retrieval_noise_rate") for row in rows]
    recovery = [row.get("conditional_recovery_rate") for row in rows]
    pairs = [(x, y, row.get("snapshot_id")) for x, y, row in zip(noise, recovery, rows) if x is not None and y is not None]
    if pairs:
        fig, axis = plt.subplots(figsize=(6.5, 4.0))
        axis.scatter([item[0] for item in pairs], [item[1] for item in pairs])
        for x_value, y_value, name in pairs: axis.annotate(str(name), (x_value, y_value), xytext=(3, 3), textcoords="offset points")
        axis.set(xlabel="Post-failure retrieval-noise rate", ylabel="Conditional recovery rate", title="Noise and recovery association (descriptive)")
        axis.grid(alpha=.25); fig.tight_layout(); path = figure_dir / "noise_vs_recovery.png"; fig.savefig(path, dpi=180); plt.close(fig); created.append(path)
    families = metrics.get("task_family_summary", [])
    if families:
        fig, axis = plt.subplots(figsize=(6.5, 4.0))
        axis.bar([str(row.get("task_family")) for row in families], [row.get("conditional_recovery_rate") or 0 for row in families])
        axis.set(xlabel="Task family", ylabel="Conditional recovery rate", title="Recovery by task family")
        axis.tick_params(axis="x", rotation=30); axis.grid(axis="y", alpha=.25); fig.tight_layout()
        path = figure_dir / "task_family_breakdown.png"; fig.savefig(path, dpi=180); plt.close(fig); created.append(path)
    manifest_path = output / "analysis_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["generated_artifacts"] = {
            str(path.relative_to(output)): sha256_file(path)
            for path in sorted(output.rglob("*")) if path.is_file() and path != manifest_path
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return created
