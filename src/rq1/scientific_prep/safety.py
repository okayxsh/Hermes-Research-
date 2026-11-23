"""Output isolation for preparation runs.

* every run writes into a NEW versioned directory ``<output-root>/<kind>-<label>``; existing directories are never
  reused or overwritten, and every file is written once (``FileExistsError`` otherwise);
* output roots inside the repository's historical/engineering areas or any configured protected path are refused;
* outputs are preparation evidence only: payloads must never carry approval or manifest markers.
"""
from __future__ import annotations

import csv
import io
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from rq1.experiment.models import canonical_hash
from rq1.scientific_prep import PREPARATION_STATUS

EXIT_OK, EXIT_PARTIAL, EXIT_BLOCKED, EXIT_REFUSED = 0, 1, 2, 3
# Repository areas holding frozen historical evidence, the closed engineering run, task lists, profiles and approvals.
PROTECTED_REPOSITORY_AREAS = ("artifacts", "results", "runs", "state", "data", "university-gpu", "hermes", ".hermes",
                              ".git", "configs", "docs", "src", "tests", "scripts")
PROTECT_ENV = "RQ1_SCI_PREP_PROTECTED_PATHS"
_LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_FORBIDDEN_KEYS = {"manifest_type", "approved_at", "approval_reference"}
_FORBIDDEN_STATUS = {"approved", "frozen"}


class OutputRefused(RuntimeError):
    pass


def protected_paths(repo_root: Path, extra: Sequence[str | Path] = ()) -> list[Path]:
    paths = [(repo_root / area).resolve() for area in PROTECTED_REPOSITORY_AREAS]
    from_env = [p for p in os.environ.get(PROTECT_ENV, "").split(os.pathsep) if p.strip()]
    return paths + [Path(p).expanduser().resolve() for p in [*extra, *from_env]]


def _inside(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def prepare_output_dir(output_root: Path, kind: str, label: str, *, repo_root: Path,
                       extra_protected: Sequence[str | Path] = ()) -> Path:
    if not _LABEL.fullmatch(label):
        raise OutputRefused(f"label must match {_LABEL.pattern}")
    root = Path(output_root).expanduser().resolve()
    for protected in protected_paths(repo_root, extra_protected):
        if _inside(root, protected):
            raise OutputRefused(f"output root {root} is inside protected path {protected}")
    target = root / f"{kind}-{label}"
    if target.exists():
        raise OutputRefused(f"output directory already exists and is immutable: {target}")
    target.mkdir(parents=True)
    return target


def assert_preparation_payload(payload: Any, *, where: str = "payload") -> None:
    """Fail closed if an output could be mistaken for an approved/frozen scientific manifest."""
    if isinstance(payload, Mapping):
        bad = _FORBIDDEN_KEYS & set(payload)
        if bad:
            raise OutputRefused(f"{where} carries scientific-manifest keys {sorted(bad)}")
        if str(payload.get("status", "")).lower() in _FORBIDDEN_STATUS or payload.get("approved") is True:
            raise OutputRefused(f"{where} claims an approved/frozen status")
        for key, value in payload.items():
            assert_preparation_payload(value, where=f"{where}.{key}")
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            assert_preparation_payload(value, where=f"{where}[{index}]")


def stamp(payload: dict[str, Any], schema: str) -> dict[str, Any]:
    """Add schema/status markers and a content hash over the payload (no timestamps: repeat runs are byte-stable)."""
    clash = {"schema", "preparation_status", "approved", "scientific_manifest", "content_sha256"} & set(payload)
    if clash:
        raise OutputRefused(f"payload may not set reserved marker keys {sorted(clash)}")
    value = {"schema": schema, "preparation_status": PREPARATION_STATUS, "approved": False, "scientific_manifest": False,
             **payload}
    value["content_sha256"] = canonical_hash({k: v for k, v in value.items() if k != "content_sha256"})
    assert_preparation_payload(value)
    return value


def write_json_once(path: Path, payload: Mapping[str, Any]) -> Path:
    if path.exists():
        raise FileExistsError(path)
    data = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    tmp = path.with_name(path.name + ".partial")
    with open(tmp, "x", encoding="utf-8", newline="\n") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    return path


def write_csv_once(path: Path, rows: Iterable[Mapping[str, Any]], columns: Sequence[str]) -> Path:
    if path.exists():
        raise FileExistsError(path)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(columns), extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: (json.dumps(v, sort_keys=True) if isinstance(v, (list, dict)) else v) for k, v in row.items()})
    with open(path, "x", encoding="utf-8", newline="") as handle:
        handle.write(buffer.getvalue())
    return path
