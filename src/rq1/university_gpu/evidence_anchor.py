"""Create and verify per-unit evidence digests in a separate operator-controlled directory."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rq1.university_gpu.models import atomic_write_json, content_hash, load_json, sha256_file


class EvidenceAnchorError(ValueError):
    pass


def _validated_manifest(run_root: Path, manifest_path: Path) -> dict:
    manifest = load_json(manifest_path)
    if manifest != load_json(run_root / "manifest.json"):
        raise EvidenceAnchorError("frozen manifest differs from run")
    from rq1.university_gpu.scheduler import _validate_run_manifest
    if (manifest.get("schema") != "rq1-feasibility-execution-manifest-v1"
            or _validate_run_manifest(manifest)):
        raise EvidenceAnchorError("feasibility manifest is invalid")
    return manifest


def _attempt_files(attempt: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(attempt.rglob("*")):
        if path.is_symlink():
            raise EvidenceAnchorError("attempt evidence contains a symlink")
        if path.is_file():
            rows.append({"path": path.relative_to(attempt).as_posix(), "sha256": sha256_file(path), "bytes": path.stat().st_size})
    if not rows:
        raise EvidenceAnchorError("attempt evidence is empty")
    return rows


def _anchor_for(run_root: Path, manifest: dict, envelope: dict) -> dict:
    result = envelope["result"]
    unit = str(envelope["unit_key"])
    attempt_id = str(envelope["attempt_id"])
    allowed = {str(item.get("unit_key")) for item in manifest.get("units", [])}
    if unit not in allowed or Path(unit).name != unit or unit in {".", ".."}:
        raise EvidenceAnchorError("anchor unit is outside the frozen manifest or unsafe")
    if not attempt_id.isalnum():
        raise EvidenceAnchorError("unsafe attempt identity")
    if envelope.get("result_sha256") != content_hash(result) or result.get("unit_key") != unit or result.get("attempt_id") != attempt_id:
        raise EvidenceAnchorError("authoritative result hash or identity is invalid")
    matches = list((run_root / "attempts" / unit).glob("*-" + attempt_id))
    if len(matches) != 1 or matches[0].is_symlink():
        raise EvidenceAnchorError("attempt directory is missing or ambiguous")
    try:
        matches[0].resolve().relative_to((run_root / "attempts").resolve())
    except ValueError as exc:
        raise EvidenceAnchorError("attempt directory escapes run") from exc
    from rq1.university_gpu.evidence import validate_attempt_evidence
    task = next((item for item in manifest.get("tasks", []) if item.get("task_id") == result.get("task_id")), None)
    if task is None or validate_attempt_evidence(matches[0], result, task, str(manifest.get("run_id", ""))):
        raise EvidenceAnchorError("attempt raw evidence does not validate against frozen task/result")
    from rq1.university_gpu.scheduler import _validate_feasibility_result
    expected_unit = next(item for item in manifest["units"] if item.get("unit_key") == unit)
    if _validate_feasibility_result(result, expected_unit, manifest):
        raise EvidenceAnchorError("result does not validate against frozen unit")
    payload = {"schema": "rq1-external-unit-evidence-anchor-v1", "run_id": manifest["run_id"],
               "manifest_sha256": manifest["manifest_sha256"], "unit_key": unit,
               "episode_id": result["unit_key"], "attempt_id": attempt_id,
               "result_sha256": sha256_file(run_root / "results" / "units" / f"{unit}.json"),
               "attempt_files": _attempt_files(matches[0])}
    payload["anchor_sha256"] = content_hash(payload)
    return payload


def verify_anchor(run_root: Path, manifest_path: Path, anchor_path: Path) -> dict:
    if anchor_path.is_symlink():
        raise EvidenceAnchorError("external anchor must not be a symlink")
    run_root, manifest_path, anchor_path = run_root.resolve(), manifest_path.resolve(), anchor_path.resolve()
    manifest = _validated_manifest(run_root, manifest_path)
    anchor_dir = anchor_path.parent.resolve()
    if anchor_dir == run_root or run_root in anchor_dir.parents:
        raise EvidenceAnchorError("anchor directory is inside mutable run root")
    if anchor_path.is_symlink() or not anchor_path.is_file():
        raise EvidenceAnchorError("external anchor must be a regular non-symlink file")
    unit = load_json(anchor_path).get("unit_key")
    if (not isinstance(unit, str) or Path(unit).name != unit or unit in {".", ".."}
            or unit not in {str(item.get("unit_key")) for item in manifest.get("units", [])}
            or anchor_path.name != f"{unit}.json"):
        raise EvidenceAnchorError("anchor unit identity is unsafe or not in manifest")
    envelope_path = run_root / "results" / "units" / f"{unit}.json"
    observed = _anchor_for(run_root, manifest, load_json(envelope_path))
    expected = load_json(anchor_path)
    if expected != observed:
        raise EvidenceAnchorError("run evidence conflicts with external unit anchor")
    return {"status": "PASS", "unit_key": unit, "anchor_sha256": expected["anchor_sha256"]}


def create_anchors(run_root: Path, manifest_path: Path, anchor_dir: Path) -> dict:
    if anchor_dir.is_symlink():
        raise EvidenceAnchorError("external anchor directory must not be a symlink")
    run_root, manifest_path, anchor_dir = run_root.resolve(), manifest_path.resolve(), anchor_dir.resolve()
    if anchor_dir == run_root or run_root in anchor_dir.parents:
        raise EvidenceAnchorError("anchor directory must be outside the mutable run root")
    manifest = _validated_manifest(run_root, manifest_path)
    anchor_dir.mkdir(parents=True, exist_ok=True)
    completed = sorted((run_root / "results" / "units").glob("*.json"))
    created = 0
    for result_path in completed:
        if result_path.is_symlink():
            raise EvidenceAnchorError("result must not be a symlink")
        envelope = load_json(result_path)
        if result_path.stem != envelope.get("unit_key"):
            raise EvidenceAnchorError("result filename differs from unit identity")
        anchor = _anchor_for(run_root, manifest, envelope)
        target = anchor_dir / f"{envelope['unit_key']}.json"
        if target.is_symlink():
            raise EvidenceAnchorError("external anchor must not be a symlink")
        if target.exists():
            if load_json(target) != anchor:
                raise EvidenceAnchorError(f"existing external anchor conflicts: {target.name}")
        else:
            atomic_write_json(target, anchor, overwrite=False)
            created += 1
    return {"status": "PASS", "integrity": "anchor records created; independent protection not verified",
            "anchored": len(completed), "created": created, "unanchored_completed_units": [],
            "pending_units": sorted({str(u["unit_key"]) for u in manifest.get("units", [])}
                                    - {p.stem for p in completed}), "anchor_dir": str(anchor_dir)}


def verify_anchor_dir(run_root: Path, manifest_path: Path, anchor_dir: Path) -> dict:
    if anchor_dir.is_symlink():
        raise EvidenceAnchorError("external anchor directory must not be a symlink")
    run_root, manifest_path, anchor_dir = run_root.resolve(), manifest_path.resolve(), anchor_dir.resolve()
    if not anchor_dir.is_dir() or anchor_dir.is_symlink():
        raise EvidenceAnchorError("external anchor directory is missing or unsafe")
    if anchor_dir == run_root or run_root in anchor_dir.parents:
        raise EvidenceAnchorError("anchor directory is inside mutable run root")
    manifest = _validated_manifest(run_root, manifest_path)
    files = sorted(anchor_dir.glob("*.json"))
    seen = set()
    for path in files:
        result = verify_anchor(run_root, manifest_path, path)
        if result["unit_key"] in seen:
            raise EvidenceAnchorError("duplicate unit anchor")
        seen.add(result["unit_key"])
    completed = {p.stem for p in (run_root / "results" / "units").glob("*.json")}
    return {"status": "PASS", "integrity": "compared_to_supplied_anchor_directory; independent protection not verified", "verified": len(seen), "unanchored_units": sorted(completed - seen),
            "missing_anchor_units": sorted({str(u["unit_key"]) for u in manifest.get("units", [])} - seen)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for action in ("create", "verify"):
        p = sub.add_parser(action)
        p.add_argument("--run-root", type=Path, required=True)
        p.add_argument("--manifest", type=Path, required=True)
        p.add_argument("--anchor-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = create_anchors(args.run_root, args.manifest, args.anchor_dir) if args.action == "create" else verify_anchor_dir(args.run_root, args.manifest, args.anchor_dir)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
