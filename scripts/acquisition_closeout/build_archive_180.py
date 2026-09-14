"""Build and verify the deterministic off-Pod export archive of the completed 180-episode acquisition.

Never modifies scientific data. Writes the archive, its .sha256 file, and a validation report under
/workspace/persistent/exports/; staging and test extraction live under /tmp and are removed afterwards.
"""
import gzip
import hashlib
import json
import pathlib
import shutil
import subprocess
import sys
import tarfile
from datetime import datetime, timezone

ROOT = pathlib.Path("/workspace/persistent/rq1-protocol-migration")
RUN_ID = "rq1-acquisition-gemma4-12b"
RUN = ROOT / "results" / "final" / RUN_ID
NAME = "rq1-acquisition-180-final-20260914"
EXPORTS = pathlib.Path("/workspace/persistent/exports")
ARCHIVE = EXPORTS / f"{NAME}.tar.gz"
SHA_FILE = EXPORTS / f"{NAME}.tar.gz.sha256"
VALIDATION = EXPORTS / f"{NAME}.validation.json"
STAGE = pathlib.Path("/tmp") / f"{NAME}-stage"
EXTRACT = pathlib.Path("/tmp") / f"{NAME}-validate"
CLOSEOUT = ROOT / "artifacts" / "acquisition-closeout" / RUN_ID
FROZEN_SHA = "8bd452e76120da21d721c6e894d1ce5af4912ca9"
FIXED_MTIME = int(datetime(2026, 9, 14, tzinfo=timezone.utc).timestamp())


def sha(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


if ARCHIVE.exists() or SHA_FILE.exists():
    sys.exit(f"refusing to overwrite an existing export: {ARCHIVE}")
closeout = json.loads((CLOSEOUT / "closeout-manifest.json").read_text(encoding="utf-8"))
if closeout.get("closeout_passed") is not True:
    sys.exit("closeout did not pass; refusing to archive")

shutil.rmtree(STAGE, ignore_errors=True)
bundle_root = STAGE / NAME
(bundle_root / "repository-at-8bd452e").mkdir(parents=True)
git_archive = subprocess.run(["git", "archive", "--format=tar", FROZEN_SHA, "docs", "configs", "hermes/prompts", "AGENTS.md", "README.md"],
                             cwd=ROOT, capture_output=True, check=True)
with tarfile.open(fileobj=__import__("io").BytesIO(git_archive.stdout), mode="r:") as snapshot:
    snapshot.extractall(bundle_root / "repository-at-8bd452e", filter="data")

sources: list[tuple[pathlib.Path, str]] = []


def add_file(source: pathlib.Path, destination: str) -> None:
    if not source.is_file():
        sys.exit(f"required archive input missing: {source}")
    sources.append((source, f"{NAME}/{destination}"))


def add_tree(source: pathlib.Path, destination: str) -> None:
    for path in sorted(source.rglob("*")):
        if path.is_file():
            sources.append((path, f"{NAME}/{destination}/{path.relative_to(source).as_posix()}"))


add_tree(RUN, f"results/final/{RUN_ID}")
for relative in (
    "artifacts/task_manifests/frozen/acquisition-b64c383e0f335b90.json",
    "artifacts/task_manifests/proposals/acquisition-b64c383e0f335b90.json",
    "artifacts/task_manifests/proposal_archive/acquisition-b64c383e0f335b90.json",
    "artifacts/freezes/acquisition-environment-freeze.json",
    "artifacts/freezes/acquisition-protocol-freeze.json",
    "artifacts/approvals/acquisition/8bd452e76120/task-freeze.approval.json",
    "artifacts/approvals/acquisition/8bd452e76120/acquisition-environment.approval.json",
    "artifacts/approvals/acquisition/8bd452e76120/acquisition-protocol.approval.json",
    "artifacts/prelaunch/acquisition-check/prelaunch-acquisition-check-20260913-final-a/acquisition-check-report.json",
    "artifacts/prelaunch/acquisition-check/prelaunch-acquisition-check-20260913-final-a/check-plan.json",
    "artifacts/prelaunch/acquisition-check/prelaunch-acquisition-check-20260913-final-a/invocations.jsonl",
    "artifacts/prelaunch/production-preflight/preflight-20260913T091328Z.json",
    "artifacts/prelaunch/production-preflight/preflight-20260913T093430Z.json",
):
    add_file(ROOT / relative, relative)
add_tree(CLOSEOUT, "closeout")
for name in ("rq1-acquisition-gemma4-12b.log", "rq1-acquisition-approval-and-freezes.log",
             "rq1-acquisition-plan-at-launch.json", "rq1-acquisition-preflight-at-launch.json"):
    add_file(pathlib.Path("/workspace/persistent/logs") / name, f"logs/{name}")
add_file(pathlib.Path("/workspace/persistent/artifacts/rq1_acquisition_launch.sh"), "logs/rq1_acquisition_launch.sh")
add_file(pathlib.Path("/workspace/persistent/backups/rq1-protocol-migration-final-8bd452e.bundle"), "git/rq1-protocol-migration-final-8bd452e.bundle")

metadata = {
    "schema_version": 1,
    "archive": f"{NAME}.tar.gz",
    "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "run_id": RUN_ID,
    "frozen_git_sha": FROZEN_SHA,
    "branch": "rq1-protocol-migration",
    "remote": "https://github.com/okayxsh/Hermes-Research-",
    "queue_hash": closeout["queue_hash"],
    "model": closeout["model"],
    "counts": closeout["counts"],
    "final_pool_hash": closeout["final_pool_hash"],
    "skills_per_family": closeout["skills_per_family"],
    "closeout_manifest_sha256": sha(CLOSEOUT / "closeout-manifest.json"),
    "excluded": ["Ollama model blobs", "virtual environments", "CUDA", "HuggingFace/SBERT caches", "ALFWorld dataset", "unrelated prelaunch artifacts"],
    "verification": "From the extraction root run: sha256sum -c rq1-acquisition-180-final-20260914/CONTENTS.sha256",
}
(bundle_root / "ARCHIVE-METADATA.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
(bundle_root / "README.txt").write_text(
    "RQ1 scientific acquisition rq1-acquisition-gemma4-12b (180 TRAIN episodes, completed 2026-09-13).\n"
    "results/final/: authoritative run directory (results.jsonl is the recovery authority).\n"
    "closeout/: closeout manifest, skill-yield report, result-directory SHA-256 listing.\n"
    "artifacts/: frozen task manifest, environment/protocol freezes, approvals, evidence and preflight reports.\n"
    "logs/: scientific console log, approval/freeze log, launch plan/preflight, launcher script.\n"
    "repository-at-8bd452e/: docs, configs, prompts at the frozen commit; git/ holds the full-history bundle.\n"
    "checkpoint.backup.json is a non-authoritative previous checkpoint generation (see closeout manifest).\n",
    encoding="utf-8")
add_tree(bundle_root, "")
sources = [(path, name.replace(f"{NAME}//", f"{NAME}/")) for path, name in sources]
sources.sort(key=lambda item: item[1])
if len({name for _, name in sources}) != len(sources):
    sys.exit("duplicate archive member names")

listing = {}
for line in (CLOSEOUT / "result-directory.sha256").read_text(encoding="utf-8").splitlines():
    digest, relative = line.split("  ", 1)
    listing[f"{NAME}/{relative}"] = digest
contents_lines = []
for path, name in sources:
    digest = listing.get(name) or sha(path)
    contents_lines.append(f"{digest}  {name}")
contents_path = bundle_root / "CONTENTS.sha256"
contents_path.write_text("\n".join(contents_lines) + "\n", encoding="utf-8")
sources.append((contents_path, f"{NAME}/CONTENTS.sha256"))

EXPORTS.mkdir(parents=True, exist_ok=True)
partial = ARCHIVE.with_name(ARCHIVE.name + ".partial")
with open(partial, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=6, mtime=0) as compressed:
    with tarfile.open(fileobj=compressed, mode="w|", format=tarfile.PAX_FORMAT) as archive:
        for path, name in sources:
            info = tarfile.TarInfo(name)
            info.size = path.stat().st_size
            info.mtime = FIXED_MTIME
            info.mode = 0o644
            info.uid = info.gid = 0
            info.uname = info.gname = "root"
            with open(path, "rb") as handle:
                archive.addfile(info, handle)
partial.rename(ARCHIVE)
archive_sha = sha(ARCHIVE)
SHA_FILE.write_text(f"{archive_sha}  {ARCHIVE.name}\n", encoding="utf-8")
print("ARCHIVE", ARCHIVE, ARCHIVE.stat().st_size, archive_sha, flush=True)

# ------------------------------------------------------------------ verification
report = {"archive": str(ARCHIVE), "size_bytes": ARCHIVE.stat().st_size, "sha256": archive_sha, "sha256_file": str(SHA_FILE)}
with tarfile.open(ARCHIVE, "r:gz") as archive:
    names = [member.name for member in archive.getmembers()]
report["members"] = len(names)
report["listing_matches_sources"] = sorted(names) == sorted(name for _, name in sources)
shutil.rmtree(EXTRACT, ignore_errors=True)
EXTRACT.mkdir(parents=True)
subprocess.run(["tar", "-xzf", str(ARCHIVE), "-C", str(EXTRACT)], check=True)
verify = subprocess.run(["sha256sum", "-c", "--quiet", f"{NAME}/CONTENTS.sha256"], cwd=EXTRACT, capture_output=True, text=True)
report["contents_sha256_verified"] = verify.returncode == 0
report["contents_sha256_failures"] = verify.stdout.strip().splitlines()[:10]
extracted_run = EXTRACT / NAME / "results" / "final" / RUN_ID
rows = [line for line in (extracted_run / "results.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
pool = json.loads((extracted_run / "skill_pool.json").read_text(encoding="utf-8"))
report["result_records"] = len(rows)
report["final_pool_size"] = pool.get("pool_size")
report["final_pool_hash"] = pool.get("pool_hash")
required = ["artifacts/freezes/acquisition-environment-freeze.json", "artifacts/freezes/acquisition-protocol-freeze.json",
            "artifacts/task_manifests/frozen/acquisition-b64c383e0f335b90.json",
            "artifacts/approvals/acquisition/8bd452e76120/task-freeze.approval.json",
            "artifacts/approvals/acquisition/8bd452e76120/acquisition-environment.approval.json",
            "artifacts/approvals/acquisition/8bd452e76120/acquisition-protocol.approval.json",
            "closeout/closeout-manifest.json", "closeout/yield-report.json", "logs/rq1-acquisition-gemma4-12b.log",
            f"results/final/{RUN_ID}/checkpoint.json", f"results/final/{RUN_ID}/checkpoint.backup.json",
            "git/rq1-protocol-migration-final-8bd452e.bundle", "repository-at-8bd452e/docs/ACQUISITION_RUNBOOK.md"]
report["required_present"] = {item: (EXTRACT / NAME / item).is_file() for item in required}
zero = [str(path.relative_to(EXTRACT)) for path in (EXTRACT / NAME).rglob("*") if path.is_file() and path.stat().st_size == 0]
report["zero_byte_files"] = {"count": len(zero), "examples": zero[:10],
                             "expected": "errors.jsonl and empty per-episode registry/hook logs are legitimately empty; every file is hash-verified"}
report["passed"] = (report["listing_matches_sources"] and report["contents_sha256_verified"] and len(rows) == 180
                    and pool.get("pool_size") == 34 and all(report["required_present"].values()))
shutil.rmtree(EXTRACT, ignore_errors=True)
shutil.rmtree(STAGE, ignore_errors=True)
report["temporary_extraction_removed"] = not EXTRACT.exists()
VALIDATION.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print("VALIDATION", json.dumps(report, sort_keys=True), flush=True)
sys.exit(0 if report["passed"] else 3)
