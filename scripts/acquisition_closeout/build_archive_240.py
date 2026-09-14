"""Build and verify the deterministic off-Pod export archive of the complete 240-episode acquisition.

Never modifies scientific data.  Writes the archive, its .sha256 file, an adjacent content
manifest, and a validation report under /workspace/persistent/exports/, plus a full-history
git bundle under /workspace/persistent/backups/.  Staging and the test extraction live under
/tmp and are removed afterwards; the earlier 180 archive is never touched.
"""
import gzip
import hashlib
import io
import json
import pathlib
import shutil
import subprocess
import sys
import tarfile
from datetime import datetime, timezone

ROOT = pathlib.Path("/workspace/persistent/rq1-protocol-migration")
PERSISTENT = pathlib.Path("/workspace/persistent")
PARENT_ID = "rq1-acquisition-gemma4-12b"
EXT_ID = "rq1-acquisition-gemma4-12b-ext-181-240"
PARENT_SHA = "8bd452e76120da21d721c6e894d1ce5af4912ca9"
EXT_SHA = "670627a9cd8800ce64e369a17263e3a3b0486dd1"
NAME = "rq1-acquisition-240-final-20260914"
EXPORTS = PERSISTENT / "exports"
ARCHIVE = EXPORTS / f"{NAME}.tar.gz"
SHA_FILE = EXPORTS / f"{NAME}.tar.gz.sha256"
MANIFEST_FILE = EXPORTS / f"{NAME}.manifest.json"
VALIDATION = EXPORTS / f"{NAME}.validation.json"
ARCHIVE_180 = EXPORTS / "rq1-acquisition-180-final-20260914.tar.gz"
STAGE = pathlib.Path("/tmp") / f"{NAME}-stage"
EXTRACT = pathlib.Path("/tmp") / f"{NAME}-validate"
CLOSEOUT_240 = ROOT / "artifacts" / "acquisition-closeout" / "rq1-acquisition-240-final"
CLOSEOUT_180 = ROOT / "artifacts" / "acquisition-closeout" / PARENT_ID
VALIDATION_PACKAGE = ROOT / "artifacts" / "skill-validation" / "rq1-acquisition-240"
FIXED_MTIME = int(datetime(2026, 9, 14, tzinfo=timezone.utc).timestamp())
FAMILY_SKILLS = {"pick_and_place": 11, "pick_two_and_place": 8, "look_at_object": 4, "clean_and_place": 9, "heat_and_place": 15, "cool_and_place": 3}


def sha(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*arguments: str, binary: bool = False):
    completed = subprocess.run(["git", *arguments], cwd=ROOT, capture_output=True, check=True)
    return completed.stdout if binary else completed.stdout.decode("utf-8").strip()


for path in (ARCHIVE, SHA_FILE, MANIFEST_FILE, VALIDATION):
    if path.exists():
        sys.exit(f"refusing to overwrite an existing export: {path}")
combined = json.loads((CLOSEOUT_240 / "combined-closeout-manifest.json").read_text(encoding="utf-8"))
hard_cap = json.loads((CLOSEOUT_240 / "acquisition-hard-cap-verification.json").read_text(encoding="utf-8"))
if combined.get("closeout_passed") is not True or hard_cap.get("passed") is not True:
    sys.exit("combined closeout or hard-cap verification did not pass; refusing to archive")
head = git("rev-parse", "HEAD")
if git("status", "--porcelain"):
    sys.exit("repository working tree is not clean; commit the closeout tooling first")

# ------------------------------------------------------------------ full-history git bundle
BUNDLE = PERSISTENT / "backups" / f"rq1-protocol-migration-full-{head[:7]}.bundle"
if not BUNDLE.exists():
    git("bundle", "create", str(BUNDLE), "rq1-protocol-migration")
git("bundle", "verify", str(BUNDLE))

# ------------------------------------------------------------------ staging: snapshots and metadata
shutil.rmtree(STAGE, ignore_errors=True)
bundle_root = STAGE / NAME
bundle_root.mkdir(parents=True)
for commit, paths in ((PARENT_SHA, ["docs", "configs", "hermes/prompts", "AGENTS.md", "README.md"]),
                      (EXT_SHA, ["docs", "configs", "hermes/prompts", "AGENTS.md", "README.md"]),
                      (head, ["docs", "configs", "hermes/prompts", "scripts/acquisition_closeout", "AGENTS.md", "README.md"])):
    target = bundle_root / f"repository-at-{commit[:7]}"
    target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(git("archive", "--format=tar", commit, *paths, binary=True)), mode="r:") as snapshot:
        snapshot.extractall(target, filter="data")

sources: list[tuple[pathlib.Path, str]] = []


def add_file(source: pathlib.Path, destination: str) -> None:
    if not source.is_file():
        sys.exit(f"required archive input missing: {source}")
    sources.append((source, f"{NAME}/{destination}"))


def add_tree(source: pathlib.Path, destination: str) -> None:
    if not source.is_dir():
        sys.exit(f"required archive directory missing: {source}")
    for path in sorted(source.rglob("*")):
        if path.is_file():
            sources.append((path, f"{NAME}/{destination}/{path.relative_to(source).as_posix()}"))


def add_repo(relative: str) -> None:
    path = ROOT / relative
    add_tree(path, relative) if path.is_dir() else add_file(path, relative)


# A. parent run 1-180
add_repo(f"results/final/{PARENT_ID}")
for relative in (
    "artifacts/task_manifests/frozen/acquisition-b64c383e0f335b90.json",
    "artifacts/task_manifests/proposals/acquisition-b64c383e0f335b90.json",
    "artifacts/task_manifests/proposal_archive/acquisition-b64c383e0f335b90.json",
    "artifacts/freezes/acquisition-environment-freeze.json",
    "artifacts/freezes/acquisition-protocol-freeze.json",
    "artifacts/approvals/acquisition/8bd452e76120",
    "artifacts/prelaunch/acquisition-check/prelaunch-acquisition-check-20260913-final-a/acquisition-check-report.json",
    "artifacts/prelaunch/acquisition-check/prelaunch-acquisition-check-20260913-final-a/check-plan.json",
    "artifacts/prelaunch/acquisition-check/prelaunch-acquisition-check-20260913-final-a/invocations.jsonl",
    "artifacts/prelaunch/production-preflight",
    f"artifacts/acquisition-closeout/{PARENT_ID}",
):
    add_repo(relative)
for name in ("rq1-acquisition-gemma4-12b.log", "rq1-acquisition-approval-and-freezes.log", "rq1-acquisition-plan-at-launch.json",
             "rq1-acquisition-preflight-at-launch.json", "rq1-acquisition-closeout.log", "rq1-acquisition-archive.log"):
    add_file(PERSISTENT / "logs" / name, f"logs/parent/{name}")
add_file(PERSISTENT / "artifacts" / "rq1_acquisition_launch.sh", "logs/parent/rq1_acquisition_launch.sh")
add_file(EXPORTS / "rq1-acquisition-180-final-20260914.tar.gz.sha256", "exports-180/rq1-acquisition-180-final-20260914.tar.gz.sha256")
add_file(EXPORTS / "rq1-acquisition-180-final-20260914.validation.json", "exports-180/rq1-acquisition-180-final-20260914.validation.json")

# B. extension 181-240
add_repo(f"results/final/{EXT_ID}")
for relative in (
    "artifacts/task_manifests/frozen_extension/acquisition-extension-4b9e3c8cda4dd809.json",
    "artifacts/task_manifests/extension_proposals/acquisition-extension-4b9e3c8cda4dd809.json",
    "artifacts/task_manifests/extension_proposal_archive/acquisition-extension-4b9e3c8cda4dd809.json",
    "artifacts/freezes/acquisition-extension-environment-freeze.json",
    "artifacts/freezes/acquisition-extension-protocol-freeze.json",
    "artifacts/approvals/acquisition-extension/670627a9cd88",
    f"artifacts/acquisition-extension/{PARENT_ID}",
    "artifacts/prelaunch/acquisition-extension-check/prelaunch-acquisition-extension-check-20260914-a",
    "artifacts/prelaunch/extension-preflight",
):
    add_repo(relative)
for name in (f"{EXT_ID}.log", "rq1-extension-prepare.log", "rq1-extension-approval-and-freezes.log", "rq1-extension-full-suite.log",
             "rq1-acquisition-240-closeout.log"):
    add_file(PERSISTENT / "logs" / name, f"logs/extension/{name}")
add_tree(PERSISTENT / "logs" / "rq1-extension-prepare", "logs/extension/rq1-extension-prepare")
add_tree(PERSISTENT / "logs" / "rq1-extension-approval", "logs/extension/rq1-extension-approval")
for name in ("rq1_acquisition_extension_launch.sh", "extension_prepare.sh", "extension_approval_and_freezes.sh", "extension_approve.py",
             "verify_extension_freezes.py", "verify_extension_units.py"):
    add_file(PERSISTENT / "artifacts" / name, f"logs/extension/scripts/{name}")
if (PERSISTENT / "logs" / "rq1-hard-cap-full-suite.log").is_file():
    add_file(PERSISTENT / "logs" / "rq1-hard-cap-full-suite.log", "logs/closeout/rq1-hard-cap-full-suite.log")

# C. combined closeout, validation package, feasibility analysis
add_repo("artifacts/acquisition-closeout/rq1-acquisition-240-final")
add_repo("artifacts/skill-validation/rq1-acquisition-240")

# D. git history
add_file(BUNDLE, f"git/{BUNDLE.name}")
add_file(PERSISTENT / "backups" / "rq1-protocol-migration-final-8bd452e.bundle", "git/rq1-protocol-migration-final-8bd452e.bundle")
add_file(PERSISTENT / "backups" / "rq1-protocol-migration-extension-prep-670627a.bundle", "git/rq1-protocol-migration-extension-prep-670627a.bundle")
git_metadata = {
    "schema_version": 1,
    "branch": "rq1-protocol-migration",
    "remote": "https://github.com/okayxsh/Hermes-Research-",
    "parent_run_commit": PARENT_SHA,
    "extension_run_commit": EXT_SHA,
    "closeout_commit": head,
    "log": git("log", "--format=%H %ad %s", "--date=iso", "-20").splitlines(),
    "full_bundle": {"file": f"git/{BUNDLE.name}", "sha256": sha(BUNDLE), "heads": git("bundle", "list-heads", str(BUNDLE)).splitlines(),
                    "restore": f"git clone -b rq1-protocol-migration {BUNDLE.name} rq1-protocol-migration"},
}
(bundle_root / "GIT-METADATA.json").write_text(json.dumps(git_metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
metadata = {
    "schema_version": 1,
    "archive": f"{NAME}.tar.gz",
    "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "description": "Complete RQ1 train-only acquisition: parent run units 1-180 plus balanced extension units 181-240 (hard cap reached).",
    "runs": combined["runs"],
    "totals": combined["totals"],
    "final_skill_pool": combined["final_skill_pool"],
    "model": combined["model"],
    "environment": combined["environment"],
    "acquisition_protocol": combined["acquisition_protocol"],
    "combined_closeout_manifest_sha256": sha(CLOSEOUT_240 / "combined-closeout-manifest.json"),
    "human_validation": combined["human_validation"],
    "evaluation_started": False,
    "new_library_sizes_frozen": False,
    "further_acquisition_authorized": False,
    "excluded": ["Ollama model blobs", "Python virtual environments", "CUDA", "HuggingFace/SBERT caches", "ALFWorld dataset",
                 "prelaunch capability checks unrelated to the freezes", "the 180 archive itself (only its hash and validation report)"],
    "verification": f"From the extraction root: sha256sum -c {NAME}/CONTENTS.sha256",
}
(bundle_root / "ARCHIVE-METADATA.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
(bundle_root / "README.txt").write_text(
    "RQ1 complete scientific acquisition (240 ALFWorld TRAIN episodes, 40 per family).\n"
    f"results/final/{PARENT_ID}/: parent run, units 1-180 (commit 8bd452e).\n"
    f"results/final/{EXT_ID}/: balanced extension, units 181-240 (commit 670627a).\n"
    "artifacts/: frozen queues, freezes, approvals, prelaunch evidence, closeouts (180 and 240), starting pool,\n"
    "  the raw 50-skill pool snapshot, yield report, library-feasibility analysis, and the human skill-validation package.\n"
    "logs/: scientific console logs, approval/freeze logs, preflight/plan outputs, launcher scripts.\n"
    "repository-at-*/: docs, configs, prompts at the parent, extension, and closeout commits; git/ holds a full-history bundle.\n"
    "checkpoint.backup.json files are non-authoritative previous checkpoint generations (see the closeout manifests).\n"
    "Human validation is PENDING. No evaluation has run. No further acquisition is authorized.\n",
    encoding="utf-8")
add_tree(bundle_root, "")
sources = [(path, name.replace(f"{NAME}//", f"{NAME}/")) for path, name in sources]
sources.sort(key=lambda item: item[1])
if len({name for _, name in sources}) != len(sources):
    sys.exit("duplicate archive member names")

# ------------------------------------------------------------------ content manifest and hash list
known = {}
for listing_path in (CLOSEOUT_180 / "result-directory.sha256", CLOSEOUT_240 / "extension-result-directory.sha256"):
    for line in listing_path.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        known[f"{NAME}/{relative}"] = digest
members = [{"path": name, "size": path.stat().st_size, "sha256": known.get(name) or sha(path)} for path, name in sources]
important = [
    f"results/final/{PARENT_ID}/results.jsonl", f"results/final/{PARENT_ID}/checkpoint.json", f"results/final/{PARENT_ID}/skill_pool.json",
    f"results/final/{EXT_ID}/results.jsonl", f"results/final/{EXT_ID}/checkpoint.json", f"results/final/{EXT_ID}/skill_pool.json",
    "artifacts/acquisition-closeout/rq1-acquisition-240-final/combined-closeout-manifest.json",
    "artifacts/acquisition-closeout/rq1-acquisition-240-final/final-skill-pool-50.json",
    "artifacts/acquisition-closeout/rq1-acquisition-240-final/acquisition-yield-240.json",
    "artifacts/acquisition-closeout/rq1-acquisition-240-final/pre-evaluation-library-feasibility.md",
    "artifacts/acquisition-closeout/rq1-acquisition-240-final/acquisition-hard-cap-verification.json",
    f"artifacts/acquisition-closeout/{PARENT_ID}/closeout-manifest.json",
    "artifacts/skill-validation/rq1-acquisition-240/skill-validation-review.csv",
    "artifacts/skill-validation/rq1-acquisition-240/skill-validation-review.json",
    "artifacts/skill-validation/rq1-acquisition-240/skill-validation-instructions.md",
    "artifacts/freezes/acquisition-environment-freeze.json", "artifacts/freezes/acquisition-protocol-freeze.json",
    "artifacts/freezes/acquisition-extension-environment-freeze.json", "artifacts/freezes/acquisition-extension-protocol-freeze.json",
    "artifacts/task_manifests/frozen/acquisition-b64c383e0f335b90.json",
    "artifacts/task_manifests/frozen_extension/acquisition-extension-4b9e3c8cda4dd809.json",
    "artifacts/approvals/acquisition/8bd452e76120/task-freeze.approval.json",
    "artifacts/approvals/acquisition/8bd452e76120/acquisition-environment.approval.json",
    "artifacts/approvals/acquisition/8bd452e76120/acquisition-protocol.approval.json",
    "artifacts/approvals/acquisition-extension/670627a9cd88/extension-task-freeze.approval.json",
    "artifacts/approvals/acquisition-extension/670627a9cd88/acquisition-extension-environment.approval.json",
    "artifacts/approvals/acquisition-extension/670627a9cd88/acquisition-extension-protocol.approval.json",
    f"logs/parent/{PARENT_ID}.log", f"logs/extension/{EXT_ID}.log",
    f"git/{BUNDLE.name}", "GIT-METADATA.json", "ARCHIVE-METADATA.json",
    "repository-at-670627a/docs/decisions/011-acquisition-extension-181-240.md", "repository-at-670627a/docs/ACQUISITION_RUNBOOK.md",
]
by_name = {member["path"]: member for member in members}
missing = [item for item in important if f"{NAME}/{item}" not in by_name]
if missing:
    sys.exit(f"important archive members missing: {missing}")
content_manifest = {
    "schema_version": 1,
    "kind": "rq1-acquisition-240-archive-content-manifest",
    "archive": f"{NAME}.tar.gz",
    "created_at": metadata["created_at"],
    "total_members": len(members) + 2,
    "members_listed": len(members),
    "unlisted_members": [f"{NAME}/ARCHIVE-MANIFEST.json", f"{NAME}/CONTENTS.sha256"],
    "total_listed_bytes": sum(member["size"] for member in members),
    "scientific_result_records": {"parent": combined["runs"]["parent"]["counts"]["records"], "extension": combined["runs"]["extension"]["counts"]["records"],
                                  "total": combined["totals"]["episodes"]},
    "totals": combined["totals"],
    "final_skill_pool": combined["final_skill_pool"],
    "provenance": {
        "parent": {"run_id": PARENT_ID, "git_sha": PARENT_SHA, "queue_sha256": combined["runs"]["parent"]["queue_sha256"],
                   "closeout_manifest_sha256": combined["runs"]["parent"]["closeout_manifest"]["sha256"]},
        "extension": {"run_id": EXT_ID, "git_sha": EXT_SHA, "queue_sha256": combined["runs"]["extension"]["queue_sha256"],
                      "starting_pool_hash": combined["runs"]["extension"]["starting_pool_hash"]},
        "closeout_commit": head,
        "combined_closeout_manifest_sha256": metadata["combined_closeout_manifest_sha256"],
    },
    "important_artifacts": {item: by_name[f"{NAME}/{item}"] for item in important},
    "members": members,
}
(bundle_root / "ARCHIVE-MANIFEST.json").write_text(json.dumps(content_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
sources.append((bundle_root / "ARCHIVE-MANIFEST.json", f"{NAME}/ARCHIVE-MANIFEST.json"))
contents_path = bundle_root / "CONTENTS.sha256"
contents_path.write_text("\n".join([*(f"{member['sha256']}  {member['path']}" for member in members),
                                    f"{sha(bundle_root / 'ARCHIVE-MANIFEST.json')}  {NAME}/ARCHIVE-MANIFEST.json"]) + "\n", encoding="utf-8")
sources.append((contents_path, f"{NAME}/CONTENTS.sha256"))

# ------------------------------------------------------------------ deterministic archive
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
MANIFEST_FILE.write_text(json.dumps({**content_manifest, "archive_file": {"path": str(ARCHIVE), "size_bytes": ARCHIVE.stat().st_size, "sha256": archive_sha,
                                                                         "sha256_file": str(SHA_FILE)}}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print("ARCHIVE", ARCHIVE, ARCHIVE.stat().st_size, archive_sha, flush=True)

# ------------------------------------------------------------------ verification by test extraction
report = {"archive": str(ARCHIVE), "size_bytes": ARCHIVE.stat().st_size, "sha256": archive_sha, "sha256_file": str(SHA_FILE),
          "content_manifest": {"path": str(MANIFEST_FILE), "sha256": sha(MANIFEST_FILE)}}
with tarfile.open(ARCHIVE, "r:gz") as archive:
    names = [member.name for member in archive.getmembers()]
report["archive_listing_opens"] = True
report["members"] = len(names)
report["listing_matches_sources"] = sorted(names) == sorted(name for _, name in sources) and len(names) == content_manifest["total_members"]
shutil.rmtree(EXTRACT, ignore_errors=True)
EXTRACT.mkdir(parents=True)
subprocess.run(["tar", "-xzf", str(ARCHIVE), "-C", str(EXTRACT)], check=True)
verify = subprocess.run(["sha256sum", "-c", "--quiet", f"{NAME}/CONTENTS.sha256"], cwd=EXTRACT, capture_output=True, text=True)
report["all_member_hashes_match"] = verify.returncode == 0
report["hash_failures"] = verify.stdout.strip().splitlines()[:10]
base = EXTRACT / NAME


def records(run: str) -> int:
    return sum(1 for line in (base / "results" / "final" / run / "results.jsonl").read_text(encoding="utf-8").splitlines() if line.strip())


pool_snapshot = json.loads((base / "artifacts/acquisition-closeout/rq1-acquisition-240-final/final-skill-pool-50.json").read_text(encoding="utf-8"))
extension_pool = json.loads((base / f"results/final/{EXT_ID}/skill_pool.json").read_text(encoding="utf-8"))
extracted_manifest = json.loads((base / "artifacts/acquisition-closeout/rq1-acquisition-240-final/combined-closeout-manifest.json").read_text(encoding="utf-8"))
report["parent_result_records"] = records(PARENT_ID)
report["extension_result_records"] = records(EXT_ID)
report["total_scientific_acquisition"] = report["parent_result_records"] + report["extension_result_records"]
report["final_pool_size"] = pool_snapshot["pool_size"]
report["final_pool_hash"] = pool_snapshot["pool_hash"]
report["family_counts"] = pool_snapshot["per_family"]
report["family_counts_match"] = pool_snapshot["per_family"] == FAMILY_SKILLS and extension_pool["pool_hash"] == pool_snapshot["pool_hash"] and extension_pool["pool_size"] == 50
report["combined_manifest_passed"] = extracted_manifest.get("closeout_passed") is True and extracted_manifest["totals"]["episodes"] == 240
groups = {
    "freezes": [item for item in important if item.startswith("artifacts/freezes/") or "/frozen" in item],
    "approvals": [item for item in important if "/approvals/" in item],
    "closeout_manifests": [f"artifacts/acquisition-closeout/{PARENT_ID}/closeout-manifest.json", "artifacts/acquisition-closeout/rq1-acquisition-240-final/combined-closeout-manifest.json"],
    "validation_package": [item for item in important if "skill-validation" in item],
    "reports": ["artifacts/acquisition-closeout/rq1-acquisition-240-final/acquisition-yield-240.json", "artifacts/acquisition-closeout/rq1-acquisition-240-final/acquisition-yield-240.md",
                "artifacts/acquisition-closeout/rq1-acquisition-240-final/pre-evaluation-library-feasibility.md"],
    "logs": [f"logs/parent/{PARENT_ID}.log", f"logs/extension/{EXT_ID}.log", "logs/extension/rq1-extension-approval-and-freezes.log", "logs/parent/rq1-acquisition-approval-and-freezes.log"],
    "git_metadata": ["GIT-METADATA.json", f"git/{BUNDLE.name}", "repository-at-8bd452e/docs/decisions/010-gemma-backbone-and-model-output-failures.md",
                     "repository-at-670627a/docs/decisions/011-acquisition-extension-181-240.md", f"repository-at-{head[:7]}/scripts/acquisition_closeout/closeout_240.py"],
}
report["required_present"] = {group: {item: (base / item).is_file() for item in items} for group, items in groups.items()}
heads = subprocess.run(["git", "bundle", "list-heads", str(base / "git" / BUNDLE.name)], capture_output=True, text=True)
report["git_bundle_lists_closeout_head"] = heads.returncode == 0 and head in heads.stdout
zero = [str(path.relative_to(EXTRACT)) for path in base.rglob("*") if path.is_file() and path.stat().st_size == 0]
report["zero_byte_files"] = {"count": len(zero), "examples": zero[:5],
                             "expected": "errors.jsonl, empty per-episode registry/hook logs, and empty .err captures are legitimately empty; every file is hash-verified"}
report["passed"] = (report["listing_matches_sources"] and report["all_member_hashes_match"] and report["parent_result_records"] == 180
                    and report["extension_result_records"] == 60 and report["final_pool_size"] == 50 and report["family_counts_match"]
                    and report["combined_manifest_passed"] and all(all(items.values()) for items in report["required_present"].values())
                    and report["git_bundle_lists_closeout_head"])
shutil.rmtree(EXTRACT, ignore_errors=True)
shutil.rmtree(STAGE, ignore_errors=True)
report["temporary_extraction_removed"] = not EXTRACT.exists() and not STAGE.exists()
report["archive_180_retained"] = ARCHIVE_180.is_file() and sha(ARCHIVE_180) == (EXPORTS / "rq1-acquisition-180-final-20260914.tar.gz.sha256").read_text().split()[0]
VALIDATION.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print("VALIDATION", json.dumps(report, sort_keys=True), flush=True)
sys.exit(0 if report["passed"] and report["archive_180_retained"] else 3)
