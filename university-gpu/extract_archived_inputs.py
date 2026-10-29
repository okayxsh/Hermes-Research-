"""Copy three verified historical inputs out of the archived evaluation, without modifying it."""
import argparse
import hashlib
import json
import tarfile
from pathlib import Path

PREFIX = "rq1-evaluation-360-final/repo/artifacts/"
MEMBERS = {
    "evaluation-task-manifest.json": PREFIX + "task_manifests/frozen_evaluation/evaluation-amended-b83a2ecacba19992.json",
    "controlled-failures.json": PREFIX + "task_manifests/frozen_evaluation/controlled-failures-69e1d7e924db8991.json",
    "evaluation-libraries.json": PREFIX + "evaluation-libraries/libraries-0030f89e76c6bc4c.json",
}
ARCHIVE_SHA256 = "d52ed82e01a073f8c3382711d03f65916cb41ca89bc0b773e6845817bd37e4fd"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--archive-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("output directory already exists")
    archive_bytes = args.archive.read_bytes()
    if hashlib.sha256(archive_bytes).hexdigest() != ARCHIVE_SHA256:
        raise SystemExit("archive SHA-256 mismatch")
    evidence = json.loads(args.archive_manifest.read_text(encoding="utf-8"))
    if evidence.get("archive_sha256") != ARCHIVE_SHA256:
        raise SystemExit("archive manifest does not match frozen archive hash")
    expected = {row["path"]: row for row in evidence["members"]}
    outputs = {}
    with tarfile.open(args.archive, "r:gz") as archive:
        for filename, member_name in MEMBERS.items():
            member = archive.getmember(member_name)
            if not member.isfile() or member_name not in expected:
                raise SystemExit(f"missing regular archived member: {member_name}")
            data = archive.extractfile(member).read()
            sha = hashlib.sha256(data).hexdigest()
            if sha != expected[member_name]["sha256"] or len(data) != expected[member_name]["size"]:
                raise SystemExit(f"archived member hash mismatch: {member_name}")
            outputs[filename] = (data, sha)
    args.output.mkdir(parents=True)
    for filename, (data, _) in outputs.items():
        with (args.output / filename).open("xb") as handle:
            handle.write(data)
    ledger = {"archive_sha256": ARCHIVE_SHA256, "files": {name: value[1] for name, value in outputs.items()},
              "scope": "historical valid-unseen inputs only; pilot inputs and new relevance ratings remain required"}
    (args.output / "SOURCE-HASHES.json").write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(ledger, indent=2))


if __name__ == "__main__":
    main()
