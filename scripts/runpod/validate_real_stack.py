#!/usr/bin/env python3
"""Fail-closed validation for an already provisioned RunPod machine.

This module performs validation only. It never creates a Pod, starts a Pod, or
launches the scientific acquisition/evaluation phases. Real ALFWorld and
Hermes checks are intentionally explicit and are not replaced by fake probes.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


class Validator:
    def __init__(self, root: Path, persistent_root: Path, backup_dir: Path | None) -> None:
        self.root = root.resolve()
        self.persistent_root = persistent_root.resolve()
        self.backup_dir = backup_dir.resolve() if backup_dir else None
        self.log_dir = self.root / "artifacts" / "runpod" / "validation"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checks: list[dict[str, Any]] = []
        self.counter = 0

    def check(self, name: str, ok: bool, detail: str, *, required: bool = True) -> None:
        self.checks.append({"name": name, "ok": bool(ok), "required": required, "detail": detail})
        marker = "PASS" if ok else ("BLOCKED" if required else "WARN")
        print(f"[{marker}] {name}: {detail}")

    def command(self, name: str, argv: list[str], *, timeout: int = 120, env: dict[str, str] | None = None) -> tuple[bool, str]:
        self.counter += 1
        log_path = self.log_dir / f"{self.counter:02d}-{name}.log"
        merged = os.environ.copy()
        if env:
            merged.update(env)
        try:
            completed = subprocess.run(
                argv,
                cwd=self.root,
                env=merged,
                text=True,
                capture_output=True,
                timeout=timeout,
                check=False,
            )
            output = (completed.stdout or "") + (completed.stderr or "")
            log_path.write_text(output, encoding="utf-8", errors="replace")
            return completed.returncode == 0, output.strip()
        except (OSError, subprocess.SubprocessError) as exc:
            log_path.write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
            return False, f"{type(exc).__name__}: {exc}"

    def cli(self, *args: str, timeout: int = 120, env: dict[str, str] | None = None) -> tuple[bool, str]:
        return self.command("rq1-" + args[0].replace("/", "-"), [sys.executable, "-m", "rq1.cli", *args], timeout=timeout, env=env)

    def report(self, status: str) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "generated_at": utc_now(),
            "status": status,
            "scientific_launch_ready": status == "passed",
            "repository": str(self.root),
            "persistent_root": str(self.persistent_root),
            "backup_dir": str(self.backup_dir) if self.backup_dir else None,
            "python": sys.version,
            "platform": platform.platform(),
            "checks": self.checks,
            "logs": str(self.log_dir.relative_to(self.root)).replace("\\", "/"),
            "scientific_invariants": {
                "valid_unseen_accessed": False,
                "final_experiment_started": False,
                "fake_evidence_promotes_real_readiness": False,
            },
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the real RQ1 stack without launching the experiment.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--persistent-root", type=Path, default=Path("/workspace/persistent"))
    parser.add_argument("--backup-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="describe checks without executing them")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.repo_root.resolve()
    persistent = args.persistent_root.resolve()
    backup = args.backup_dir.resolve() if args.backup_dir else None
    if args.dry_run:
        print(json.dumps({"dry_run": True, "checks": [
            "GPU visibility and resources", "Python 3.11 and locked environment", "Git commit",
            "ALFWorld 0.4.2/data/valid_seen real smoke", "Ollama hermes3:8b READY smoke",
            "Hermes CLI and real integration capability", "checkpoint and interruption resume",
            "persistent output and backup paths",
        ]}, indent=2))
        return 0

    validator = Validator(root, persistent, backup)
    if not root.is_dir():
        print(f"repository does not exist: {root}", file=sys.stderr)
        return 2

    system = platform.system()
    architecture = platform.machine().lower()
    validator.check("linux", system == "Linux", system)
    validator.check("x86_64", architecture in {"x86_64", "amd64"}, architecture)

    nvidia = shutil.which("nvidia-smi")
    if nvidia:
        ok, output = validator.command("nvidia-smi", [nvidia, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"])
        validator.check("gpu", ok, output.splitlines()[0] if output else "nvidia-smi returned no GPU")
    else:
        validator.check("gpu", False, "nvidia-smi is not on PATH")

    free_bytes = shutil.disk_usage(persistent if persistent.exists() else root).free
    validator.check("disk", free_bytes >= 80 * 1024**3, f"{free_bytes / 1024**3:.1f} GiB free; 80 GiB required")
    meminfo = Path("/proc/meminfo")
    memory_gib = None
    if meminfo.exists():
        for line in meminfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("MemTotal:"):
                memory_gib = int(line.split()[1]) / 1024 / 1024
                break
    validator.check("ram", memory_gib is not None and memory_gib >= 32, f"{memory_gib:.1f} GiB detected; 32 GiB required" if memory_gib else "MemTotal unavailable")

    commit_ok, commit = validator.command("git-commit", ["git", "rev-parse", "--verify", "HEAD"], timeout=15)
    validator.check("git_commit", commit_ok and bool(commit), commit or "repository has no resolvable commit")

    py_ok = sys.version_info[:2] == (3, 11)
    validator.check("python311", py_ok, platform.python_version())
    uv = shutil.which("uv")
    if uv:
        uv_ok, uv_output = validator.command("uv-lock", [uv, "lock", "--check"], timeout=120)
        validator.check("locked_environment", uv_ok, uv_output or "uv lock check passed")
    else:
        validator.check("locked_environment", False, "uv is not on PATH")

    config_ok, config_output = validator.cli("validate-config")
    validator.check("config", config_ok, config_output[-300:] if config_output else "configuration validation failed")

    package_ok, package_output = validator.command("alfworld-package", [sys.executable, "-c", "import importlib.metadata as m; print(m.version('alfworld'))"], timeout=30)
    validator.check("alfworld_042", package_ok and package_output.strip() == "0.4.2", package_output or "alfworld package unavailable")

    # The first probe imports ALFWorld and scans the persistent network volume;
    # allow enough time for a cold start on RunPod's mounted storage.
    capability_ok, capability_output = validator.cli("alfworld", "capabilities", timeout=1800)
    validator.check("alfworld_capabilities", capability_ok, capability_output[-300:] if capability_output else "real adapter capability probe failed")
    index_ok, index_output = validator.cli("alfworld", "index", "--split", "valid_seen", timeout=1800)
    validator.check("alfworld_valid_seen_index", index_ok, "valid_seen index constructed" if index_ok else index_output[-300:])
    smoke_ok, smoke_output = validator.cli("alfworld", "smoke-test", "--split", "valid_seen", "--yes", timeout=1800)
    validator.check("alfworld_real_smoke", smoke_ok, "real start/step/status/reset/abort completed" if smoke_ok else smoke_output[-300:])

    model_smoke = (
        "import json,urllib.request; "
        "p=json.dumps({'model':'hermes3:8b','prompt':'Reply with READY only.','stream':False}).encode(); "
        "r=json.load(urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:11434/api/generate',data=p,headers={'Content-Type':'application/json'}),timeout=60)); "
        "assert r.get('response','').strip()=='READY', repr(r.get('response'))"
    )
    ollama_ok, ollama_output = validator.command("ollama-ready", [sys.executable, "-c", model_smoke], timeout=180)
    validator.check("ollama_hermes3_ready", ollama_ok, "hermes3:8b returned READY" if ollama_ok else ollama_output[-300:])
    hermes = shutil.which("hermes")
    validator.check("hermes_cli", hermes is not None, hermes or "hermes executable unavailable")
    if hermes:
        version_ok, version_output = validator.command("hermes-version", [hermes, "--version"], timeout=30)
        help_ok, help_output = validator.command("hermes-help", [hermes, "--help"], timeout=30)
        validator.check("hermes_version", version_ok, version_output[:200])
        validator.check("hermes_help", help_ok, "help command completed" if help_ok else help_output[-300:])
    hermes_cap_ok, hermes_cap_output = validator.cli("hermes-capabilities")
    validator.check("hermes_capabilities", hermes_cap_ok, hermes_cap_output[-300:] if hermes_cap_output else "probe failed")
    real_hermes_ok, real_hermes_output = validator.cli("verify-hermes-integration", "--mode", "real", timeout=180)
    validator.check("hermes_real_integration", real_hermes_ok, "real Hermes integration evidence passed" if real_hermes_ok else "blocked: real Hermes dispatch/skill observation is not validated; " + real_hermes_output[-250:])

    retrieval_ok, retrieval_output = validator.command("retrieval", [sys.executable, "-c", "import rq1.hermes, rq1.skills; print('native Hermes skill boundary available')"], timeout=30)
    validator.check("retrieval_components", retrieval_ok, retrieval_output or "retrieval imports failed")

    validation_id = f"runpod-validation-{int(time.time())}"
    env = {"RQ1_ALFWORLD_DATA_DIR": str(persistent / "alfworld_data")}
    first_args = ["experiment", "checkpoint-test", "--run-id", validation_id, "--max-runs", "3", "--total-runs", "6"]
    if backup:
        first_args += ["--backup-dir", str(backup), "--require-backup"]
    first_ok, first_output = validator.cli(*first_args, timeout=300, env=env)
    resume_args = ["experiment", "checkpoint-test", "--run-id", validation_id, "--resume", "--max-runs", "3", "--total-runs", "6"]
    if backup:
        resume_args += ["--backup-dir", str(backup), "--require-backup"]
    resume_ok, resume_output = validator.cli(*resume_args, timeout=300, env=env)
    validator.check("checkpoint_resume", first_ok and resume_ok, "synthetic checkpoint and resume completed" if first_ok and resume_ok else (resume_output or first_output)[-300:])

    interrupt_id = f"runpod-interrupt-{int(time.time())}"
    interrupt_args = [sys.executable, "-m", "rq1.cli", "experiment", "checkpoint-test", "--run-id", interrupt_id, "--total-runs", "6", "--delay-ms", "500"]
    try:
        process = subprocess.Popen(interrupt_args, cwd=validator.root, env={**os.environ, **env}, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        time.sleep(0.75)
        process.send_signal(signal.SIGINT)
        process.wait(timeout=30)
        interrupted_ok = process.returncode in {0, 130}
        resume_interrupt_ok, resume_interrupt_output = validator.cli("experiment", "checkpoint-test", "--run-id", interrupt_id, "--resume", "--total-runs", "6", "--delay-ms", "500", timeout=300, env=env)
        validator.check("interrupt_resume", interrupted_ok and resume_interrupt_ok, "SIGINT followed by resume completed" if interrupted_ok and resume_interrupt_ok else resume_interrupt_output[-300:])
    except (OSError, subprocess.SubprocessError) as exc:
        validator.check("interrupt_resume", False, f"{type(exc).__name__}: {exc}")

    for label, path in (("persistent_output", root / "results"), ("backup_path", backup)):
        if path is None:
            validator.check(label, True, "not configured; local durable state remains authoritative", required=False)
            continue
        try:
            path.mkdir(parents=True, exist_ok=True)
            probe = path / ".runpod-write-test"
            probe.write_text("ok\n", encoding="utf-8")
            probe.unlink()
            validator.check(label, True, str(path))
        except OSError as exc:
            validator.check(label, False, f"{path}: {exc}")

    failed = [item for item in validator.checks if item["required"] and not item["ok"]]
    status = "blocked" if failed else "passed"
    report = validator.report(status)
    report_path = validator.root / "artifacts" / "runpod" / "validation-report.json"
    temporary = report_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(report_path)
    if status == "passed":
        marker = validator.persistent_root / "rq1-validation-passed.json"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps({"schema_version": 1, "validated_at": utc_now(), "git_commit": commit, "report": str(report_path.relative_to(validator.root))}, indent=2) + "\n", encoding="utf-8")
        print(f"Validation passed; launch marker: {marker}")
        return 0
    print(f"Validation is blocked; no launch marker was written. Report: {report_path}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
