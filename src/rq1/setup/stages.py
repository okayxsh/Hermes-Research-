from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from urllib.error import URLError

from rq1.bridge.app import create_bridge_server
from rq1.bridge.environment import real_adapter_capability
from rq1.setup.models import ProbeResult, SetupOptions
from rq1.setup.probes import (
    REQUIRED_HOSTS,
    SUPPORTED_UBUNTU,
    command_probe,
    data_inventory,
    executable_in_venv,
    is_wsl,
    network_probe,
    nvidia_probe,
    python_executable,
    read_json_response,
    read_os_release,
    total_ram_gib,
    wsl_generation,
    write_machine_yaml,
)
from rq1.setup.runner import CommandRunner, redact, redact_command
from rq1.utils.time import utc_now


OLLAMA_INSTALL_URL = "https://ollama.com/install.sh"
HERMES_INSTALL_URL = "https://hermes-agent.nousresearch.com/install.sh"
UV_INSTALL_URL = "https://astral.sh/uv/install.sh"
OLLAMA_HOST = "http://127.0.0.1:11434"
PRIMARY_MODEL = "hermes3:8b"
FALLBACK_MODEL = "llama3.1:8b"
MINIMUM_CONTEXT = 65536
SYSTEM_PACKAGES = (
    "ca-certificates",
    "curl",
    "git",
    "xz-utils",
    "build-essential",
    "libffi-dev",
    "python3-dev",
)


class StageFailure(RuntimeError):
    def __init__(self, message: str, remediation: str | None = None) -> None:
        super().__init__(message)
        self.remediation = remediation


@dataclass
class StageOutcome:
    status: str = "passed"
    probes: list[ProbeResult] = field(default_factory=list)
    artifacts: list[Path] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    skip_reason: str | None = None
    remediation: str | None = None


@dataclass
class StageContext:
    root: Path
    options: SetupOptions
    runner: CommandRunner
    network: Callable[[str, int], ProbeResult] = network_probe
    http_json: Callable[[str, bytes | None, int], dict[str, Any]] = read_json_response

    @property
    def artifacts(self) -> Path:
        return self.root / "artifacts"

    @property
    def data_dir(self) -> Path:
        configured = os.environ.get("RQ1_ALFWORLD_DATA_DIR")
        return Path(configured).expanduser() if configured else Path.home() / ".cache" / "rq1-experiment" / "alfworld"

    def portable(self, path: Path) -> str:
        resolved = path.expanduser().resolve()
        home = Path.home().resolve()
        try:
            return "$REPO/" + resolved.relative_to(self.root.resolve()).as_posix()
        except ValueError:
            pass
        try:
            return "$HOME/" + resolved.relative_to(home).as_posix()
        except ValueError:
            return f"$EXTERNAL_PATH/{resolved.name}"

    def sanitize_text(self, value: str) -> str:
        sanitized = redact(value)
        replacements = (
            (str(self.root.resolve()), "$REPO"),
            (str(Path.home().resolve()), "$HOME"),
        )
        for source, replacement in replacements:
            sanitized = sanitized.replace(source, replacement)
            sanitized = sanitized.replace(source.replace("\\", "/"), replacement)
        machine_name = platform.node()
        if machine_name:
            sanitized = sanitized.replace(machine_name, "$HOSTNAME")
        for username in filter(None, (os.environ.get("USER"), os.environ.get("USERNAME"))):
            sanitized = re.sub(rf"(?i)(?<![\w-]){re.escape(username)}(?![\w-])", "$USER", sanitized)
        sanitized = re.sub(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])", "$IP_ADDRESS", sanitized)
        return sanitized

    def sanitize_command(self, command: tuple[str, ...]) -> list[str]:
        return [self.sanitize_text(part) for part in redact_command(command)]

    def sanitize_payload(self, value: Any) -> Any:
        if isinstance(value, str):
            return self.sanitize_text(value)
        if isinstance(value, dict):
            return {str(key): self.sanitize_payload(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self.sanitize_payload(item) for item in value]
        return value


def _write_json_yaml(path: Path, payload: dict[str, Any]) -> Path:
    """Write deterministic JSON, which is valid YAML 1.2, without bootstrap dependencies."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


def _installer(ctx: StageContext, name: str, url: str) -> tuple[Path, str]:
    destination = ctx.artifacts / "installers" / f"{name}-install.sh"
    destination.parent.mkdir(parents=True, exist_ok=True)
    ctx.runner.run(("curl", "-fL", "--retry", "3", "--output", str(destination), url), timeout=180, check=True)
    if ctx.options.dry_run:
        return destination, "dry-run"
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    return destination, digest


def _version(ctx: StageContext, command: str, *args: str) -> str | None:
    path = ctx.runner.which(command)
    if not path:
        return None
    result = ctx.runner.run((path, *(args or ("--version",))), timeout=30)
    lines = (result.stdout or result.stderr).strip().splitlines()
    return lines[0][:300] if result.ok and lines else None


def _software_manifest(ctx: StageContext, updates: dict[str, Any]) -> Path:
    path = ctx.artifacts / "manifests" / "software_versions.yaml"
    current: dict[str, Any] = {"schema_version": 1, "software": {}}
    if path.exists():
        try:
            current = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            pass
    current.setdefault("software", {}).update(ctx.sanitize_payload(updates))
    current["generated_at"] = utc_now()
    return _write_json_yaml(path, current)


def run_preflight(ctx: StageContext) -> StageOutcome:
    release = read_os_release()
    architecture = platform.machine().lower()
    ram = total_ram_gib()
    free = shutil.disk_usage(ctx.root).free / 1024**3
    probes = [
        ProbeResult("ubuntu", release.get("ID") == "ubuntu" and release.get("VERSION_ID") in SUPPORTED_UBUNTU,
                    f"{release.get('ID', 'unknown')} {release.get('VERSION_ID', 'unknown')}", release.get("VERSION_ID")),
        ProbeResult("architecture", architecture in {"x86_64", "amd64"}, architecture),
        ProbeResult("ram", ram is not None and ram >= 16, f"{ram} GiB detected; 16 GiB required", metadata={"ram_gib": ram}),
        ProbeResult("disk", free >= 25, f"{free:.2f} GiB free; 25 GiB required", metadata={"free_gib": round(free, 2)}),
        ProbeResult("wsl2", not is_wsl() or wsl_generation() == 2, "native Ubuntu or WSL2 detected"),
        command_probe(ctx.runner, "git"),
        command_probe(ctx.runner, "curl"),
        command_probe(ctx.runner, "python3"),
        nvidia_probe(ctx.runner),
    ]
    git_commit = ctx.runner.run(("git", "rev-parse", "--verify", "HEAD"), cwd=ctx.root, timeout=15)
    probes.append(ProbeResult("git-commit", git_commit.ok and bool(git_commit.stdout.strip()), "repository commit resolved" if git_commit.ok else "repository has no resolvable commit"))
    probes.append(ProbeResult("repository-writable", os.access(ctx.root, os.W_OK), "repository root is writable" if os.access(ctx.root, os.W_OK) else "repository root is not writable"))
    root_or_sudo = hasattr(os, "geteuid") and os.geteuid() == 0 or bool(ctx.runner.which("sudo"))
    probes.append(ProbeResult("sudo", root_or_sudo, "root or sudo command available" if root_or_sudo else "sudo unavailable"))
    probes.extend(ctx.network(url, 8) for url in REQUIRED_HOSTS)
    hard = [probe for probe in probes if not probe.available and probe.name != "nvidia"]
    if hard:
        names = ", ".join(probe.name for probe in hard)
        raise StageFailure(
            f"Preflight failed: {names}",
            "Run on x86_64 Ubuntu 22.04/24.04 with at least 16 GiB RAM, 25 GiB free disk, sudo, and official-host connectivity.",
        )
    manifest = write_machine_yaml(ctx.root, ctx.runner)
    warnings = []
    if ram is not None and ram < 24:
        warnings.append("Less than the recommended 24 GiB RAM; the 65,536-token model smoke test may fail.")
    if not next(item for item in probes if item.name == "nvidia").available:
        warnings.append("No usable NVIDIA GPU was detected; Ollama may use CPU fallback.")
    return StageOutcome(probes=probes, artifacts=[manifest], warnings=warnings)


def preflight_available(ctx: StageContext) -> bool:
    release = read_os_release()
    architecture = platform.machine().lower()
    ram = total_ram_gib()
    try:
        enough_disk = shutil.disk_usage(ctx.root).free / 1024**3 >= 25
    except OSError:
        return False
    root_or_sudo = (hasattr(os, "geteuid") and os.geteuid() == 0) or bool(ctx.runner.which("sudo"))
    commit = ctx.runner.run(("git", "rev-parse", "--verify", "HEAD"), cwd=ctx.root, timeout=15)
    return all(
        (
            release.get("ID") == "ubuntu" and release.get("VERSION_ID") in SUPPORTED_UBUNTU,
            architecture in {"x86_64", "amd64"},
            ram is not None and ram >= 16,
            enough_disk,
            not is_wsl() or wsl_generation() == 2,
            all(ctx.runner.which(command) for command in ("git", "curl", "python3")),
            root_or_sudo,
            commit.ok and bool(commit.stdout.strip()),
            os.access(ctx.root, os.W_OK),
            all(ctx.network(url, 8).available for url in REQUIRED_HOSTS),
        )
    )


def run_system_packages(ctx: StageContext) -> StageOutcome:
    if ctx.options.skip_system_packages:
        available, versions = _system_packages_installed(ctx)
        return StageOutcome(
            status="passed" if available else "skipped",
            skip_reason=None if available else "--skip-system-packages requested and prerequisites are incomplete",
            warnings=[] if available else ["Later stages may fail because system packages were skipped."],
            metadata={"packages": versions},
        )
    sudo = () if hasattr(os, "geteuid") and os.geteuid() == 0 else ("sudo",)
    ctx.runner.run((*sudo, "apt-get", "update"), timeout=900, check=True)
    yes = ("-y",) if ctx.options.yes else ()
    ctx.runner.run((*sudo, "apt-get", "install", *yes, "--no-install-recommends", *SYSTEM_PACKAGES), timeout=1800, check=True)
    result = ctx.runner.run(("dpkg-query", "-W", "-f=${Package}=${Version}\\n", *SYSTEM_PACKAGES), timeout=30, check=True)
    versions = {line.split("=", 1)[0]: line.split("=", 1)[1] for line in result.stdout.splitlines() if "=" in line}
    manifest = _software_manifest(ctx, {"apt": versions})
    return StageOutcome(artifacts=[manifest], metadata={"packages": versions})


def _system_packages_installed(ctx: StageContext) -> tuple[bool, dict[str, str]]:
    if not ctx.runner.which("dpkg-query"):
        return False, {}
    result = ctx.runner.run(("dpkg-query", "-W", "-f=${Package}=${Version}\\n", *SYSTEM_PACKAGES), timeout=30)
    versions = {
        line.split("=", 1)[0]: line.split("=", 1)[1]
        for line in result.stdout.splitlines()
        if "=" in line
    }
    return result.ok and all(name in versions for name in SYSTEM_PACKAGES), versions


def _resolve_uv(ctx: StageContext) -> str:
    path = ctx.runner.which("uv")
    if path:
        return path
    local = Path.home() / ".local" / "bin" / "uv"
    if local.exists():
        return str(local)
    installer, digest = _installer(ctx, "uv", UV_INSTALL_URL)
    ctx.runner.run(("sh", str(installer)), env={"UV_UNMANAGED_INSTALL": str(Path.home() / ".local" / "bin")}, timeout=300, check=True)
    if ctx.options.dry_run:
        return str(local)
    if not local.exists():
        raise StageFailure("uv installer completed but the expected executable was not found", "Inspect the uv installer output and PATH.")
    _software_manifest(ctx, {"uv_installer": {"url": UV_INSTALL_URL, "sha256": digest}})
    return str(local)


def run_python_environment(ctx: StageContext) -> StageOutcome:
    uv = _resolve_uv(ctx)
    ctx.runner.run((uv, "python", "install", "3.11"), cwd=ctx.root, timeout=900, check=True)
    ctx.runner.run((uv, "sync", "--locked"), cwd=ctx.root, timeout=1800, check=True)
    python = python_executable(ctx.root)
    version_result = ctx.runner.run((str(python), "--version"), timeout=30, check=True)
    uv_version = _version(ctx, "uv") or "resolved by setup runner"
    manifest = _software_manifest(ctx, {"python": version_result.stdout.strip() or version_result.stderr.strip(), "uv": uv_version})
    return StageOutcome(
        probes=[ProbeResult("python-environment", ctx.options.dry_run or python.exists(), f"project interpreter: {ctx.portable(python)}")],
        artifacts=[manifest, ctx.root / "uv.lock"],
    )


def _ollama_api(ctx: StageContext, path: str, payload: dict[str, Any] | None = None, timeout: int = 15) -> dict[str, Any]:
    encoded = json.dumps(payload).encode("utf-8") if payload is not None else None
    return ctx.http_json(f"{OLLAMA_HOST}{path}", encoded, timeout)


def _ollama_available(ctx: StageContext) -> bool:
    try:
        _ollama_api(ctx, "/api/version")
        return True
    except (OSError, URLError, RuntimeError, ValueError):
        return False


def _systemd_available(ctx: StageContext) -> bool:
    systemctl = ctx.runner.which("systemctl")
    if not systemctl:
        return False
    probe = ctx.runner.run((systemctl, "is-system-running"), timeout=15)
    state = (probe.stdout or probe.stderr).strip().lower()
    return state in {"running", "degraded", "starting", "maintenance"}


def run_ollama(ctx: StageContext) -> StageOutcome:
    installer_hash = None
    if not ctx.runner.which("ollama"):
        installer, installer_hash = _installer(ctx, "ollama", OLLAMA_INSTALL_URL)
        sudo = () if hasattr(os, "geteuid") and os.geteuid() == 0 else ("sudo",)
        ctx.runner.run((*sudo, "sh", str(installer)), timeout=900, check=True)
    if not _ollama_available(ctx) and not ctx.options.dry_run:
        systemctl = ctx.runner.which("systemctl")
        systemd = _systemd_available(ctx)
        if systemd:
            dropin = ctx.artifacts / "service-config" / "ollama-rq1.conf"
            dropin.parent.mkdir(parents=True, exist_ok=True)
            dropin.write_text(
                "[Service]\n"
                f"Environment=\"OLLAMA_CONTEXT_LENGTH={MINIMUM_CONTEXT}\"\n"
                "Environment=\"OLLAMA_HOST=127.0.0.1:11434\"\n",
                encoding="utf-8",
            )
            sudo = () if hasattr(os, "geteuid") and os.geteuid() == 0 else ("sudo",)
            ctx.runner.run((*sudo, "install", "-D", "-m", "0644", str(dropin), "/etc/systemd/system/ollama.service.d/rq1.conf"), check=True)
            ctx.runner.run((*sudo, "systemctl", "daemon-reload"), check=True)
            ctx.runner.run((*sudo, "systemctl", "enable", "--now", "ollama"), timeout=120, check=True)
        else:
            ollama = ctx.runner.which("ollama") or "ollama"
            ctx.runner.start_background(
                (ollama, "serve"), cwd=ctx.root,
                env={"OLLAMA_CONTEXT_LENGTH": str(MINIMUM_CONTEXT), "OLLAMA_HOST": "127.0.0.1:11434"},
                log_path=ctx.artifacts / "logs" / "ollama.log",
                pid_path=ctx.root / "state" / "ollama.pid",
            )
        for _ in range(30):
            if _ollama_available(ctx):
                break
            time.sleep(1)
    available = ctx.options.dry_run or _ollama_available(ctx)
    if not available:
        raise StageFailure("Ollama did not become healthy on 127.0.0.1:11434", "Inspect systemd status or artifacts/logs/ollama.log.")
    api_version = {} if ctx.options.dry_run else _ollama_api(ctx, "/api/version")
    manifest = _software_manifest(ctx, {"ollama": {"cli": _version(ctx, "ollama"), "api": api_version, "installer_sha256": installer_hash, "context_length": MINIMUM_CONTEXT}})
    return StageOutcome(probes=[ProbeResult("ollama-api", available, "localhost API responsive")], artifacts=[manifest])


def _resolve_hermes(ctx: StageContext) -> str | None:
    return ctx.runner.which("hermes") or (str(Path.home() / ".local" / "bin" / "hermes") if (Path.home() / ".local" / "bin" / "hermes").exists() else None)


def _hermes_help_capabilities(ctx: StageContext, hermes: str) -> tuple[dict[str, Any], dict[str, str]]:
    outputs: dict[str, str] = {}
    commands = {
        "main": (hermes, "--help"),
        "profile_create": (hermes, "profile", "create", "--help"),
        "config": (hermes, "config", "--help"),
        "config_set": (hermes, "config", "set", "--help"),
        "config_get": (hermes, "config", "get", "--help"),
    }
    for name, command in commands.items():
        result = ctx.runner.run(command, timeout=30)
        if not result.ok:
            return {"supported": False, "failed_help": name}, outputs
        outputs[name] = result.stdout or result.stderr
    main = outputs["main"]
    capabilities = {
        "profile_create": "--no-skills" in outputs["profile_create"],
        "profile_selector": "-p" in main or "--profile" in main,
        "profile_selector_flag": "-p" if "-p" in main else ("--profile" if "--profile" in main else None),
        "config_set": "config" in outputs["config_set"].lower() or "value" in outputs["config_set"].lower(),
        "config_get": "config" in outputs["config_get"].lower() or "key" in outputs["config_get"].lower(),
    }
    capabilities["supported"] = all(
        capabilities[key] for key in ("profile_create", "profile_selector", "config_set", "config_get")
    )
    return capabilities, outputs


def run_hermes(ctx: StageContext) -> StageOutcome:
    hermes = _resolve_hermes(ctx)
    installer_hash = None
    if not hermes:
        hermes_home = Path.home() / ".hermes"
        if hermes_home.exists() and any(hermes_home.iterdir()):
            raise StageFailure("Hermes is not on PATH but ~/.hermes already contains data", "Resolve the existing installation manually; setup will not modify a personal/default Hermes profile.")
        installer, installer_hash = _installer(ctx, "hermes", HERMES_INSTALL_URL)
        installer_text = "" if ctx.options.dry_run else installer.read_text(encoding="utf-8", errors="replace")
        if not ctx.options.dry_run and (
            "--no-skills" not in installer_text or "--skip-browser" not in installer_text
        ):
            raise StageFailure(
                "Hermes installer does not advertise the required --no-skills and --skip-browser flags",
                "Review the captured installer checksum and current official installation documentation before proceeding.",
            )
        ctx.runner.run(("bash", str(installer), "--no-skills", "--skip-browser"), timeout=1800, check=True)
        hermes = _resolve_hermes(ctx) or str(Path.home() / ".local" / "bin" / "hermes")
    version = ctx.runner.run((hermes, "--version"), timeout=30, check=True)
    detected, help_outputs = _hermes_help_capabilities(ctx, hermes)
    capabilities = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "version": (version.stdout or version.stderr).strip(),
        **detected,
        "config_command": bool(detected.get("config_set") and detected.get("config_get")),
        "default_profile_modified": False,
        "installer_sha256": installer_hash,
        "help_sha256": {name: hashlib.sha256(output.encode("utf-8")).hexdigest() for name, output in help_outputs.items()},
        "memory_controls": "unverified",
        "curator_controls": "unverified",
        "tool_allowlist_controls": "unverified",
    }
    path = ctx.artifacts / "manifests" / "hermes_capabilities.json"
    _write_json_yaml(path, capabilities)
    if not capabilities["supported"] and not ctx.options.dry_run:
        raise StageFailure("Installed Hermes CLI lacks required profile/config capabilities", "Review artifacts/manifests/hermes_capabilities.json and the installed Hermes help output.")
    manifest = _software_manifest(ctx, {"hermes": {"version": capabilities["version"], "installer_sha256": installer_hash}})
    return StageOutcome(probes=[ProbeResult("hermes-cli", True, "required help surfaces detected", str(capabilities["version"]))], artifacts=[path, manifest])


def run_alfworld_package(ctx: StageContext) -> StageOutcome:
    uv = _resolve_uv(ctx)
    ctx.runner.run((uv, "sync", "--locked", "--extra", "alfworld"), cwd=ctx.root, timeout=1800, check=True)
    python = python_executable(ctx.root)
    probe = ctx.runner.run((str(python), "-c", "import importlib.metadata as m; import alfworld; print(m.version('alfworld'))"), timeout=60, check=True)
    version = probe.stdout.strip() or "0.4.2"
    downloader = executable_in_venv(ctx.root, "alfworld-download")
    if not ctx.options.dry_run and not downloader.exists():
        raise StageFailure("ALFWorld imports but alfworld-download is missing", "Recreate the locked environment and inspect the ALFWorld package installation.")
    manifest = _software_manifest(ctx, {"alfworld": {"version": version, "mode": "text-only", "import_tested": True}})
    return StageOutcome(probes=[ProbeResult("alfworld-package", True, "import and downloader command verified", version)], artifacts=[manifest])


def _alfworld_package_available(ctx: StageContext) -> tuple[bool, str | None]:
    python = python_executable(ctx.root)
    downloader = executable_in_venv(ctx.root, "alfworld-download")
    if not python.exists() or not downloader.is_file():
        return False, None
    result = ctx.runner.run(
        (str(python), "-c", "import importlib.metadata as m; import alfworld; print(m.version('alfworld'))"),
        timeout=60,
    )
    version = result.stdout.strip() if result.ok else None
    return result.ok and version == "0.4.2", version


def _valid_alfworld_data(path: Path) -> bool:
    required = (path / "json_2.1.1", path / "logic")
    return path.is_dir() and all(
        directory.is_dir() and any(item.is_file() for item in directory.rglob("*"))
        for directory in required
    )


def run_alfworld_data(ctx: StageContext) -> StageOutcome:
    data_dir = ctx.data_dir
    reused = _valid_alfworld_data(data_dir)
    if ctx.options.skip_alfworld_data and not _valid_alfworld_data(data_dir):
        return StageOutcome(status="skipped", skip_reason="--skip-alfworld-data requested", warnings=["Real ALFWorld remains unavailable until data is downloaded."])
    if not _valid_alfworld_data(data_dir):
        downloader = executable_in_venv(ctx.root, "alfworld-download")
        ctx.runner.run((str(downloader),), env={"ALFWORLD_DATA": str(data_dir)}, timeout=7200, check=True)
    if not ctx.options.dry_run and not _valid_alfworld_data(data_dir):
        raise StageFailure("ALFWorld data download completed without required text-world directories", "Set RQ1_ALFWORLD_DATA_DIR to valid data or rerun this stage after removing only the incomplete download.")
    inventory = data_inventory(data_dir) if not ctx.options.dry_run else {"path": ctx.portable(data_dir), "exists": False, "dry_run": True}
    inventory["path"] = ctx.portable(data_dir)
    inventory.update({"schema_version": 1, "generated_at": utc_now(), "package_version": "0.4.2", "download_command": "alfworld-download"})
    manifest = _write_json_yaml(ctx.artifacts / "manifests" / "alfworld_data_manifest.yaml", inventory)
    return StageOutcome(probes=[ProbeResult("alfworld-data", ctx.options.dry_run or _valid_alfworld_data(data_dir), "required text data directories verified")], artifacts=[manifest], metadata={"reused": reused})


def _model_info(ctx: StageContext, model: str) -> dict[str, Any] | None:
    try:
        shown = _ollama_api(ctx, "/api/show", {"model": model}, 30)
        tags = _ollama_api(ctx, "/api/tags", None, 30).get("models", [])
        tagged = next(
            (
                item
                for item in tags
                if isinstance(item, dict) and item.get("name") in {model, f"{model}:latest"}
            ),
            None,
        )
        return {"show": shown, "tag": tagged}
    except (OSError, URLError, RuntimeError, ValueError):
        return None


def _smoke_model(ctx: StageContext, model: str) -> tuple[bool, dict[str, Any], str]:
    info = _model_info(ctx, model)
    if not info:
        return False, {}, f"Ollama does not report {model}"
    try:
        smoke = _ollama_api(
            ctx,
            "/api/generate",
            {
                "model": model,
                "prompt": "Reply with READY only.",
                "stream": False,
                "keep_alive": "2m",
                "options": {"temperature": 0, "seed": 1},
            },
            600,
        )
        if not isinstance(smoke.get("response"), str) or not smoke["response"].strip():
            return False, info, "model returned no text"
        running = _ollama_api(ctx, "/api/ps", None, 30).get("models", [])
        active = next(
            (item for item in running if isinstance(item, dict) and item.get("name") in {model, f"{model}:latest"}),
            None,
        )
        context_length = active.get("context_length") if isinstance(active, dict) else None
        metadata = {"metadata": info, "raw_inference_smoke_tested": True, "configured_context": context_length}
        if not isinstance(context_length, int) or context_length < MINIMUM_CONTEXT:
            return False, metadata, f"context length is below {MINIMUM_CONTEXT}"
        return True, metadata, "model metadata, inference, and context verified"
    except (OSError, URLError, RuntimeError, ValueError) as exc:
        return False, info, f"model smoke test failed: {type(exc).__name__}"


def run_candidate_models(ctx: StageContext) -> StageOutcome:
    if ctx.options.skip_model:
        valid, metadata, detail = _smoke_model(ctx, PRIMARY_MODEL)
        return StageOutcome(
            status="passed" if valid else "skipped",
            skip_reason=None if valid else "--skip-model requested and the existing model failed verification",
            probes=[ProbeResult("primary-model", valid, detail)],
            metadata={"primary_present": bool(metadata), "verification": metadata},
        )
    ollama = ctx.runner.which("ollama") or "ollama"
    ctx.runner.run((ollama, "pull", PRIMARY_MODEL), timeout=7200, check=True)
    if ctx.options.install_fallback_model:
        ctx.runner.run((ollama, "pull", FALLBACK_MODEL), timeout=7200, check=True)
    if ctx.options.dry_run:
        models = {PRIMARY_MODEL: {"dry_run": True}}
    else:
        valid, primary, detail = _smoke_model(ctx, PRIMARY_MODEL)
        if not valid:
            raise StageFailure(detail, "Inspect `ollama ps`, `ollama list`, and the Ollama service logs.")
        models = {PRIMARY_MODEL: primary}
        if ctx.options.install_fallback_model:
            models[FALLBACK_MODEL] = {"metadata": _model_info(ctx, FALLBACK_MODEL), "raw_inference_smoke_tested": False}
    manifest = _write_json_yaml(
        ctx.artifacts / "manifests" / "model_manifest.yaml",
        ctx.sanitize_payload(
            {
                "schema_version": 1,
                "generated_at": utc_now(),
                "primary": PRIMARY_MODEL,
                "fallback_policy": "explicit-flag-only",
                "models": models,
            }
        ),
    )
    return StageOutcome(probes=[ProbeResult("primary-model", True, f"{PRIMARY_MODEL} present and raw-smoke-tested")], artifacts=[manifest])


PROFILE_NAMES = ("rq1-pilot", "rq1-acquisition")


def _profile_settings(ctx: StageContext) -> dict[str, str]:
    return {
        "model.default": PRIMARY_MODEL,
        "model.provider": "custom",
        "model.base_url": f"{OLLAMA_HOST}/v1",
        "model.context_length": str(MINIMUM_CONTEXT),
        "terminal.backend": "local",
        "terminal.cwd": str(ctx.root.resolve()),
    }


def _active_profile(list_output: str) -> str | None:
    return next(
        (line.strip().lstrip("*").strip().split()[0] for line in list_output.splitlines() if line.strip().startswith("*")),
        None,
    )


def _profile_names(list_output: str) -> set[str]:
    return {
        line.strip().lstrip("*").strip().split()[0]
        for line in list_output.splitlines()
        if line.strip() and not line.lower().lstrip().startswith("profile")
    }


def _verify_profiles(
    ctx: StageContext,
    hermes: str,
    selector: str,
    *,
    expected_active: str | None,
) -> tuple[bool, dict[str, Any], str]:
    listed = ctx.runner.run((hermes, "profile", "list"), timeout=30)
    if not listed.ok:
        return False, {}, "Hermes profile list failed"
    active = _active_profile(listed.stdout)
    if expected_active is None or active != expected_active:
        return False, {"active_profile": active}, "Hermes active/default profile could not be proven unchanged"
    existing = _profile_names(listed.stdout)
    metadata: dict[str, Any] = {}
    for name in PROFILE_NAMES:
        marker = Path.home() / ".hermes" / "profiles" / name / ".no-bundled-skills"
        if name not in existing or not marker.is_file():
            return False, metadata, f"Hermes profile {name} or its no-skills marker is missing"
        verified: dict[str, str] = {}
        for key, expected in _profile_settings(ctx).items():
            result = ctx.runner.run((hermes, selector, name, "config", "get", key), timeout=30)
            actual = (result.stdout or result.stderr).strip()
            if not result.ok or expected not in actual:
                return False, metadata, f"Hermes profile {name} did not verify {key}"
            verified[key] = ctx.sanitize_text(actual)
        metadata[name] = {"settings": verified, "no_bundled_skills": True}
    return True, metadata, "two isolated non-default profiles verified"


def run_base_profiles(ctx: StageContext) -> StageOutcome:
    hermes = _resolve_hermes(ctx)
    if not hermes and ctx.options.dry_run:
        hermes = str(Path.home() / ".local" / "bin" / "hermes")
    if not hermes:
        raise StageFailure("Hermes command is unavailable", "Run the Hermes installation stage first.")
    capabilities, _ = _hermes_help_capabilities(ctx, hermes)
    if not capabilities.get("supported") and not ctx.options.dry_run:
        raise StageFailure("Hermes profile/config command shapes are unsupported")
    selector = str(capabilities.get("profile_selector_flag") or "-p")
    listed = ctx.runner.run((hermes, "profile", "list"), timeout=30, check=True).stdout
    active_before = _active_profile(listed)
    existing_profiles = _profile_names(listed)
    for name in PROFILE_NAMES:
        if name not in existing_profiles:
            ctx.runner.run(
                (hermes, "profile", "create", name, "--no-skills", "--description", "Isolated reproducible experiment profile"),
                timeout=120,
                check=True,
            )
        for key, value in _profile_settings(ctx).items():
            ctx.runner.run((hermes, selector, name, "config", "set", key, value), timeout=30, check=True)
    if ctx.options.dry_run:
        valid, profile_metadata, detail = True, {}, "profile operations planned and capability-gated"
    else:
        valid, profile_metadata, detail = _verify_profiles(
            ctx, hermes, selector, expected_active=active_before
        )
    if not valid:
        raise StageFailure(detail, "Restore the prior active profile and inspect the installed Hermes command help.")
    path = ctx.artifacts / "manifests" / "hermes_profiles.json"
    _write_json_yaml(
        path,
        {
            "schema_version": 1,
            "generated_at": utc_now(),
            "active_profile_changed": False,
            "active_profile": active_before,
            "profiles": profile_metadata,
        },
    )
    return StageOutcome(probes=[ProbeResult("hermes-profiles", True, detail)], artifacts=[path])


def _verify_fake_bridge(ctx: StageContext) -> dict[str, Any]:
    log_root = ctx.artifacts / "verification" / "bridge-events"
    server = create_bridge_server(log_root, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        health = read_json_response(f"{base}/health", b"{}", 5)
        started = read_json_response(f"{base}/episode/start", json.dumps({"task_id": "installation-smoke", "split": "valid_seen", "seed": 1, "action_limit": 6}).encode(), 5)
        episode_id = str(started["episode_id"])
        stepped = read_json_response(f"{base}/episode/step", json.dumps({"episode_id": episode_id, "action": "go to countertop 1"}).encode(), 5)
        status = read_json_response(f"{base}/episode/{episode_id}/status", None, 5)
        reset = read_json_response(f"{base}/episode/{episode_id}/reset", b"{}", 5)
        abort = read_json_response(f"{base}/episode/{episode_id}/abort", json.dumps({"reason": "installation verification"}).encode(), 5)
        return {"healthy": health.get("bridge_available") is True, "episode_id": episode_id, "step_number": stepped.get("step_number"), "status_step_number": status.get("step_number"), "reset_count": reset.get("reset_count"), "aborted": abort.get("aborted"), "raw_event_log": ctx.portable(log_root / f"{episode_id}.jsonl")}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _python_environment_available(ctx: StageContext) -> tuple[bool, str]:
    python = python_executable(ctx.root)
    if not python.is_file():
        return False, "project .venv interpreter is missing"
    result = ctx.runner.run(
        (str(python), "-c", "import sys, rq1; print('.'.join(map(str, sys.version_info[:3])))"),
        cwd=ctx.root,
        timeout=60,
    )
    version = result.stdout.strip()
    return result.ok and version.startswith("3.11."), version or "project import failed"


def _hermes_profiles_available(ctx: StageContext) -> tuple[bool, dict[str, Any], str]:
    hermes = _resolve_hermes(ctx)
    if not hermes:
        return False, {}, "Hermes command is unavailable"
    capabilities, _ = _hermes_help_capabilities(ctx, hermes)
    selector = capabilities.get("profile_selector_flag")
    if not capabilities.get("supported") or not isinstance(selector, str):
        return False, capabilities, "Hermes profile/config command shapes are unsupported"
    listed = ctx.runner.run((hermes, "profile", "list"), timeout=30)
    if not listed.ok:
        return False, capabilities, "Hermes profile list failed"
    valid, metadata, detail = _verify_profiles(
        ctx, hermes, selector, expected_active=_active_profile(listed.stdout)
    )
    return valid, {"capabilities": capabilities, "profiles": metadata}, detail


def run_installation_verification(ctx: StageContext) -> StageOutcome:
    if ctx.options.dry_run:
        return StageOutcome(
            status="skipped",
            probes=[ProbeResult("installation-verification", False, "dry-run; no local server was started")],
            metadata={"installation_ready": False, "pilot_ready": False, "dry_run": True},
        )
    bridge = _verify_fake_bridge(ctx)
    real = real_adapter_capability()
    data_ok = _valid_alfworld_data(ctx.data_dir)
    python_ok, python_detail = _python_environment_available(ctx)
    alfworld_ok, alfworld_version = _alfworld_package_available(ctx)
    model_ok, model_metadata, model_detail = _smoke_model(ctx, PRIMARY_MODEL)
    hermes_ok, hermes_metadata, hermes_detail = _hermes_profiles_available(ctx)
    required = {
        "python_environment": python_ok,
        "ollama_primary_model": model_ok,
        "hermes_profiles": hermes_ok,
        "alfworld_package": alfworld_ok,
        "alfworld_data": data_ok,
        "fake_bridge": bridge.get("healthy") is True,
    }
    installation_ready = all(required.values())
    metadata = {
        "required_capabilities": required,
        "installation_ready": installation_ready,
        "pilot_ready": False,
        "verification_levels": {
            "python": ["installed", "configured", "smoke_tested"],
            "ollama": ["installed", "configured", "smoke_tested"] if model_ok else [],
            "hermes": ["installed", "configured"] if hermes_ok else [],
            "alfworld": ["installed", "import_tested"] if required["alfworld_package"] else [],
            "fake_bridge": ["smoke_tested"] if bridge.get("healthy") else [],
            "real_alfworld_bridge": ["unverified"],
        },
        "real_alfworld": {
            "available": real.available,
            "status": "unverified",
            "details": real.details,
            "required_gate": "real start -> step -> reset",
        },
        "fake_bridge": bridge,
        "verification_details": {
            "python": python_detail,
            "alfworld_version": alfworld_version,
            "model": model_metadata,
            "model_detail": model_detail,
            "hermes": hermes_metadata,
            "hermes_detail": hermes_detail,
        },
    }
    warnings = [] if installation_ready else ["Installation is incomplete; inspect required_capabilities in installation.json."]
    probes = [
        ProbeResult("python-environment", python_ok, python_detail),
        ProbeResult("ollama-primary-model", model_ok, model_detail),
        ProbeResult("hermes-profiles", hermes_ok, hermes_detail),
        ProbeResult("alfworld-package", alfworld_ok, f"version: {alfworld_version or 'unavailable'}"),
        ProbeResult("alfworld-data", data_ok, "non-empty text data directories verified" if data_ok else "required data is missing or incomplete"),
        ProbeResult("fake-bridge", bool(bridge.get("healthy")), "HTTP lifecycle completed"),
        ProbeResult("real-alfworld", False, real.details),
    ]
    return StageOutcome(
        status="passed" if installation_ready else "blocked",
        probes=probes,
        warnings=warnings,
        metadata=metadata,
        remediation=None if installation_ready else "Run setup_machine.sh --resume after resolving missing capabilities.",
    )


STAGE_HANDLERS: dict[str, Callable[[StageContext], StageOutcome]] = {
    "preflight": run_preflight,
    "system-packages": run_system_packages,
    "python-environment": run_python_environment,
    "ollama": run_ollama,
    "hermes": run_hermes,
    "alfworld-package": run_alfworld_package,
    "alfworld-data": run_alfworld_data,
    "candidate-models": run_candidate_models,
    "base-profiles": run_base_profiles,
    "installation-verification": run_installation_verification,
}


def commands_since(ctx: StageContext, runner: CommandRunner, offset: int) -> list[list[str]]:
    return [ctx.sanitize_command(command) for command in runner.commands[offset:]]
