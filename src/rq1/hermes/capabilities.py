"""Read-only installed-Hermes capability probing and version-adapter selection."""
from __future__ import annotations

import hashlib
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from rq1.utils.time import utc_now


CommandRunner = Callable[[tuple[str, ...]], tuple[int, str, str]]


def _run_command(command: tuple[str, ...]) -> tuple[int, str, str]:
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=15, check=False)
        return completed.returncode, completed.stdout, completed.stderr
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 127, "", f"{type(exc).__name__}: {exc}"


@dataclass(frozen=True)
class HermesCapabilityReport:
    schema_version: int
    generated_at: str
    installed: bool
    executable: str | None
    version: str | None
    source_revision: str | None
    cli_supported: bool
    profile_supported: bool
    no_skills_supported: bool
    profile_inspection_supported: bool
    profile_location_supported: bool
    project_plugin_activation_supported: bool
    plugin_supported: bool
    hook_supported: bool
    config_validation_supported: bool
    skill_supported: bool
    project_plugin_locations: tuple[str, ...]
    selected_version_adapter: str
    unsupported_requirements: tuple[str, ...]
    evidence_sha256: dict[str, str]
    details: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _contains(output: str, value: str) -> bool:
    return value.lower() in output.lower()


def select_version_adapter(version: str | None, help_text: str) -> str:
    """Select only adapters whose minimal surface is present in captured help."""
    if version and _contains(help_text, "plugin") and _contains(help_text, "hook"):
        return "hermes-plugin-v1"
    return "unsupported"


def probe_hermes_capabilities(
    executable: str = "hermes",
    *,
    runner: CommandRunner | None = None,
    project_root: Path | None = None,
) -> HermesCapabilityReport:
    """Probe only read-only help/version commands advertised by the executable."""
    runner = runner or _run_command
    resolved = shutil.which(executable) if runner is _run_command else executable
    if runner is _run_command and not resolved:
        return HermesCapabilityReport(
            1, utc_now(), False, None, None, None, False, False, False, False, False, False, False, False, False, False,
            (), "unsupported", ("Hermes executable is unavailable",), {}, "Hermes is not installed or not on PATH."
        )
    target = resolved or executable
    version_code, version_out, version_err = runner((target, "--version"))
    help_code, help_out, help_err = runner((target, "--help"))
    combined_help = help_out + "\n" + help_err
    evidence = {"version": version_out + version_err, "help": combined_help}
    # The command shapes below are probed only after their immediate parent is
    # advertised in help. They are all help-only and cannot alter profiles.
    profile_help = ""
    profile_create_help = ""
    profile_show_help = ""
    if help_code == 0 and _contains(combined_help, "profile"):
        code, out, err = runner((target, "profile", "--help"))
        evidence["profile_help"] = out + "\n" + err
        if code == 0:
            profile_help = evidence["profile_help"]
        if code == 0 and _contains(profile_help, "create"):
            code, out, err = runner((target, "profile", "create", "--help"))
            evidence["profile_create_help"] = out + "\n" + err
            if code == 0:
                profile_create_help = evidence["profile_create_help"]
        if code == 0 and _contains(profile_help, "show"):
            code, out, err = runner((target, "profile", "show", "--help"))
            evidence["profile_show_help"] = out + "\n" + err
            if code == 0:
                profile_show_help = evidence["profile_show_help"]
    plugin_help = ""
    if help_code == 0 and _contains(combined_help, "plugin"):
        code, out, err = runner((target, "plugins", "--help"))
        evidence["plugins_help"] = out + "\n" + err
        if code == 0:
            plugin_help = evidence["plugins_help"]
    skill_help = ""
    if help_code == 0 and _contains(combined_help, "skill"):
        code, out, err = runner((target, "skills", "--help"))
        evidence["skills_help"] = out + "\n" + err
        if code == 0:
            skill_help = evidence["skills_help"]

    no_skills = "--no-skills" in (profile_create_help + profile_help)
    profile_inspection = bool(profile_show_help) and "--json" in profile_show_help
    profile_location = profile_inspection and _contains(profile_show_help, "path")
    plugin = bool(plugin_help)
    hooks = plugin and _contains(combined_help + plugin_help, "hook")
    adapter = select_version_adapter((version_out or version_err).strip() or None, combined_help + plugin_help)
    locations = (str((project_root or Path.cwd()) / ".hermes" / "plugins"),) if plugin else ()
    project_plugin_activation = plugin and bool(locations)
    requirements: list[str] = []
    for name, present in (
        ("profile", bool(profile_help)),
        ("--no-skills", no_skills),
        ("profile inspection JSON", profile_inspection),
        ("profile location discovery", profile_location),
        ("project plugin activation", project_plugin_activation),
        ("plugin", plugin),
        ("pre_tool_call/post_tool_call hooks", hooks),
    ):
        if not present:
            requirements.append(name)
    return HermesCapabilityReport(
        1,
        utc_now(),
        help_code == 0 or version_code == 0,
        target,
        (version_out or version_err).strip() or None,
        None,
        help_code == 0,
        bool(profile_help),
        no_skills,
        profile_inspection,
        profile_location,
        project_plugin_activation,
        plugin,
        hooks,
        _contains(combined_help + profile_help, "config"),
        bool(skill_help) or _contains(combined_help, "skill"),
        locations,
        adapter,
        tuple(requirements),
        {name: hashlib.sha256(value.encode("utf-8")).hexdigest() for name, value in evidence.items()},
        "Capability evidence was collected using only version and help commands; no profile or plugin was modified.",
    )
