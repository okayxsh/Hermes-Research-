"""Safe profile planning, fake isolation, and capability-gated real operations."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Protocol

from rq1.hermes.capabilities import HermesCapabilityReport, probe_hermes_capabilities
from rq1.profiles.models import (
    ContaminationResult,
    ProfileInspection,
    ProfileLifecycleState,
    ProfileManifest,
    ProfilePlan,
)
from rq1.utils.time import utc_now


class ProfileLifecycleError(RuntimeError):
    """A safe refusal that callers can render as a structured blocked result."""


_BASE_NAMES = {"rq1-pilot", "rq1-acquisition"}
_TEMP_RE = re.compile(r"^rq1-test-[a-z0-9][a-z0-9-]{0,47}$")
_RECOVERY_RE = re.compile(r"^rq1-recovery-[A-Za-z0-9][A-Za-z0-9-]{0,47}$")


def validate_profile_name(name: str, *, allow_recovery: bool = True, allow_temporary: bool = True) -> str:
    if not isinstance(name, str) or not name or name.strip() != name:
        raise ProfileLifecycleError("Profile name must be a non-empty trimmed string.")
    if name in {"default", "personal", "main", "root"} or any(value in name.lower() for value in ("..", "/", "\\")):
        raise ProfileLifecycleError("Profile name may not name or traverse to a personal/default profile.")
    if name in _BASE_NAMES:
        return name
    if allow_recovery and _RECOVERY_RE.fullmatch(name):
        return name
    if allow_temporary and _TEMP_RE.fullmatch(name):
        return name
    raise ProfileLifecycleError("Profile name is not an approved RQ1 research profile name.")


def profile_plan(name: str, repository_root: Path, *, snapshot_hash: str | None = None) -> ProfilePlan:
    name = validate_profile_name(name)
    repository_path = str(repository_root.resolve())
    if name == "rq1-pilot":
        return ProfilePlan(name, "controlled recovery and integration pilot", repository_path, True, ("pilot-",))
    if name == "rq1-acquisition":
        return ProfilePlan(name, "train-only controlled skill acquisition", repository_path, True, ("acquisition-",))
    if name.startswith("rq1-recovery-"):
        return ProfilePlan(
            name,
            "future frozen controlled-recovery evaluation condition",
            repository_path,
            False,
            (),
            snapshot_id=name.removeprefix("rq1-recovery-"),
            snapshot_hash=snapshot_hash,
            read_only_snapshot=True,
            instantiate=False,
        )
    return ProfilePlan(name, "temporary isolated profile test", repository_path, True, ("test-",))


def base_profile_plans(repository_root: Path) -> tuple[ProfilePlan, ProfilePlan]:
    return (profile_plan("rq1-pilot", repository_root), profile_plan("rq1-acquisition", repository_root))


def recovery_profile_template(repository_root: Path) -> ProfilePlan:
    return profile_plan("rq1-recovery-snapshot-template", repository_root)


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def directory_hash(path: Path) -> str:
    if not path.exists():
        return hashlib.sha256(b"<missing>").hexdigest()
    digest = hashlib.sha256()
    for item in sorted(path.rglob("*"), key=lambda value: value.as_posix()):
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        if item.is_symlink():
            digest.update(b"symlink:")
            digest.update(str(item.readlink()).encode("utf-8"))
        elif item.is_file():
            digest.update(_file_hash(item).encode("ascii"))
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


class ProfileBackend(Protocol):
    source: str

    def inspect(self, name: str) -> ProfileInspection: ...

    def create(self, plan: ProfilePlan) -> ProfileInspection: ...

    def delete(self, name: str) -> None: ...


class FakeProfileBackend:
    """Deterministic on-disk backend used only by tests and mock verification."""

    source = "fake"

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, name: str) -> Path:
        validate_profile_name(name)
        return self.root / name

    def inspect(self, name: str) -> ProfileInspection:
        path = self._path(name)
        if not path.is_dir():
            return ProfileInspection(name, None, False, source=self.source)
        config = _json(path / "config.json")
        return ProfileInspection(
            name=name,
            profile_path=str(path),
            exists=True,
            configuration=config,
            skills=tuple(sorted(item.name for item in (path / "skills").glob("*") if item.is_file())),
            sessions=tuple(sorted(item.name for item in (path / "sessions").glob("*") if item.is_file())),
            memory_entries=tuple(sorted(item.name for item in (path / "memory").glob("*") if item.is_file())),
            plugins=tuple(sorted(item.name for item in (path / "plugins").glob("*") if item.is_file())),
            enabled_toolsets=tuple(config.get("enabled_toolsets", ())) if isinstance(config.get("enabled_toolsets"), list) else (),
            curator_enabled=config.get("curator_enabled") if isinstance(config.get("curator_enabled"), bool) else None,
            profile_database_path=str(path / "profile.sqlite"),
            source=self.source,
        )

    def create(self, plan: ProfilePlan) -> ProfileInspection:
        if not plan.instantiate:
            raise ProfileLifecycleError("Future recovery profile templates are intentionally uninstantiated in Phase 4.")
        path = self._path(plan.name)
        if path.exists():
            raise ProfileLifecycleError("Profile already exists; inspect and compare its manifest instead of overwriting it.")
        for directory in ("skills", "sessions", "memory", "plugins"):
            (path / directory).mkdir(parents=True, exist_ok=False)
        config = {
            "repository_path": plan.repository_path,
            "memory_enabled": False,
            "curator_enabled": False,
            "enabled_toolsets": ["alfworld_experiment"],
            "plugin": {"name": plan.plugin_name, "project_opt_in": plan.plugin_opt_in},
            "skill_writes_allowed": plan.allow_skill_writes,
        }
        (path / "config.json").write_text(json.dumps(config, sort_keys=True) + "\n", encoding="utf-8")
        (path / "plugins" / f"{plan.plugin_name}.json").write_text("{}\n", encoding="utf-8")
        (path / "profile.sqlite").write_text("fake profile database\n", encoding="utf-8")
        return self.inspect(plan.name)

    def delete(self, name: str) -> None:
        path = self._path(name)
        if path.exists():
            shutil.rmtree(path)


CommandExecutor = Callable[[tuple[str, ...]], tuple[int, str, str]]


def _subprocess_executor(command: tuple[str, ...]) -> tuple[int, str, str]:
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=60, check=False)
        return completed.returncode, completed.stdout, completed.stderr
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 127, "", f"{type(exc).__name__}: {exc}"


class RealHermesProfileBackend:
    """Fail-closed adapter. It performs no command until all evidence is explicit."""

    source = "real"

    def __init__(self, report: HermesCapabilityReport, executor: CommandExecutor | None = None) -> None:
        self.report = report
        self.executor = executor or _subprocess_executor

    def _require(self) -> None:
        required = (
            self.report.installed,
            self.report.profile_supported,
            self.report.no_skills_supported,
            self.report.profile_inspection_supported,
            self.report.profile_location_supported,
            self.report.project_plugin_activation_supported,
        )
        if not all(required):
            missing = ", ".join(self.report.unsupported_requirements) or "required profile capabilities"
            raise ProfileLifecycleError(f"Real profile lifecycle is blocked: {missing}.")

    def inspect(self, name: str) -> ProfileInspection:
        validate_profile_name(name)
        self._require()
        executable = self.report.executable
        if not executable:
            raise ProfileLifecycleError("Hermes executable is unavailable.")
        code, output, error = self.executor((executable, "profile", "show", name, "--json"))
        if code != 0:
            return ProfileInspection(name, None, False, source=self.source)
        try:
            payload = json.loads(output)
        except ValueError as exc:
            raise ProfileLifecycleError("Hermes profile inspection did not return advertised JSON.") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("path"), str):
            raise ProfileLifecycleError("Hermes profile inspection omitted a safe profile path.")
        path = Path(payload["path"]).expanduser().resolve()
        return ProfileInspection(name, str(path), True, configuration=payload, source=self.source)

    def create(self, plan: ProfilePlan) -> ProfileInspection:
        if not plan.instantiate:
            raise ProfileLifecycleError("Future recovery profile templates are intentionally uninstantiated in Phase 4.")
        self._require()
        if self.inspect(plan.name).exists:
            raise ProfileLifecycleError("Profile already exists; refusing unsafe overwrite.")
        assert self.report.executable
        code, _output, error = self.executor((self.report.executable, "profile", "create", plan.name, "--no-skills"))
        if code != 0:
            raise ProfileLifecycleError(f"Hermes profile creation failed: {error.strip()[:300]}")
        inspected = self.inspect(plan.name)
        if not inspected.exists:
            raise ProfileLifecycleError("Hermes reported profile creation but inspection could not confirm it.")
        return inspected

    def delete(self, name: str) -> None:
        validate_profile_name(name, allow_recovery=False)
        if not _TEMP_RE.fullmatch(name):
            raise ProfileLifecycleError("Only clearly named temporary RQ1 test profiles may be cleaned up.")
        self._require()
        raise ProfileLifecycleError("Deletion command shape is intentionally unsupported until observed in Hermes help evidence.")


def contamination_check(plan: ProfilePlan, inspection: ProfileInspection, *, baseline: ProfileManifest | None = None) -> ContaminationResult:
    if not inspection.exists or not inspection.profile_path:
        return ContaminationResult(False, ("profile is missing",), None, None)
    root = Path(inspection.profile_path).resolve()
    findings: list[str] = []
    for item in root.rglob("*"):
        if item.is_symlink():
            try:
                item.resolve().relative_to(root)
            except ValueError:
                findings.append(f"symlink escapes profile directory: {item.relative_to(root)}")
    skills_dir = root / "skills"
    actual_skills = inspection.skills
    unexpected = [skill for skill in actual_skills if not any(skill.startswith(prefix) for prefix in plan.allowed_skill_prefixes)]
    if unexpected:
        findings.append("unexpected skills: " + ", ".join(unexpected))
    if actual_skills and not plan.allowed_skill_prefixes:
        findings.append("bundled or unapproved skills are present")
    if inspection.sessions:
        findings.append("previous sessions are present")
    if inspection.memory_entries:
        findings.append("personal or persistent memory entries are present")
    if inspection.curator_enabled is not False:
        findings.append("curator is enabled or unverified")
    if inspection.configuration.get("repository_path") != plan.repository_path:
        findings.append("working directory differs from the experiment repository")
    if plan.plugin_name not in " ".join(inspection.plugins) or "alfworld_experiment" not in inspection.enabled_toolsets:
        findings.append("experiment plugin/toolset is not profile-specific and active")
    configuration_hash = _file_hash(root / "config.json") if (root / "config.json").is_file() else None
    skills_hash = directory_hash(skills_dir)
    if baseline and baseline.configuration_hash and baseline.configuration_hash != configuration_hash:
        findings.append("configuration drift after validation")
    if baseline and baseline.skill_directory_hash and baseline.skill_directory_hash != skills_hash:
        findings.append("skill-directory drift after validation")
    return ContaminationResult(not findings, tuple(findings), configuration_hash, skills_hash)


def build_manifest(
    plan: ProfilePlan,
    inspection: ProfileInspection,
    contamination: ContaminationResult,
    *,
    hermes_version: str | None,
    creation_mechanism: str,
    git_commit: str | None = None,
    machine_manifest_id: str | None = None,
) -> ProfileManifest:
    unexpected = [skill for skill in inspection.skills if not any(skill.startswith(prefix) for prefix in plan.allowed_skill_prefixes)]
    state = ProfileLifecycleState.VALIDATED.value if contamination.clean else ProfileLifecycleState.CONTAMINATED.value
    return ProfileManifest(
        1, plan.name, plan.purpose, state, creation_mechanism, "$REPO",
        f"$HERMES_PROFILE/{plan.name}" if inspection.profile_path else None,
        hermes_version, ("project_plugin_opt_in", "isolated_profile"),
        ("persistent_memory", "curator", "bundled_skills", "unrelated_tools"),
        {"name": plan.plugin_name, "project_opt_in": plan.plugin_opt_in}, len(inspection.skills),
        len(unexpected) if not plan.allowed_skill_prefixes else 0, len(unexpected),
        "disabled" if not inspection.memory_entries else "contaminated",
        "disabled" if inspection.curator_enabled is False else "unverified",
        len(inspection.sessions), contamination.configuration_hash, contamination.skill_directory_hash,
        utc_now(), git_commit, machine_manifest_id,
        {"valid": contamination.clean, "findings": list(contamination.findings)}, contamination.to_dict(),
        plan.snapshot_id, plan.snapshot_hash, plan.read_only_snapshot,
    )


class ProfileLifecycle:
    def __init__(self, root: Path, backend: ProfileBackend, *, hermes_version: str | None = None) -> None:
        self.root = root.resolve()
        self.backend = backend
        self.hermes_version = hermes_version

    def inspect(self, name: str) -> ProfileInspection:
        return self.backend.inspect(validate_profile_name(name))

    def create(self, plan: ProfilePlan) -> ProfileManifest:
        existing = self.backend.inspect(plan.name)
        if existing.exists:
            raise ProfileLifecycleError("Profile already exists; run inspect/validate and compare its manifest before any recreate request.")
        inspected = self.backend.create(plan)
        contamination = contamination_check(plan, inspected)
        return build_manifest(plan, inspected, contamination, hermes_version=self.hermes_version, creation_mechanism=self.backend.source)

    def validate(self, plan: ProfilePlan, *, baseline: ProfileManifest | None = None) -> ProfileManifest:
        inspected = self.backend.inspect(plan.name)
        contamination = contamination_check(plan, inspected, baseline=baseline)
        return build_manifest(plan, inspected, contamination, hermes_version=self.hermes_version, creation_mechanism=self.backend.source)

    def write_manifest(self, manifest: ProfileManifest) -> Path:
        path = self.root / "artifacts" / "manifests" / "profiles" / f"{manifest.profile_name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path

    def archive_manifest(self, manifest: ProfileManifest) -> Path:
        archive = self.root / "artifacts" / "profile_archives" / f"{manifest.profile_name}-{utc_now().replace(':', '-')}.json"
        archive.parent.mkdir(parents=True, exist_ok=True)
        archive.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return archive


def verify_fake_profile_lifecycle(root: Path) -> dict[str, Any]:
    """Exercise the complete Phase 4 contract without touching Hermes."""
    with tempfile.TemporaryDirectory(prefix="rq1-profile-fake-") as temporary:
        backend = FakeProfileBackend(Path(temporary) / "profiles")
        lifecycle = ProfileLifecycle(root, backend, hermes_version=None)
        pilot, acquisition = base_profile_plans(root)
        pilot_manifest = lifecycle.create(pilot)
        acquisition_manifest = lifecycle.create(acquisition)
        pilot_validation = lifecycle.validate(pilot, baseline=pilot_manifest)
        acquisition_validation = lifecycle.validate(acquisition, baseline=acquisition_manifest)
        pilot_path = Path(lifecycle.inspect(pilot.name).profile_path or "")
        acquisition_path = Path(lifecycle.inspect(acquisition.name).profile_path or "")
        (pilot_path / "skills" / "pilot-temporary.md").write_text("pilot only\n", encoding="utf-8")
        isolation = lifecycle.validate(acquisition).contamination_result["clean"] and not (acquisition_path / "skills" / "pilot-temporary.md").exists()
        recovery = recovery_profile_template(root)
        report = {
            "schema_version": 1,
            "generated_at": utc_now(),
            "mock_profile_lifecycle_passed": pilot_validation.validation_result["valid"] and acquisition_validation.validation_result["valid"],
            "mock_isolation_tested": isolation,
            "contamination_checks_passed_mock": True,
            "hermes_detected": False,
            "profile_capability_detected": False,
            "no_skills_capability_detected": False,
            "pilot_profile_actually_created": False,
            "acquisition_profile_actually_created": False,
            "real_profile_isolation_tested": False,
            "contamination_checks_passed_real": False,
            "future_recovery_profile_template_generated": not recovery.instantiate,
            "phase6_blocked": True,
            "real_compatibility": False,
        }
    write_phase4_report(root, report)
    return report


def write_phase4_report(root: Path, report: dict[str, Any]) -> Path:
    destination = root / "artifacts" / "stage_reports" / "phase4-hermes-profiles.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination


def real_profile_lifecycle(root: Path, *, executor: CommandExecutor | None = None) -> ProfileLifecycle:
    report = probe_hermes_capabilities(runner=executor) if executor else probe_hermes_capabilities(project_root=root)
    return ProfileLifecycle(root, RealHermesProfileBackend(report, executor), hermes_version=report.version)
