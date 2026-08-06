from __future__ import annotations

import json
import ssl
import sys
import tempfile
import tomllib
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Mapping
from unittest import mock


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rq1.setup.models import (  # noqa: E402
    SETUP_STAGES,
    CommandResult,
    ProbeResult,
    SetupOptions,
    SetupState,
)
from rq1.setup.orchestrator import SetupError, SetupOrchestrator  # noqa: E402
from rq1.cli import command_setup_machine  # noqa: E402
from rq1.setup.probes import python_executable, read_os_release, total_ram_gib, wsl_generation  # noqa: E402
from rq1.setup.probes import network_probe  # noqa: E402
from urllib.error import HTTPError, URLError  # noqa: E402
from rq1.setup.registry import SetupRegistry  # noqa: E402
from rq1.setup.runner import redact, redact_command  # noqa: E402
from rq1.setup.stages import (  # noqa: E402
    SYSTEM_PACKAGES,
    StageContext,
    StageFailure,
    StageOutcome,
    _system_packages_installed,
    _systemd_available,
    _hermes_profiles_available,
    run_candidate_models,
    _valid_alfworld_data,
)


EXPECTED_STAGE_ORDER = [
    "preflight",
    "system-packages",
    "python-environment",
    "ollama",
    "hermes",
    "alfworld-package",
    "alfworld-data",
    "candidate-models",
    "base-profiles",
    "installation-verification",
]


class FakeCommandRunner:
    """In-memory CommandRunner that can never invoke a host command."""

    def __init__(
        self,
        *,
        available: Mapping[str, str] | None = None,
        responses: Mapping[tuple[str, ...], CommandResult] | None = None,
        reject_unconfigured: bool = False,
    ) -> None:
        self.available = dict(available or {})
        self.responses = dict(responses or {})
        self.reject_unconfigured = reject_unconfigured
        self.commands: list[tuple[str, ...]] = []
        self.background_commands: list[tuple[str, ...]] = []

    def which(self, command: str) -> str | None:
        return self.available.get(command)

    def run(
        self,
        command: tuple[str, ...],
        *,
        cwd: Path | None = None,
        env: Mapping[str, str] | None = None,
        timeout: int = 300,
        check: bool = False,
    ) -> CommandResult:
        del cwd, env, timeout
        command = tuple(command)
        self.commands.append(command)
        if command in self.responses:
            result = self.responses[command]
        elif self.reject_unconfigured:
            raise AssertionError(f"Unexpected external command request: {command!r}")
        else:
            result = CommandResult(command, 0, "fake command output\n", "")
        if check and not result.ok:
            raise RuntimeError(f"Fake command failed: {command!r}")
        return result

    def start_background(
        self,
        command: tuple[str, ...],
        *,
        cwd: Path,
        env: Mapping[str, str],
        log_path: Path,
        pid_path: Path,
    ) -> int:
        del cwd, env, log_path, pid_path
        command = tuple(command)
        self.background_commands.append(command)
        if self.reject_unconfigured:
            raise AssertionError(f"Unexpected background command request: {command!r}")
        return 4242


def guarded_context_factory(
    root: Path, options: SetupOptions, runner: FakeCommandRunner
) -> StageContext:
    def reject_network(url: str, timeout: int) -> ProbeResult:
        raise AssertionError(f"Unexpected network probe: {url} ({timeout}s)")

    def reject_http(url: str, payload: bytes | None, timeout: int) -> dict[str, object]:
        del payload
        raise AssertionError(f"Unexpected HTTP request: {url} ({timeout}s)")

    return StageContext(root, options, runner, network=reject_network, http_json=reject_http)


def make_handlers(
    calls: list[str],
    *,
    final_ready: bool = True,
    outcomes: Mapping[str, StageOutcome] | None = None,
    failures: Mapping[str, Exception] | None = None,
) -> dict[str, object]:
    configured_outcomes = dict(outcomes or {})
    configured_failures = dict(failures or {})
    handlers: dict[str, object] = {}

    for stage_name in EXPECTED_STAGE_ORDER:
        def handler(context: StageContext, name: str = stage_name) -> StageOutcome:
            del context
            calls.append(name)
            if name in configured_failures:
                raise configured_failures[name]
            if name in configured_outcomes:
                return configured_outcomes[name]
            if name == "installation-verification" and final_ready:
                return StageOutcome(
                    metadata={
                        "installation_ready": True,
                        "pilot_ready": False,
                        "required_capabilities": {
                            "python_environment": True,
                            "ollama_primary_model": True,
                            "hermes_profiles": True,
                            "alfworld_package": True,
                            "alfworld_data": True,
                            "fake_bridge": True,
                        },
                        "real_alfworld": {
                            "available": False,
                            "status": "unverified",
                            "required_gate": "real start -> step -> reset",
                        },
                    }
                )
            return StageOutcome()

        handlers[stage_name] = handler

    return handlers


class SetupModelsAndParsingTests(unittest.TestCase):
    @mock.patch("rq1.setup.probes.urlopen")
    def test_network_probe_accepts_http_responses_including_404(self, urlopen) -> None:
        response = mock.MagicMock(status=200)
        urlopen.return_value.__enter__.return_value = response
        self.assertTrue(network_probe("https://registry.ollama.ai").available)

        urlopen.reset_mock()
        urlopen.side_effect = HTTPError("https://registry.ollama.ai", 404, "Not Found", {}, None)
        result = network_probe("https://registry.ollama.ai")
        self.assertTrue(result.available)
        self.assertEqual("HTTP 404", result.details)

    @mock.patch("rq1.setup.probes.urlopen")
    def test_network_probe_rejects_transport_failures(self, urlopen) -> None:
        for failure in (TimeoutError("timed out"), URLError("DNS failure"), ssl.SSLError("TLS failure")):
            urlopen.reset_mock()
            urlopen.side_effect = failure
            result = network_probe("https://registry.ollama.ai")
            self.assertFalse(result.available)
    def test_uv_lock_is_python311_and_contains_complete_alfworld_closure(self) -> None:
        root = Path(__file__).resolve().parents[1]
        lock = tomllib.loads((root / "uv.lock").read_text(encoding="utf-8"))
        packages = lock["package"]
        names = {package["name"] for package in packages}
        referenced = {
            dependency["name"]
            for package in packages
            for dependency in package.get("dependencies", [])
        }
        referenced.update(
            dependency["name"]
            for package in packages
            for dependencies in package.get("optional-dependencies", {}).values()
            for dependency in dependencies
        )
        versions = {package["name"]: package.get("version") for package in packages}
        self.assertEqual(">=3.11, <3.12", lock["requires-python"])
        self.assertEqual("0.4.2", versions["alfworld"])
        self.assertEqual("1.7.0", versions["textworld"])
        self.assertEqual(set(), referenced - names)

    def test_installation_example_matches_its_machine_readable_schema_contract(self) -> None:
        root = Path(__file__).resolve().parents[1]
        schema = json.loads(
            (root / "data" / "schemas" / "installation_report.schema.json").read_text(
                encoding="utf-8"
            )
        )
        example = json.loads(
            (root / "docs" / "examples" / "installation.example.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertTrue(set(schema["required"]).issubset(example))
        for name, definition in schema["properties"].items():
            if name not in example:
                continue
            if "const" in definition:
                self.assertEqual(definition["const"], example[name])
            if "enum" in definition:
                self.assertIn(example[name], definition["enum"])

    def test_setup_options_defaults_explicit_values_and_immutability(self) -> None:
        defaults = SetupOptions()
        self.assertFalse(defaults.dry_run)
        self.assertFalse(defaults.yes)
        self.assertFalse(defaults.resume)
        self.assertFalse(defaults.skip_system_packages)
        self.assertFalse(defaults.skip_model)
        self.assertFalse(defaults.skip_alfworld_data)
        self.assertFalse(defaults.install_fallback_model)
        self.assertIsNone(defaults.force_stage)
        self.assertFalse(defaults.verbose)

        configured = SetupOptions(
            dry_run=True,
            yes=True,
            resume=True,
            skip_system_packages=True,
            skip_model=True,
            skip_alfworld_data=True,
            install_fallback_model=True,
            force_stage="hermes",
            verbose=True,
        )
        self.assertEqual("hermes", configured.force_stage)
        self.assertTrue(all(value for name, value in configured.__dict__.items() if name != "force_stage"))
        with self.assertRaises(FrozenInstanceError):
            configured.yes = False  # type: ignore[misc]

    def test_setup_stage_order_and_prerequisites(self) -> None:
        self.assertEqual(EXPECTED_STAGE_ORDER, [stage.name for stage in SETUP_STAGES])
        prerequisites = {stage.name: stage.prerequisites for stage in SETUP_STAGES}
        self.assertEqual((), prerequisites["preflight"])
        self.assertEqual(("preflight",), prerequisites["system-packages"])
        self.assertEqual(("system-packages",), prerequisites["python-environment"])
        self.assertEqual(("python-environment",), prerequisites["ollama"])
        self.assertEqual(("python-environment",), prerequisites["hermes"])
        self.assertEqual(("python-environment",), prerequisites["alfworld-package"])
        self.assertEqual(("alfworld-package",), prerequisites["alfworld-data"])
        self.assertEqual(("ollama",), prerequisites["candidate-models"])
        self.assertEqual(("hermes", "ollama"), prerequisites["base-profiles"])
        self.assertEqual(
            ("alfworld-data", "candidate-models", "base-profiles"),
            prerequisites["installation-verification"],
        )

    def test_redaction_removes_assignments_and_bearer_credentials(self) -> None:
        command = (
            "tool",
            "API_KEY=alpha",
            "token=beta",
            "--password=gamma",
            "authorization=delta",
            "Bearer epsilon",
            "ordinary-value",
        )
        redacted = redact_command(command)
        joined = " ".join(redacted)
        for secret in ("alpha", "beta", "gamma", "delta", "epsilon"):
            self.assertNotIn(secret, joined)
        self.assertEqual("API_KEY=<redacted>", redacted[1])
        self.assertEqual("--password=<redacted>", redacted[3])
        self.assertEqual("Bearer <redacted>", redacted[5])
        self.assertEqual("ordinary-value", redacted[6])
        self.assertEqual("not-a-secret", redact("not-a-secret"))

        extended = redact_command(
            ("tool", "--token", "separate-secret", '{"api_key":"json-secret"}')
        )
        self.assertNotIn("separate-secret", " ".join(extended))
        self.assertNotIn("json-secret", " ".join(extended))

    def test_os_release_parsing_handles_comments_quotes_and_embedded_equals(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "os-release"
            path.write_text(
                '# comment\nID="ubuntu"\nVERSION_ID="24.04"\nNAME=Ubuntu\nURL="https://example.invalid/a=b"\n',
                encoding="utf-8",
            )
            self.assertEqual(
                {
                    "ID": "ubuntu",
                    "VERSION_ID": "24.04",
                    "NAME": "Ubuntu",
                    "URL": "https://example.invalid/a=b",
                },
                read_os_release(path),
            )
            self.assertEqual({}, read_os_release(Path(directory) / "missing"))

    def test_total_ram_parsing_returns_gib_and_handles_missing_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            meminfo = Path(directory) / "meminfo"
            meminfo.write_text(
                "MemTotal:       16777216 kB\nMemFree:         1024 kB\n",
                encoding="utf-8",
            )
            self.assertEqual(16.0, total_ram_gib(meminfo))
            self.assertIsNone(total_ram_gib(Path(directory) / "missing"))

    def test_wsl_generation_distinguishes_wsl1_wsl2_and_native(self) -> None:
        with mock.patch("rq1.setup.probes.is_wsl", return_value=False):
            self.assertIsNone(wsl_generation())
        with mock.patch("rq1.setup.probes.is_wsl", return_value=True), mock.patch(
            "rq1.setup.probes.platform.release", return_value="4.4.0-Microsoft"
        ):
            self.assertEqual(1, wsl_generation())
        with mock.patch("rq1.setup.probes.is_wsl", return_value=True), mock.patch(
            "rq1.setup.probes.platform.release", return_value="5.15.90.1-microsoft-standard-WSL2"
        ):
            self.assertEqual(2, wsl_generation())

    def test_project_python_never_falls_back_to_system_interpreter(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = python_executable(Path(directory))
            self.assertFalse(path.exists())
            self.assertIn(".venv", path.parts)

    def test_alfworld_data_validation_rejects_empty_marker_directories(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "json_2.1.1").mkdir()
            (root / "logic").mkdir()
            self.assertFalse(_valid_alfworld_data(root))
            (root / "json_2.1.1" / "game.tw-pddl").write_text("game", encoding="utf-8")
            (root / "logic" / "alfred.pddl").write_text("logic", encoding="utf-8")
            self.assertTrue(_valid_alfworld_data(root))

    def test_skip_package_probe_requires_every_declared_apt_package(self) -> None:
        output = "\n".join(f"{name}=1.0" for name in SYSTEM_PACKAGES) + "\n"
        command = ("dpkg-query", "-W", "-f=${Package}=${Version}\\n", *SYSTEM_PACKAGES)
        complete = FakeCommandRunner(
            available={"dpkg-query": "dpkg-query"},
            responses={command: CommandResult(command, 0, output)},
        )
        ctx = StageContext(Path.cwd(), SetupOptions(), complete)
        self.assertTrue(_system_packages_installed(ctx)[0])
        incomplete = FakeCommandRunner(
            available={"dpkg-query": "dpkg-query"},
            responses={command: CommandResult(command, 0, "git=1.0\n")},
        )
        self.assertFalse(_system_packages_installed(StageContext(Path.cwd(), SetupOptions(), incomplete))[0])

    def test_systemd_probe_rejects_offline_exit_one(self) -> None:
        command = ("systemctl", "is-system-running")
        offline = FakeCommandRunner(
            available={"systemctl": "systemctl"},
            responses={command: CommandResult(command, 1, "offline\n")},
        )
        self.assertFalse(_systemd_available(StageContext(Path.cwd(), SetupOptions(), offline)))
        degraded = FakeCommandRunner(
            available={"systemctl": "systemctl"},
            responses={command: CommandResult(command, 1, "degraded\n")},
        )
        self.assertTrue(_systemd_available(StageContext(Path.cwd(), SetupOptions(), degraded)))

    def test_model_stage_never_pulls_fallback_without_explicit_flag(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner = FakeCommandRunner(available={"ollama": "ollama"})

            def fake_http(url: str, payload: bytes | None, timeout: int) -> dict[str, object]:
                del payload, timeout
                if url.endswith("/api/tags"):
                    return {"models": [{"name": "hermes3:8b", "digest": "sha256:test"}]}
                if url.endswith("/api/show"):
                    return {"details": {"family": "llama"}}
                if url.endswith("/api/generate"):
                    return {"response": "READY"}
                if url.endswith("/api/ps"):
                    return {"models": [{"name": "hermes3:8b", "context_length": 65536}]}
                raise AssertionError(url)

            ctx = StageContext(root, SetupOptions(), runner, http_json=fake_http)
            outcome = run_candidate_models(ctx)
            self.assertEqual("passed", outcome.status)
            self.assertIn(("ollama", "pull", "hermes3:8b"), runner.commands)
            self.assertNotIn(("ollama", "pull", "llama3.1:8b"), runner.commands)

    def test_profile_verification_fails_closed_without_observed_json_inspection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "repo"
            root.mkdir()
            responses: dict[tuple[str, ...], CommandResult] = {}

            def response(command: tuple[str, ...], stdout: str) -> None:
                responses[command] = CommandResult(command, 0, stdout)

            response(("hermes", "--version"), "Hermes 1.2.3")
            response(("hermes", "--help"), "usage: hermes profile plugins")
            response(("hermes", "profile", "--help"), "create show")
            response(("hermes", "profile", "create", "--help"), "--no-skills")
            response(("hermes", "profile", "show", "--help"), "show profile")
            response(("hermes", "plugins", "--help"), "hooks")
            runner = FakeCommandRunner(available={"hermes": "hermes"}, responses=responses, reject_unconfigured=True)
            valid, metadata, detail = _hermes_profiles_available(StageContext(root, SetupOptions(), runner))
            self.assertFalse(valid)
            self.assertEqual({}, metadata)
            self.assertIn("blocked", detail)


class SetupRegistryTests(unittest.TestCase):
    def test_malformed_state_returns_a_clean_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "setup_status.json"
            path.write_text("{broken", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "Invalid setup state"):
                SetupRegistry(path).load()

    def test_invalidate_from_preserves_upstream_and_resets_selected_and_downstream(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            registry = SetupRegistry(Path(directory) / "state" / "setup_status.json")
            states = {
                stage.name: SetupState(
                    status="passed",
                    attempt_id=f"attempt-{index}",
                    report=f"report-{index}.json",
                    completed_at="2026-08-05T00:00:00Z",
                    input_fingerprint=f"fingerprint-{index}",
                )
                for index, stage in enumerate(SETUP_STAGES)
            }
            registry.save(states)

            invalidated = registry.invalidate_from("hermes")

            expected = EXPECTED_STAGE_ORDER[EXPECTED_STAGE_ORDER.index("hermes") :]
            self.assertEqual(expected, invalidated)
            reloaded = registry.load()
            for name in EXPECTED_STAGE_ORDER[: EXPECTED_STAGE_ORDER.index("hermes")]:
                self.assertEqual("passed", reloaded[name].status)
            for name in expected:
                self.assertEqual(SetupState(), reloaded[name])

    def test_invalidate_rejects_unknown_stage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            registry = SetupRegistry(Path(directory) / "setup_status.json")
            with self.assertRaisesRegex(ValueError, "Unknown setup stage"):
                registry.invalidate_from("not-a-stage")


class SetupOrchestratorTests(unittest.TestCase):
    def test_cli_requires_yes_before_real_setup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(SetupError, "requires --yes"):
                command_setup_machine(Path(directory), SetupOptions())

    def make_orchestrator(
        self,
        root: Path,
        options: SetupOptions,
        calls: list[str],
        *,
        final_ready: bool = True,
        outcomes: Mapping[str, StageOutcome] | None = None,
        failures: Mapping[str, Exception] | None = None,
        runner: FakeCommandRunner | None = None,
    ) -> SetupOrchestrator:
        fake_runner = runner or FakeCommandRunner(reject_unconfigured=True)
        return SetupOrchestrator(
            root,
            options,
            runner=fake_runner,
            handlers=make_handlers(
                calls,
                final_ready=final_ready,
                outcomes=outcomes,
                failures=failures,
            ),
            context_factory=guarded_context_factory,
        )

    def test_dry_run_never_records_passed_setup_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calls: list[str] = []
            orchestrator = self.make_orchestrator(
                root, SetupOptions(dry_run=True), calls, final_ready=False
            )

            aggregate = orchestrator.run()

            self.assertEqual(EXPECTED_STAGE_ORDER, calls)
            self.assertTrue(aggregate["dry_run"])
            self.assertFalse(aggregate["installation_ready"])
            self.assertEqual("skipped", aggregate["status"])
            self.assertTrue(all(item["status"] == "skipped" for item in aggregate["attempts"]))
            self.assertTrue(
                all(item["skip_reason"] == "dry-run; no external mutation was executed" for item in aggregate["attempts"])
            )
            self.assertTrue(all(state["status"] == "pending" for state in aggregate["stages"].values()))
            self.assertFalse((root / "state" / "setup_status.json").exists())

    def test_normal_mocked_sequence_writes_passing_aggregate_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calls: list[str] = []
            orchestrator = self.make_orchestrator(root, SetupOptions(), calls)

            aggregate = orchestrator.run()

            self.assertEqual(EXPECTED_STAGE_ORDER, calls)
            self.assertEqual("passed", aggregate["status"])
            self.assertTrue(aggregate["installation_ready"])
            self.assertFalse(aggregate["pilot_ready"])
            self.assertFalse(aggregate["real_integration_tested"])
            self.assertIn("No real Hermes-to-ALFWorld compatibility claim", aggregate["compatibility_claim"])
            self.assertEqual(10, len(aggregate["attempts"]))
            self.assertTrue(all(item["status"] == "passed" for item in aggregate["attempts"]))
            self.assertTrue(all(state["status"] == "passed" for state in aggregate["stages"].values()))
            self.assertEqual(
                aggregate,
                json.loads(
                    (root / "artifacts" / "stage_reports" / "installation.json").read_text(
                        encoding="utf-8"
                    )
                ),
            )

    def test_stage_failure_is_structured_and_stops_downstream_execution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calls: list[str] = []
            orchestrator = self.make_orchestrator(
                root,
                SetupOptions(),
                calls,
                failures={
                    "hermes": StageFailure(
                        "Hermes capability probe failed", "Inspect the captured help output."
                    )
                },
            )

            aggregate = orchestrator.run()

            self.assertEqual(EXPECTED_STAGE_ORDER[:5], calls)
            self.assertEqual("failed", aggregate["status"])
            self.assertFalse(aggregate["installation_ready"])
            failure = aggregate["attempts"][-1]
            self.assertEqual("hermes", failure["stage"])
            self.assertEqual(["Hermes capability probe failed"], failure["errors"])
            self.assertEqual("Inspect the captured help output.", failure["remediation"])
            states = orchestrator.status()
            self.assertEqual("failed", states["hermes"]["status"])
            self.assertEqual("pending", states["alfworld-package"]["status"])

    def test_skipped_stage_is_recorded_and_allows_explicit_stop(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calls: list[str] = []
            skipped = StageOutcome(
                status="skipped",
                skip_reason="--skip-system-packages requested and prerequisites are incomplete",
                warnings=["Later stages may fail because system packages were skipped."],
            )
            orchestrator = self.make_orchestrator(
                root,
                SetupOptions(skip_system_packages=True),
                calls,
                outcomes={"system-packages": skipped},
            )

            aggregate = orchestrator.run(stop_after="system-packages")

            self.assertEqual(["preflight", "system-packages"], calls)
            self.assertEqual("skipped", aggregate["status"])
            attempt = aggregate["attempts"][-1]
            self.assertEqual("skipped", attempt["status"])
            self.assertEqual(skipped.skip_reason, attempt["skip_reason"])
            self.assertEqual("skipped", orchestrator.status()["system-packages"]["status"])

    def test_resume_reuses_current_passed_stages_without_calling_handlers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            initial_calls: list[str] = []
            self.make_orchestrator(root, SetupOptions(), initial_calls).run()
            self.assertEqual(EXPECTED_STAGE_ORDER, initial_calls)

            resumed_calls: list[str] = []
            resumed = self.make_orchestrator(root, SetupOptions(resume=True), resumed_calls)
            with mock.patch.object(resumed, "_current", return_value=True) as current:
                aggregate = resumed.run()

            self.assertEqual([], resumed_calls)
            self.assertEqual([], aggregate["attempts"])
            self.assertEqual(len(EXPECTED_STAGE_ORDER), current.call_count)
            self.assertTrue(all(state["status"] == "passed" for state in resumed.status().values()))

    def test_force_stage_requires_yes_before_state_is_changed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calls: list[str] = []
            orchestrator = self.make_orchestrator(
                root, SetupOptions(force_stage="hermes"), calls
            )

            with self.assertRaisesRegex(SetupError, "--force-stage requires --yes"):
                orchestrator.run()

            self.assertEqual([], calls)
            self.assertFalse((root / "state" / "setup_status.json").exists())

    def test_force_stage_with_yes_invalidates_downstream_and_reruns_selected_stage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = SetupRegistry(root / "state" / "setup_status.json")
            registry.save(
                {
                    stage.name: SetupState(
                        status="passed",
                        attempt_id=f"old-{index}",
                        report=f"old-{index}.json",
                        completed_at="2026-08-05T00:00:00Z",
                        input_fingerprint=f"old-{index}",
                    )
                    for index, stage in enumerate(SETUP_STAGES)
                }
            )
            calls: list[str] = []
            orchestrator = self.make_orchestrator(
                root, SetupOptions(yes=True, force_stage="hermes"), calls
            )

            aggregate = orchestrator.run(only_stage="hermes")

            downstream = EXPECTED_STAGE_ORDER[EXPECTED_STAGE_ORDER.index("hermes") :]
            self.assertEqual(downstream, aggregate["invalidated_stages"])
            self.assertEqual(["hermes"], calls)
            states = orchestrator.status()
            for name in EXPECTED_STAGE_ORDER[: EXPECTED_STAGE_ORDER.index("hermes")]:
                self.assertEqual("passed", states[name]["status"])
            self.assertEqual("passed", states["hermes"]["status"])
            for name in downstream[1:]:
                self.assertEqual("pending", states[name]["status"])

    def test_stage_report_has_required_fields_and_redacted_commands(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            command = ("fake-tool", "API_KEY=do-not-record", "Bearer hidden-token")
            runner = FakeCommandRunner(
                responses={command: CommandResult(command, 0, "ok", "")}
            )
            calls: list[str] = []

            def preflight(context: StageContext) -> StageOutcome:
                calls.append("preflight")
                context.runner.run(command, check=True)
                return StageOutcome(
                    probes=[ProbeResult("fake-probe", True, "host 192.0.2.10 token=json-secret", "1.2.3")],
                    artifacts=[context.root / "artifacts" / "manifests" / "fixture.json"],
                    warnings=["fixture warning"],
                    metadata={"source": "unit-test", "authorization": "Bearer metadata-secret"},
                )

            orchestrator = SetupOrchestrator(
                root,
                SetupOptions(),
                runner=runner,
                handlers={"preflight": preflight},
                context_factory=guarded_context_factory,
            )
            aggregate = orchestrator.run(only_stage="preflight")
            report_files = list((root / "artifacts" / "stage_reports").glob("setup-preflight-*.json"))

            self.assertEqual(["preflight"], calls)
            self.assertEqual(1, len(report_files))
            report = json.loads(report_files[0].read_text(encoding="utf-8"))
            required_fields = {
                "stage",
                "status",
                "started_at",
                "completed_at",
                "run_id",
                "attempt_id",
                "dry_run",
                "input_fingerprint",
                "commands",
                "probes",
                "artifacts",
                "warnings",
                "errors",
                "remediation",
                "skip_reason",
                "metadata",
            }
            self.assertEqual(required_fields, set(report))
            self.assertEqual("passed", report["status"])
            self.assertTrue(report["started_at"].endswith("Z"))
            self.assertTrue(report["completed_at"].endswith("Z"))
            self.assertEqual(64, len(report["input_fingerprint"]))
            self.assertEqual(
                [["fake-tool", "API_KEY=<redacted>", "Bearer <redacted>"]],
                report["commands"],
            )
            self.assertNotIn("do-not-record", json.dumps(report))
            self.assertNotIn("hidden-token", json.dumps(report))
            self.assertNotIn("json-secret", json.dumps(report))
            self.assertNotIn("metadata-secret", json.dumps(report))
            self.assertNotIn("192.0.2.10", json.dumps(report))
            self.assertEqual("fake-probe", report["probes"][0]["name"])
            self.assertEqual("$REPO/artifacts/manifests/fixture.json", report["artifacts"][0])
            self.assertEqual(report, aggregate["attempts"][0])

    def test_mocked_orchestration_never_uses_subprocess_network_or_downloads(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner = FakeCommandRunner(reject_unconfigured=True)
            calls: list[str] = []
            orchestrator = self.make_orchestrator(
                root,
                SetupOptions(),
                calls,
                runner=runner,
            )

            with (
                mock.patch("rq1.setup.runner.subprocess.run") as subprocess_run,
                mock.patch("rq1.setup.runner.subprocess.Popen") as subprocess_popen,
                mock.patch("rq1.setup.probes.urlopen") as urlopen,
            ):
                result = orchestrator.run()

            self.assertTrue(result["installation_ready"])
            self.assertEqual(EXPECTED_STAGE_ORDER, calls)
            self.assertEqual([], runner.commands)
            self.assertEqual([], runner.background_commands)
            subprocess_run.assert_not_called()
            subprocess_popen.assert_not_called()
            urlopen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
