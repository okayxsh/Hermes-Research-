from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Mapping, Protocol

from rq1.setup.models import CommandResult


_SECRET_NAME = r"api[_-]?key|token|password|secret|authorization"
_SECRET_ASSIGNMENT = re.compile(rf"(?i)({_SECRET_NAME})=([^\s]+)")
_SECRET_JSON = re.compile(
    rf'(?i)(["\']?(?:{_SECRET_NAME})["\']?\s*:\s*)'
    r'("(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|[^,}\s]+)'
)
_BEARER = re.compile(r"(?i)\bbearer\s+[^\s,}\"]+")
_SECRET_FLAGS = {
    "--api-key",
    "--api_key",
    "--token",
    "--password",
    "--secret",
    "--authorization",
}


def redact(value: str) -> str:
    value = _SECRET_ASSIGNMENT.sub(r"\1=<redacted>", value)
    value = _SECRET_JSON.sub(r"\1\"<redacted>\"", value)
    return _BEARER.sub("Bearer <redacted>", value)


def redact_command(command: tuple[str, ...] | list[str]) -> list[str]:
    redacted: list[str] = []
    hide_next = False
    for part in command:
        value = str(part)
        if hide_next:
            redacted.append("<redacted>")
            hide_next = False
            continue
        redacted.append(redact(value))
        hide_next = value.lower() in _SECRET_FLAGS
    return redacted


class CommandRunner(Protocol):
    commands: list[tuple[str, ...]]

    def which(self, command: str) -> str | None: ...

    def run(
        self,
        command: tuple[str, ...],
        *,
        cwd: Path | None = None,
        env: Mapping[str, str] | None = None,
        timeout: int = 300,
        check: bool = False,
    ) -> CommandResult: ...

    def start_background(
        self,
        command: tuple[str, ...],
        *,
        cwd: Path,
        env: Mapping[str, str],
        log_path: Path,
        pid_path: Path,
    ) -> int: ...


class SubprocessRunner:
    def __init__(self, dry_run: bool = False, verbose: bool = False) -> None:
        self.dry_run = dry_run
        self.verbose = verbose
        self.commands: list[tuple[str, ...]] = []

    def which(self, command: str) -> str | None:
        from shutil import which

        return which(command)

    def run(
        self,
        command: tuple[str, ...],
        *,
        cwd: Path | None = None,
        env: Mapping[str, str] | None = None,
        timeout: int = 300,
        check: bool = False,
    ) -> CommandResult:
        self.commands.append(command)
        if self.dry_run:
            return CommandResult(command, 0, "dry-run")
        merged_env = os.environ.copy()
        if env:
            merged_env.update(env)
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=merged_env,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        result = CommandResult(command, completed.returncode, completed.stdout, completed.stderr)
        if check and not result.ok:
            detail = redact((result.stderr or result.stdout).strip())[:1000]
            raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(redact_command(command))}: {detail}")
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
        self.commands.append(command)
        if self.dry_run:
            return 0
        merged_env = os.environ.copy()
        merged_env.update(env)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        pid_path.parent.mkdir(parents=True, exist_ok=True)
        output = log_path.open("ab")
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=merged_env,
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        output.close()
        pid_path.write_text(f"{process.pid}\n", encoding="utf-8")
        return process.pid
