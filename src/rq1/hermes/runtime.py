"""Resolve and probe Hermes's application venv; never fall back to base Python."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


class HermesRuntimeError(RuntimeError):
    pass


def resolve_hermes_python(*, home: Path | None = None) -> Path:
    explicit = os.environ.get("RQ1_HERMES_PYTHON")
    if explicit:
        candidates = [Path(explicit).expanduser()]
    else:
        candidates = [Path("/usr/local/lib/hermes-agent/venv/bin/python")]
        candidates.extend(sorted(((home or Path.home()) / ".hermes/installs").glob("*/environments/*/venv/bin/python")))
        candidates = [path for path in candidates if path.is_file()]
    if len(candidates) != 1:
        raise HermesRuntimeError("Set RQ1_HERMES_PYTHON to one installed Hermes application venv interpreter; found " + str(len(candidates)))
    candidate = candidates[0]
    if not candidate.is_absolute() or not candidate.is_file() or not (candidate.parent.parent / "pyvenv.cfg").is_file():
        raise HermesRuntimeError("Hermes interpreter must be an absolute existing venv Python with pyvenv.cfg")
    # Preserve the venv entry-point path: resolving its Python symlink can select base Python.
    return candidate


def validate_hermes_python(candidate: Path) -> dict:
    script = """
import json, sys
assert sys.prefix != sys.base_prefix, 'base Python is forbidden'
from hermes_cli.plugins import PluginContext, _dispatch_pre_tool_call_hooks, collect_directory_manifests, get_plugin_manager
from model_tools import _emit_post_tool_call_hook
sys.path.insert(0, sys.argv[1])
from rq1.hermes.plugin_runtime import TOOL_SCHEMAS
assert 'alfworld_start' in TOOL_SCHEMAS
print('RQ1_HERMES_PROBE=' + json.dumps({'python':sys.executable,'prefix':sys.prefix,'imports_verified':True,'plugin_dispatch_verified':False}))
"""
    try:
        result = subprocess.run([str(candidate), "-B", "-c", script, str(Path(__file__).resolve().parents[2])],
                                capture_output=True, text=True, timeout=30, check=False,
                                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    except (OSError, subprocess.SubprocessError) as exc:
        raise HermesRuntimeError(f"Hermes import probe failed: {exc}") from exc
    for line in reversed(result.stdout.splitlines()):
        if result.returncode == 0 and line.startswith("RQ1_HERMES_PROBE="):
            return json.loads(line.split("=", 1)[1])
    raise HermesRuntimeError("Hermes venv lacks required runtime/plugin imports: " + result.stderr[-1500:])


def checked_hermes_python() -> Path:
    candidate = resolve_hermes_python()
    validate_hermes_python(candidate)
    return candidate


def hermes_runtime_available() -> bool:
    try:
        checked_hermes_python()
        return True
    except HermesRuntimeError:
        return False


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    try:
        report = validate_hermes_python(resolve_hermes_python())
    except HermesRuntimeError as exc:
        print(json.dumps({"status": "BLOCKED", "reason": str(exc)}))
        raise SystemExit(1)
    print(json.dumps({"status": "PASS_IMPORTS_ONLY", **report}, indent=2))


if __name__ == "__main__":
    main()
