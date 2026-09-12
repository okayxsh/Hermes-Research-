"""Harness-owned real ALFWorld episode control through the Hermes registry.

The experiment harness owns the episode identifier and all environment state.
The local Hermes model selects one already-admissible textual action at a time;
each action is then dispatched through the *real* project plugin registry.  No
``AIAgent.run_conversation`` loop is involved, which avoids allowing a model to
invent an episode identifier or silently lose the active episode.
"""
from __future__ import annotations

import json
import os
import select
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.error import URLError
from urllib.request import Request, urlopen
from uuid import uuid4

from rq1.bridge.app import create_bridge_server
from rq1.bridge.adapters.capabilities import default_data_dir


class EpisodeDriverError(RuntimeError):
    """Raised when the observed Hermes/bridge contract cannot be honoured."""


@dataclass(frozen=True)
class ActionRecord:
    step: int
    phase: str
    action: str
    action_valid: bool | None
    done: bool
    success: bool | None
    observation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "phase": self.phase,
            "action": self.action,
            "action_valid": self.action_valid,
            "done": self.done,
            "success": self.success,
            "observation": self.observation,
        }


def parse_model_action(response: str, admissible_actions: Sequence[str]) -> str | None:
    """Accept only one exact, machine-parseable legal action.

    This intentionally has no case-insensitive/sub-string matching and no
    fallback action.  A malformed model response is a model-selection failure,
    not permission for the controller to choose on the model's behalf.
    """
    lines = [line.strip() for line in response.splitlines() if line.strip()]
    if len(lines) != 1 or not lines[0].startswith("ACTION:"):
        return None
    action = lines[0].removeprefix("ACTION:").strip()
    return action if action in admissible_actions else None


_REGISTRY_WORKER = r'''
import json
import os
import sys

sys.path.insert(0, os.environ["RQ1_PROJECT_SOURCE"])
from hermes_cli.plugins import (
    PluginContext,
    _dispatch_pre_tool_call_hooks,
    collect_directory_manifests,
    get_plugin_manager,
)
from model_tools import _emit_post_tool_call_hook

manager = get_plugin_manager()
manager.discover_and_load()
manifest = next(
    item for item in collect_directory_manifests()
    if item.name == "alfworld-experiment" and item.source == "project"
)
context = PluginContext(manifest, manager)
plugin = next(item for item in manager.list_plugins() if item["name"] == "alfworld-experiment")
if not plugin.get("enabled"):
    raise RuntimeError("Project ALFWorld plugin was discovered but is not enabled")

for line in sys.stdin:
    try:
        request = json.loads(line)
        if request.get("command") != "dispatch":
            raise RuntimeError("Unsupported registry worker command")
        name = request["tool"]
        args = request["args"]
        hook_kwargs = request["hook_kwargs"]
        blocked, modified = _dispatch_pre_tool_call_hooks(name, args, **hook_kwargs)
        if blocked:
            raise RuntimeError(f"Hermes pre_tool_call blocked {name}: {blocked}")
        effective = modified if modified is not None else args
        raw = context.dispatch_tool(name, effective, **hook_kwargs)
        _emit_post_tool_call_hook(
            function_name=name,
            function_args=effective,
            result=raw,
            **hook_kwargs,
        )
        parsed = json.loads(raw)
        if not isinstance(parsed, dict) or parsed.get("ok") is not True:
            raise RuntimeError(f"Hermes registry dispatch failed for {name}")
        response = {"ok": True, "payload": parsed}
    except Exception as exc:
        response = {"ok": False, "error_type": type(exc).__name__, "error": str(exc)}
    print("RQ1_REGISTRY_RESULT=" + json.dumps(response, sort_keys=True), flush=True)
'''


class _RegistryWorker:
    """One fresh installed-Hermes registry process for one experimental task."""

    def __init__(
        self,
        *,
        root: Path,
        bridge_url: str,
        output_dir: Path,
        run_id: str,
        attempt_id: str,
        profile: str,
        timeout_seconds: float,
    ) -> None:
        self.root = root
        self.bridge_url = bridge_url
        self.output_dir = output_dir
        self.run_id = run_id
        self.attempt_id = attempt_id
        self.profile = profile
        self.timeout_seconds = timeout_seconds
        self.process: subprocess.Popen[str] | None = None
        self._stderr: Any = None

    @staticmethod
    def _hermes_python() -> str:
        """Use Hermes's interpreter, never the repository virtual environment."""
        candidate = Path("/usr/local/lib/hermes-agent/venv/bin/python")
        if candidate.is_file():
            return str(candidate)
        raise EpisodeDriverError("Installed Hermes Python runtime was not found")

    def start(self) -> None:
        home = self.output_dir / "hermes-home"
        home.mkdir(parents=True, exist_ok=True)
        (home / "config.yaml").write_text(
            "plugins:\n  enabled:\n    - alfworld-experiment\n"
            "tools:\n  tool_search:\n    enabled: \"off\"\n",
            encoding="utf-8",
        )
        self._stderr = (self.output_dir / "registry.stderr.log").open("w", encoding="utf-8")
        environment = os.environ.copy()
        environment.update(
            {
                "HERMES_HOME": str(home),
                "HERMES_ENABLE_PROJECT_PLUGINS": "1",
                "RQ1_PROJECT_SOURCE": str(self.root / "src"),
                "RQ1_BRIDGE_URL": self.bridge_url,
                "RQ1_BRIDGE_TIMEOUT_SECONDS": str(int(self.timeout_seconds)),
                "RQ1_HERMES_EVENT_LOG": str(self.output_dir / "plugin-events.jsonl"),
                "RQ1_RUN_ID": self.run_id,
                "RQ1_ATTEMPT_ID": self.attempt_id,
                "RQ1_PROFILE": self.profile,
            }
        )
        self.process = subprocess.Popen(
            (self._hermes_python(), "-u", "-c", _REGISTRY_WORKER),
            cwd=self.root,
            env=environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr,
            text=True,
            bufsize=1,
        )

    def dispatch(self, tool: str, args: Mapping[str, Any], hook_kwargs: Mapping[str, str]) -> dict[str, Any]:
        if self.process is None or self.process.stdin is None or self.process.stdout is None:
            raise EpisodeDriverError("Hermes registry worker is not running")
        self.process.stdin.write(
            json.dumps(
                {"command": "dispatch", "tool": tool, "args": dict(args), "hook_kwargs": dict(hook_kwargs)},
                sort_keys=True,
            )
            + "\n"
        )
        self.process.stdin.flush()
        deadline = time.monotonic() + self.timeout_seconds
        while time.monotonic() < deadline:
            ready, _, _ = select.select([self.process.stdout], [], [], min(1.0, deadline - time.monotonic()))
            if not ready:
                continue
            line = self.process.stdout.readline()
            if not line:
                raise EpisodeDriverError("Hermes registry worker exited before returning a tool result")
            if not line.startswith("RQ1_REGISTRY_RESULT="):
                continue
            response = json.loads(line.split("=", 1)[1])
            if response.get("ok") is not True:
                raise EpisodeDriverError(
                    f"Hermes registry {tool} failed: {response.get('error_type')}: {response.get('error')}"
                )
            payload = response.get("payload")
            if not isinstance(payload, dict):
                raise EpisodeDriverError("Hermes registry returned a malformed payload")
            return payload
        raise EpisodeDriverError(f"Hermes registry {tool} exceeded {self.timeout_seconds:.0f}s")

    def close(self) -> None:
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
            self.process = None
        if self._stderr is not None:
            self._stderr.close()
            self._stderr = None


class RealEpisodeSession:
    """A live, single-task session whose identifier is never model-visible."""

    def __init__(
        self,
        driver: "RealEpisodeDriver",
        *,
        output_dir: Path,
        run_id: str,
        attempt_id: str,
        profile: str,
    ) -> None:
        self.driver = driver
        self.output_dir = output_dir
        self.run_id = run_id
        self.attempt_id = attempt_id
        self.profile = profile
        self.worker = _RegistryWorker(
            root=driver.root,
            bridge_url=driver.bridge_url,
            output_dir=output_dir,
            run_id=run_id,
            attempt_id=attempt_id,
            profile=profile,
            timeout_seconds=driver.bridge_timeout_seconds,
        )
        self.episode_id: str | None = None
        self.state: dict[str, Any] | None = None
        self.records: list[ActionRecord] = []
        self.invalid_model_actions = 0
        self._call_number = 0
        self._events = output_dir / "episode-events.jsonl"

    def __enter__(self) -> "RealEpisodeSession":
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self.worker.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.worker.close()

    def _event(self, event: str, payload: Mapping[str, Any]) -> None:
        record = {"event": event, "timestamp": time.time(), "payload": dict(payload)}
        with self._events.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _dispatch(self, tool: str, args: Mapping[str, Any]) -> dict[str, Any]:
        self._call_number += 1
        hooks = {
            "task_id": str(args.get("task_id") or self.state and self.state.get("task_id") or ""),
            "session_id": self.attempt_id,
            "tool_call_id": f"rq1-driver-tool-{self._call_number}",
            "api_request_id": f"rq1-driver-request-{self._call_number}",
        }
        response = self.worker.dispatch(tool, args, hooks)
        result = response.get("result")
        if not isinstance(result, dict):
            raise EpisodeDriverError(f"Hermes {tool} response lacks a result object")
        self._event("tool_result", {"tool": tool, "request": dict(args), "response": result})
        return result

    def start(self, task_id: str, split: str, seed: int, action_limit: int) -> dict[str, Any]:
        result = self._dispatch(
            "alfworld_start",
            {"task_id": task_id, "split": split, "seed": seed, "action_limit": action_limit},
        )
        episode_id = result.get("episode_id")
        if not isinstance(episode_id, str) or not episode_id:
            raise EpisodeDriverError("Real alfworld_start returned no valid episode_id")
        self.episode_id = episode_id
        self.state = result
        return result

    def step(self, action: str, *, phase: str) -> dict[str, Any]:
        if self.episode_id is None or self.state is None:
            raise EpisodeDriverError("Cannot step before real episode start")
        admissible = self.state.get("admissible_actions")
        if not isinstance(admissible, list) or action not in admissible:
            raise EpisodeDriverError("Controller refused to dispatch a non-admissible action")
        result = self._dispatch("alfworld_step", {"episode_id": self.episode_id, "action": action})
        if result.get("episode_id") != self.episode_id:
            raise EpisodeDriverError("Bridge response changed the Python-owned episode_id")
        self.state = result
        self.records.append(
            ActionRecord(
                step=len(self.records) + 1,
                phase=phase,
                action=action,
                action_valid=result.get("action_valid"),
                done=bool(result.get("done")),
                success=result.get("success") if isinstance(result.get("success"), bool) else None,
                observation=str(result.get("observation", "")),
            )
        )
        return result

    def abort(self, reason: str) -> None:
        if self.episode_id is None or self.state is None or self.state.get("done"):
            return
        try:
            self._dispatch("alfworld_abort", {"episode_id": self.episode_id, "reason": reason})
        except EpisodeDriverError as exc:
            self._event("abort_error", {"error": str(exc)})

    def apply_scripted(self, actions: Sequence[str], *, phase: str) -> dict[str, Any]:
        for action in actions:
            result = self.step(action, phase=phase)
            if result.get("action_valid") is not True or result.get("done"):
                raise EpisodeDriverError("Reference/oracle action was invalid or terminal before completion")
        assert self.state is not None
        return self.state

    def _model_response(self, prompt: str) -> str:
        payload = {
            "model": self.driver.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0},
        }
        request = Request(
            self.driver.ollama_url + "/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urlopen(request, timeout=self.driver.model_timeout_seconds) as response:
                value = json.loads(response.read().decode("utf-8"))
        except (URLError, OSError, TimeoutError, json.JSONDecodeError) as exc:
            raise EpisodeDriverError(f"Ollama action selection failed: {type(exc).__name__}") from exc
        content = value.get("message", {}).get("content") if isinstance(value, dict) else None
        if not isinstance(content, str):
            raise EpisodeDriverError("Ollama action selection returned no text response")
        return content

    def choose_action(self, recovery_memory: Mapping[str, Any] | None = None) -> str | None:
        if self.state is None:
            raise EpisodeDriverError("Cannot select an action before start")
        admissible = self.state.get("admissible_actions")
        if not isinstance(admissible, list) or not all(isinstance(item, str) for item in admissible):
            raise EpisodeDriverError("Real ALFWorld state lacks admissible actions")
        memory = None
        if recovery_memory is not None:
            memory = recovery_memory.get("recovery_memory", recovery_memory)
        context = {
            "task_instruction": self.state.get("instruction"),
            "observation": self.state.get("observation"),
            "inventory": self.state.get("inventory") or [],
            "admissible_actions": admissible,
            "recovery_memory": memory,
        }
        base = (
            "Choose exactly one legal ALFWorld action. Do not explain or reason aloud. "
            "Return exactly one line in this format: ACTION: <exact action>.\n"
            + json.dumps(context, ensure_ascii=False, sort_keys=True)
        )
        for selection_attempt in range(2):
            prompt = base if selection_attempt == 0 else (
                "Your prior response was not an exact legal ACTION line. Return only "
                "ACTION: <exact action from admissible_actions>.\n" + json.dumps(context, ensure_ascii=False, sort_keys=True)
            )
            response = self._model_response(prompt)
            action = parse_model_action(response, admissible)
            self._event(
                "model_selection",
                {
                    "selection_attempt": selection_attempt + 1,
                    "context": context,
                    "response_format_valid": action is not None,
                    "selected_action": action,
                },
            )
            if action is not None:
                return action
        self.invalid_model_actions += 1
        return None

    def run_model_loop(
        self,
        action_budget: int,
        *,
        phase: str,
        recovery_memory: Mapping[str, Any] | None = None,
    ) -> list[ActionRecord]:
        if action_budget < 1:
            raise ValueError("action_budget must be positive")
        starting = len(self.records)
        for _ in range(action_budget):
            assert self.state is not None
            if self.state.get("done"):
                break
            action = self.choose_action(recovery_memory)
            if action is None:
                self.abort("model returned no exact admissible action after bounded clarification")
                break
            self.step(action, phase=phase)
        return self.records[starting:]


class RealEpisodeDriver:
    """Long-lived real bridge with fresh Hermes/model context per episode."""

    def __init__(
        self,
        root: Path,
        *,
        data_dir: Path | None = None,
        model_name: str = "hermes3:8b",
        ollama_url: str = "http://127.0.0.1:11434",
        bridge_timeout_seconds: float = 300,
        model_timeout_seconds: float = 180,
    ) -> None:
        self.root = root.resolve()
        self.data_dir = (data_dir or default_data_dir()).resolve()
        self.model_name = model_name
        self.ollama_url = ollama_url.rstrip("/")
        self.bridge_timeout_seconds = bridge_timeout_seconds
        self.model_timeout_seconds = model_timeout_seconds
        self._server: Any = None
        self._thread: threading.Thread | None = None

    @property
    def bridge_url(self) -> str:
        if self._server is None:
            raise EpisodeDriverError("Real episode driver bridge is not running")
        return f"http://127.0.0.1:{self._server.server_port}"

    def __enter__(self) -> "RealEpisodeDriver":
        self._server = create_bridge_server(
            self.root / "artifacts" / "prelaunch" / "bridge",
            port=0,
            mode="real",
            data_dir=self.data_dir,
        )
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=10)
        self._server = None
        self._thread = None

    def session(
        self,
        *,
        output_dir: Path,
        run_id: str,
        attempt_id: str | None = None,
        profile: str = "rq1-prelaunch-pilot",
    ) -> RealEpisodeSession:
        if self._server is None:
            raise EpisodeDriverError("Use RealEpisodeDriver as a context manager")
        return RealEpisodeSession(
            self,
            output_dir=output_dir,
            run_id=run_id,
            attempt_id=attempt_id or str(uuid4()),
            profile=profile,
        )
