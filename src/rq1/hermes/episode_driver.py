"""Harness-owned real ALFWorld episode control through the Hermes registry.

The experiment harness owns the episode identifier and all environment state.
The local Hermes model chooses the index of one currently admissible action;
Python maps that index back to the exact action string and dispatches it
through the *real* project plugin registry.  No ``AIAgent.run_conversation``
loop is involved, so the model can never invent an episode identifier, tool
arguments, or a free-text command, or silently lose the active episode.
"""
from __future__ import annotations

import json
import os
import re
import select
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib.error import URLError
from urllib.request import Request, urlopen
from uuid import uuid4

from rq1.bridge.app import create_bridge_server
from rq1.bridge.adapters.capabilities import default_data_dir
from rq1.retrieval.query import INVENTORY_NOT_OBSERVED_MARKER

# Canonical Ollama inference seed shared by every RQ1 condition.  It is
# independent of the frozen experimental task seeds (11, 29, 47).
INFERENCE_SEED = 42
ACTION_SELECTION_PROTOCOL = "action-index-history-v2"
# Decision 008: every decision sees the complete observable history of the
# current episode (prior actions and their resulting observations only).
ACTION_HISTORY_POLICY = "full_within_episode_actions_and_observations"
# Decision 009: interface corrections found in non-scientific prelaunch checks.
INITIAL_OBSERVATION_POLICY = "verbatim_reset_observation_at_every_decision"
INVENTORY_POLICY = "not_observed_marker_inventory_only_via_inventory_action"
ACTION_INDEX_PARSING_POLICY = "exactly_one_action_index_line_surrounding_prose_ignored"
EMPTY_HISTORY_MARKER = "(no previous steps)"
MAX_SELECTION_ATTEMPTS = 3
RETRY_CLARIFICATION = (
    "Your previous response was invalid. Return only ACTION_INDEX: <integer> "
    "where the integer is one of the listed indices."
)
_GOAL_PATTERN = re.compile(r"^Your task is to:[ \t]*(\S.*?)[ \t]*$", re.MULTILINE)
ACTION_INDEX_TOKEN = "ACTION_INDEX"
_ACTION_INDEX_PATTERN = re.compile(r"ACTION_INDEX:[ \t]*([0-9]+)")


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


@dataclass(frozen=True)
class ActionSelection:
    action: str | None
    index: int | None
    attempts: tuple[dict[str, Any], ...]


def extract_task_goal(initial_observation: str) -> str:
    """Return the natural-language goal from the ALFWorld initial observation.

    ALFWorld 0.4.2 exposes the goal only inside the reset observation.  This is
    an exact-marker match; it never falls back to the indexed task ID and never
    paraphrases the goal.
    """
    matches = _GOAL_PATTERN.findall(initial_observation)
    if len(matches) != 1:
        raise EpisodeDriverError("Initial ALFWorld observation does not contain exactly one task goal")
    return matches[0]


def render_episode_history(history: Sequence[tuple[str, str]]) -> str:
    """Every executed step of this episode, oldest first: action, then its observation."""
    if not history:
        return EMPTY_HISTORY_MARKER
    return "\n\n".join(
        f"Step {number}\nAction: {action}\nObservation: {observation}"
        for number, (action, observation) in enumerate(history, 1)
    )


def render_action_prompt(
    *,
    task_goal: str,
    initial_observation: str,
    observation: str,
    inventory: Sequence[str],
    admissible_actions: Sequence[str],
    recovery_memory: Mapping[str, Any] | None = None,
    attempt: int = 1,
    history: Sequence[tuple[str, str]] = (),
) -> str:
    """Build the condition-identical action-selection prompt.

    Only the injected recovery-memory block may differ between conditions.
    The initial observation is this episode's verbatim ALFWorld reset text.
    The history holds only already-executed actions and their observations;
    scores, library names, episode identifiers, world facts, and
    reference/oracle actions are never part of the prompt.  Inventory is never
    inferred: it is observed only when the agent chooses the ``inventory``
    action, and then only as that step's observation.
    """
    sections = [
        "TASK GOAL:\n" + task_goal,
        "INITIAL OBSERVATION:\n" + initial_observation,
        "EPISODE HISTORY:\n" + render_episode_history(history),
        "CURRENT OBSERVATION:\n" + observation,
        "CURRENT INVENTORY:\n" + (", ".join(inventory) if inventory else INVENTORY_NOT_OBSERVED_MARKER),
    ]
    if recovery_memory is not None:
        sections.append("RECOVERY MEMORY:\n" + json.dumps(recovery_memory, ensure_ascii=False, sort_keys=True))
    sections.append(
        "ADMISSIBLE ACTIONS:\n"
        + "\n".join(f"{index}. {action}" for index, action in enumerate(admissible_actions))
    )
    instruction = "Return exactly:\nACTION_INDEX: <integer>"
    if attempt > 1:
        # Retries differ only by this format clarification; the attempt number
        # keeps seeded retries from deterministically repeating the same output.
        instruction = f"{RETRY_CLARIFICATION} (attempt {attempt} of {MAX_SELECTION_ATTEMPTS})\n" + instruction
    sections.append(instruction)
    return "\n\n".join(sections)


def classify_action_index(response: str, admissible_actions: Sequence[str]) -> tuple[int | None, str | None]:
    """Return ``(index, None)`` or ``(None, rejection_reason)`` for one response.

    The token ``ACTION_INDEX`` must occur exactly once in the whole response,
    on a line that is exactly ``ACTION_INDEX: <integer>`` naming a listed
    index; prose on other lines is ignored (Decision 009).  A second mention,
    even of the same index, is ambiguous.  There is no string/fuzzy matching
    of action names and no fallback: anything else is an invalid selection,
    not permission to choose on the model's behalf.
    """
    mentions = response.count(ACTION_INDEX_TOKEN)
    if mentions == 0:
        return None, "no_action_index"
    if mentions > 1:
        return None, "multiple_action_index_mentions"
    line = next(line.strip() for line in response.splitlines() if ACTION_INDEX_TOKEN in line)
    match = _ACTION_INDEX_PATTERN.fullmatch(line)
    if match is None:
        return None, "malformed_action_index"
    index = int(match.group(1))
    if index >= len(admissible_actions):
        return None, "index_out_of_range"
    return index, None


def parse_action_index(response: str, admissible_actions: Sequence[str]) -> int | None:
    return classify_action_index(response, admissible_actions)[0]


def select_admissible_action(
    ask: Callable[[str], str],
    *,
    task_goal: str,
    initial_observation: str,
    observation: str,
    inventory: Sequence[str],
    admissible_actions: Sequence[str],
    recovery_memory: Mapping[str, Any] | None = None,
    max_attempts: int = MAX_SELECTION_ATTEMPTS,
    history: Sequence[tuple[str, str]] = (),
) -> ActionSelection:
    """Map a model-chosen index to one exact admissible action with bounded retry."""
    if not admissible_actions:
        raise EpisodeDriverError("Cannot select from an empty admissible-action list")
    attempts: list[dict[str, Any]] = []
    for attempt in range(1, max_attempts + 1):
        prompt = render_action_prompt(
            task_goal=task_goal,
            initial_observation=initial_observation,
            observation=observation,
            inventory=inventory,
            admissible_actions=admissible_actions,
            recovery_memory=recovery_memory,
            attempt=attempt,
            history=history,
        )
        response = ask(prompt)
        index, rejection = classify_action_index(response, admissible_actions)
        action = admissible_actions[index] if index is not None else None
        attempts.append(
            {
                "attempt": attempt,
                "prompt": prompt,
                "response": response,
                "parsed_index": index,
                "rejection_reason": rejection,
                "selected_action": action,
                "valid": action is not None,
            }
        )
        if action is not None:
            return ActionSelection(action, index, tuple(attempts))
    return ActionSelection(None, None, tuple(attempts))


def ollama_chat_payload(model: str, prompt: str, seed: int) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        # Provider setting: thinking-capable models (e.g. Gemma 4) enable hidden
        # reasoning by default in Ollama; the controller adds no chain-of-thought.
        "think": False,
        "options": {"temperature": 0, "seed": seed},
    }


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
        self.task_goal: str | None = None
        self.initial_observation: str | None = None
        self.records: list[ActionRecord] = []
        self.invalid_model_actions = 0
        self.selection_failures: list[dict[str, Any]] = []
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
        self.initial_observation = str(result.get("observation", ""))
        self.task_goal = extract_task_goal(self.initial_observation)
        self._event(
            "task_goal_frozen",
            {"task_id": task_id, "task_goal": self.task_goal, "source": "initial_observation"},
        )
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
        payload = ollama_chat_payload(self.driver.model_name, prompt, self.driver.inference_seed)
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

    def complete_post_success_learning(self, prompt: str) -> str:
        """Ask the same experimental model for its post-success candidate skill."""
        if self.state is None or self.state.get("done") is not True or self.state.get("success") is not True:
            raise EpisodeDriverError("Post-success learning requires a successful episode")
        response = self._model_response(prompt)
        self._event(
            "post_success_learning",
            {
                "model": self.driver.model_name,
                "inference_seed": self.driver.inference_seed,
                "prompt": prompt,
                "response": response,
            },
        )
        return response

    def choose_action(self, recovery_memory: Mapping[str, Any] | None = None) -> str | None:
        if self.state is None or self.task_goal is None or self.initial_observation is None:
            raise EpisodeDriverError("Cannot select an action before start")
        admissible = self.state.get("admissible_actions")
        if not isinstance(admissible, list) or not all(isinstance(item, str) for item in admissible):
            raise EpisodeDriverError("Real ALFWorld state lacks admissible actions")
        memory = None
        if recovery_memory is not None:
            memory = recovery_memory.get("recovery_memory", recovery_memory)
        # Every step already executed in this session, including any frozen
        # checkpoint replay and controlled detour; never oracle or future steps.
        history = [(record.action, record.observation) for record in self.records]
        selection = select_admissible_action(
            self._model_response,
            task_goal=self.task_goal,
            initial_observation=self.initial_observation,
            observation=str(self.state.get("observation", "")),
            inventory=[str(item) for item in self.state.get("inventory") or []],
            admissible_actions=admissible,
            recovery_memory=memory,
            history=history,
        )
        for attempt in selection.attempts:
            self._event(
                "model_selection",
                {
                    "protocol": ACTION_SELECTION_PROTOCOL,
                    "action_history_policy": ACTION_HISTORY_POLICY,
                    "initial_observation_policy": INITIAL_OBSERVATION_POLICY,
                    "inventory_policy": INVENTORY_POLICY,
                    "action_index_parsing": ACTION_INDEX_PARSING_POLICY,
                    "history_steps": len(history),
                    "inference_seed": self.driver.inference_seed,
                    "max_attempts": MAX_SELECTION_ATTEMPTS,
                    "step_number": self.state.get("step_number"),
                    "task_goal": self.task_goal,
                    **attempt,
                },
            )
        if selection.action is None:
            self.invalid_model_actions += 1
        return selection.action

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
                self.selection_failures.append(
                    {
                        "phase": phase,
                        "step_number": self.state.get("step_number"),
                        "attempts": MAX_SELECTION_ATTEMPTS,
                        "reason": "action_selection_invalid_after_bounded_retry",
                    }
                )
                self.abort("model returned no valid ACTION_INDEX after bounded retry")
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
        inference_seed: int = INFERENCE_SEED,
        bridge_log_root: Path | None = None,
    ) -> None:
        self.root = root.resolve()
        self.data_dir = (data_dir or default_data_dir()).resolve()
        self.model_name = model_name
        self.inference_seed = inference_seed
        self.bridge_log_root = bridge_log_root
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
            self.bridge_log_root or (self.root / "artifacts" / "prelaunch" / "bridge"),
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
