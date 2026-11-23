"""Deterministic test doubles for rq1.scientific_prep (synthetic fixtures; NEVER scientific evidence).

FakeDriver mirrors the RealEpisodeDriver.session(...) context-manager surface used by the oracle validator:
session.start / session.step / session.state, with admissible actions, action_valid, done and success.
"""
from __future__ import annotations

import json
from pathlib import Path

ROUTE = ("go to drawer 1", "go to countertop 1", "take apple 1 from countertop 1", "go to cabinet 1",
         "put apple 1 in/on cabinet 1")


class FakeWorld:
    def __init__(self, *, locations=("cabinet 1", "countertop 1", "drawer 1", "fridge 1"), nondeterministic=False,
                 detour_noop=False, oracle_breaks=False, session_number=0):
        self.locations, self.nondeterministic, self.detour_noop = tuple(locations), nondeterministic, detour_noop
        self.oracle_breaks, self.session_number = oracle_breaks, session_number
        self.location, self.holding, self.apple_at, self.done, self.success, self.steps = None, False, "countertop 1", False, False, 0
        self.detoured, self.last_text = False, "You are in the middle of a room."

    def _admissible(self):
        if self.done:
            return []
        go = [f"go to {loc}" for loc in self.locations if loc != self.location]
        if self.oracle_breaks and self.detoured:
            go = [a for a in go if a != "go to cabinet 1"]
        acts = ["look", "inventory", *go]
        if self.location == self.apple_at and not self.holding:
            acts.append(f"take apple 1 from {self.location}")
        if self.holding and self.location == "cabinet 1":
            acts.append("put apple 1 in/on cabinet 1")
        return sorted(acts)

    def observation(self, text):
        return text + (f" [session {self.session_number}]" if self.nondeterministic and self.session_number > 1 else "")

    def state(self, text, valid=None):
        return {"observation": self.observation(text), "admissible_actions": self._admissible(), "step_number": self.steps,
                "done": self.done, "success": self.success, "action_valid": valid}

    def act(self, action, limit):
        self.steps += 1
        if action.startswith("go to "):
            target = action[6:]
            if self.detour_noop and target == "drawer 1" and self.location == "countertop 1":
                text = self.last_text                      # observation unchanged: not a real state change
            else:
                self.location, text = target, f"You arrive at {target}."
            if self.location == "drawer 1" and self.steps > 2:
                self.detoured = True
        elif action.startswith("take "):
            self.holding, self.apple_at, text = True, None, "You pick up the apple 1."
        elif action.startswith("put "):
            self.holding, self.done, self.success, text = False, True, True, "You put the apple 1 in the cabinet 1."
        else:
            text = "Nothing happens."
        if self.steps >= limit and not self.done:
            self.done, text = True, "Action limit reached."
        self.last_text = text
        return self.state(text, valid=True)


class FakeSession:
    def __init__(self, driver, output_dir):
        self.driver, self.output_dir, self.state, self.limit, self.world = driver, Path(output_dir), None, 0, None

    def __enter__(self):
        self.output_dir.mkdir(parents=True, exist_ok=False)
        return self

    def __exit__(self, *_):
        return None

    def _log(self, tool, request, response):
        # same record shape as RealEpisodeSession._event("tool_result", ...)
        if self.driver.log_fault == "omit":
            return
        if self.driver.log_fault == "unsuccessful" and response.get("success"):
            response = {**response, "success": False}
        with (self.output_dir / "episode-events.jsonl").open("a", encoding="utf-8") as h:
            h.write(json.dumps({"event": "tool_result", "payload": {"tool": tool, "request": request, "response": response}},
                               sort_keys=True) + "\n")

    def start(self, task_id, split, seed, action_limit):
        self.driver.sessions += 1
        self.world = FakeWorld(**self.driver.world_kwargs, session_number=self.driver.sessions)
        self.limit = action_limit
        self.state = {**self.world.state("You are in the middle of a room."), "task_id": task_id, "split": split}
        self._log("alfworld_start", {"task_id": task_id, "split": split, "seed": seed}, self.state)
        return self.state

    def step(self, action, *, phase):
        if action not in (self.state or {}).get("admissible_actions", []):
            raise AssertionError("validator dispatched a non-admissible action")
        self.state = self.world.act(action, self.limit)
        self._log("alfworld_step", {"action": action}, self.state)
        return self.state


class FakeDriver:
    def __init__(self, *, log_fault=None, **world_kwargs):
        # log_fault: None | "omit" (session writes no bridge log) | "unsuccessful" (log contradicts in-memory success)
        self.world_kwargs, self.sessions, self.log_fault = world_kwargs, 0, log_fault

    def session(self, *, output_dir, run_id, attempt_id=None, profile=""):
        return FakeSession(self, output_dir)


def write_task(root: Path, split: str, relative: str, task_type: str, *, game=True, game_text=None, traj_extra=None):
    d = root / "json_2.1.1" / split / relative
    d.mkdir(parents=True)
    (d / "traj_data.json").write_text(json.dumps({"task_type": task_type, **(traj_extra or {})}), encoding="utf-8")
    if game:
        (d / "game.tw-pddl").write_text(game_text if game_text is not None else json.dumps({"game": relative}), encoding="utf-8")
    return d
