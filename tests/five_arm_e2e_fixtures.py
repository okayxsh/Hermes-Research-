"""SYNTHETIC end-to-end fixtures: a scripted ALFWorld-like world behind the REAL session/harness/prompt code, a scripted
provider, and a builder for a complete synthetic five-arm pilot (rankings, selections, audits, manifest).

Nothing here is scientific evidence.  The world is a tiny deterministic navigation puzzle; the "model" is a script.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Callable

from five_arm_fixtures import SKILL_TEXTS, TOLERANCE, SyntheticTokenizer, corpus
from rq1.evaluation.amended_libraries import AmendedLibrary
from rq1.evaluation.amended_matrix import EvaluationTask
from rq1.five_arm import cpu_checks
from rq1.five_arm.manifest import REQUIRED_INPUT_HASHES, DIAGNOSTIC_INPUT_HASHES, build_diagnostic_manifest, build_manifest
from rq1.five_arm.memory import Skill, construct_arm_memory
from rq1.five_arm.prompting import LeakageProbe
from rq1.five_arm.selection import natural_ranking, select_interventions
from rq1.hermes.episode_driver import EpisodeDriverError, ProviderText, RealEpisodeSession
from rq1.pilot.real_runtime.harnesses import RealRecoveryHarness

TOK = SyntheticTokenizer()
LOCATIONS = ("bed 1", "desk 1", "drawer 1", "shelf 1", "sidetable 1")
ROUTE = ("go to bed 1", "go to desk 1", "go to sidetable 1", "go to shelf 1")
PREFIX = ("go to bed 1",)
GOAL = "visit the sidetable and then rest at the shelf."
COUNTER_SPEC = {"prefix": "<turn user> ", "suffix": " <end turn><turn model>", "add_bos": True}


def counter(prompt: str) -> int:
    """Synthetic 'verified' chat-template counter (prefix + prompt + suffix + BOS) over the synthetic tokenizer."""
    return len(TOK.encode(f"{COUNTER_SPEC['prefix']}{prompt}{COUNTER_SPEC['suffix']}")) + 1


class ScriptedWorld:
    """Deterministic stand-in for the Hermes registry worker + bridge (same response shape and action-limit rule)."""

    def __init__(self, task_id: str, *, fail_at_step: int | None = None) -> None:
        self.task_id, self.fail_at_step = task_id, fail_at_step
        self.position, self.visited, self.count, self.limit, self.done = None, set(), 0, 0, False

    def start(self) -> None:
        return None

    def close(self) -> None:
        return None

    def _state(self, observation: str, *, valid: bool | None, success: bool = False) -> dict[str, Any]:
        actions = [] if self.done else [f"go to {loc}" for loc in LOCATIONS if loc != self.position] + ["look"]
        return {"result": {"episode_id": "ep-1", "task_id": self.task_id, "split": self.task_id.split(":")[0],
                           "task_family": "look_at_obj_in_light", "instruction": self.task_id, "inventory": [],
                           "admissible_actions": actions, "done": self.done, "success": success, "observation": observation,
                           "step_number": self.count, "action_valid": valid, "action_count": self.count, "action_limit": self.limit}}

    def dispatch(self, tool: str, args: dict[str, Any], hook_kwargs: dict[str, Any]) -> dict[str, Any]:
        if tool == "alfworld_start":
            self.limit = int(args["action_limit"])
            return self._state("-= Welcome to TextWorld =-\nYou are in the middle of a room. You see a bed 1, a desk 1, a "
                               f"drawer 1, a shelf 1 and a sidetable 1.\n\nYour task is to: {GOAL}", valid=None)
        if tool == "alfworld_abort":
            self.done = True
            return self._state("Episode aborted by repository controller.", valid=None)
        if tool != "alfworld_step":
            raise AssertionError(tool)
        if self.fail_at_step is not None and self.count + 1 >= self.fail_at_step:
            raise EpisodeDriverError("bridge step failed: synthetic environment fault")
        action = args["action"]
        self.count += 1
        success = False
        if action == "look":
            observation = f"You are at {self.position or 'the middle of the room'}. Nothing happens."
        else:
            self.position = action[len("go to "):]
            self.visited.add(self.position)
            observation = f"You arrive at {self.position}."
            if self.position == "shelf 1" and "sidetable 1" in self.visited:
                success, self.done, observation = True, True, "You arrive at shelf 1. You rest. Task complete."
        if self.count >= self.limit and not self.done:
            self.done = True
            return self._state("Action limit reached.", valid=True)
        return self._state(observation, valid=True, success=success)


class ScriptedDriver:
    """Real ``RealEpisodeSession`` code with the scripted world and a scripted provider (``respond(prompt)``)."""

    root = Path(".")
    bridge_url = "http://127.0.0.1:9"
    bridge_timeout_seconds = 5.0
    model_name = "gemma4:12b"
    ollama_url = "http://127.0.0.1:11434"
    model_timeout_seconds = 5.0

    def __init__(self, respond: Callable[[str], Any], *, env_fail: Callable[[str], int | None] = lambda _t: None) -> None:
        self.respond, self.env_fail = respond, env_fail
        self.inference_seed = 42
        self.prompts: list[str] = []
        self.sessions = 0

    def session(self, *, output_dir, run_id, attempt_id=None, profile="test"):
        session = RealEpisodeSession(self, output_dir=output_dir, run_id=run_id, attempt_id=attempt_id or "a", profile=profile)
        session.worker = _LazyWorld(self)
        self.sessions += 1

        def model(prompt: str):
            self.prompts.append(prompt)
            text = self.respond(prompt)
            if isinstance(text, Exception):
                raise text
            return ProviderText(text, done_reason="stop", eval_count=4, prompt_eval_count=counter(prompt))

        session._model_response = model
        return session


class _LazyWorld:
    def __init__(self, driver: ScriptedDriver) -> None:
        self.driver, self.world = driver, None

    def start(self):
        return None

    def close(self):
        return None

    def dispatch(self, tool, args, hooks):
        if self.world is None:
            task_id = args["task_id"]
            self.world = ScriptedWorld(task_id, fail_at_step=self.driver.env_fail(task_id))
        return self.world.dispatch(tool, args, hooks)


# --- scripted provider policies ----------------------------------------------------------------------------------------
def _admissible(prompt: str) -> list[str]:
    block = prompt.split("ADMISSIBLE ACTIONS:\n", 1)[1].split("\n\n", 1)[0]
    return [line.split(". ", 1)[1] for line in block.splitlines()]


def _visited(prompt: str) -> set[str]:
    return set(re.findall(r"^Action: go to (.+)$", prompt, re.MULTILINE))


def solver(prompt: str) -> str:
    """Goes to the sidetable (if not yet visited) and then to the shelf."""
    actions = _admissible(prompt)
    target = "go to shelf 1" if "sidetable 1" in _visited(prompt) else "go to sidetable 1"
    if target not in actions:
        target = "look"
    return f"ACTION_INDEX: {actions.index(target)}"


def looker(prompt: str) -> str:
    return f"ACTION_INDEX: {_admissible(prompt).index('look')}"


def memory_sensitive(prompt: str) -> str:
    """Solves only when a RECOVERY MEMORY section is present (so D0 wanders and memory-bearing arms solve)."""
    return solver(prompt) if "RECOVERY MEMORY:" in prompt else looker(prompt)


def garbage(_prompt: str) -> str:
    return "I am not sure."


def provider_down(_prompt: str):
    return EpisodeDriverError("Ollama action selection failed: URLError")


# --- synthetic pilot construction -------------------------------------------------------------------------------------
class HashEmbedder:
    """Deterministic 8-d embedding from SHA-256 (no ties in practice); SYNTHETIC, not Sentence-BERT."""

    def encode(self, texts):
        out = []
        for text in texts:
            digest = hashlib.sha256(text.encode()).digest()
            out.append([b / 255.0 for b in digest[:8]])
        return out


def library() -> AmendedLibrary:
    return AmendedLibrary("Accum-18", tuple({"skill_id": sid, "task_family": "look_at_obj_in_light", "text": text, "role": "core"}
                                            for sid, text in sorted(SKILL_TEXTS.items())))


def freeze_task(driver, task_id: str, root: Path, cap: int) -> EvaluationTask:
    """Run the real harness once WITHOUT a frozen perturbation (real oracle validation path) to obtain the digests."""
    harness = RealRecoveryHarness(driver, output_dir=root / f"freeze-{abs(hash(task_id)) % 10**8}", run_id="freeze",
                                  attempt_id="freeze", reference_actions=ROUTE, total_action_limit=cap,
                                  allowed_split=task_id.split(":")[0])
    harness.start_and_replay(task_id, task_id.split(":")[0], 0, PREFIX)
    perturbation = harness.apply_controlled_action_perturbation("cp-1")
    harness.close()
    return EvaluationTask(task_id, "look_at_obj_in_light", 1, "cp-1", harness.checkpoint_digest, perturbation.post_state_digest,
                          perturbation.action, ROUTE[len(PREFIX)], PREFIX, ROUTE, cap - len(PREFIX) - 1)


def task_meta(task: EvaluationTask, cap: int, *, template: str, scene: str) -> dict[str, Any]:
    return {"task_id": task.task_id, "goal_template": template, "scene": scene, "family": "look_at_obj_in_light",
            "oracle_sha256": hashlib.sha256(task.task_id.encode()).hexdigest(), "checkpoint_digest": task.checkpoint_digest,
            "post_detour_digest": task.post_detour_digest, "prefix_actions": len(task.prefix_actions), "detour_actions": 1,
            "continuation_actions": len(task.reference_actions) - len(task.prefix_actions), "oracle_replay_total_action_limit": cap}


def build_pilot(root: Path, *, task_ids=("valid_seen:alpha/trial_T1", "valid_seen:beta/trial_T2"), cap: int = 12,
                seeds=(11,), labels: Callable[[str, list[str]], dict[str, str]] | None = None, corp=None) -> dict[str, Any]:
    """Complete synthetic five-arm pilot: real CPU ranking export + selection + memory audits + manifest."""
    free = ScriptedDriver(solver)
    lib, emb = library(), HashEmbedder()
    corp = corp or corpus()
    tasks, metas, selections, audits, rankings = {}, [], {}, {}, {}
    for index, task_id in enumerate(task_ids):
        task = freeze_task(free, task_id, root, cap)
        tasks[task_id] = task
        ranking = cpu_checks.checkpoint_ranking(free, task, lib, emb, output_dir=root / f"ranking-{index}", split="valid_seen",
                                                total_action_cap=cap)
        rankings[task_id] = ranking
        ranked = natural_ranking({r["skill_id"]: r["score"] for r in ranking["ranking"]})
        top = [r.skill_id for r in ranked[:3]]
        task_labels = (labels or (lambda _t, t3: {t3[0]: "IRRELEVANT", t3[1]: "RELEVANT", t3[2]: "IRRELEVANT"}))(task_id, top)
        full = {s: task_labels.get(s, "IRRELEVANT") for s in SKILL_TEXTS}
        selections[task_id] = select_interventions(task_id, ranked, full, rel_scope="natural_top3", irr_scope="natural_top3")
        natural = [Skill(r.skill_id, SKILL_TEXTS[r.skill_id], r.score, r.rank) for r in ranked[:3]]
        audits[task_id] = {}
        for arm in ("D0", "D18", "Rel", "Irr", "Fill"):
            sel = selections[task_id].get(arm.lower()) if arm in ("Rel", "Irr") else None
            if arm in ("Rel", "Irr") and sel is None:
                continue
            active = Skill(sel["skill_id"], SKILL_TEXTS[sel["skill_id"]], sel["score"], sel["rank"]) if sel else None
            _, audit = construct_arm_memory(arm, natural_top3=natural, active_skill=active,
                                            corpus=corp if arm in ("Rel", "Irr", "Fill") else None, tokenizer=TOK,
                                            tolerance=TOLERANCE, d0_representation="absent_memory_section", salt=task_id)
            audits[task_id][arm] = audit
        metas.append(task_meta(task, cap, template=f"tpl-{index % 2}", scene=str(10 + index % 3)))
    manifest = build_manifest(stage="pilot", run_id="synthetic-pilot", tasks=metas, selections=selections, memory_audits=audits,
                              seeds=seeds, total_action_cap=cap, source_commit="c" * 40,
                              input_hashes={k: "f" * 64 for k in REQUIRED_INPUT_HASHES}, synthetic=True)
    probes = {t: LeakageProbe(oracle_continuation=tuple(task.reference_actions[len(task.prefix_actions):]),
                              expected_next_action=task.expected_next_action, skill_ids=tuple(SKILL_TEXTS),
                              library_texts=tuple(SKILL_TEXTS.values())) for t, task in tasks.items()}
    return {"manifest": manifest, "tasks": tasks, "metas": {m["task_id"]: m for m in metas}, "library": lib, "embedder": emb,
            "corpus": corp, "rankings": rankings, "selections": selections, "probes": probes}


def build_diagnostic(root: Path, *, task_ids=("valid_seen:alpha/trial_T1", "valid_seen:beta/trial_T2"), cap: int = 12,
                     seeds=(11, 29)) -> dict[str, Any]:
    free = ScriptedDriver(solver)
    tasks = {t: freeze_task(free, t, root, cap) for t in task_ids}
    metas = [task_meta(task, cap, template="tpl-0", scene="10") for task in tasks.values()]
    manifest = build_diagnostic_manifest(run_id="synthetic-diagnostic", tasks=metas, seeds=seeds, total_action_cap=cap,
                                         source_commit="c" * 40, input_hashes={k: "f" * 64 for k in DIAGNOSTIC_INPUT_HASHES},
                                         synthetic=True)
    return {"manifest": manifest, "tasks": tasks, "metas": {m["task_id"]: m for m in metas}}
