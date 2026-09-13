from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rq1.evaluation.recovery_executor import RecoveryEpisodeSpec, run_recovery_episode
from rq1.hermes import episode_driver
from rq1.hermes.episode_driver import (
    ACTION_SELECTION_PROTOCOL,
    INFERENCE_SEED,
    MAX_SELECTION_ATTEMPTS,
    RETRY_CLARIFICATION,
    EpisodeDriverError,
    RealEpisodeSession,
    extract_task_goal,
    ollama_chat_payload,
    parse_action_index,
    render_action_prompt,
    select_admissible_action,
)
from rq1.pilot.real_runtime.harnesses import RealRecoveryHarness
from rq1.recovery.controlled_failure import ControlledFailureError, select_reversible_navigation_action
from rq1.retrieval import RetrievalQuery, build_retrieval_boundary
from rq1.retrieval.query import CANONICAL_FAILURE_MESSAGE, query_template_hash
from rq1.tasks.selection import FROZEN_SEEDS
from rq1.utils.config import load_json_yaml

ROOT = Path(__file__).resolve().parents[1]
TASK_ID = "valid_seen:look_at_obj_in_light-AlarmClock-None-DeskLamp-323/trial_T20190909_044715_250790"
GOAL = "look at alarmclock under the desklamp."
INITIAL_OBSERVATION = (
    "-= Welcome to TextWorld, ALFRED! =-\n\n"
    "You are in the middle of a room. Looking quickly around you, you see a bed 1, a desk 1, "
    "and a sidetable 1.\n\n"
    "Your task is to: " + GOAL
)
ACTIONS = ("go to bed 1", "go to desk 1", "go to sidetable 1", "look")
EPISODE_ID = "python-owned-episode-7"
NOLIB = {"recovery_memory": {"retrieved_skills": [], "no_retrieved_skills_available": True}}
PILOT_SKILLS = [
    ("skill-navigation", "TITLE: Navigation\nBODY: Move to the named receptacle."),
    ("skill-handling", "TITLE: Object handling\nBODY: Locate the object before taking it."),
    ("skill-lighting", "TITLE: Lighting\nBODY: Use the light source."),
]


class FakeWorker:
    """Deterministic stand-in for the Hermes registry worker (no subprocess)."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.step_number = 0

    def start(self) -> None:
        return None

    def close(self) -> None:
        return None

    def dispatch(self, tool, args, hook_kwargs):
        self.calls.append((tool, dict(args)))
        base = {
            "episode_id": EPISODE_ID,
            "task_id": TASK_ID,
            "split": "valid_seen",
            "task_family": "look_at_obj_in_light",
            "instruction": TASK_ID,
            "inventory": [],
            "admissible_actions": list(ACTIONS),
            "done": False,
            "success": False,
        }
        if tool == "alfworld_start":
            return {"result": {**base, "observation": INITIAL_OBSERVATION, "step_number": 0, "action_valid": None}}
        if tool == "alfworld_step":
            self.step_number += 1
            return {
                "result": {
                    **base,
                    "observation": f"You arrive at waypoint {self.step_number}.",
                    "step_number": self.step_number,
                    "action_valid": True,
                }
            }
        if tool == "alfworld_abort":
            return {
                "result": {
                    **base,
                    "observation": "Episode aborted by repository controller.",
                    "admissible_actions": [],
                    "done": True,
                    "step_number": self.step_number,
                    "action_valid": None,
                }
            }
        raise AssertionError(f"unexpected tool {tool}")


class FakeDriver:
    """Real session/controller code with a scripted worker and scripted model."""

    root = Path(".")
    bridge_url = "http://127.0.0.1:9"
    bridge_timeout_seconds = 5.0
    model_name = "hermes3:8b"
    ollama_url = "http://127.0.0.1:11434"
    model_timeout_seconds = 5.0
    inference_seed = INFERENCE_SEED

    def __init__(self, respond=None) -> None:
        self.respond = respond
        self.prompts: list[str] = []
        self.workers: list[FakeWorker] = []

    def session(self, *, output_dir, run_id, attempt_id=None, profile="test"):
        session = RealEpisodeSession(
            self, output_dir=output_dir, run_id=run_id, attempt_id=attempt_id or "attempt", profile=profile
        )
        session.worker = FakeWorker()
        self.workers.append(session.worker)
        if self.respond is not None:
            def model(prompt: str) -> str:
                self.prompts.append(prompt)
                return self.respond(prompt)

            session._model_response = model
        return session


class NoOracleHarness(RealRecoveryHarness):
    """The real solvability oracle is exercised by the RunPod pilot, not here."""

    def _oracle(self, detour: str) -> None:
        return None


class LengthEmbedder:
    def encode(self, texts):
        return [[1.0, float(len(text) % 7), 1.0] for text in texts]


def _without_memory(prompt: str) -> str:
    return "\n\n".join(section for section in prompt.split("\n\n") if not section.startswith("RECOVERY MEMORY:\n"))


def _state_sections(prompt: str) -> str:
    return prompt.rsplit("\n\n", 1)[0]


class ActionSelectionControllerTests(unittest.TestCase):
    def test_goal_is_extracted_from_initial_observation_not_task_id(self) -> None:
        self.assertEqual(GOAL, extract_task_goal(INITIAL_OBSERVATION))
        with self.assertRaises(EpisodeDriverError):
            extract_task_goal("You arrive at desk 1. On the desk 1, you see nothing.")
        with self.assertRaises(EpisodeDriverError):
            extract_task_goal(TASK_ID)

    def test_numbered_prompt_is_deterministic_and_exact(self) -> None:
        kwargs = dict(task_goal=GOAL, observation="You arrive at bed 1.", inventory=(), admissible_actions=ACTIONS)
        prompt = render_action_prompt(**kwargs)
        self.assertEqual(prompt, render_action_prompt(**kwargs))
        self.assertEqual(
            "TASK GOAL:\nlook at alarmclock under the desklamp.\n\n"
            "EPISODE HISTORY:\n(no previous steps)\n\n"
            "CURRENT OBSERVATION:\nYou arrive at bed 1.\n\n"
            "CURRENT INVENTORY:\n<empty>\n\n"
            "ADMISSIBLE ACTIONS:\n0. go to bed 1\n1. go to desk 1\n2. go to sidetable 1\n3. look\n\n"
            "Return exactly:\nACTION_INDEX: <integer>",
            prompt,
        )

    def test_prompt_contains_full_prior_history_in_chronological_order(self) -> None:
        history = [("go to bed 1", "You arrive at bed 1."), ("look", "You are facing the bed 1.")]
        prompt = render_action_prompt(
            task_goal=GOAL, observation="You are facing the bed 1.", inventory=("alarmclock 1",),
            admissible_actions=ACTIONS, history=history,
        )
        self.assertIn(
            "EPISODE HISTORY:\nStep 1\nAction: go to bed 1\nObservation: You arrive at bed 1.\n\n"
            "Step 2\nAction: look\nObservation: You are facing the bed 1.\n\n"
            "CURRENT OBSERVATION:\nYou are facing the bed 1.\n\nCURRENT INVENTORY:\nalarmclock 1\n\n",
            prompt,
        )
        self.assertEqual(prompt, render_action_prompt(
            task_goal=GOAL, observation="You are facing the bed 1.", inventory=("alarmclock 1",),
            admissible_actions=ACTIONS, history=list(history),
        ))

    def test_session_history_is_complete_chronological_and_never_future(self) -> None:
        driver = FakeDriver(lambda prompt: "ACTION_INDEX: 1")
        with tempfile.TemporaryDirectory() as tmp:
            with driver.session(output_dir=Path(tmp) / "episode", run_id="run") as session:
                session.start(TASK_ID, "valid_seen", 11, 10)
                records = session.run_model_loop(4, phase="acquisition")
            events = [json.loads(line) for line in (Path(tmp) / "episode" / "episode-events.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(4, len(records))
        self.assertEqual(4, len(driver.prompts))
        for decision, prompt in enumerate(driver.prompts):
            expected = "\n\n".join(
                f"Step {step}\nAction: go to desk 1\nObservation: You arrive at waypoint {step}."
                for step in range(1, decision + 1)
            ) or "(no previous steps)"
            self.assertIn("EPISODE HISTORY:\n" + expected + "\n\nCURRENT OBSERVATION:\n", prompt)
            self.assertNotIn(f"Step {decision + 1}\n", prompt)
            self.assertNotIn(f"waypoint {decision + 1}.", prompt)
        selections = [event["payload"] for event in events if event["event"] == "model_selection"]
        self.assertEqual([0, 1, 2, 3], [payload["history_steps"] for payload in selections])
        self.assertEqual({"full_within_episode_actions_and_observations"}, {payload["action_history_policy"] for payload in selections})

    def test_valid_index_maps_to_exact_admissible_action(self) -> None:
        self.assertEqual(2, parse_action_index("ACTION_INDEX: 2", ACTIONS))
        self.assertEqual(0, parse_action_index("  ACTION_INDEX:0  \n", ACTIONS))
        selection = select_admissible_action(
            lambda prompt: "ACTION_INDEX: 2",
            task_goal=GOAL,
            observation="You arrive at bed 1.",
            inventory=(),
            admissible_actions=ACTIONS,
        )
        self.assertEqual("go to sidetable 1", selection.action)
        self.assertEqual(2, selection.index)
        self.assertEqual(1, len(selection.attempts))

    def test_out_of_range_and_malformed_indices_are_rejected(self) -> None:
        for response in (
            "ACTION_INDEX: 4",
            "ACTION_INDEX: -1",
            "ACTION_INDEX: 1.0",
            "ACTION_INDEX: one",
            "action_index: 1",
            "ACTION_INDEX: 1 (go to desk 1)",
            "ACTION_INDEX: 1\nextra",
            "ACTION: go to desk 1",
            "go to desk 1",
            "",
        ):
            with self.subTest(response=response):
                self.assertIsNone(parse_action_index(response, ACTIONS))

    def test_malformed_response_retries_with_same_state_and_format_clarification_only(self) -> None:
        responses = iter(["ACTION: go to desk 1", "ACTION_INDEX: 4", "ACTION_INDEX: 1"])
        prompts: list[str] = []

        def ask(prompt: str) -> str:
            prompts.append(prompt)
            return next(responses)

        kwargs = dict(task_goal=GOAL, observation="You arrive at bed 1.", inventory=(), admissible_actions=ACTIONS)
        selection = select_admissible_action(ask, **kwargs)
        self.assertEqual("go to desk 1", selection.action)
        self.assertEqual([False, False, True], [attempt["valid"] for attempt in selection.attempts])
        self.assertNotIn(RETRY_CLARIFICATION, prompts[0])
        for prompt in prompts[1:]:
            self.assertIn(RETRY_CLARIFICATION, prompt)
            self.assertEqual(_state_sections(render_action_prompt(**kwargs)), _state_sections(prompt))

    def test_no_fallback_action_after_three_invalid_attempts(self) -> None:
        self.assertEqual(3, MAX_SELECTION_ATTEMPTS)
        calls: list[str] = []

        def ask(prompt: str) -> str:
            calls.append(prompt)
            return "ACTION: go to alarmclock 3"

        selection = select_admissible_action(
            ask, task_goal=GOAL, observation="You arrive at bed 1.", inventory=(), admissible_actions=ACTIONS
        )
        self.assertIsNone(selection.action)
        self.assertIsNone(selection.index)
        self.assertEqual(MAX_SELECTION_ATTEMPTS, len(calls))

    def test_fixed_inference_seed_is_sent_and_logged(self) -> None:
        self.assertEqual(42, INFERENCE_SEED)
        self.assertNotIn(INFERENCE_SEED, FROZEN_SEEDS)
        self.assertEqual({"temperature": 0, "seed": INFERENCE_SEED}, ollama_chat_payload("hermes3:8b", "p", INFERENCE_SEED)["options"])
        self.assertIs(False, ollama_chat_payload("gemma4:12b", "p", INFERENCE_SEED)["think"])
        config = load_json_yaml(ROOT / "configs" / "base.yaml")["model_inference"]
        self.assertEqual(INFERENCE_SEED, config["seed"])
        self.assertEqual(0, config["temperature"])
        self.assertEqual(ACTION_SELECTION_PROTOCOL, config["action_selection_protocol"])
        self.assertEqual("action-index-history-v1", ACTION_SELECTION_PROTOCOL)
        self.assertEqual("full_within_episode_actions_and_observations", config["action_history"])
        self.assertEqual(MAX_SELECTION_ATTEMPTS, config["max_selection_attempts"])

        captured: list[dict] = []

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

            def read(self):
                return json.dumps({"message": {"content": "ACTION_INDEX: 1"}}).encode("utf-8")

        def fake_urlopen(request, timeout):
            captured.append(json.loads(request.data.decode("utf-8")))
            return Response()

        with tempfile.TemporaryDirectory() as tmp, patch.object(episode_driver, "urlopen", fake_urlopen):
            output = Path(tmp) / "episode"
            with FakeDriver().session(output_dir=output, run_id="run") as session:
                session.start(TASK_ID, "valid_seen", 11, 10)
                self.assertEqual("go to desk 1", session.choose_action())
            events = [json.loads(line) for line in (output / "episode-events.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual([INFERENCE_SEED], [payload["options"]["seed"] for payload in captured])
        selections = [event["payload"] for event in events if event["event"] == "model_selection"]
        self.assertEqual([INFERENCE_SEED], [payload["inference_seed"] for payload in selections])
        self.assertEqual(ACTION_SELECTION_PROTOCOL, selections[0]["protocol"])
        self.assertEqual("ACTION_INDEX: 1", selections[0]["response"])
        frozen = [event["payload"] for event in events if event["event"] == "task_goal_frozen"]
        self.assertEqual([GOAL], [payload["task_goal"] for payload in frozen])

    def test_exhausted_selection_records_controller_failure_without_dispatch(self) -> None:
        driver = FakeDriver(lambda prompt: "ACTION: go to alarmclock 3")
        with tempfile.TemporaryDirectory() as tmp:
            harness = RealRecoveryHarness(
                driver,
                output_dir=Path(tmp) / "episode",
                run_id="run",
                attempt_id="exhausted",
                reference_actions=("look", "go to desk 1"),
            )
            try:
                harness.start_and_replay(TASK_ID, "valid_seen", 11, ("look",))
                harness.inject_recovery_memory(NOLIB)
                steps = harness.run_recovery(5)
                invalid_selections = harness.session.invalid_model_actions
            finally:
                harness.close()
        self.assertEqual(MAX_SELECTION_ATTEMPTS, len(driver.prompts))
        self.assertEqual(1, invalid_selections)
        self.assertEqual(1, len(steps))
        self.assertIsNone(steps[0]["action"])
        self.assertIs(False, steps[0]["action_valid"])
        self.assertEqual("action_selection_invalid_after_bounded_retry", steps[0]["controller_failure"])
        self.assertEqual(
            ["alfworld_start", "alfworld_step", "alfworld_abort"],
            [tool for tool, _ in driver.workers[0].calls],
        )

    def test_memory_and_nolib_share_one_controller_and_differ_only_in_memory(self) -> None:
        results = {}
        prompts = {}
        dispatched = {}
        with patch.object(
            episode_driver, "select_admissible_action", wraps=episode_driver.select_admissible_action
        ) as controller:
            for condition, skills in (("Pilot-Memory", PILOT_SKILLS), ("NoLib", [])):
                driver = FakeDriver(lambda prompt: "ACTION_INDEX: 3")
                with tempfile.TemporaryDirectory() as tmp:
                    harness = NoOracleHarness(
                        driver,
                        output_dir=Path(tmp) / "episode",
                        run_id="run",
                        attempt_id=condition,
                        reference_actions=("look", "go to desk 1", "go to sidetable 1"),
                    )
                    boundary = build_retrieval_boundary(skills, LengthEmbedder(), embedding_model="test-embedder", top_k=3)
                    spec = RecoveryEpisodeSpec(
                        run_id="run",
                        attempt_id=condition,
                        task_id=TASK_ID,
                        task_family="look_at_obj_in_light",
                        split="valid_seen",
                        seed=11,
                        condition=condition,
                        library_name=condition,
                        library_size=len(skills),
                        library_hash=None,
                        checkpoint_id="checkpoint-1",
                        prefix_actions=("look",),
                        action_budget=2,
                    )
                    try:
                        results[condition] = run_recovery_episode(harness, spec, boundary, log_dir=Path(tmp) / "result")
                    finally:
                        harness.close()
                prompts[condition] = list(driver.prompts)
                dispatched[condition] = driver.workers[0].calls
        self.assertEqual(4, controller.call_count)

        memory, nolib = results["Pilot-Memory"], results["NoLib"]
        self.assertEqual((1, False), (memory.retrieval_count, memory.no_retrieval))
        self.assertTrue(nolib.no_retrieval)
        for result in (memory, nolib):
            self.assertEqual(GOAL, result.failure_context["task_instruction"])
            self.assertEqual(2, result.outcome.actions)
            self.assertEqual(0, result.outcome.invalid_actions)

        self.assertEqual(
            [_without_memory(prompt) for prompt in prompts["Pilot-Memory"]],
            [_without_memory(prompt) for prompt in prompts["NoLib"]],
        )
        self.assertIn("Use the light source.", prompts["Pilot-Memory"][0])
        self.assertIn('"no_retrieved_skills_available": true', prompts["NoLib"][0])
        replay_and_detour = (
            "EPISODE HISTORY:\nStep 1\nAction: look\nObservation: You arrive at waypoint 1.\n\n"
            "Step 2\nAction: go to bed 1\nObservation: You arrive at waypoint 2.\n\n"
        )
        for condition in ("Pilot-Memory", "NoLib"):
            self.assertIn(replay_and_detour, prompts[condition][0])
            self.assertIn("Step 3\nAction: look\nObservation: You arrive at waypoint 3.", prompts[condition][1])
            self.assertNotIn("go to desk 1", prompts[condition][0].split("ADMISSIBLE ACTIONS:")[0])

        # The Sentence-BERT query stays the frozen four-field query-v1: no action history.
        for result in (memory, nolib):
            context = result.failure_context
            query = RetrievalQuery(
                task_instruction=GOAL, observation="You arrive at waypoint 2.", inventory=(),
                failure_message=CANONICAL_FAILURE_MESSAGE,
            )
            self.assertEqual((GOAL, "You arrive at waypoint 2."), (context["task_instruction"], context["observation"]))
            self.assertEqual(query.text_hash(), result.retrieval_event["query_text_hash"])
            self.assertEqual(query_template_hash(), result.retrieval_event["query_template_hash"])
            text = query.text()
            self.assertEqual(["TASK:", "OBSERVATION:", "INVENTORY:", "FAILURE:"], [line for line in text.splitlines() if line.endswith(":")])
            for leaked in ("EPISODE HISTORY", "Action:", "Step 1", "go to bed 1", "waypoint 1"):
                self.assertNotIn(leaked, text)
        for condition, condition_prompts in prompts.items():
            for prompt in condition_prompts:
                self.assertIn("TASK GOAL:\n" + GOAL, prompt)
                for hidden in (EPISODE_ID, TASK_ID, "score", "Pilot-Memory", "NoLib"):
                    self.assertNotIn(hidden, prompt)
            for tool, args in dispatched[condition]:
                if tool == "alfworld_step":
                    self.assertEqual(EPISODE_ID, args["episode_id"])
                    self.assertIn(args["action"], ACTIONS)

    def test_navigation_detour_is_sorted_and_never_the_expected_action(self) -> None:
        action = select_reversible_navigation_action(
            ("go to pantry 1", "go to desk 1", "look", "go to bed 1"), "go to desk 1"
        )
        self.assertEqual("go to bed 1", action)

    def test_non_navigation_checkpoint_fails_closed(self) -> None:
        with self.assertRaises(ControlledFailureError) as caught:
            select_reversible_navigation_action(("go to desk 1",), "take mug 1")
        self.assertEqual("checkpoint_not_navigation_eligible", caught.exception.code)


if __name__ == "__main__":
    unittest.main()
