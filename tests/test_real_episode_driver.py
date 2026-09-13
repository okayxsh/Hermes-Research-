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
    EXPERIMENT_MODEL,
    INFERENCE_SEED,
    MAX_SELECTION_ATTEMPTS,
    OUTPUT_CAP_REJECTION,
    OUTPUT_TOKEN_CAP,
    RETRY_CLARIFICATION,
    EpisodeDriverError,
    ProviderText,
    RealEpisodeSession,
    classify_action_index,
    extract_task_goal,
    ollama_chat_payload,
    provider_settings,
    reached_output_cap,
    parse_action_index,
    render_action_prompt,
    select_admissible_action,
)
from rq1.pilot.real_runtime.harnesses import RealRecoveryHarness
from rq1.recovery.controlled_failure import ControlledFailureError, select_reversible_navigation_action
from rq1.retrieval import RetrievalQuery, build_retrieval_boundary
from rq1.retrieval.query import CANONICAL_FAILURE_MESSAGE, INVENTORY_NOT_OBSERVED_MARKER, query_template_hash
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
    model_name = "gemma4:12b"
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
        kwargs = dict(
            task_goal=GOAL, initial_observation=INITIAL_OBSERVATION, observation="You arrive at bed 1.",
            inventory=(), admissible_actions=ACTIONS,
        )
        prompt = render_action_prompt(**kwargs)
        self.assertEqual(prompt, render_action_prompt(**kwargs))
        self.assertEqual(
            "TASK GOAL:\nlook at alarmclock under the desklamp.\n\n"
            "INITIAL OBSERVATION:\n" + INITIAL_OBSERVATION + "\n\n"
            "EPISODE HISTORY:\n(no previous steps)\n\n"
            "CURRENT OBSERVATION:\nYou arrive at bed 1.\n\n"
            "CURRENT INVENTORY:\n<not observed — use the inventory action>\n\n"
            "ADMISSIBLE ACTIONS:\n0. go to bed 1\n1. go to desk 1\n2. go to sidetable 1\n3. look\n\n"
            "Return exactly:\nACTION_INDEX: <integer>",
            prompt,
        )

    def test_prompt_contains_full_prior_history_in_chronological_order(self) -> None:
        history = [("go to bed 1", "You arrive at bed 1."), ("look", "You are facing the bed 1.")]
        prompt = render_action_prompt(
            task_goal=GOAL, initial_observation=INITIAL_OBSERVATION, observation="You are facing the bed 1.", inventory=("alarmclock 1",),
            admissible_actions=ACTIONS, history=history,
        )
        self.assertIn(
            "EPISODE HISTORY:\nStep 1\nAction: go to bed 1\nObservation: You arrive at bed 1.\n\n"
            "Step 2\nAction: look\nObservation: You are facing the bed 1.\n\n"
            "CURRENT OBSERVATION:\nYou are facing the bed 1.\n\nCURRENT INVENTORY:\nalarmclock 1\n\n",
            prompt,
        )
        self.assertEqual(prompt, render_action_prompt(
            task_goal=GOAL, initial_observation=INITIAL_OBSERVATION, observation="You are facing the bed 1.", inventory=("alarmclock 1",),
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
            # The verbatim reset observation is present at every decision, before the history.
            self.assertTrue(prompt.startswith(
                "TASK GOAL:\n" + GOAL + "\n\nINITIAL OBSERVATION:\n" + INITIAL_OBSERVATION + "\n\nEPISODE HISTORY:\n"
            ))
            self.assertIn("CURRENT INVENTORY:\n<not observed — use the inventory action>\n\n", prompt)
            self.assertNotIn("<empty>", prompt)
        selections = [event["payload"] for event in events if event["event"] == "model_selection"]
        self.assertEqual([0, 1, 2, 3], [payload["history_steps"] for payload in selections])
        self.assertEqual({"full_within_episode_actions_and_observations"}, {payload["action_history_policy"] for payload in selections})
        self.assertEqual(
            {("verbatim_reset_observation_at_every_decision", "not_observed_marker_inventory_only_via_inventory_action")},
            {(payload["initial_observation_policy"], payload["inventory_policy"]) for payload in selections},
        )

    def test_valid_index_maps_to_exact_admissible_action(self) -> None:
        self.assertEqual(2, parse_action_index("ACTION_INDEX: 2", ACTIONS))
        self.assertEqual(0, parse_action_index("  ACTION_INDEX:0  \n", ACTIONS))
        selection = select_admissible_action(
            lambda prompt: "ACTION_INDEX: 2",
            task_goal=GOAL,
            initial_observation=INITIAL_OBSERVATION,
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
            "**ACTION_INDEX: 1**",
            "ACTION_INDEX: <integer>",
            "ACTION_INDEX: 1\nACTION_INDEX: 2",
            "ACTION_INDEX: 2\nACTION_INDEX: 2",
            "I would pick ACTION_INDEX: 1.\nACTION_INDEX: 2",
            "Return exactly:\nACTION_INDEX: <integer>\nACTION_INDEX: 1",
            "ACTION_INDEX: 99999999999999999999",
            "ACTION: go to desk 1",
            "go to desk 1",
            "",
        ):
            with self.subTest(response=response):
                self.assertIsNone(parse_action_index(response, ACTIONS))

    def test_one_valid_index_line_is_accepted_despite_surrounding_prose(self) -> None:
        for response, expected in (
            ("I should inspect the sidetable.\nACTION_INDEX: 2", 2),
            ("ACTION_INDEX: 2\nThis seems appropriate.", 2),
            ("The desk is wrong.\n\n  ACTION_INDEX:1  \n\nI will check it next.", 1),
            ("ACTION_INDEX: 3", 3),
        ):
            with self.subTest(response=response):
                self.assertEqual((expected, None), classify_action_index(response, ACTIONS))
                self.assertEqual(expected, parse_action_index(response, ACTIONS))

    def test_rejections_are_classified_and_never_inferred_from_prose(self) -> None:
        for response, reason in (
            ("I will go to desk 1.", "no_action_index"),
            ("go to sidetable 1", "no_action_index"),
            ("ACTION_INDEX: 1\nACTION_INDEX: 2", "multiple_action_index_mentions"),
            ("ACTION_INDEX: 2 is best.\nACTION_INDEX: 2", "multiple_action_index_mentions"),
            ("Thinking about it.\nACTION_INDEX: two", "malformed_action_index"),
            ("ACTION_INDEX: -1", "malformed_action_index"),
            ("Prose first.\nACTION_INDEX: 4", "index_out_of_range"),
        ):
            with self.subTest(response=response):
                self.assertEqual((None, reason), classify_action_index(response, ACTIONS))

    def test_malformed_response_retries_with_same_state_and_format_clarification_only(self) -> None:
        responses = iter(["ACTION: go to desk 1", "ACTION_INDEX: 4", "ACTION_INDEX: 1"])
        prompts: list[str] = []

        def ask(prompt: str) -> str:
            prompts.append(prompt)
            return next(responses)

        kwargs = dict(
            task_goal=GOAL, initial_observation=INITIAL_OBSERVATION, observation="You arrive at bed 1.",
            inventory=(), admissible_actions=ACTIONS,
        )
        selection = select_admissible_action(ask, **kwargs)
        self.assertEqual("go to desk 1", selection.action)
        self.assertEqual([False, False, True], [attempt["valid"] for attempt in selection.attempts])
        self.assertEqual(["no_action_index", "index_out_of_range", None], [attempt["rejection_reason"] for attempt in selection.attempts])
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
            ask, task_goal=GOAL, initial_observation=INITIAL_OBSERVATION, observation="You arrive at bed 1.",
            inventory=(), admissible_actions=ACTIONS,
        )
        self.assertIsNone(selection.action)
        self.assertIsNone(selection.index)
        self.assertEqual(MAX_SELECTION_ATTEMPTS, len(calls))
        # Naming an exact admissible action in prose is never mapped to an action.
        prose_only = select_admissible_action(
            lambda prompt: "I choose go to desk 1, which is listed as 1.",
            task_goal=GOAL, initial_observation=INITIAL_OBSERVATION, observation="You arrive at bed 1.",
            inventory=(), admissible_actions=ACTIONS,
        )
        self.assertIsNone(prose_only.action)
        self.assertEqual(["no_action_index"] * MAX_SELECTION_ATTEMPTS, [attempt["rejection_reason"] for attempt in prose_only.attempts])

    def test_fixed_inference_seed_is_sent_and_logged(self) -> None:
        self.assertEqual(42, INFERENCE_SEED)
        self.assertNotIn(INFERENCE_SEED, FROZEN_SEEDS)
        self.assertEqual(
            {"temperature": 0, "seed": INFERENCE_SEED, "num_predict": 2048, "num_ctx": 32768},
            ollama_chat_payload(EXPERIMENT_MODEL, "p", INFERENCE_SEED)["options"],
        )
        self.assertIs(False, ollama_chat_payload("gemma4:12b", "p", INFERENCE_SEED)["think"])
        config = load_json_yaml(ROOT / "configs" / "base.yaml")["model_inference"]
        self.assertEqual(INFERENCE_SEED, config["seed"])
        self.assertEqual(0, config["temperature"])
        self.assertEqual(ACTION_SELECTION_PROTOCOL, config["action_selection_protocol"])
        self.assertEqual("action-index-history-v3", ACTION_SELECTION_PROTOCOL)
        self.assertEqual(("gemma4:12b", 2048, 32768), (config["model"], config["output_token_cap"], config["model_context_length"]))
        self.assertEqual("verbatim_reset_observation_at_every_decision", config["initial_observation"])
        self.assertEqual("not_observed_marker_inventory_only_via_inventory_action", config["inventory"])
        self.assertEqual("exactly_one_action_index_line_surrounding_prose_ignored", config["action_index_parsing"])
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

        # The Sentence-BERT query keeps the frozen four-field template: no history, no initial observation.
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
            self.assertEqual("d519a394b6ba45ce88e427f5bedcd3f18f4fc1b6bc9bf576531f87d8dc4a1275", query_template_hash())
            self.assertIn("INVENTORY:\n<not observed — use the inventory action>\nFAILURE:\n", text)
            for absent in ("<empty>", "INITIAL OBSERVATION", "Welcome to TextWorld"):
                self.assertNotIn(absent, text)
            for leaked in ("EPISODE HISTORY", "Action:", "Step 1", "go to bed 1", "waypoint 1"):
                self.assertNotIn(leaked, text)
        for condition, condition_prompts in prompts.items():
            for prompt in condition_prompts:
                self.assertIn("TASK GOAL:\n" + GOAL, prompt)
                self.assertIn("INITIAL OBSERVATION:\n" + INITIAL_OBSERVATION + "\n\nEPISODE HISTORY:\n", prompt)
                self.assertIn("CURRENT INVENTORY:\n" + INVENTORY_NOT_OBSERVED_MARKER + "\n\n", prompt)
                self.assertNotIn("<empty>", prompt)
                for hidden in (EPISODE_ID, TASK_ID, "score", "Pilot-Memory", "NoLib"):
                    self.assertNotIn(hidden, prompt)
            for tool, args in dispatched[condition]:
                if tool == "alfworld_step":
                    self.assertEqual(EPISODE_ID, args["episode_id"])
                    self.assertIn(args["action"], ACTIONS)

    def test_output_cap_hit_consumes_one_attempt_and_is_never_parsed(self) -> None:
        capped = ProviderText("ACTION_INDEX: 2\n" + "more " * 50, done_reason="length", eval_count=OUTPUT_TOKEN_CAP)
        responses = iter([capped, ProviderText("I choose.\nACTION_INDEX: 1", done_reason="stop", eval_count=9)])
        selection = select_admissible_action(
            lambda prompt: next(responses), task_goal=GOAL, initial_observation=INITIAL_OBSERVATION,
            observation="You arrive at bed 1.", inventory=(), admissible_actions=ACTIONS,
        )
        self.assertEqual(("go to desk 1", 1), (selection.action, selection.index))
        self.assertEqual([OUTPUT_CAP_REJECTION, None], [attempt["rejection_reason"] for attempt in selection.attempts])
        self.assertEqual(["length", "stop"], [attempt["done_reason"] for attempt in selection.attempts])

    def test_three_capped_responses_exhaust_selection_without_infrastructure_failure(self) -> None:
        capped = ProviderText("thinking " * 100, done_reason="length", eval_count=OUTPUT_TOKEN_CAP)
        driver = FakeDriver(lambda prompt: capped)
        with tempfile.TemporaryDirectory() as tmp:
            with driver.session(output_dir=Path(tmp) / "episode", run_id="run") as session:
                session.start(TASK_ID, "valid_seen", 11, 10)
                records = session.run_model_loop(5, phase="acquisition")
        self.assertEqual([], records)
        self.assertEqual(MAX_SELECTION_ATTEMPTS, len(driver.prompts))
        self.assertEqual("action_selection_invalid_after_bounded_retry", session.selection_failures[0]["reason"])
        self.assertEqual({OUTPUT_CAP_REJECTION: MAX_SELECTION_ATTEMPTS}, session.rejection_counts)
        self.assertEqual(["alfworld_start", "alfworld_abort"], [tool for tool, _ in driver.workers[0].calls])

    def test_provider_reply_keeps_stop_evidence_and_provider_errors_stay_infrastructure(self) -> None:
        settings = provider_settings()
        self.assertEqual(
            ("gemma4:12b", "options.num_predict", 2048, 180, 3),
            (EXPERIMENT_MODEL, settings["output_cap_parameter"], settings["options"]["num_predict"],
             settings["model_timeout_seconds"], settings["max_selection_attempts"]),
        )
        self.assertIn("not_claimed_deterministic", settings["determinism"])

        class Reply:
            def __init__(self, body):
                self.body = body

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

            def read(self):
                return json.dumps(self.body).encode("utf-8")

        body = {"message": {"content": "partial"}, "done_reason": "length", "eval_count": 2048, "prompt_eval_count": 900}
        with tempfile.TemporaryDirectory() as tmp, patch.object(episode_driver, "urlopen", lambda request, timeout: Reply(body)):
            with FakeDriver().session(output_dir=Path(tmp) / "episode", run_id="run") as session:
                reply = session._model_response("p")
        self.assertEqual(
            ("partial", "length", 2048, 900, True),
            (reply, reply.done_reason, reply.eval_count, reply.prompt_eval_count, reached_output_cap(reply)),
        )

        def unavailable(request, timeout):
            raise TimeoutError("timed out")

        with tempfile.TemporaryDirectory() as tmp, patch.object(episode_driver, "urlopen", unavailable):
            with FakeDriver().session(output_dir=Path(tmp) / "episode", run_id="run") as session:
                with self.assertRaises(EpisodeDriverError):
                    session._model_response("p")

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
