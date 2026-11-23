"""Controlled-failure oracle validation for scientific candidate tasks (CPU-only; no model is ever called).

Reuses the frozen historical definitions -- unchanged, so no new failure definition is introduced:
  * expert route: ALFWorld 0.4.2 hand-coded expert (rq1.recovery.reference_route.derive_handcoded_reference);
  * checkpoint order: rq1.recovery.reference_route.midpoint_candidates (CHECKPOINT_POLICY midpoint-navigation-detour-v1);
  * detour: rq1.recovery.controlled_failure.select_reversible_navigation_action;
  * state digest: rq1.pilot.real_runtime.harnesses.bridge_state_digest.
Strengthening relative to the historical preparation (flagged for scientific review, not a new definition):
  1. the complete expert route is first replayed from reset and must reach environment-reported success;
  2. the oracle runs in an INDEPENDENT second session that must reproduce the checkpoint and post-detour digests of
     the first session before executing the continuation to environment-reported success;
  3. the persisted bridge log of the route replay and of the oracle session must itself confirm the step count and
     the final environment-reported done+success (in-memory state alone never yields ENVIRONMENT_VALIDATED).
ENVIRONMENT_VALIDATED is evidence about whatever backend the driver wraps; the CLI always records the backend class,
so results produced with the synthetic test doubles are identifiable as fixtures and are never scientific evidence.
Statuses: ENVIRONMENT_VALIDATED | FAILED | BLOCKED (missing input/route/identity) | UNVERIFIED (not executed).
Oracle actions/continuations are evidence only and are never written to any agent-visible artifact.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from rq1.experiment.models import canonical_hash
from rq1.pilot.real_runtime.harnesses import bridge_state_digest
from rq1.recovery.controlled_failure import ControlledFailureError, select_reversible_navigation_action
from rq1.recovery.reference_route import ReferenceRoute, ReferenceRouteError, midpoint_candidates
from rq1.scientific_prep.horizons import BLOCKED, DEFAULT_CAPS, ENVIRONMENT_VALIDATED, FAILED, horizon_rows
from rq1.utils.hashing import sha256_file

DETOUR_ACTIONS = 1
POLICY_NOTE = ("historical midpoint-navigation-detour-v1 definitions; strengthened by full-route replay and "
               "independent-session digest reproduction (flagged for scientific review)")
RouteProvider = Callable[[str, str], Sequence[str]]


class _Reject(Exception):
    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(code)
        self.code, self.detail = code, detail


def _session_files(directory: Path) -> list[dict[str, Any]]:
    if not directory.is_dir():
        return []
    return [{"path": p.relative_to(directory).as_posix(), "sha256": sha256_file(p)}
            for p in sorted(directory.rglob("*")) if p.is_file()]


def _step_checked(session: Any, action: str, phase: str, *, allow_done: bool) -> dict[str, Any]:
    admissible = (session.state or {}).get("admissible_actions")
    if not isinstance(admissible, list) or action not in admissible:
        raise _Reject(f"{phase}_action_not_admissible", action)
    state = session.step(action, phase=phase)
    if state.get("action_valid") is not True:
        raise _Reject(f"{phase}_action_invalid", action)
    if state.get("done") and not allow_done:
        raise _Reject(f"{phase}_terminal_too_early", action)
    return state


def _logged_environment_outcome(directory: Path, expected_steps: int) -> dict[str, Any]:
    """Independent confirmation from the PERSISTED bridge log (``episode-events.jsonl``, written by every real
    RealEpisodeSession): exactly one ``alfworld_start``, ``expected_steps`` ``alfworld_step`` results, and the last
    environment response reporting done and success.  In-memory session state alone never validates an oracle."""
    path = directory / "episode-events.jsonl"
    if not path.is_file():
        raise _Reject("environment_log_missing", f"{path.name} not written by the session")
    starts, steps, last = 0, 0, None
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            payload = record.get("payload") or {}
            if record.get("event") != "tool_result":
                continue
            if payload.get("tool") == "alfworld_start":
                starts += 1
            elif payload.get("tool") == "alfworld_step":
                steps, last = steps + 1, payload.get("response") or {}
    except (OSError, ValueError, AttributeError) as exc:
        raise _Reject("environment_log_unreadable", str(exc)) from exc
    if starts != 1 or steps != expected_steps:
        raise _Reject("environment_log_disagrees", f"log has {starts} start(s) and {steps} step(s); expected 1 and {expected_steps}")
    if not isinstance(last, dict) or last.get("done") is not True or last.get("success") is not True:
        raise _Reject("environment_log_does_not_confirm_success", "last logged environment response is not done+success")
    return {"log_file": path.name, "log_sha256": sha256_file(path), "logged_steps": steps,
            "logged_final_done": True, "logged_final_success": True}


def _replay_route(driver: Any, task_id: str, split: str, seed: int, actions: Sequence[str], directory: Path,
                  max_total: int) -> dict[str, Any]:
    with driver.session(output_dir=directory, run_id="rq1-sci-prep-route", attempt_id=directory.name,
                        profile="rq1-sci-prep-route-validation") as session:
        session.start(task_id, split, seed, max_total)
        for index, action in enumerate(actions):
            state = _step_checked(session, action, "route_replay", allow_done=index == len(actions) - 1)
        final = session.state or {}
        if final.get("done") is not True or final.get("success") is not True:
            raise _Reject("route_not_completing", "full expert route did not reach environment-reported success")
        return {"actions_executed": len(actions), "env_done": True, "env_success": True,
                "final_digest": bridge_state_digest(final)}


def _checkpoint_session(driver: Any, task_id: str, split: str, seed: int, prefix: Sequence[str],
                        continuation: Sequence[str], directory: Path, max_total: int, *,
                        reference: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Session A (reference=None): checkpoint + detour digests.  Session B (reference=A): an INDEPENDENT replay that
    must reproduce A's checkpoint digest, detour and post-detour digest, then run the oracle continuation to
    environment-reported success -- all inside one live session."""
    expected = continuation[0]
    with driver.session(output_dir=directory, run_id="rq1-sci-prep-oracle", attempt_id=directory.name,
                        profile="rq1-sci-prep-oracle-validation") as session:
        session.start(task_id, split, seed, max_total)
        for action in prefix:
            _step_checked(session, action, "checkpoint_replay", allow_done=False)
        checkpoint = dict(session.state or {})
        admissible = checkpoint.get("admissible_actions")
        if not isinstance(admissible, list):
            raise _Reject("admissible_actions_unavailable")
        digest = bridge_state_digest(checkpoint)
        if reference is not None and digest != reference["checkpoint_digest"]:
            raise _Reject("checkpoint_state_hash_mismatch", "independent sessions disagree on the checkpoint state")
        chosen = select_reversible_navigation_action(admissible, expected)
        if reference is not None and chosen != reference["detour_action"]:
            raise _Reject("detour_not_reproducible", f"{chosen} != {reference['detour_action']}")
        before = str(checkpoint.get("observation", ""))
        post = _step_checked(session, chosen, "controlled_detour", allow_done=False)
        if str(post.get("observation", "")) == before:
            raise _Reject("detour_no_state_change", chosen)
        post_digest = bridge_state_digest(post)
        result = {"checkpoint_digest": digest, "detour_action": chosen, "post_detour_digest": post_digest}
        if reference is None:
            return result
        if post_digest != reference["post_detour_digest"]:
            raise _Reject("post_detour_state_hash_mismatch", "independent sessions disagree on the post-detour state")
        for position, action in enumerate(continuation):
            _step_checked(session, action, "oracle_continuation", allow_done=position == len(continuation) - 1)
        final = session.state or {}
        if final.get("done") is not True or final.get("success") is not True:
            raise _Reject("oracle_unsolved", "continuation did not reach environment-reported success")
        return {**result, "env_done": True, "env_success": True}


def validate_candidate(record: Mapping[str, Any], *, driver: Any, route_provider: RouteProvider, data_root: Path,
                       output_dir: Path, seed: int = 0, max_total_actions: int = max(DEFAULT_CAPS),
                       caps: Sequence[int] = DEFAULT_CAPS, selection_cap: int | None = None,
                       max_checkpoints: int | None = None) -> dict[str, Any]:
    task_id, split = record["task_id"], record["split"]
    out: dict[str, Any] = {"task_id": task_id, "split": split, "family": record.get("family"), "policy": POLICY_NOTE,
                           "model_calls": 0, "model_visible": False, "stages": {}, "checkpoint_attempts": []}

    def finish(status: str, code: str | None = None, detail: str = "") -> dict[str, Any]:
        out.update(status=status, code=code, detail=detail)
        return out

    # 1. exact source identity (re-hashed now; the inventory hashes are the reference)
    if not record.get("installed"):
        return finish(BLOCKED, "task_not_installed")
    root = Path(data_root)
    for field, rel in (("trajectory_sha256", record.get("trajectory_path")), ("game_sha256", record.get("game_path"))):
        path = root / rel if rel else None
        if path is None or not path.is_file():
            return finish(BLOCKED, "source_file_missing", str(rel))
        if sha256_file(path) != record.get(field):
            return finish(BLOCKED, "source_identity_mismatch", f"{field} differs from the inventory")
    out["stages"]["identity"] = {"trajectory_sha256": record["trajectory_sha256"], "game_sha256": record["game_sha256"]}
    # 2. authorized expert route (never synthesized)
    try:
        actions = tuple(str(a) for a in route_provider(task_id, split))
    except (ReferenceRouteError, OSError, ValueError, RuntimeError) as exc:
        return finish(BLOCKED, "no_expert_route", f"{type(exc).__name__}: {exc}")
    if not actions:
        return finish(BLOCKED, "no_expert_route", "route provider returned no actions")
    out["stages"]["route"] = {"length": len(actions), "actions": list(actions), "route_sha256": canonical_hash(list(actions)),
                              "source": "alfworld_0.4.2_handcoded_expert"}
    # 3. route applies to THIS task instance: full replay from reset to environment-reported success
    try:
        out["stages"]["route_replay"] = _replay_route(driver, task_id, split, seed, actions, output_dir / "route-replay",
                                                      max_total_actions)
        out["stages"]["route_replay"]["environment_log"] = _logged_environment_outcome(output_dir / "route-replay", len(actions))
    except _Reject as exc:
        return finish(FAILED, exc.code, exc.detail)
    out["stages"]["route_replay"]["evidence_files"] = _session_files(output_dir / "route-replay")
    # 4-9. checkpoints in historical midpoint order
    try:
        candidates = list(midpoint_candidates(ReferenceRoute(task_id, split, actions)))
    except ReferenceRouteError as exc:
        return finish(FAILED, "route_too_short", str(exc))
    for count, (index, prefix, continuation) in enumerate(candidates, 1):
        if max_checkpoints is not None and count > max_checkpoints:
            break
        attempt: dict[str, Any] = {"checkpoint_index": index, "prefix_actions": len(prefix),
                                   "continuation_actions": len(continuation),
                                   "horizons": horizon_rows(len(prefix), DETOUR_ACTIONS, len(continuation), caps=caps)}
        out["checkpoint_attempts"].append(attempt)
        if not continuation[0].startswith("go to "):
            attempt.update(result="rejected", code="checkpoint_not_navigation_eligible")
            continue
        if selection_cap is not None and len(prefix) + DETOUR_ACTIONS + len(continuation) > selection_cap:
            attempt.update(result="rejected", code="reference_rejoin_exceeds_recovery_budget")
            continue
        base = output_dir / f"checkpoint-{index}"
        try:
            first = _checkpoint_session(driver, task_id, split, seed, prefix, continuation, base / "session-a",
                                        max_total_actions)
            _checkpoint_session(driver, task_id, split, seed, prefix, continuation, base / "session-b-oracle",
                                max_total_actions, reference=first)
        except ControlledFailureError as exc:
            attempt.update(result="rejected", code=exc.code, detail=str(exc))
            continue
        except _Reject as exc:
            attempt.update(result="rejected", code=exc.code, detail=exc.detail)
            if exc.code.endswith("state_hash_mismatch"):
                return finish(FAILED, exc.code, "environment state is not reproducible across independent sessions")
            continue
        try:   # the persisted bridge log of session B must itself confirm the environment-reported success
            logged = _logged_environment_outcome(base / "session-b-oracle", len(prefix) + DETOUR_ACTIONS + len(continuation))
        except _Reject as exc:
            attempt.update(result="rejected", code=exc.code, detail=exc.detail)
            return finish(FAILED, exc.code, "persisted environment evidence does not confirm the oracle")
        validated = horizon_rows(len(prefix), DETOUR_ACTIONS, len(continuation), caps=caps, environment_validated=True)
        attempt.update(result="accepted", horizons=validated)
        out["oracle"] = {
            "checkpoint_index": index, "prefix_actions": list(prefix), "expected_next_action": continuation[0],
            "detour_action": first["detour_action"], "continuation_actions": list(continuation),
            "checkpoint_digest": first["checkpoint_digest"], "post_detour_digest": first["post_detour_digest"],
            "digests_reproduced_in_independent_session": True, "env_done": True, "env_success": True,
            "environment_log_confirmation": logged,
            "prefix_length": len(prefix), "detour_length": DETOUR_ACTIONS, "continuation_length": len(continuation),
            "total_oracle_actions": len(prefix) + DETOUR_ACTIONS + len(continuation), "horizons": validated,
            "evidence_files": {name: _session_files(base / name) for name in ("session-a", "session-b-oracle")},
        }
        return finish(ENVIRONMENT_VALIDATED)
    return finish(FAILED, "no_checkpoint_admits_validated_controlled_failure",
                  f"{len(out['checkpoint_attempts'])} checkpoint(s) attempted")
