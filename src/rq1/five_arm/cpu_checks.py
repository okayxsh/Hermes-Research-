"""CPU-only university integration tools (no model inference): frozen rankings, exact-cap oracle replay, cap invariance.

All three drive the REAL bridge/ALFWorld path through the frozen harness and are exercised locally only against a
scripted fake environment.  None of them may call the provider: every session they open is guarded, and any model call
raises ``ProtocolViolation``.

* ``checkpoint_ranking``      -- runs the authoritative ``run_recovery_episode`` order up to and including the single
  retrieval (replay -> frozen detour -> failure context -> retrieval), with a harness whose recovery phase is empty.
  The full 18-skill ranking, the rank-3/rank-4 margin and a tie-fragility flag are returned for freezing.
* ``exact_cap_oracle_replay`` -- prefix + frozen detour + oracle continuation at EXACTLY the given total action cap.
* ``cap_invariance_replay``   -- the same scripted route at two caps: every pre-cap bridge state digest must be identical,
  and the smaller cap must terminate exactly at its limit when the route is padded with a harmless action.  This is the
  environment-side evidence required for analysing cap-100 trajectories truncated to cap 50.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

from rq1.acquisition.environment import SBERT_MODEL, SBERT_REVISION
from rq1.evaluation.recovery_executor import RecoveryEpisodeSpec, run_recovery_episode
from rq1.five_arm.executor import ProtocolViolation
from rq1.pilot.real_runtime.harnesses import RealRecoveryHarness, bridge_state_digest
from rq1.retrieval.controller import build_retrieval_boundary

TIE_MARGIN = 1e-6


class _NoModelDriver:
    def __init__(self, driver: Any) -> None:
        self._driver = driver

    def session(self, **kwargs: Any) -> Any:
        session = self._driver.session(**kwargs)

        def refuse(_prompt: str):
            raise ProtocolViolation("CPU-only integration tool attempted a model call")

        session._model_response = refuse
        return session

    def __getattr__(self, name: str) -> Any:
        return getattr(self._driver, name)


class _RankingOnlyHarness(RealRecoveryHarness):
    def run_recovery(self, action_budget: int):
        return ()


def checkpoint_ranking(driver: Any, task: Any, library: Any, embedder: Any, *, output_dir: Path, split: str,
                       total_action_cap: int) -> dict[str, Any]:
    harness = _RankingOnlyHarness(_NoModelDriver(driver), output_dir=output_dir / "episode", run_id="cpu-ranking",
                                  attempt_id=hashlib.sha256(task.task_id.encode()).hexdigest()[:16],
                                  reference_actions=task.reference_actions, total_action_limit=total_action_cap,
                                  allowed_split=split, profile_prefix="rq1-five-arm-cpu",
                                  frozen_perturbation=task.frozen_perturbation())
    boundary = build_retrieval_boundary(library.retrieval_documents(), embedder, embedding_model=SBERT_MODEL, top_k=3,
                                        embedding_model_revision=SBERT_REVISION)
    spec = RecoveryEpisodeSpec(run_id="cpu-ranking", attempt_id="ranking", task_id=task.task_id, task_family=task.task_family,
                               split=split, seed=0, condition="ranking-export", library_name="Accum-18",
                               library_size=library.size, library_hash=library.content_sha256, checkpoint_id=task.checkpoint_id,
                               prefix_actions=task.prefix_actions, action_budget=1, episode_id=f"ranking-{task.task_id}")
    try:
        result = run_recovery_episode(harness, spec, boundary, log_dir=output_dir / "result")
        checkpoint, post = harness.checkpoint_digest, (result.failure or {}).get("post_state_digest")
    finally:
        harness.close()
    ranking = [(item["skill_id"], float(item["score"])) for item in result.retrieval_event["ranking"]]
    margin = ranking[2][1] - ranking[3][1] if len(ranking) > 3 else None
    return {"task_id": task.task_id, "library_sha256": library.content_sha256, "checkpoint_digest": checkpoint,
            "post_detour_digest": post, "query_text_hash": result.retrieval_event["query_text_hash"],
            "ranking": [{"rank": i, "skill_id": s, "score": v} for i, (s, v) in enumerate(ranking, 1)],
            "natural_top3": [s for s, _ in ranking[:3]], "rank3_rank4_margin": margin,
            "tie_fragile": margin is not None and margin < TIE_MARGIN, "model_calls": 0,
            "embedding_model": SBERT_MODEL, "embedding_model_revision": SBERT_REVISION}


def _scripted(driver: Any, task: Any, *, cap: int, actions: Sequence[str], output_dir: Path, split: str) -> list[dict[str, Any]]:
    states = []
    with _NoModelDriver(driver).session(output_dir=output_dir, run_id="cpu-replay", attempt_id=f"cap{cap}",
                                        profile="rq1-five-arm-cpu-replay") as session:
        session.start(task.task_id, split, 0, cap)
        for action in actions:
            if session.state.get("done"):
                break
            if action not in (session.state.get("admissible_actions") or []):
                states.append({"action": action, "admissible": False})
                break
            state = session.step(action, phase="cpu_replay")
            states.append({"action": action, "admissible": True, "digest": bridge_state_digest(state),
                           "done": bool(state.get("done")), "success": state.get("success") is True,
                           "action_valid": state.get("action_valid"), "observation": str(state.get("observation", ""))})
    return states


def exact_cap_oracle_replay(driver: Any, task: Any, *, cap: int, output_dir: Path, split: str) -> dict[str, Any]:
    route = (*task.prefix_actions, task.detour_action, *task.reference_actions[len(task.prefix_actions):])
    states = _scripted(driver, task, cap=cap, actions=route, output_dir=output_dir, split=split)
    detour_state = states[len(task.prefix_actions)] if len(states) > len(task.prefix_actions) else {}
    success = bool(states) and states[-1].get("success") is True and len(states) == len(route)
    return {"task_id": task.task_id, "total_action_limit": cap, "route_actions": len(route), "executed_actions": len(states),
            "post_detour_digest_matches": detour_state.get("digest") == task.post_detour_digest,
            "success": success, "validated": success and detour_state.get("digest") == task.post_detour_digest,
            "model_calls": 0}


def cap_invariance_replay(driver: Any, task: Any, *, caps: Sequence[int], output_dir: Path, split: str,
                          padding_action: str = "look") -> dict[str, Any]:
    small, large = sorted(int(c) for c in caps)[:2]
    route = (*task.prefix_actions, task.detour_action)
    padded = (*route, *([padding_action] * (large - len(route))))
    runs = {cap: _scripted(driver, task, cap=cap, actions=padded, output_dir=output_dir / f"cap{cap}", split=split)
            for cap in (small, large)}
    a, b = runs[small], runs[large]
    detour_index = len(route) - 1                       # the post-detour transition is state index len(route) - 1

    def reached_post_detour(states: Sequence[Mapping[str, Any]]) -> bool:
        return (len(states) > detour_index and all(s.get("admissible") is True for s in states[: detour_index + 1])
                and states[detour_index].get("done") is False)

    shared = min(len(a), len(b), small - 1)
    identical = all(a[i].get("digest") == b[i].get("digest") for i in range(shared))
    small_terminates_at_cap = len(a) == small and a[-1].get("done") is True and a[-1].get("success") is False
    large_live_at_small_cap = len(b) > small - 1 and b[small - 1].get("done") is False
    post_detour = {cap: reached_post_detour(states) for cap, states in runs.items()}
    digest_ok = {cap: (post_detour[cap] and states[detour_index].get("digest") == task.post_detour_digest)
                 for cap, states in runs.items()}
    # Classification (Phase 2E fix): a comparison that never reached the post-detour transition under BOTH caps -- or that
    # compared no post-detour state at all -- is UNTESTED, never PASS (identity over zero or pre-detour states is vacuous).
    if detour_index + 1 >= small:
        status, reason = "UNTESTED", (f"prefix + detour ({detour_index + 1} actions) does not leave a post-detour decision "
                                      f"below the smaller cap {small}")
    elif not all(post_detour.values()):
        status, reason = "UNTESTED", "the post-detour transition was not reached under both caps (inadmissible or terminal route)"
    elif not all(digest_ok.values()):
        status, reason = "FAIL", "the replay did not reproduce the frozen post-detour state digest"
    elif shared <= detour_index:
        status, reason = "UNTESTED", "no post-detour state was compared before the smaller cap"
    elif not identical:
        status, reason = "FAIL", "pre-cap bridge states differ between caps"
    elif not small_terminates_at_cap or not large_live_at_small_cap:
        status, reason = "FAIL", "the smaller cap did not terminate exactly at its limit while the larger cap continued"
    else:
        status, reason = "PASS", "post-detour states identical before the smaller cap; termination only at the cap"
    return {"task_id": task.task_id, "caps": [small, large], "status": status, "reason": reason,
            "compared_pre_cap_steps": shared, "post_detour_states_compared": max(0, shared - detour_index),
            "post_detour_reached": {str(c): v for c, v in post_detour.items()},
            "pre_cap_states_identical": identical, "small_cap_terminates_exactly_at_cap": small_terminates_at_cap,
            "large_cap_live_at_small_cap": large_live_at_small_cap,
            "truncation_environment_assumptions_hold": status == "PASS",
            "model_trajectory_equivalence": "UNVERIFIED: deterministic environment invariance only; model trajectories "
                                            "are not executed by this CPU check",
            "padding_action": padding_action, "model_calls": 0}
