"""Action-horizon accounting for controlled-failure oracles.

``remaining_recovery_actions = total_action_cap - checkpoint_prefix_actions - detour_actions``.  The oracle fits a
cap when its post-detour continuation fits the remaining recovery actions.  Arithmetic alone yields at most
ARITHMETICALLY_FEASIBLE; ENVIRONMENT_VALIDATED requires an environment-executed oracle (oracle.py) that fits the cap.
"""
from __future__ import annotations

from typing import Any, Sequence

DEFAULT_CAPS = (50, 100, 150, 200)
ARITHMETICALLY_FEASIBLE, ENVIRONMENT_VALIDATED = "ARITHMETICALLY_FEASIBLE", "ENVIRONMENT_VALIDATED"
FAILED, BLOCKED, UNVERIFIED = "FAILED", "BLOCKED", "UNVERIFIED"


def horizon_rows(prefix_actions: int, detour_actions: int, continuation_actions: int, *,
                 caps: Sequence[int] = DEFAULT_CAPS, environment_validated: bool = False) -> list[dict[str, Any]]:
    for name, value in (("prefix_actions", prefix_actions), ("detour_actions", detour_actions),
                        ("continuation_actions", continuation_actions)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    rows = []
    for cap in sorted(set(caps)):
        if not isinstance(cap, int) or cap < 1:
            raise ValueError("caps must be positive integers")
        remaining = cap - prefix_actions - detour_actions
        checkpoint_feasible = remaining >= 1
        fits = checkpoint_feasible and continuation_actions <= remaining
        if not checkpoint_feasible:
            status, reason = FAILED, "checkpoint prefix plus detour leaves no recovery action within this cap"
        elif not fits:
            status, reason = FAILED, "oracle continuation exceeds the remaining recovery actions"
        elif environment_validated:
            status, reason = ENVIRONMENT_VALIDATED, "environment-executed oracle completes within this cap"
        else:
            status, reason = ARITHMETICALLY_FEASIBLE, "length arithmetic only; not replay-validated"
        rows.append({"total_action_cap": cap, "prefix_actions": prefix_actions, "detour_actions": detour_actions,
                     "remaining_recovery_actions": remaining, "oracle_continuation_actions": continuation_actions,
                     "total_oracle_actions": prefix_actions + detour_actions + continuation_actions,
                     "checkpoint_feasible": checkpoint_feasible, "oracle_fits": fits, "status": status, "reason": reason})
    return rows
