"""Safe, structured capability evidence for the ALFWorld 0.4.2 adapter."""
from __future__ import annotations

import importlib.metadata
import importlib.util
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from rq1.bridge.adapters.task_index import TaskIndexError, build_task_index


def default_data_dir() -> Path:
    configured = os.environ.get("RQ1_ALFWORLD_DATA_DIR")
    return Path(configured).expanduser() if configured else Path.home() / ".cache" / "rq1-experiment" / "alfworld"


@dataclass(frozen=True)
class ALFWorldCapabilityReport:
    package_detected: bool
    version_supported: bool
    data_detected: bool
    text_environment_constructible: bool
    task_index_constructible: bool
    real_start_supported: bool
    real_step_supported: bool
    real_reset_supported: bool
    admissible_actions_observable: bool
    inventory_observable: bool
    success_observable: bool
    deterministic_replay_candidate: bool
    direct_state_mutation_supported: bool
    target_relocation_supported: bool
    real_adapter_ready: bool
    version: str | None
    details: str
    data_identity: str | None = None

    @property
    def available(self) -> bool:
        return self.real_adapter_ready

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def probe_alfworld_capabilities(data_dir: Path | None = None) -> ALFWorldCapabilityReport:
    root = data_dir or default_data_dir()
    detected = importlib.util.find_spec("alfworld") is not None
    try:
        version = importlib.metadata.version("alfworld") if detected else None
    except importlib.metadata.PackageNotFoundError:
        version = None
    version_supported = version == "0.4.2"
    data_detected = (root / "json_2.1.1").is_dir() and (root / "logic").is_dir()
    index = None
    index_error = None
    if data_detected:
        try:
            index = build_task_index(root)
        except TaskIndexError as exc:
            index_error = str(exc)
    indexed = index is not None and bool(index.entries)
    surface = False
    if detected and version_supported:
        try:
            module = __import__("alfworld.agents.environment.alfred_tw_env", fromlist=["AlfredTWEnv"])
            surface = all(hasattr(getattr(module, "AlfredTWEnv"), name) for name in ("init_env",))
        except Exception:
            surface = False
    ready = bool(detected and version_supported and indexed and surface)
    detail = "ALFWorld package/data/surface are eligible for an explicitly opted-in smoke test." if ready else (
        index_error or "ALFWorld 0.4.2 package and indexed text data are required; real adapter remains unverified and unavailable."
    )
    return ALFWorldCapabilityReport(
        detected, version_supported, data_detected, surface, indexed, surface, surface, surface,
        surface, False, surface, False, False, False, ready, version, detail, index.identity if index else None,
    )
