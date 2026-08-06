"""Version-specific ALFWorld adapters; generic bridge code stays dependency-free."""

from rq1.bridge.adapters.alfworld_v042 import RealALFWorldAdapter
from rq1.bridge.adapters.capabilities import ALFWorldCapabilityReport, probe_alfworld_capabilities
from rq1.bridge.adapters.task_index import TaskIndex, TaskIndexError, build_task_index

__all__ = ["ALFWorldCapabilityReport", "RealALFWorldAdapter", "TaskIndex", "TaskIndexError", "build_task_index", "probe_alfworld_capabilities"]
