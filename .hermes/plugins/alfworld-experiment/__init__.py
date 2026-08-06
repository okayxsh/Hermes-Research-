"""Explicitly trusted project-local Hermes plugin for the local ALFWorld bridge."""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
_SOURCE = _ROOT / "src"
if str(_SOURCE) not in sys.path:
    sys.path.insert(0, str(_SOURCE))

from rq1.hermes.plugin_runtime import register_plugin


def register(ctx):
    """Hermes plugin entry point; capability checks happen in register_plugin."""
    register_plugin(ctx)
