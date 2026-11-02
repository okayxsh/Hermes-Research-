"""Reserve a private JSONL stream for the engineering worker process."""
from __future__ import annotations

import os
import sys
from typing import TextIO


def isolate_protocol_stdout() -> TextIO:
    """Keep the original stdout for JSONL; redirect inherited stdout to stderr.

    This must run before loading the episode runtime, so Python, native libraries,
    and children inheriting fd 1 cannot write onto the protocol stream.
    """
    sys.stdout.flush()
    protocol_fd = os.dup(1)
    try:
        os.set_inheritable(protocol_fd, False)
        os.dup2(2, 1)
        return os.fdopen(protocol_fd, 'w', encoding='utf-8', buffering=1)
    except BaseException:
        os.close(protocol_fd)
        raise
