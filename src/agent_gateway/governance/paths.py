"""Where the event ledger, loop counters and checkpoints live.

A hook usually passes a relative path such as ``events.jsonl``. If that resolved
against the working directory, two checkouts of one repository would keep two
ledgers, and a reader of either would see a populated file that was not the
current one. So a relative path resolves against a single run directory:

1. an explicit ``override`` argument
2. ``AGENT_GATEWAY_RUN_DIR`` in the environment
3. ``~/.agent_gateway/runs/current``

An absolute path is always honoured as given.
"""
from __future__ import annotations

import os
from pathlib import Path

from .._config import RUN_DIR

EVENTS_FILENAME = "events.jsonl"
CHECKPOINT_FILENAME = "checkpoint.json"


def run_dir(override: Path | None = None) -> Path:
    """The run directory, independent of the process working directory."""
    if override is not None:
        return Path(override)
    configured = os.environ.get(RUN_DIR, "").strip()
    if configured:
        return Path(configured)
    return Path.home() / ".agent_gateway" / "runs" / "current"


def resolve_run_path(path: Path | str) -> Path:
    """Resolve ``path`` against the run directory unless it is absolute."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else run_dir() / candidate


def events_path() -> Path:
    return run_dir() / EVENTS_FILENAME


def checkpoint_path() -> Path:
    return run_dir() / CHECKPOINT_FILENAME
