"""Where configuration comes from.

Every location is read at call time, never at import time, and every variable
carries the ``AGENT_GATEWAY_`` prefix. A value set in the real environment beats
one in a ``.env`` file.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ENV_FILE = "AGENT_GATEWAY_ENV_FILE"
RUN_DIR = "AGENT_GATEWAY_RUN_DIR"
CAPABILITIES = "AGENT_GATEWAY_CAPABILITIES"
POOL_REGISTRY = "AGENT_GATEWAY_POOL_REGISTRY"
POOL_ROLES = "AGENT_GATEWAY_POOL_ROLES"
POOL_ENABLED = "AGENT_GATEWAY_POOL_ENABLED"
POOL_MODE = "AGENT_GATEWAY_POOL_MODE"
POOL_TELEMETRY = "AGENT_GATEWAY_POOL_TELEMETRY"
ALLOW_CLOUD = "AGENT_GATEWAY_ALLOW_CLOUD"
OLLAMA_URL = "AGENT_GATEWAY_OLLAMA_URL"

DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
_TRUE = {"1", "true", "yes", "on"}


def ensure_env_loaded() -> None:
    """Load a ``.env`` file into the environment without overriding anything.

    Reads ``AGENT_GATEWAY_ENV_FILE`` when set, else ``.env`` in the working
    directory. Called by the command line entry point only, so that importing
    the library never changes the process environment.
    """
    configured = os.environ.get(ENV_FILE, "").strip()
    load_dotenv(Path(configured) if configured else Path.cwd() / ".env", override=False)


def env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in _TRUE


def _configured_path(name: str, default_filename: str) -> Path:
    raw = os.environ.get(name, "").strip()
    return Path(raw) if raw else Path.cwd() / default_filename


def capabilities_path() -> Path:
    return _configured_path(CAPABILITIES, "capabilities.yaml")


def pool_registry_path() -> Path:
    return _configured_path(POOL_REGISTRY, "agent-pool.yaml")


def pool_roles_path() -> Path:
    return _configured_path(POOL_ROLES, "agent-roles.yaml")
