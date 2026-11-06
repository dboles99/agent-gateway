"""Dispatch one task packet to an Ollama-backed agent and validate what comes back.

Registry checks raise before any HTTP call. Transport and response failures return
a ``failed`` result packet instead, because a delegated worker must never crash the
orchestrator that called it.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator

from .._config import (
    ALLOW_CLOUD,
    DEFAULT_OLLAMA_URL,
    OLLAMA_URL,
    env_bool,
    pool_registry_path,
)
from .agent_registry import AgentDefinition


class OllamaResolutionError(Exception):
    """The registry does not permit this dispatch. Raised before any HTTP call."""


# Ascending write scope. An agent may never claim broader write authority than the
# registry grants the model backing it.
_WRITE_SCOPE_RANK = {
    "none": 0,
    "artifact-only": 1,
    "staging-only": 2,
    "full": 3,
}


def _cloud_calls_allowed(defaults: dict[str, Any]) -> bool:
    """Cloud dispatch is off unless the operator or the registry opts in."""
    return env_bool(ALLOW_CLOUD, bool(defaults.get("cloud_calls", False)))


def _write_scope_rank(value: str, source: str) -> int:
    try:
        return _WRITE_SCOPE_RANK[value]
    except KeyError:
        raise OllamaResolutionError(
            f"unknown write scope {value!r} in {source} "
            f"(expected one of {sorted(_WRITE_SCOPE_RANK)})"
        ) from None


def _load_registry(registry_path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
    except OSError as exc:
        raise OllamaResolutionError(f"cannot read registry {registry_path}: {exc}") from exc
    if not isinstance(data, dict):
        raise OllamaResolutionError(f"{registry_path} did not parse to a mapping")
    return data


def _find_model_entry(registry: dict[str, Any], model_tag: str | None) -> dict[str, Any]:
    for entry in (registry.get("models") or {}).values():
        if entry.get("model_tag") == model_tag:
            return entry
    raise OllamaResolutionError(f"model_tag {model_tag!r} is not registered")


def _check_registry_permits(
    agent: AgentDefinition, role: str, registry: dict[str, Any]
) -> dict[str, Any]:
    defaults = registry.get("defaults") or {}
    if not defaults.get("enabled", False):
        raise OllamaResolutionError(
            "Agent pool is disabled (defaults.enabled is false in the registry)"
        )
    model_entry = _find_model_entry(registry, agent.model_tag)
    if not model_entry.get("enabled", False):
        raise OllamaResolutionError(f"model_tag {agent.model_tag!r} is registered but not enabled")

    if model_entry.get("execution") == "cloud" and not _cloud_calls_allowed(defaults):
        raise OllamaResolutionError(
            f"model_tag {agent.model_tag!r} is a cloud-execution model but cloud "
            f"calls are not allowed: prompt content would leave this machine. Set "
            f"{ALLOW_CLOUD}=true (or defaults.cloud_calls in the registry) to permit it"
        )
    if role not in (model_entry.get("roles") or []):
        raise OllamaResolutionError(
            f"model_tag {agent.model_tag!r} does not permit role {role!r} "
            f"(permitted: {model_entry.get('roles')!r})"
        )

    granted = model_entry.get("write_permission", "none")
    claimed = agent.write_authority
    if _write_scope_rank(claimed, f"agent {agent.name!r}") > _write_scope_rank(
        granted, f"registry entry for {agent.model_tag!r}"
    ):
        raise OllamaResolutionError(
            f"agent {agent.name!r} claims write_authority={claimed!r} but model_tag "
            f"{agent.model_tag!r} is registered with write_permission={granted!r}: an "
            "agent may not out-scope the write permission granted to its model"
        )

    return model_entry


def _failed_result(task_packet: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "execution_id": task_packet.get("execution_id", ""),
        "task_id": task_packet.get("task_id", ""),
        "status": "failed",
        "artifacts": [],
        "validation": {},
        "unresolved": [reason],
    }


def dispatch_task(
    agent: AgentDefinition,
    role: str,
    task_packet: dict[str, Any],
    result_schema: dict[str, Any],
    *,
    registry_path: Path | None = None,
    base_url: str | None = None,
) -> dict[str, Any]:
    """Dispatch one task packet and return a result packet.

    Raises :class:`OllamaResolutionError` before any HTTP call if the registry does
    not permit this model and role. Failures after that point (unreachable, timeout,
    malformed or schema-invalid response) return ``{"status": "failed", ...}``.
    """
    if agent.provider != "ollama" or not agent.model_tag:
        raise OllamaResolutionError(f"Agent {agent.name!r} is not an Ollama-provider agent")

    registry = _load_registry(registry_path or pool_registry_path())
    model_entry = _check_registry_permits(agent, role, registry)
    resolved_base_url = (
        base_url
        or os.environ.get(OLLAMA_URL)
        or (registry.get("defaults") or {}).get("base_url", DEFAULT_OLLAMA_URL)
    )
    timeout_seconds = model_entry.get("timeout_seconds", 600)

    payload = {
        "model": agent.model_tag,
        "messages": [
            {
                "role": "system",
                "content": (
                    f"Role: {role}. The next message is a JSON task packet. "
                    "Reply with JSON matching the supplied schema and nothing else."
                ),
            },
            {"role": "user", "content": json.dumps(task_packet)},
        ],
        "stream": False,
        "format": result_schema,
    }
    request = urllib.request.Request(
        f"{resolved_base_url.rstrip('/')}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            body = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return _failed_result(task_packet, f"ollama request failed: {exc}")
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return _failed_result(task_packet, f"ollama response envelope was not valid JSON: {exc}")

    if not isinstance(body, dict):
        return _failed_result(task_packet, "ollama response envelope was not a JSON object")

    content = (body.get("message") or {}).get("content") or ""
    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        return _failed_result(task_packet, f"ollama response was not valid JSON: {content[:500]!r}")

    errors = list(Draft202012Validator(result_schema).iter_errors(result))
    if errors:
        messages = "; ".join(e.message for e in errors)
        return _failed_result(task_packet, f"ollama response failed schema validation: {messages}")

    return result
