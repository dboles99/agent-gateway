"""An optional, advisory pool of local Ollama agents.

Importing this package does not contact Ollama, discover models, change environment
variables, start background work or write telemetry. The pool is off until a
registry file turns it on, and its results always require review by a primary
orchestrator.
"""
from __future__ import annotations

from .contracts import TaskContract, TaskValidationError
from .delegation import OllamaAgentPool, delegate_task, run_shadow_review
from .registry import AgentPoolConfig, ModelEntry, RoleEntry, load_agent_pool_config
from .result_normalizer import ResultEnvelope, normalize_result

__all__ = [
    "AgentPoolConfig",
    "ModelEntry",
    "OllamaAgentPool",
    "ResultEnvelope",
    "RoleEntry",
    "TaskContract",
    "TaskValidationError",
    "delegate_task",
    "load_agent_pool_config",
    "normalize_result",
    "run_shadow_review",
]
