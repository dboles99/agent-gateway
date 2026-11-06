from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

AUTO_MODEL = "auto"
SUPPORTED_MODES = {
    "disabled",
    "manual",
    "delegated-subagent",
    "shadow",
    "debate",
    "ensemble",
    "benchmark",
    "bounded-fallback",
}
SUPPORTED_EXECUTION = {"local", "cloud"}
SUPPORTED_ORCHESTRATORS = {"claude", "codex", "other"}
SUPPORTED_PRIVACY = {"public", "internal", "confidential", "restricted"}
SUPPORTED_WRITE_PERMISSION = {"none", "artifact-only", "staging-only"}
ADVISORY_MODES = SUPPORTED_MODES - {"disabled"}


class TaskValidationError(ValueError):
    """An Ollama subagent task violates the safety contract."""


@dataclass(frozen=True)
class TaskContract:
    task_id: str
    orchestrator: str
    mode: str
    agent_role: str
    objective: str
    model_tag: str = AUTO_MODEL
    execution: str = "local"
    parent_task_id: str | None = None
    allowed_context: list[str] = field(default_factory=list)
    excluded_context: list[str] = field(default_factory=list)
    expected_output_schema: dict[str, Any] = field(default_factory=dict)
    timeout_seconds: int = 600
    maximum_output_tokens: int = 2000
    write_permission: str = "none"
    requires_primary_review: bool = True
    privacy_classification: str = "internal"
    success_criteria: list[str] = field(default_factory=list)
    stop_conditions: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TaskContract:
        """Build a contract. Raises TypeError on missing or unknown fields."""
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "parent_task_id": self.parent_task_id,
            "orchestrator": self.orchestrator,
            "mode": self.mode,
            "agent_role": self.agent_role,
            "model_tag": self.model_tag,
            "execution": self.execution,
            "objective": self.objective,
            "allowed_context": list(self.allowed_context),
            "excluded_context": list(self.excluded_context),
            "expected_output_schema": dict(self.expected_output_schema),
            "timeout_seconds": self.timeout_seconds,
            "maximum_output_tokens": self.maximum_output_tokens,
            "write_permission": self.write_permission,
            "requires_primary_review": self.requires_primary_review,
            "privacy_classification": self.privacy_classification,
            "success_criteria": list(self.success_criteria),
            "stop_conditions": list(self.stop_conditions),
        }


def validate_task_contract(task: TaskContract, *, known_roles: set[str], known_models: set[str]) -> None:
    if not task.task_id.strip():
        raise TaskValidationError("task_id is required")
    if not task.objective.strip():
        raise TaskValidationError("objective is required")
    if task.orchestrator not in SUPPORTED_ORCHESTRATORS:
        raise TaskValidationError(f"unsupported orchestrator: {task.orchestrator}")
    if task.mode not in SUPPORTED_MODES:
        raise TaskValidationError(f"unsupported mode: {task.mode}")
    if task.mode == "disabled":
        raise TaskValidationError("disabled mode cannot call an Ollama subagent")
    if task.agent_role not in known_roles:
        raise TaskValidationError(f"unknown role: {task.agent_role}")
    if task.model_tag not in known_models:
        raise TaskValidationError(f"unknown or unvalidated model: {task.model_tag}")
    if task.execution not in SUPPORTED_EXECUTION:
        raise TaskValidationError(f"unsupported execution: {task.execution}")
    if task.privacy_classification not in SUPPORTED_PRIVACY:
        raise TaskValidationError(f"unsupported privacy class: {task.privacy_classification}")
    if task.execution == "cloud" and task.privacy_classification in {"confidential", "restricted"}:
        raise TaskValidationError("cloud execution is blocked for confidential or restricted work")
    if task.write_permission not in SUPPORTED_WRITE_PERMISSION:
        raise TaskValidationError(f"unsupported write permission: {task.write_permission}")
    if task.write_permission == "staging-only" and task.mode in ADVISORY_MODES:
        raise TaskValidationError("advisory tasks may not request staging write permission")
    if not task.requires_primary_review:
        raise TaskValidationError("Ollama subagent results require primary review")
    if task.timeout_seconds < 1 or task.timeout_seconds > 3600:
        raise TaskValidationError("timeout_seconds must be between 1 and 3600")
    if task.maximum_output_tokens < 1 or task.maximum_output_tokens > 32000:
        raise TaskValidationError("maximum_output_tokens must be between 1 and 32000")
