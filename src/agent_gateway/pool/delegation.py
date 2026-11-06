from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

from .contracts import (
    AUTO_MODEL,
    TaskContract,
    TaskValidationError,
    validate_task_contract,
)
from .local_client import LocalOllamaClient, OllamaClient, OllamaClientError
from .registry import AgentPoolConfig, load_agent_pool_config
from .result_normalizer import ResultEnvelope, normalize_result, rejected_result
from .telemetry import TelemetryRecorder


class OllamaAgentPool:
    def __init__(
        self,
        *,
        config: AgentPoolConfig | None = None,
        client: OllamaClient | None = None,
        telemetry: TelemetryRecorder | None = None,
    ):
        self.config = config or load_agent_pool_config()
        self.client = client
        self.telemetry = telemetry or TelemetryRecorder()

    def list_models(self) -> list[dict[str, Any]]:
        return [
            {
                "model_id": model.model_id,
                "model_tag": model.model_tag,
                "execution": model.execution,
                "enabled": model.enabled,
                "status": model.status,
                "lifecycle": model.lifecycle,
                "roles": list(model.roles),
                "privacy_classes": list(model.privacy_classes),
                "requires_primary_review": model.requires_primary_review,
                "fallback_eligible": model.fallback_eligible,
                "write_permission": model.write_permission,
            }
            for model in self.config.list_models()
        ]

    def delegate_task(
        self, payload: dict[str, Any] | TaskContract, *, mandatory: bool = False
    ) -> ResultEnvelope:
        """Run one task. A refusal or failure comes back as a result, not an exception.

        ``mandatory=True`` lets an unexpected error propagate, for a caller that
        cannot continue without this result.
        """
        try:
            task = _coerce_task(payload)
        except TypeError as exc:
            data = payload if isinstance(payload, dict) else {}
            return rejected_result(
                task_id=str(data.get("task_id", "")),
                model_tag=str(data.get("model_tag", "")),
                agent_role=str(data.get("agent_role", "")),
                summary=f"Task does not match the contract: {exc}",
                error_category="contract_rejected",
            )

        if not self.config.defaults.enabled or self.config.defaults.mode == "disabled":
            return _reject(task, "Ollama agent pool is disabled.", "disabled")

        if task.model_tag == AUTO_MODEL:
            chosen = self.config.select_model_for_role(
                task.agent_role, privacy_classification=task.privacy_classification
            )
            if chosen is None:
                return _reject(
                    task, "No enabled model may take this role and privacy class.", "no_eligible_model"
                )
            task = replace(task, model_tag=chosen.model_tag, execution=chosen.execution)

        model = self.config.model_by_tag(task.model_tag)
        if model is None:
            return _reject(task, "Requested model is not registered.", "unknown_model")
        if model.execution == "cloud" and not self.config.defaults.cloud_calls:
            return _reject(task, "Cloud calls are disabled.", "cloud_disabled")

        try:
            validate_task_contract(
                task,
                known_roles=set(self.config.roles),
                known_models=self.config.known_model_tags(),
            )
        except TaskValidationError as exc:
            return _reject(task, str(exc), "contract_rejected")

        # The privacy rule above reads the task's own `execution` field, so the
        # registry has the last word on where this model really runs.
        if task.execution != model.execution:
            return _reject(
                task,
                f"Task declares execution {task.execution!r}, model runs {model.execution!r}.",
                "execution_mismatch",
            )
        if not self.config.model_allowed(model, task.agent_role, task.privacy_classification):
            return _reject(
                task,
                "Model is not enabled for this role and privacy class.",
                "model_not_permitted",
            )

        client = self.client or LocalOllamaClient(self.config.defaults.base_url)
        try:
            response = client.generate(
                model_tag=task.model_tag,
                prompt=_render_prompt(task),
                timeout_seconds=task.timeout_seconds,
                maximum_output_tokens=task.maximum_output_tokens,
            )
            result = normalize_result(
                response.text,
                task_id=task.task_id,
                model_tag=task.model_tag,
                agent_role=task.agent_role,
                metrics=response.metrics,
            )
        except OllamaClientError as exc:
            status = "timeout" if exc.category == "timeout" else "failed"
            result = _reject(task, str(exc), exc.category, status=status)
        except Exception as exc:
            if mandatory:
                raise
            result = _reject(task, str(exc), "unexpected_error", status="failed")

        self.telemetry.record(task, result)
        return result

    def run_shadow_review(
        self, payload: dict[str, Any], *, primary_result_summary: str = ""
    ) -> ResultEnvelope:
        """Ask for an advisory review of finished work. Never gates the caller."""
        try:
            task = _coerce_task(payload)
        except TypeError:
            return self.delegate_task(payload)
        shadow_task = replace(
            task,
            mode="shadow",
            agent_role=task.agent_role or "shadow-reviewer",
            objective=(
                f"{task.objective}\n\nPrimary result summary for advisory shadow review:\n"
                f"{primary_result_summary}"
                if primary_result_summary
                else task.objective
            ),
        )
        return self.delegate_task(shadow_task)


def delegate_task(payload: dict[str, Any], *, client: OllamaClient | None = None) -> ResultEnvelope:
    return OllamaAgentPool(client=client).delegate_task(payload)


def run_shadow_review(
    payload: dict[str, Any],
    *,
    primary_result_summary: str = "",
    client: OllamaClient | None = None,
) -> ResultEnvelope:
    return OllamaAgentPool(client=client).run_shadow_review(
        payload,
        primary_result_summary=primary_result_summary,
    )


def _reject(
    task: TaskContract, summary: str, category: str, *, status: str = "rejected"
) -> ResultEnvelope:
    return rejected_result(
        task_id=task.task_id,
        model_tag=task.model_tag,
        agent_role=task.agent_role,
        summary=summary,
        status=status,
        error_category=category,
    )


def _coerce_task(payload: dict[str, Any] | TaskContract) -> TaskContract:
    if isinstance(payload, TaskContract):
        return payload
    return TaskContract.from_dict(payload)


def _render_prompt(task: TaskContract) -> str:
    return json.dumps(
        {
            "review": "A primary orchestrator reviews this output before any use.",
            "task": task.to_dict(),
            "output": (
                "Return JSON with keys: summary, recommendations, findings, risks, "
                "assumptions, proposed_changes, tests_proposed, confidence."
            ),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
