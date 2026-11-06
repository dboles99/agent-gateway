"""The pool's registry of models and roles, read from two YAML files.

With no files present the pool is disabled and has no models. Fields a model entry
carries beyond those used in ``model_allowed`` are descriptive and are not enforced.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .._config import (
    ALLOW_CLOUD,
    DEFAULT_OLLAMA_URL,
    OLLAMA_URL,
    POOL_ENABLED,
    POOL_MODE,
    env_bool,
    pool_registry_path,
    pool_roles_path,
)


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data if isinstance(data, dict) else {}


@dataclass(frozen=True)
class PoolDefaults:
    enabled: bool = False
    mode: str = "disabled"
    cloud_calls: bool = False
    base_url: str = DEFAULT_OLLAMA_URL

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PoolDefaults:
        return cls(
            enabled=env_bool(POOL_ENABLED, bool(data.get("enabled", False))),
            mode=os.environ.get(POOL_MODE, str(data.get("mode", "disabled"))),
            cloud_calls=env_bool(ALLOW_CLOUD, bool(data.get("cloud_calls", False))),
            base_url=os.environ.get(OLLAMA_URL, str(data.get("base_url", DEFAULT_OLLAMA_URL))),
        )


@dataclass(frozen=True)
class ModelEntry:
    model_id: str
    model_tag: str
    provider: str = "ollama"
    execution: str = "local"
    enabled: bool = False
    status: str = "unvalidated"
    lifecycle: str = "unvalidated"
    maturity: str = "experimental"
    roles: tuple[str, ...] = ()
    privacy_classes: tuple[str, ...] = ("public",)
    timeout_seconds: int = 600
    max_output_tokens: int = 2000
    preferred_context_size: int | None = None
    write_permission: str = "none"
    requires_primary_review: bool = True
    fallback_eligible: bool = False
    cost_class: str = "unknown"
    approved_task_types: tuple[str, ...] = ()
    prohibited_task_types: tuple[str, ...] = ()
    known_weaknesses: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, model_id: str, data: dict[str, Any]) -> ModelEntry:
        tag = data.get("model_tag") or f"unconfigured:{model_id}"
        return cls(
            model_id=model_id,
            model_tag=str(tag),
            provider=str(data.get("provider", "ollama")),
            execution=str(data.get("execution", "local")),
            enabled=bool(data.get("enabled", False)),
            status=str(data.get("status", "unvalidated")),
            lifecycle=str(data.get("lifecycle", data.get("status", "unvalidated"))),
            maturity=str(data.get("maturity", "experimental")),
            roles=tuple(data.get("roles", []) or []),
            privacy_classes=tuple(data.get("privacy_classes", ["public"]) or ["public"]),
            timeout_seconds=int(data.get("timeout_seconds", 600)),
            max_output_tokens=int(data.get("max_output_tokens", 2000)),
            preferred_context_size=(
                int(data["preferred_context_size"]) if data.get("preferred_context_size") else None
            ),
            write_permission=str(data.get("write_permission", "none")),
            requires_primary_review=bool(data.get("requires_primary_review", True)),
            fallback_eligible=bool(data.get("fallback_eligible", False)),
            cost_class=str(data.get("cost_class", "unknown")),
            approved_task_types=tuple(data.get("approved_task_types", []) or []),
            prohibited_task_types=tuple(data.get("prohibited_task_types", []) or []),
            known_weaknesses=tuple(data.get("known_weaknesses", []) or []),
        )

    def advisory_safe(self) -> bool:
        return self.requires_primary_review and self.write_permission == "none"


@dataclass(frozen=True)
class RoleEntry:
    role_id: str
    purpose: str
    allowed_task_types: tuple[str, ...] = ()
    prohibited_task_types: tuple[str, ...] = ()
    write_permission: str = "none"
    expected_output_structure: str = "advisory-result-envelope"
    requires_primary_review: bool = True
    default_timeout_seconds: int = 600
    privacy_classes: tuple[str, ...] = ("public", "internal")

    @classmethod
    def from_dict(cls, role_id: str, data: dict[str, Any]) -> RoleEntry:
        return cls(
            role_id=role_id,
            purpose=str(data.get("purpose", "")),
            allowed_task_types=tuple(data.get("allowed_task_types", []) or []),
            prohibited_task_types=tuple(data.get("prohibited_task_types", []) or []),
            write_permission=str(data.get("write_permission", "none")),
            expected_output_structure=str(
                data.get("expected_output_structure", "advisory-result-envelope")
            ),
            requires_primary_review=bool(data.get("requires_primary_review", True)),
            default_timeout_seconds=int(data.get("default_timeout_seconds", 600)),
            privacy_classes=tuple(data.get("privacy_classes", ["public", "internal"]) or []),
        )


@dataclass(frozen=True)
class AgentPoolConfig:
    defaults: PoolDefaults = field(default_factory=PoolDefaults)
    models: dict[str, ModelEntry] = field(default_factory=dict)
    roles: dict[str, RoleEntry] = field(default_factory=dict)

    def known_model_tags(self) -> set[str]:
        return {m.model_tag for m in self.models.values() if m.model_tag}

    def model_by_tag(self, model_tag: str) -> ModelEntry | None:
        for model in self.models.values():
            if model.model_tag == model_tag or model.model_id == model_tag:
                return model
        return None

    def unknown_model_entry(self, model_tag: str) -> ModelEntry:
        return ModelEntry(
            model_id=f"unknown:{model_tag}",
            model_tag=model_tag,
            enabled=False,
            status="unvalidated",
            lifecycle="unvalidated",
            maturity="experimental",
            roles=(),
            privacy_classes=("public",),
            write_permission="none",
            requires_primary_review=True,
            fallback_eligible=False,
            known_weaknesses=("not registered or validated",),
        )

    def list_models(self) -> list[ModelEntry]:
        return sorted(self.models.values(), key=lambda item: item.model_id)

    def select_model_for_role(
        self,
        role: str,
        *,
        privacy_classification: str = "internal",
        execution: str | None = None,
        explicit_model: str | None = None,
    ) -> ModelEntry | None:
        if explicit_model and explicit_model != "auto":
            model = self.model_by_tag(explicit_model)
            return model if model and self.model_allowed(model, role, privacy_classification) else None

        for model in self.list_models():
            if execution and model.execution != execution:
                continue
            if self.model_allowed(model, role, privacy_classification):
                return model
        return None

    def model_allowed(self, model: ModelEntry, role: str, privacy_classification: str) -> bool:
        """The rule both automatic selection and dispatch apply to a model."""
        if not model.enabled:
            return False
        if role not in model.roles:
            return False
        if privacy_classification not in model.privacy_classes:
            return False
        if model.execution == "cloud" and not self.defaults.cloud_calls:
            return False
        if privacy_classification in {"confidential", "restricted"} and model.execution == "cloud":
            return False
        if not model.requires_primary_review:
            return False
        if model.write_permission != "none":
            return False
        return model.lifecycle not in {"blocked", "deprecated", "unvalidated", "discovered"}


def load_agent_pool_config(
    registry_path: Path | None = None,
    roles_path: Path | None = None,
) -> AgentPoolConfig:
    """Load the pool config. An absent file means an empty, disabled pool."""
    registry_raw = _read_yaml(Path(registry_path) if registry_path else pool_registry_path())
    roles_raw = _read_yaml(Path(roles_path) if roles_path else pool_roles_path())
    defaults = PoolDefaults.from_dict(registry_raw.get("defaults", {}) or {})
    models = {
        model_id: ModelEntry.from_dict(model_id, model_data or {})
        for model_id, model_data in (registry_raw.get("models", {}) or {}).items()
    }
    roles = {
        role_id: RoleEntry.from_dict(role_id, role_data or {})
        for role_id, role_data in (roles_raw.get("roles", {}) or {}).items()
    }
    return AgentPoolConfig(defaults=defaults, models=models, roles=roles)
