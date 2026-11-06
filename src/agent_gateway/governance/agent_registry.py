"""Load agent definitions from Markdown files with YAML frontmatter.

The format is the one used for Claude Code subagents, with extra fields for model
provider, write authority and review. Loading applies governance rules, so a
definition that tries to limit its write scope while skipping review is refused.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator

from ..schemas import AGENT_DEFINITION, load_schema


class AgentDefinitionError(Exception):
    """An agent's frontmatter is malformed or fails schema validation."""


class AgentGovernanceError(AgentDefinitionError):
    """An agent definition violates a governance rule."""


class AgentResolutionError(AgentDefinitionError):
    """A named agent cannot be found."""


@dataclass(frozen=True)
class AgentDefinition:
    name: str
    description: str
    model: str | None
    tools: list[str]
    provider: str
    model_tag: str | None
    skill_allowlist: list[str]
    privacy_policy: str
    hooks: list[str]
    loop_max_iterations: int
    write_authority: str
    requires_primary_review: bool
    runtime_target: str
    source_path: Path


def _split_frontmatter(text: str, source_path: Path) -> dict[str, Any]:
    if not text.startswith("---"):
        raise AgentDefinitionError(
            f"{source_path}: agent file must start with YAML frontmatter (---)"
        )
    parts = text.split("---", 2)
    if len(parts) < 3:
        raise AgentDefinitionError(
            f"{source_path}: frontmatter is not closed with a second '---'"
        )
    data = yaml.safe_load(parts[1]) or {}
    if not isinstance(data, dict):
        raise AgentDefinitionError(f"{source_path}: frontmatter must be a YAML mapping")
    return data


def _normalize_tools(value: Any, source_path: Path) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value]
    if isinstance(value, str):
        return [t.strip() for t in value.split(",") if t.strip()]
    raise AgentDefinitionError(f"{source_path}: unsupported tools field type {type(value)!r}")


def load_agent_definition(path: Path) -> AgentDefinition:
    """Parse, schema-validate and apply governance rules to one agent file."""
    text = path.read_text(encoding="utf-8")
    data = _split_frontmatter(text, path)

    errors = list(Draft202012Validator(load_schema(AGENT_DEFINITION)).iter_errors(data))
    if errors:
        messages = "; ".join(e.message for e in errors)
        raise AgentDefinitionError(f"{path}: schema validation failed: {messages}")

    model_selector = data.get("model_selector") or {}
    provider = model_selector.get("provider", "claude")
    model_tag = model_selector.get("model") if provider != "claude" else None

    write_authority = data.get("write_authority", "full")
    default_review = write_authority != "full"
    requires_primary_review = data.get("requires_primary_review", default_review)

    if write_authority != "full" and not requires_primary_review:
        raise AgentGovernanceError(
            f"{path}: write_authority={write_authority!r} requires "
            "requires_primary_review: true (an agent cannot limit its own "
            "write scope while also skipping primary review)"
        )

    tools = _normalize_tools(data.get("tools"), path)
    if provider != "claude" and ("tools" not in data or tools):
        raise AgentGovernanceError(
            f"{path}: provider={provider} agents must declare `tools: []` "
            "explicitly. A coding-agent host may read an omitted tools field as "
            "full tool access, which would let this agent run directly instead of "
            "through its governed dispatch, bypassing write_authority and "
            "requires_primary_review"
        )

    loop_policy = data.get("loop_policy") or {}

    return AgentDefinition(
        name=data["name"],
        description=data["description"],
        model=data.get("model"),
        tools=tools,
        provider=provider,
        model_tag=model_tag,
        skill_allowlist=list(data.get("skill_allowlist") or []),
        privacy_policy=data.get("privacy_policy", "internal"),
        hooks=list(data.get("hooks") or []),
        loop_max_iterations=int(loop_policy.get("max_iterations", 1)),
        write_authority=write_authority,
        requires_primary_review=bool(requires_primary_review),
        runtime_target=data.get("runtime_target", "any"),
        source_path=path,
    )


def resolve_agent(name: str, agents_dir: Path | None = None) -> AgentDefinition:
    """Find and load ``<agents_dir>/<name>.md``. Defaults to ``./.claude/agents``."""
    directory = agents_dir if agents_dir is not None else Path.cwd() / ".claude" / "agents"
    path = directory / f"{name}.md"
    if not path.is_file():
        raise AgentResolutionError(f"No agent definition found at {path}")
    return load_agent_definition(path)
