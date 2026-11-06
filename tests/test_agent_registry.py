from __future__ import annotations

from pathlib import Path

import pytest

from agent_gateway.governance.agent_registry import (
    AgentDefinitionError,
    AgentGovernanceError,
    AgentResolutionError,
    load_agent_definition,
    resolve_agent,
)

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "agents"


def write_agent(tmp_path: Path, name: str, frontmatter: str) -> Path:
    path = tmp_path / f"{name}.md"
    path.write_text(f"---\n{frontmatter}\n---\n\n# {name}\n", encoding="utf-8")
    return path


def test_the_example_agent_loads():
    agent = resolve_agent("local-critic", agents_dir=EXAMPLES)
    assert agent.provider == "ollama"
    assert agent.model_tag == "your-model:latest"
    assert agent.write_authority == "none"
    assert agent.requires_primary_review is True
    assert agent.privacy_policy == "internal"


def test_a_plain_agent_defaults(tmp_path: Path):
    path = write_agent(
        tmp_path,
        "worker",
        "name: worker\ndescription: Does the work.\nmodel: small\ntools: Read, Edit, Bash\n",
    )
    agent = load_agent_definition(path)
    assert agent.model == "small"
    assert agent.tools == ["Read", "Edit", "Bash"]
    assert agent.provider == "claude"
    assert agent.model_tag is None
    assert agent.write_authority == "full"
    assert agent.requires_primary_review is False
    assert agent.loop_max_iterations == 1
    assert agent.runtime_target == "any"


def test_the_default_directory_is_relative_to_the_working_directory(tmp_path, monkeypatch):
    agents = tmp_path / ".claude" / "agents"
    agents.mkdir(parents=True)
    write_agent(agents, "here", "name: here\ndescription: Found here.\n")
    monkeypatch.chdir(tmp_path)
    assert resolve_agent("here").name == "here"


def test_resolve_raises_when_not_found(tmp_path: Path):
    with pytest.raises(AgentResolutionError):
        resolve_agent("does-not-exist", agents_dir=tmp_path)


def test_a_non_claude_agent_without_explicit_tools_is_refused(tmp_path: Path):
    path = write_agent(
        tmp_path,
        "no-tools",
        "name: no-tools\ndescription: Forgot tools.\n"
        "model_selector:\n  provider: ollama\n  model: demo:latest\n"
        "write_authority: none\nrequires_primary_review: true\n",
    )
    with pytest.raises(AgentGovernanceError):
        load_agent_definition(path)


def test_a_non_claude_agent_with_tools_is_refused(tmp_path: Path):
    path = write_agent(
        tmp_path,
        "with-tools",
        "name: with-tools\ndescription: Declares tools.\n"
        "model_selector:\n  provider: ollama\n  model: demo:latest\n"
        "tools:\n  - Read\nwrite_authority: none\nrequires_primary_review: true\n",
    )
    with pytest.raises(AgentGovernanceError):
        load_agent_definition(path)


def test_limited_write_authority_defaults_review_to_true(tmp_path: Path):
    path = write_agent(
        tmp_path,
        "artifact",
        "name: artifact\ndescription: Writes artifacts.\nwrite_authority: artifact-only\n",
    )
    assert load_agent_definition(path).requires_primary_review is True


def test_limiting_write_scope_while_skipping_review_is_refused(tmp_path: Path):
    path = write_agent(
        tmp_path,
        "bad",
        "name: bad\ndescription: Skips review.\n"
        "write_authority: staging-only\nrequires_primary_review: false\n",
    )
    with pytest.raises(AgentGovernanceError):
        load_agent_definition(path)


def test_missing_closing_frontmatter_raises(tmp_path: Path):
    path = tmp_path / "broken.md"
    path.write_text("---\nname: broken\n\n# no closing marker\n", encoding="utf-8")
    with pytest.raises(AgentDefinitionError):
        load_agent_definition(path)


def test_no_frontmatter_raises(tmp_path: Path):
    path = tmp_path / "plain.md"
    path.write_text("# just a heading\n", encoding="utf-8")
    with pytest.raises(AgentDefinitionError):
        load_agent_definition(path)


def test_a_schema_violation_raises(tmp_path: Path):
    path = write_agent(tmp_path, "typo", "name: typo\ndescription: x\nwrite_authority: everything\n")
    with pytest.raises(AgentDefinitionError):
        load_agent_definition(path)


def test_tools_as_a_yaml_list(tmp_path: Path):
    path = write_agent(
        tmp_path, "list", "name: list\ndescription: A list.\ntools:\n  - Read\n  - Grep\n"
    )
    assert load_agent_definition(path).tools == ["Read", "Grep"]
