"""The schema-validated dispatch client, over a real local socket."""
from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest
import yaml
from fake_ollama import FakeOllama, json_reply, raw_reply

from agent_gateway.governance.agent_registry import AgentDefinition
from agent_gateway.governance.ollama_client import OllamaResolutionError, dispatch_task
from agent_gateway.schemas import RESULT_PACKET, load_schema

RESULT_SCHEMA = load_schema(RESULT_PACKET)
PACKET = {"execution_id": "e1", "task_id": "t1"}
GOOD_RESULT = {
    "execution_id": "e1",
    "task_id": "t1",
    "status": "passed",
    "artifacts": [],
    "validation": {"agreement": True},
    "unresolved": [],
}


def agent(model_tag: str = "demo:latest", write_authority: str = "none") -> AgentDefinition:
    return AgentDefinition(
        name="test-critic",
        description="test",
        model=None,
        tools=[],
        provider="ollama",
        model_tag=model_tag,
        skill_allowlist=[],
        privacy_policy="internal",
        hooks=[],
        loop_max_iterations=1,
        write_authority=write_authority,
        requires_primary_review=True,
        runtime_target="any",
        source_path=Path("unused.md"),
    )


def registry(tmp_path: Path, *, enabled=True, model_enabled=True, roles=("critic",), **entry) -> Path:
    path = tmp_path / "registry.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "defaults": {"enabled": enabled},
                "models": {
                    "m": {
                        "model_tag": "demo:latest",
                        "enabled": model_enabled,
                        "roles": list(roles),
                        "timeout_seconds": 5,
                        **entry,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def chat(content: str):
    return json_reply({"message": {"role": "assistant", "content": content}})


def unused_port_url() -> str:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return f"http://127.0.0.1:{sock.getsockname()[1]}"


@pytest.mark.parametrize(
    "settings",
    [{"enabled": False}, {"model_enabled": False}, {"roles": ("test-designer",)}],
)
def test_the_registry_refuses_before_any_http_call(tmp_path: Path, settings: dict):
    with FakeOllama(chat("{}")) as server:
        with pytest.raises(OllamaResolutionError):
            dispatch_task(
                agent(),
                "critic",
                PACKET,
                RESULT_SCHEMA,
                registry_path=registry(tmp_path, **settings),
                base_url=server.url,
            )
        assert server.requests == []


def test_an_unregistered_model_is_refused(tmp_path: Path):
    with pytest.raises(OllamaResolutionError):
        dispatch_task(
            agent("unknown:tag"), "critic", PACKET, RESULT_SCHEMA, registry_path=registry(tmp_path)
        )


def test_a_missing_registry_is_a_resolution_error(tmp_path: Path):
    with pytest.raises(OllamaResolutionError):
        dispatch_task(
            agent(), "critic", PACKET, RESULT_SCHEMA, registry_path=tmp_path / "absent.yaml"
        )


def test_a_non_ollama_agent_is_refused(tmp_path: Path):
    other = AgentDefinition(**{**agent().__dict__, "provider": "claude"})
    with pytest.raises(OllamaResolutionError):
        dispatch_task(other, "critic", PACKET, RESULT_SCHEMA, registry_path=registry(tmp_path))


def test_an_agent_may_not_out_scope_its_model(tmp_path: Path):
    with pytest.raises(OllamaResolutionError, match="out-scope"):
        dispatch_task(
            agent(write_authority="staging-only"),
            "critic",
            PACKET,
            RESULT_SCHEMA,
            registry_path=registry(tmp_path, write_permission="artifact-only"),
        )


def test_an_unknown_write_scope_is_refused(tmp_path: Path):
    with pytest.raises(OllamaResolutionError, match="unknown write scope"):
        dispatch_task(
            agent(write_authority="everything"),
            "critic",
            PACKET,
            RESULT_SCHEMA,
            registry_path=registry(tmp_path),
        )


def test_a_cloud_model_needs_an_opt_in(tmp_path: Path, monkeypatch):
    path = registry(tmp_path, execution="cloud")
    with pytest.raises(OllamaResolutionError, match="leave this machine"):
        dispatch_task(agent(), "critic", PACKET, RESULT_SCHEMA, registry_path=path)

    monkeypatch.setenv("AGENT_GATEWAY_ALLOW_CLOUD", "true")
    with FakeOllama(chat(json.dumps(GOOD_RESULT))) as server:
        result = dispatch_task(
            agent(), "critic", PACKET, RESULT_SCHEMA, registry_path=path, base_url=server.url
        )
    assert result == GOOD_RESULT


def test_dispatch_returns_the_parsed_result_and_sends_the_schema(tmp_path: Path):
    with FakeOllama(chat(json.dumps(GOOD_RESULT))) as server:
        result = dispatch_task(
            agent(),
            "critic",
            PACKET,
            RESULT_SCHEMA,
            registry_path=registry(tmp_path),
            base_url=server.url,
        )
    assert result == GOOD_RESULT
    path, body = server.requests[0]
    assert path == "/api/chat"
    assert body["model"] == "demo:latest"
    assert body["format"] == RESULT_SCHEMA
    assert body["stream"] is False


def test_the_base_url_can_come_from_the_environment(tmp_path: Path, monkeypatch):
    with FakeOllama(chat(json.dumps(GOOD_RESULT))) as server:
        monkeypatch.setenv("AGENT_GATEWAY_OLLAMA_URL", server.url)
        result = dispatch_task(
            agent(), "critic", PACKET, RESULT_SCHEMA, registry_path=registry(tmp_path)
        )
    assert result["status"] == "passed"


@pytest.mark.parametrize(
    "reply",
    [
        chat("not json"),
        chat(json.dumps({"status": "passed"})),
        raw_reply(b"<html>Bad Gateway</html>"),
        json_reply(["a", "list"]),
        json_reply({"message": {"content": None}}),
        raw_reply(b"{}", status=500),
    ],
    ids=["bad-content", "schema-invalid", "bad-envelope", "list-envelope", "null-content", "http-500"],
)
def test_a_bad_response_returns_failed_not_an_exception(tmp_path: Path, reply):
    with FakeOllama(reply) as server:
        result = dispatch_task(
            agent(),
            "critic",
            PACKET,
            RESULT_SCHEMA,
            registry_path=registry(tmp_path),
            base_url=server.url,
        )
    assert result["status"] == "failed"
    assert result["unresolved"]
    assert result["task_id"] == "t1"


def test_a_refused_connection_returns_failed(tmp_path: Path):
    result = dispatch_task(
        agent(),
        "critic",
        PACKET,
        RESULT_SCHEMA,
        registry_path=registry(tmp_path),
        base_url=unused_port_url(),
    )
    assert result["status"] == "failed"
    assert "request failed" in result["unresolved"][0]
