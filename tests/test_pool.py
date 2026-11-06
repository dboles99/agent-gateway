from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest
from fake_ollama import FakeOllama, json_reply, raw_reply
from typer.testing import CliRunner

from agent_gateway.cli import app
from agent_gateway.pool import TaskContract, TaskValidationError, normalize_result
from agent_gateway.pool.contracts import validate_task_contract
from agent_gateway.pool.delegation import OllamaAgentPool
from agent_gateway.pool.local_client import (
    LocalOllamaClient,
    OllamaClientError,
    OllamaResponse,
)
from agent_gateway.pool.registry import (
    AgentPoolConfig,
    ModelEntry,
    PoolDefaults,
    RoleEntry,
    load_agent_pool_config,
)
from agent_gateway.pool.telemetry import TelemetryRecorder

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "pool"
LOCAL = "demo-local:latest"
CLOUD = "demo-cloud:latest"
runner = CliRunner()


class FakeClient:
    def __init__(self, response: str | None = None, error: OllamaClientError | None = None):
        self.response = response or json.dumps(
            {
                "summary": "Found missing tests.",
                "findings": [{"severity": "medium", "detail": "No timeout test."}],
                "recommendations": ["Add timeout coverage."],
                "tests_proposed": ["test_timeout"],
                "confidence": 0.4,
            }
        )
        self.error = error
        self.calls: list[dict] = []

    def generate(self, *, model_tag, prompt, timeout_seconds, maximum_output_tokens):
        self.calls.append(
            {
                "model_tag": model_tag,
                "prompt": prompt,
                "timeout_seconds": timeout_seconds,
                "maximum_output_tokens": maximum_output_tokens,
            }
        )
        if self.error:
            raise self.error
        return OllamaResponse(
            text=self.response,
            metrics={"latency_ms": 12, "prompt_tokens": 10, "output_tokens": 20},
        )


def model(model_id: str, tag: str, execution: str, **overrides) -> ModelEntry:
    values = {
        "model_id": model_id,
        "model_tag": tag,
        "execution": execution,
        "enabled": True,
        "status": "experimental",
        "lifecycle": "candidate",
        "roles": ("critic", "shadow-reviewer"),
        "privacy_classes": ("public", "internal", "confidential"),
        "write_permission": "none",
        "requires_primary_review": True,
    }
    values.update(overrides)
    return ModelEntry(**values)


def config(*, enabled: bool = True, cloud_calls: bool = False, local: dict | None = None) -> AgentPoolConfig:
    return AgentPoolConfig(
        defaults=PoolDefaults(enabled=enabled, mode="manual", cloud_calls=cloud_calls),
        models={
            "a-local": model("a-local", LOCAL, "local", **(local or {})),
            "b-cloud": model("b-cloud", CLOUD, "cloud"),
        },
        roles={
            "critic": RoleEntry(role_id="critic", purpose="Find flaws"),
            "shadow-reviewer": RoleEntry(role_id="shadow-reviewer", purpose="Review"),
        },
    )


def task(**overrides) -> dict:
    data = {
        "task_id": "t-001",
        "orchestrator": "codex",
        "mode": "delegated-subagent",
        "agent_role": "critic",
        "model_tag": LOCAL,
        "execution": "local",
        "objective": "Identify likely defects and missing tests.",
        "allowed_context": ["synthetic context only"],
        "excluded_context": ["credentials"],
        "timeout_seconds": 60,
        "maximum_output_tokens": 500,
        "success_criteria": ["Returns findings."],
        "stop_conditions": ["Timeout."],
    }
    data.update(overrides)
    return data


class TestDelegation:
    def test_listing_models_does_not_call_the_client(self):
        client = FakeClient()
        pool = OllamaAgentPool(config=config(enabled=False), client=client)
        assert len(pool.list_models()) == 2
        assert client.calls == []

    def test_a_disabled_pool_rejects_without_a_call(self):
        client = FakeClient()
        result = OllamaAgentPool(config=config(enabled=False), client=client).delegate_task(task())
        assert result.status == "rejected"
        assert result.metrics["error_category"] == "disabled"
        assert client.calls == []

    def test_delegation_normalises_the_result(self):
        client = FakeClient()
        result = OllamaAgentPool(config=config(), client=client).delegate_task(task())
        assert result.status == "completed"
        assert result.requires_review is True
        assert result.findings[0]["detail"] == "No timeout test."
        assert len(client.calls) == 1

    def test_auto_selects_an_eligible_model(self):
        client = FakeClient()
        payload = task()
        del payload["model_tag"], payload["execution"]
        result = OllamaAgentPool(config=config(), client=client).delegate_task(payload)
        assert result.status == "completed"
        assert client.calls[0]["model_tag"] == LOCAL

    def test_auto_with_no_eligible_model_is_rejected(self):
        client = FakeClient()
        payload = task(agent_role="nobody", model_tag="auto")
        result = OllamaAgentPool(config=config(), client=client).delegate_task(payload)
        assert result.metrics["error_category"] == "no_eligible_model"
        assert client.calls == []

    def test_a_task_with_unknown_fields_is_rejected_not_raised(self):
        client = FakeClient()
        result = OllamaAgentPool(config=config(), client=client).delegate_task(task(surprise=1))
        assert result.status == "rejected"
        assert result.metrics["error_category"] == "contract_rejected"
        assert client.calls == []

    def test_an_unknown_role_is_rejected_before_a_call(self):
        client = FakeClient()
        result = OllamaAgentPool(config=config(), client=client).delegate_task(
            task(agent_role="primary-orchestrator")
        )
        assert result.metrics["error_category"] == "contract_rejected"
        assert client.calls == []

    def test_an_unknown_model_is_rejected(self):
        client = FakeClient()
        result = OllamaAgentPool(config=config(), client=client).delegate_task(
            task(model_tag="new:latest")
        )
        assert result.metrics["error_category"] == "unknown_model"

    def test_cloud_confidential_work_is_blocked(self):
        client = FakeClient()
        result = OllamaAgentPool(config=config(cloud_calls=True), client=client).delegate_task(
            task(model_tag=CLOUD, execution="cloud", privacy_classification="confidential")
        )
        assert result.status == "rejected"
        assert "cloud execution is blocked" in result.summary
        assert client.calls == []

    def test_a_task_cannot_claim_local_to_send_confidential_work_to_the_cloud(self):
        """The contract rule reads the task's field, so the registry must check it."""
        client = FakeClient()
        result = OllamaAgentPool(config=config(cloud_calls=True), client=client).delegate_task(
            task(model_tag=CLOUD, execution="local", privacy_classification="confidential")
        )
        assert result.metrics["error_category"] == "execution_mismatch"
        assert client.calls == []

    def test_cloud_calls_are_off_by_default(self):
        client = FakeClient()
        result = OllamaAgentPool(config=config(), client=client).delegate_task(
            task(model_tag=CLOUD, execution="cloud")
        )
        assert result.metrics["error_category"] == "cloud_disabled"
        assert client.calls == []

    @pytest.mark.parametrize(
        "settings",
        [
            {"enabled": False},
            {"roles": ("something-else",)},
            {"privacy_classes": ("public",)},
            {"lifecycle": "blocked"},
            {"requires_primary_review": False},
            {"write_permission": "artifact-only"},
        ],
        ids=["disabled", "role", "privacy", "lifecycle", "review", "write"],
    )
    def test_a_named_model_must_still_be_permitted(self, settings):
        client = FakeClient()
        result = OllamaAgentPool(config=config(local=settings), client=client).delegate_task(task())
        assert result.metrics["error_category"] == "model_not_permitted"
        assert client.calls == []

    def test_staging_write_permission_is_blocked(self):
        client = FakeClient()
        result = OllamaAgentPool(config=config(), client=client).delegate_task(
            task(write_permission="staging-only")
        )
        assert "may not request staging" in result.summary
        assert client.calls == []

    def test_dropping_primary_review_is_blocked(self):
        client = FakeClient()
        result = OllamaAgentPool(config=config(), client=client).delegate_task(
            task(requires_primary_review=False)
        )
        assert "require primary review" in result.summary
        assert client.calls == []

    def test_a_timeout_is_normalised(self):
        client = FakeClient(error=OllamaClientError("timeout", "model timed out"))
        result = OllamaAgentPool(config=config(), client=client).delegate_task(task())
        assert result.status == "timeout"
        assert result.metrics["error_category"] == "timeout"

    def test_a_connection_failure_is_nonfatal(self):
        client = FakeClient(error=OllamaClientError("connection_failure", "refused"))
        result = OllamaAgentPool(config=config(), client=client).delegate_task(task())
        assert result.status == "failed"

    def test_an_unexpected_error_raises_only_when_mandatory(self):
        class Broken(FakeClient):
            def generate(self, **_kwargs):
                raise ValueError("boom")

        pool = OllamaAgentPool(config=config(), client=Broken())
        assert pool.delegate_task(task()).metrics["error_category"] == "unexpected_error"
        with pytest.raises(ValueError):
            pool.delegate_task(task(), mandatory=True)

    def test_a_shadow_review_carries_the_primary_summary(self):
        client = FakeClient()
        result = OllamaAgentPool(config=config(), client=client).run_shadow_review(
            task(mode="manual", agent_role="shadow-reviewer"),
            primary_result_summary="One file changed.",
        )
        assert result.status == "completed"
        assert result.requires_review is True
        assert "Primary result summary" in client.calls[0]["prompt"]


class TestTelemetry:
    def test_it_records_metadata_and_never_context(self, tmp_path: Path):
        path = tmp_path / "telemetry.jsonl"
        pool = OllamaAgentPool(
            config=config(), client=FakeClient(), telemetry=TelemetryRecorder(path=path)
        )
        pool.delegate_task(task(allowed_context=["PRIVATE_MARKER"]))
        event = json.loads(path.read_text(encoding="utf-8").strip())
        assert event["task_id"] == "t-001"
        assert event["role"] == "critic"
        assert "PRIVATE_MARKER" not in json.dumps(event)

    def test_a_telemetry_failure_does_not_break_the_task(self, tmp_path: Path):
        blocker = tmp_path / "a-file"
        blocker.write_text("", encoding="utf-8")
        pool = OllamaAgentPool(
            config=config(),
            client=FakeClient(),
            telemetry=TelemetryRecorder(path=blocker / "below" / "telemetry.jsonl"),
        )
        assert pool.delegate_task(task()).status == "completed"

    def test_it_is_off_without_a_path(self):
        assert TelemetryRecorder().record(TaskContract.from_dict(task()), normalize_result(
            "{}", task_id="t", model_tag="m", agent_role="r")) is False


class TestNormalizer:
    def normalize(self, raw):
        return normalize_result(raw, task_id="t-001", model_tag=LOCAL, agent_role="critic")

    def test_plain_text_becomes_a_partial_result(self):
        result = self.normalize("plain text finding")
        assert result.status == "partial"
        assert "plain text" in result.summary
        assert result.requires_review is True

    def test_a_json_list_is_not_treated_as_a_result(self):
        assert self.normalize("[1, 2]").status == "partial"

    def test_a_model_cannot_relabel_its_own_result(self):
        result = self.normalize(
            json.dumps(
                {
                    "task_id": "other",
                    "model_tag": "other",
                    "agent_role": "other",
                    "requires_review": False,
                    "metrics": {"latency_ms": 0, "output_tokens": 1},
                }
            )
        )
        assert (result.task_id, result.model_tag, result.agent_role) == ("t-001", LOCAL, "critic")
        assert result.requires_review is True
        assert result.metrics["output_tokens"] is None

    def test_an_invented_status_is_not_passed_through(self):
        assert self.normalize(json.dumps({"status": "approved"})).status == "partial"

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("0.5", 0.5), ("2", 1.0), ("-1", 0.0), ('"high"', 0.0), ("NaN", 0.0), ("Infinity", 0.0)],
    )
    def test_confidence_is_clamped_to_a_real_number(self, raw, expected):
        assert self.normalize(f'{{"confidence": {raw}}}').confidence == expected

    def test_single_values_become_lists(self):
        assert self.normalize(json.dumps({"risks": "one risk"})).risks == ["one risk"]


class TestContract:
    KNOWN = {"known_roles": {"critic"}, "known_models": {LOCAL}}

    def check(self, **overrides) -> None:
        validate_task_contract(TaskContract.from_dict(task(**overrides)), **self.KNOWN)

    def test_a_good_task_passes(self):
        self.check()

    @pytest.mark.parametrize(
        "overrides",
        [
            {"task_id": " "},
            {"objective": ""},
            {"orchestrator": "someone"},
            {"mode": "free-running"},
            {"mode": "disabled"},
            {"execution": "space"},
            {"privacy_classification": "secretive"},
            {"write_permission": "full"},
            {"timeout_seconds": 0},
            {"timeout_seconds": 3601},
            {"maximum_output_tokens": 0},
            {"maximum_output_tokens": 32001},
        ],
        ids=lambda o: f"{next(iter(o))}={next(iter(o.values()))}",
    )
    def test_a_bad_field_is_refused(self, overrides):
        with pytest.raises(TaskValidationError):
            self.check(**overrides)

    def test_round_trip(self):
        contract = TaskContract.from_dict(task())
        assert TaskContract.from_dict(contract.to_dict()) == contract


class TestRegistry:
    def test_no_files_means_an_empty_disabled_pool(self):
        loaded = load_agent_pool_config()
        assert loaded.defaults.enabled is False
        assert loaded.defaults.mode == "disabled"
        assert loaded.defaults.cloud_calls is False
        assert loaded.models == {}

    def test_the_example_config_loads_and_stays_off(self):
        loaded = load_agent_pool_config(
            EXAMPLES / "agent-pool.yaml", EXAMPLES / "agent-roles.yaml"
        )
        assert loaded.defaults.mode == "disabled"
        assert set(loaded.roles) == {"critic", "shadow-reviewer"}
        assert loaded.model_by_tag("your-model:latest") is not None

    def test_the_environment_overrides_the_file(self, monkeypatch):
        monkeypatch.setenv("AGENT_GATEWAY_POOL_MODE", "manual")
        monkeypatch.setenv("AGENT_GATEWAY_ALLOW_CLOUD", "yes")
        monkeypatch.setenv("AGENT_GATEWAY_OLLAMA_URL", "http://127.0.0.1:1")
        loaded = load_agent_pool_config(EXAMPLES / "agent-pool.yaml", EXAMPLES / "agent-roles.yaml")
        assert loaded.defaults.mode == "manual"
        assert loaded.defaults.cloud_calls is True
        assert loaded.defaults.base_url == "http://127.0.0.1:1"

    def test_an_unknown_model_entry_has_no_power(self):
        entry = config().unknown_model_entry("new:latest")
        assert entry.enabled is False
        assert entry.lifecycle == "unvalidated"
        assert entry.write_permission == "none"
        assert entry.requires_primary_review is True
        assert entry.fallback_eligible is False

    def test_selection_respects_privacy_and_execution(self):
        pool = config(cloud_calls=True)
        assert pool.select_model_for_role("critic").model_tag == LOCAL
        assert pool.select_model_for_role("critic", execution="cloud").model_tag == CLOUD
        assert pool.select_model_for_role("critic", privacy_classification="restricted") is None

    def test_explicit_selection_is_still_checked(self):
        assert config().select_model_for_role("critic", explicit_model=CLOUD) is None

    def test_fallback_is_off_by_default(self):
        assert config().model_by_tag(LOCAL).fallback_eligible is False


class TestLocalClient:
    def generate(self, url: str) -> OllamaResponse:
        return LocalOllamaClient(url).generate(
            model_tag="demo:latest", prompt="hi", timeout_seconds=5, maximum_output_tokens=7
        )

    def test_a_response_and_its_metrics(self):
        reply = json_reply({"response": "hello", "prompt_eval_count": 3, "eval_count": 4})
        with FakeOllama(reply) as server:
            response = self.generate(server.url + "/")
        assert response.text == "hello"
        assert response.metrics["prompt_tokens"] == 3
        assert response.metrics["output_tokens"] == 4
        path, body = server.requests[0]
        assert path == "/api/generate"
        assert body["options"]["num_predict"] == 7

    @pytest.mark.parametrize(
        ("status", "category"),
        [
            (401, "cloud_authentication_required"),
            (404, "model_not_found"),
            (413, "context_overflow"),
            (429, "rate_limit"),
            (500, "http_error"),
        ],
    )
    def test_http_errors_are_classified(self, status: int, category: str):
        with FakeOllama(raw_reply(b"{}", status=status)) as server:
            with pytest.raises(OllamaClientError) as caught:
                self.generate(server.url)
        assert caught.value.category == category

    @pytest.mark.parametrize("data", [b"<html>", b"[1]", b"\xff\xfe"], ids=["html", "list", "bytes"])
    def test_a_malformed_body_is_classified(self, data: bytes):
        with FakeOllama(raw_reply(data)) as server:
            with pytest.raises(OllamaClientError) as caught:
                self.generate(server.url)
        assert caught.value.category == "malformed_response"

    def test_a_refused_connection_is_classified(self):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            url = f"http://127.0.0.1:{sock.getsockname()[1]}"
        with pytest.raises(OllamaClientError) as caught:
            self.generate(url)
        assert caught.value.category == "connection_failure"

    def test_the_pool_works_through_the_real_client(self):
        reply = json_reply({"response": json.dumps({"summary": "Fine.", "confidence": 0.5})})
        with FakeOllama(reply) as server:
            defaults = PoolDefaults(enabled=True, mode="manual", base_url=server.url)
            pool_config = AgentPoolConfig(
                defaults=defaults, models=config().models, roles=config().roles
            )
            result = OllamaAgentPool(config=pool_config).delegate_task(task())
        assert result.status == "completed"
        assert result.summary == "Fine."
        assert result.confidence == 0.5


class TestCommandLine:
    def test_models_lists_the_registry(self, monkeypatch):
        monkeypatch.setenv("AGENT_GATEWAY_POOL_REGISTRY", str(EXAMPLES / "agent-pool.yaml"))
        result = runner.invoke(app, ["pool", "models"])
        assert result.exit_code == 0, result.output
        assert json.loads(result.output)["models"][0]["model_id"] == "local-reviewer"

    def test_delegate_prints_a_rejection_for_a_disabled_pool(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("AGENT_GATEWAY_POOL_REGISTRY", str(EXAMPLES / "agent-pool.yaml"))
        monkeypatch.setenv("AGENT_GATEWAY_POOL_ROLES", str(EXAMPLES / "agent-roles.yaml"))
        path = tmp_path / "task.json"
        path.write_text(json.dumps(task()), encoding="utf-8")
        result = runner.invoke(app, ["pool", "delegate", "--file", str(path)])
        assert result.exit_code == 0, result.output
        assert json.loads(result.output)["metrics"]["error_category"] == "disabled"

    def test_shadow_runs_through_the_cli(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("AGENT_GATEWAY_POOL_REGISTRY", str(EXAMPLES / "agent-pool.yaml"))
        monkeypatch.setenv("AGENT_GATEWAY_POOL_ROLES", str(EXAMPLES / "agent-roles.yaml"))
        monkeypatch.setenv("AGENT_GATEWAY_POOL_MODE", "manual")
        path = tmp_path / "task.yaml"
        path.write_text(
            "task_id: t-9\norchestrator: other\nmode: manual\nagent_role: critic\n"
            "objective: Review.\nmodel_tag: your-model:latest\n",
            encoding="utf-8",
        )
        reply = json_reply({"response": json.dumps({"summary": "Looks fine."})})
        with FakeOllama(reply) as server:
            monkeypatch.setenv("AGENT_GATEWAY_OLLAMA_URL", server.url)
            result = runner.invoke(app, ["pool", "shadow", "--file", str(path)])
        assert result.exit_code == 0, result.output
        assert json.loads(result.output)["summary"] == "Looks fine."
