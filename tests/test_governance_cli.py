from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from jsonschema import Draft202012Validator
from typer.testing import CliRunner

from agent_gateway import schemas
from agent_gateway._config import ensure_env_loaded
from agent_gateway.cli import app

runner = CliRunner()
EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
HOOK_PAYLOAD = json.dumps(
    {"tool_name": "Grep", "tool_input": {"pattern": "x"}, "session_id": "sess-1"}
)


def invoke(*args: str, stdin: str | None = None):
    return runner.invoke(app, ["governance", *args], input=stdin)


def test_log_event_appends_jsonl(tmp_path: Path):
    ledger = tmp_path / "events.jsonl"
    result = invoke("log-event", str(ledger), stdin=json.dumps({"tool": "Bash"}))
    assert result.exit_code == 0
    event = json.loads(ledger.read_text(encoding="utf-8").splitlines()[0])
    assert event["tool"] == "Bash"
    assert "event_id" in event


def test_check_loop_warns_after_the_limit(tmp_path: Path):
    state = tmp_path / "loop_state.json"
    payload = json.dumps({"tool": "Grep", "arguments": {"pattern": "x"}, "task_state": "s"})
    for _ in range(2):
        assert invoke("check-loop", str(state), stdin=payload).exit_code == 0
    result = invoke("check-loop", str(state), stdin=payload)
    assert result.exit_code == 1
    assert "Repeated tool call" in result.output


def test_check_loop_hook_warns_but_never_denies(tmp_path: Path):
    state = tmp_path / "loop_state.json"
    for _ in range(2):
        result = invoke("check-loop-hook", str(state), stdin=HOOK_PAYLOAD)
        output = json.loads(result.output)["hookSpecificOutput"]
        assert result.exit_code == 0
        assert output["permissionDecision"] == "allow"
        assert "additionalContext" not in output

    result = invoke("check-loop-hook", str(state), stdin=HOOK_PAYLOAD)
    output = json.loads(result.output)["hookSpecificOutput"]
    assert result.exit_code == 0
    assert output["permissionDecision"] == "allow"
    assert "Repeated tool call" in output["additionalContext"]


def test_a_corrupt_loop_counter_does_not_take_the_hook_down(tmp_path: Path):
    state = tmp_path / "loop_state.json"
    state.write_text("{not json", encoding="utf-8")
    result = invoke("check-loop-hook", str(state), stdin=HOOK_PAYLOAD)
    assert result.exit_code == 0
    assert json.loads(state.read_text(encoding="utf-8"))


def test_a_relative_state_path_ignores_the_working_directory(tmp_path: Path, monkeypatch):
    canonical = tmp_path / "canonical"
    checkout = tmp_path / "some-checkout"
    checkout.mkdir()
    monkeypatch.setenv("AGENT_GATEWAY_RUN_DIR", str(canonical))
    monkeypatch.chdir(checkout)

    assert invoke("check-loop-hook", "loop_state.json", stdin=HOOK_PAYLOAD).exit_code == 0

    assert (canonical / "loop_state.json").is_file()
    assert list(checkout.iterdir()) == []


def test_two_checkouts_share_one_repeat_counter(tmp_path: Path, monkeypatch):
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    monkeypatch.setenv("AGENT_GATEWAY_RUN_DIR", str(tmp_path / "canonical"))

    monkeypatch.chdir(first)
    for _ in range(2):
        invoke("check-loop-hook", "loop_state.json", stdin=HOOK_PAYLOAD)

    monkeypatch.chdir(second)
    result = invoke("check-loop-hook", "loop_state.json", stdin=HOOK_PAYLOAD)
    assert "Repeated tool call" in json.loads(result.output)["hookSpecificOutput"]["additionalContext"]


def test_write_checkpoint_round_trip(tmp_path: Path):
    path = tmp_path / "checkpoint.json"
    payload = {"checkpoint_version": "0.1", "active_model": "primary"}
    assert invoke("write-checkpoint", str(path), stdin=json.dumps(payload)).exit_code == 0
    assert json.loads(path.read_text(encoding="utf-8")) == payload


class TestAdmit:
    def capsule(self, tmp_path: Path, **changes) -> Path:
        data = yaml.safe_load((EXAMPLES / "context_capsule.yaml").read_text(encoding="utf-8"))
        data.update(changes)
        path = tmp_path / "capsule.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def test_the_example_downroutes(self):
        result = invoke("admit", str(EXAMPLES / "context_capsule.yaml"))
        assert result.exit_code == 0, result.output
        decision = json.loads(result.output)
        assert decision["status"] == "DOWNROUTE"
        assert decision["model"] == "worker"

    def test_deep_reasoning_keeps_the_primary_model(self, tmp_path: Path):
        result = invoke("admit", str(self.capsule(tmp_path, requires_deep_reasoning=True)))
        assert json.loads(result.output)["status"] == "ALLOW"

    def test_a_capsule_failing_its_schema_goes_back_for_repair(self, tmp_path: Path):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"capsule_id": "x"}), encoding="utf-8")
        result = invoke("admit", str(path))
        assert result.exit_code == 1
        assert json.loads(result.output)["status"] == "REQUIRE_REPAIR"

    def test_a_missing_stop_condition_is_not_assumed(self, tmp_path: Path):
        result = invoke("admit", str(self.capsule(tmp_path, stop_condition="")))
        assert result.exit_code == 1
        assert json.loads(result.output)["reason"] == "missing stop condition"

    def test_an_oversized_capsule_is_rejected(self, tmp_path: Path):
        estimate = {"fresh_dynamic": 3000, "total": 5000}
        result = invoke("admit", str(self.capsule(tmp_path, token_estimate=estimate)))
        assert result.exit_code == 1
        assert json.loads(result.output)["status"] == "REJECT"


class TestRulesCommands:
    PACK = str(EXAMPLES / "rules" / "release-gate.yaml")

    def test_a_pass_exits_0(self):
        result = invoke(
            "evaluate-rules", self.PACK, "--context", str(EXAMPLES / "rules" / "context-pass.json")
        )
        assert result.exit_code == 0, result.output
        assert "DECISION ACCEPT" in result.output

    def test_a_blocking_decision_exits_1(self, tmp_path: Path):
        context = tmp_path / "c.yaml"
        context.write_text("change: {touches_credentials: true}\n", encoding="utf-8")
        result = invoke("evaluate-rules", self.PACK, "--context", str(context), "--json")
        assert result.exit_code == 1
        data = json.loads(result.output)
        assert (data["rule"], data["blocking"]) == ("RG-001", True)

    def test_compile_writes_the_output_file(self, tmp_path: Path):
        output = tmp_path / "out" / "pack.json"
        assert invoke("compile-rules", self.PACK, "--output", str(output)).exit_code == 0
        assert json.loads(output.read_text(encoding="utf-8"))["id"] == "RELEASE-GATE"

    def test_render(self):
        assert "RULEPACK RELEASE-GATE" in invoke("render-rules", self.PACK).output


class TestSchemasAndExamples:
    def test_validate_schemas(self):
        result = invoke("validate-schemas")
        assert result.exit_code == 0, result.output
        assert "6 schemas valid" in result.output

    def test_the_example_packet_is_valid(self):
        result = invoke("validate-task", str(EXAMPLES / "task_packet.json"))
        assert result.exit_code == 0, result.output

    def test_an_invalid_packet_fails(self, tmp_path: Path):
        path = tmp_path / "bad.yaml"
        path.write_text("objective: only this\n", encoding="utf-8")
        assert invoke("validate-task", str(path)).exit_code == 1

    @pytest.mark.parametrize("model", ["primary", "fallback"])
    def test_a_checkpoint_matches_its_schema(self, model: str):
        checkpoint = {
            "checkpoint_version": "0.1",
            "project_id": "demo",
            "active_model": model,
            "scope": {"included": ["gateway"]},
            "completed_tasks": [],
            "active_tasks": [],
            "blocked_tasks": [],
            "next_actions": ["collect the result"],
        }
        schema = schemas.load_schema(schemas.SCHEMA_DIR / "checkpoint.schema.json")
        assert list(Draft202012Validator(schema).iter_errors(checkpoint)) == []


class TestEnvFile:
    def test_env_file_is_loaded_without_overriding(self, tmp_path: Path, monkeypatch):
        env_file = tmp_path / "settings.env"
        env_file.write_text(
            "AGENT_GATEWAY_POOL_MODE=manual\nAGENT_GATEWAY_RUN_DIR=/from/file\n", encoding="utf-8"
        )
        monkeypatch.setenv("AGENT_GATEWAY_ENV_FILE", str(env_file))
        monkeypatch.setenv("AGENT_GATEWAY_RUN_DIR", "/from/real/environment")

        ensure_env_loaded()

        import os

        try:
            assert os.environ["AGENT_GATEWAY_POOL_MODE"] == "manual"
            assert os.environ["AGENT_GATEWAY_RUN_DIR"] == "/from/real/environment"
        finally:
            os.environ.pop("AGENT_GATEWAY_POOL_MODE", None)


def test_importing_the_library_does_not_load_a_env_file(tmp_path: Path):
    """Only the command line loads .env. A library import has no side effects."""
    import subprocess
    import sys

    (tmp_path / ".env").write_text("AGENT_GATEWAY_SENTINEL=loaded\n", encoding="utf-8")
    code = (
        "import os, agent_gateway.gateway, agent_gateway.governance.paths, agent_gateway.pool;"
        "print(os.environ.get('AGENT_GATEWAY_SENTINEL', 'absent'))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "absent"
