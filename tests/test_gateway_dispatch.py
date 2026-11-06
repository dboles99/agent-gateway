"""The gateway's enforcement layer and its command line.

Every refusal must be a valid result packet, and no two refusal reasons may
collapse into one: an unbuilt capability is normal, a broken handler is a defect.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from typer.testing import CliRunner

from agent_gateway.cli import app
from agent_gateway.gateway import dispatch
from agent_gateway.gateway.dispatch import DEFAULT_RESULT_SCHEMA, run_task_packet
from agent_gateway.gateway.executor import execute_task
from agent_gateway.gateway.progress import progress_event
from agent_gateway.gateway.registry import (
    CapabilityConfigError,
    CapabilityRegistry,
    DeclaredCapability,
    HandlerNotDeclared,
    HandlerUnresolvable,
    handler_status,
    load_declared_capabilities,
    load_declared_registry,
    resolve_declared_handler,
)

CAPABILITY = "text.summarise"
REQUIRED = ("summary", "audit_event")
runner = CliRunner()

CONFIG_BODY = (
    "capabilities:\n"
    f"  - id: {CAPABILITY}\n"
    "    executor: Summariser\n"
    "    required_outputs: [summary, audit_event]\n"
    "  - id: text.classify\n"
    "    executor: classifier\n"
)


def a_task_packet(**overrides) -> dict:
    packet = {
        "protocol_version": "1.0",
        "execution_id": "exec-1",
        "task_id": "t-1",
        "capability": CAPABILITY,
        "objective": "Summarise the text.",
        "inputs": {"text": "synthetic"},
        "acceptance": ["summary exists"],
        "budget": {"max_turns": 4},
        "stop_condition": "acceptance criteria pass",
    }
    packet.update(overrides)
    return packet


def a_complete_handler(_packet: dict) -> dict:
    return {
        "status": "passed",
        "artifacts": [{"name": name} for name in REQUIRED],
        "validation": {"summary_present": True},
    }


def registry_with(handler, config: Path | None = None) -> CapabilityRegistry:
    registry = CapabilityRegistry(config_path=config)
    registry.register(CAPABILITY, handler)
    return registry


def write_config(tmp: Path, body: str, name: str = "capabilities.yaml") -> Path:
    path = tmp / name
    path.write_text(body, encoding="utf-8")
    return path


@pytest.fixture
def config(tmp_path: Path) -> Path:
    return write_config(tmp_path, CONFIG_BODY)


class TestHandlerResolution:
    def test_absent_and_broken_raise_different_exceptions(self):
        with pytest.raises(HandlerNotDeclared):
            resolve_declared_handler(DeclaredCapability(id="x"))
        with pytest.raises(HandlerUnresolvable):
            resolve_declared_handler(DeclaredCapability(id="x", handler="no.such.module:thing"))

    def test_a_handler_that_is_not_callable_is_unresolvable(self):
        with pytest.raises(HandlerUnresolvable):
            resolve_declared_handler(DeclaredCapability(id="x", handler="json:__doc__"))

    def test_a_missing_attribute_is_unresolvable(self):
        with pytest.raises(HandlerUnresolvable):
            resolve_declared_handler(DeclaredCapability(id="x", handler="json:nothing_here"))

    def test_dotted_and_colon_forms_resolve(self):
        assert resolve_declared_handler(DeclaredCapability(id="x", handler="json:dumps")) is json.dumps
        assert resolve_declared_handler(DeclaredCapability(id="x", handler="json.dumps")) is json.dumps

    def test_a_malformed_path_is_unresolvable_not_a_crash(self):
        with pytest.raises(HandlerUnresolvable):
            resolve_declared_handler(DeclaredCapability(id="x", handler="nomodulehere"))

    def test_handler_status_names_all_three_states(self):
        assert handler_status(DeclaredCapability(id="x"))[0] == "unbuilt"
        assert handler_status(DeclaredCapability(id="x", handler="json:dumps"))[0] == "ready"
        assert handler_status(DeclaredCapability(id="x", handler="no.such:y"))[0] == "unresolvable"


class TestEnforcement:
    def test_a_complete_handler_passes(self, config: Path):
        result = run_task_packet(
            a_task_packet(), registry_with(a_complete_handler), config_path=config
        )
        assert result["status"] == "passed"
        assert len(result["artifacts"]) == len(REQUIRED)

    def test_required_outputs_are_enforced(self, config: Path):
        handler = lambda _p: {"status": "passed", "artifacts": [{"name": "summary"}]}  # noqa: E731
        result = run_task_packet(a_task_packet(), registry_with(handler), config_path=config)
        assert result["status"] == "failed"
        assert result["unresolved"][0]["reason"] == "required_outputs_missing"
        assert result["unresolved"][0]["errors"] == ["audit_event"]

    def test_a_handler_saying_passed_is_not_the_last_word(self, config: Path):
        result = run_task_packet(
            a_task_packet(), registry_with(lambda _p: {"status": "passed"}), config_path=config
        )
        assert result["status"] == "failed"

    def test_an_unnamed_artifact_satisfies_no_requirement(self, config: Path):
        handler = lambda _p: {"status": "passed", "artifacts": [{"path": "out.txt"}]}  # noqa: E731
        result = run_task_packet(a_task_packet(), registry_with(handler), config_path=config)
        assert result["unresolved"][0]["reason"] == "required_outputs_missing"

    def test_a_handler_reporting_blocked_keeps_its_own_reason(self, config: Path):
        handler = lambda _p: {  # noqa: E731
            "status": "blocked",
            "unresolved": [{"reason": "waiting_on_input"}],
        }
        result = run_task_packet(a_task_packet(), registry_with(handler), config_path=config)
        assert result["status"] == "blocked"
        assert result["unresolved"][0]["reason"] == "waiting_on_input"

    def test_an_invalid_packet_is_refused_before_the_handler_runs(self, config: Path):
        ran: list[bool] = []

        def handler(packet: dict) -> dict:
            ran.append(True)
            return a_complete_handler(packet)

        result = run_task_packet(
            {"capability": CAPABILITY, "task_id": "t"}, registry_with(handler), config_path=config
        )
        assert result["unresolved"][0]["reason"] == "task_packet_invalid"
        assert ran == []

    def test_a_handler_that_raises_does_not_crash_the_gateway(self, config: Path):
        def boom(_packet: dict) -> dict:
            raise RuntimeError("the model timed out")

        result = run_task_packet(a_task_packet(), registry_with(boom), config_path=config)
        assert result["status"] == "failed"
        assert result["unresolved"][0]["reason"] == "handler_raised"
        assert "the model timed out" in result["unresolved"][0]["detail"]

    def test_a_handler_returning_a_non_mapping_is_named(self, config: Path):
        result = run_task_packet(
            a_task_packet(), registry_with(lambda _p: "done"), config_path=config
        )
        assert result["unresolved"][0]["reason"] == "handler_result_malformed"

    def test_a_status_outside_the_protocol_is_refused(self, config: Path):
        handler = lambda p: {**a_complete_handler(p), "status": "great"}  # noqa: E731
        result = run_task_packet(a_task_packet(), registry_with(handler), config_path=config)
        assert result["unresolved"][0]["reason"] == "result_packet_invalid"

    def test_a_declared_schema_missing_on_disk_is_refused_not_skipped(self, tmp_path: Path):
        path = write_config(
            tmp_path,
            f"capabilities:\n  - id: {CAPABILITY}\n    input_schema: absent.json\n",
        )
        result = run_task_packet(
            a_task_packet(), registry_with(a_complete_handler), config_path=path
        )
        assert result["unresolved"][0]["reason"] == "declared_schema_missing"

    def test_a_declared_schema_is_read_relative_to_the_config(self, tmp_path: Path):
        (tmp_path / "strict.json").write_text(
            json.dumps({"type": "object", "required": ["only_this_field"]}), encoding="utf-8"
        )
        path = write_config(
            tmp_path,
            f"capabilities:\n  - id: {CAPABILITY}\n    input_schema: strict.json\n",
        )
        result = run_task_packet(
            a_task_packet(), registry_with(a_complete_handler), config_path=path
        )
        assert result["unresolved"][0]["reason"] == "task_packet_invalid"

    def test_a_completed_dispatch_emits_a_progress_proof(self, config: Path):
        result = run_task_packet(
            a_task_packet(), registry_with(a_complete_handler), config_path=config
        )
        assert [p["event_type"] for p in result["progress_proofs"]] == ["capability_completed"]
        assert result["progress_proofs"][0]["run_id"] == "exec-1"

    def test_handler_progress_proofs_are_kept(self, config: Path):
        handler = lambda p: {  # noqa: E731
            **a_complete_handler(p),
            "progress_proofs": [{"event_type": "from_the_handler"}],
        }
        result = run_task_packet(a_task_packet(), registry_with(handler), config_path=config)
        assert [p["event_type"] for p in result["progress_proofs"]] == [
            "from_the_handler",
            "capability_completed",
        ]

    def test_a_declared_handler_is_bound_from_the_config(self, tmp_path: Path):
        path = write_config(
            tmp_path,
            "capabilities:\n"
            "  - id: text.word_count\n"
            "    handler: agent_gateway.demo:word_count\n"
            "    required_outputs: [word_count]\n",
        )
        packet = a_task_packet(capability="text.word_count", inputs={"text": "one two three"})
        result = run_task_packet(packet, config_path=path)
        assert result["status"] == "passed"
        assert result["artifacts"][0]["value"] == 3


class TestRefusalContract:
    def every_refusal(self, tmp_path: Path) -> dict[str, dict]:
        config = write_config(tmp_path, CONFIG_BODY)
        registry = registry_with(a_complete_handler)
        cap = f"capabilities:\n  - id: {CAPABILITY}\n"

        def boom(_packet: dict) -> dict:
            raise ValueError("nope")

        return {
            "not an object": run_task_packet(["nope"]),
            "config unreadable": run_task_packet(
                a_task_packet(), config_path=write_config(tmp_path, "a: [", "bad.yaml")
            ),
            "invalid packet": run_task_packet({"capability": CAPABILITY}, config_path=config),
            "undeclared": run_task_packet(a_task_packet(capability="typo.id"), config_path=config),
            "unbuilt": run_task_packet(a_task_packet(), config_path=config),
            "handler raised": run_task_packet(
                a_task_packet(), registry_with(boom), config_path=config
            ),
            "malformed": run_task_packet(
                a_task_packet(), registry_with(lambda _p: 1), config_path=config
            ),
            "outputs missing": run_task_packet(
                a_task_packet(), registry_with(lambda _p: {"status": "passed"}), config_path=config
            ),
            "invalid result": run_task_packet(
                a_task_packet(),
                registry_with(lambda p: {**a_complete_handler(p), "artifacts": "x"}),
                config_path=write_config(tmp_path, cap, "plain.yaml"),
            ),
            "unresolvable": run_task_packet(
                a_task_packet(),
                config_path=write_config(
                    tmp_path, cap + "    handler: no.such.module:thing\n", "broken.yaml"
                ),
            ),
            "schema missing": run_task_packet(
                a_task_packet(),
                registry,
                config_path=write_config(
                    tmp_path, cap + "    output_schema: absent.json\n", "noschema.yaml"
                ),
            ),
        }

    def test_every_refusal_is_a_valid_result_packet(self, tmp_path: Path):
        validator = Draft202012Validator(
            json.loads(DEFAULT_RESULT_SCHEMA.read_text(encoding="utf-8"))
        )
        for label, packet in self.every_refusal(tmp_path).items():
            errors = [e.message for e in validator.iter_errors(packet)]
            assert errors == [], label

    def test_the_reasons_are_all_distinct(self, tmp_path: Path):
        reasons = [p["unresolved"][0]["reason"] for p in self.every_refusal(tmp_path).values()]
        assert sorted(reasons) == sorted(
            {
                dispatch.REASON_UNREADABLE,
                dispatch.REASON_CONFIG_UNREADABLE,
                dispatch.REASON_INVALID,
                dispatch.REASON_UNDECLARED,
                dispatch.REASON_UNBUILT,
                dispatch.REASON_HANDLER_RAISED,
                dispatch.REASON_RESULT_MALFORMED,
                dispatch.REASON_OUTPUTS_MISSING,
                dispatch.REASON_RESULT_INVALID,
                dispatch.REASON_UNRESOLVABLE,
                dispatch.REASON_SCHEMA_MISSING,
            }
        )

    def test_an_unreadable_packet_still_returns_a_packet(self):
        result = run_task_packet("not a dict at all")
        assert result["status"] == "blocked"
        assert result["execution_id"] == ""
        assert result["unresolved"][0]["reason"] == "task_packet_unreadable"

    def test_the_protocol_schema_is_the_floor_when_none_is_declared(self, config: Path):
        result = run_task_packet({"capability": "text.classify"}, config_path=config)
        assert result["unresolved"][0]["reason"] == "task_packet_invalid"


class TestRegistry:
    def test_the_config_is_read(self, config: Path):
        declared = load_declared_capabilities(config)
        assert set(declared) == {CAPABILITY, "text.classify"}
        assert declared[CAPABILITY].executor == "Summariser"
        assert declared[CAPABILITY].required_outputs == REQUIRED

    def test_a_missing_file_is_an_empty_declaration(self, tmp_path: Path):
        assert load_declared_capabilities(tmp_path / "absent.yaml") == {}

    def test_an_unparseable_file_raises(self, tmp_path: Path):
        with pytest.raises(CapabilityConfigError):
            load_declared_capabilities(write_config(tmp_path, "a: [", "bad.yaml"))

    def test_an_edit_to_the_config_is_seen(self, tmp_path: Path):
        path = write_config(tmp_path, "capabilities:\n  - id: one\n")
        assert set(load_declared_capabilities(path)) == {"one"}
        path.write_text("capabilities:\n  - id: one\n  - id: three\n", encoding="utf-8")
        assert set(load_declared_capabilities(path)) == {"one", "three"}

    def test_the_default_location_comes_from_the_environment(self, monkeypatch, config: Path):
        monkeypatch.setenv("AGENT_GATEWAY_CAPABILITIES", str(config))
        assert CAPABILITY in load_declared_capabilities()

    def test_strict_registration(self, config: Path):
        registry = CapabilityRegistry(config_path=config)
        with pytest.raises(KeyError):
            registry.register("text.typo", a_complete_handler, strict=True)
        registry.register(CAPABILITY, a_complete_handler, strict=True)
        assert registry.resolve(CAPABILITY) is a_complete_handler

    def test_lenient_registration_and_duplicates(self):
        registry = CapabilityRegistry()
        registry.register("x", a_complete_handler)
        with pytest.raises(ValueError):
            registry.register("x", a_complete_handler)

    def test_declared_but_unbuilt_reads_differently_from_a_bad_id(self, config: Path):
        registry = CapabilityRegistry(config_path=config)
        with pytest.raises(KeyError, match="declared but has no handler"):
            registry.resolve("text.classify")
        with pytest.raises(KeyError, match="Unknown capability"):
            registry.resolve("text.not_a_thing")

    def test_gaps_are_listed_in_both_directions(self, config: Path):
        registry = CapabilityRegistry(config_path=config)
        registry.register(CAPABILITY, a_complete_handler)
        registry.register("text.invented", a_complete_handler)
        assert registry.unimplemented() == ("text.classify",)
        assert registry.undeclared() == ("text.invented",)

    def test_the_registry_says_why_each_capability_did_not_bind(self, config: Path):
        registry, unbound = load_declared_registry(config)
        assert registry.capabilities == {}
        assert "declares no handler" in unbound[CAPABILITY]


def test_execute_task_returns_a_compact_result():
    registry = CapabilityRegistry()
    registry.register(CAPABILITY, a_complete_handler)
    result = execute_task({"execution_id": "e", "task_id": "t", "capability": CAPABILITY}, registry)
    assert (result["execution_id"], result["task_id"], result["status"]) == ("e", "t", "passed")
    assert result["unresolved"] == []


def test_progress_event_shape():
    event = progress_event(
        run_id="r", task_id="t", event_type="artifact_created", status="ok", weight=3
    )
    assert event["progress_weight"] == 3
    assert "event_id" in event
    assert "timestamp" in event


class TestCommandLine:
    def test_capabilities_reports_the_config(self, config: Path):
        result = runner.invoke(
            app, ["gateway", "capabilities", "--json", "--config", str(config)]
        )
        assert result.exit_code == 0, result.output
        rows = json.loads(result.stdout)["capabilities"]
        assert {row["id"] for row in rows} == {CAPABILITY, "text.classify"}
        assert {row["state"] for row in rows} == {"unbuilt"}

    def test_capabilities_exits_1_only_for_an_unresolvable_handler(self, tmp_path: Path):
        broken = write_config(
            tmp_path, "capabilities:\n  - id: x\n    handler: no.such.module:thing\n", "b.yaml"
        )
        unbuilt = write_config(tmp_path, "capabilities:\n  - id: x\n", "u.yaml")
        assert runner.invoke(app, ["gateway", "capabilities", "--config", str(broken)]).exit_code == 1
        assert runner.invoke(app, ["gateway", "capabilities", "--config", str(unbuilt)]).exit_code == 0

    def test_validate(self):
        good = runner.invoke(app, ["gateway", "validate", "-"], input=json.dumps(a_task_packet()))
        assert good.exit_code == 0, good.output
        bad = runner.invoke(app, ["gateway", "validate", "-"], input=json.dumps({"a": 1}))
        assert bad.exit_code == 1, bad.output

    def test_unreadable_input_exits_2_not_1(self):
        result = runner.invoke(app, ["gateway", "validate", "-"], input="{not json")
        assert result.exit_code == 2, result.output

    def test_execute_prints_a_result_packet_and_exits_1_when_blocked(self, config: Path):
        result = runner.invoke(
            app,
            ["gateway", "execute", "-", "--config", str(config)],
            input=json.dumps(a_task_packet()),
        )
        assert result.exit_code == 1, result.output
        packet = json.loads(result.stdout)
        assert packet["status"] == "blocked"
        assert packet["unresolved"][0]["reason"] == "capability_unbuilt"
        assert packet["task_id"] == "t-1"

    def test_execute_exits_0_on_a_pass(self):
        examples = Path(__file__).resolve().parents[1] / "examples"
        result = runner.invoke(
            app,
            [
                "gateway",
                "execute",
                str(examples / "task_packet.json"),
                "--config",
                str(examples / "capabilities.yaml"),
            ],
        )
        assert result.exit_code == 0, result.output
        assert json.loads(result.stdout)["artifacts"][0]["value"] == 4

    def test_execute_can_append_proofs_to_a_ledger(self, tmp_path: Path, config: Path):
        ledger = tmp_path / "events.jsonl"
        result = runner.invoke(
            app,
            ["gateway", "execute", "-", "--config", str(config), "--progress-ledger", str(ledger)],
            input=json.dumps(a_task_packet()),
        )
        assert result.exit_code == 1, result.output
        rows = [json.loads(line) for line in ledger.read_text().splitlines() if line]
        assert [row["event_type"] for row in rows] == ["dispatch_refused"]
