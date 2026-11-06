"""Admission, budgets, routing, the watchdog, loops, cache, leases and files."""
from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from agent_gateway.governance import paths
from agent_gateway.governance.admission import AdmissionRequest, admit
from agent_gateway.governance.budget import evaluate_budget
from agent_gateway.governance.cache import (
    CacheEntry,
    CacheState,
    dependencies_match,
    make_cache_key,
)
from agent_gateway.governance.checkpoint import read_checkpoint, write_checkpoint
from agent_gateway.governance.context import (
    ContextCapsule,
    ContextLease,
    assert_path_allowed,
)
from agent_gateway.governance.ledger import EventLedger
from agent_gateway.governance.loop_fingerprinter import LoopTracker, fingerprint
from agent_gateway.governance.models import AdmissionStatus, Budget, ModelRole, Usage
from agent_gateway.governance.router import choose_orchestrator
from agent_gateway.governance.watchdog import (
    WatchdogPolicy,
    WatchdogState,
    evaluate_watchdog,
)


def request(**overrides) -> AdmissionRequest:
    values = {
        "requested_model": ModelRole.PRIMARY,
        "estimated_fresh_tokens": 1000,
        "estimated_total_tokens": 2000,
        "maximum_fresh_tokens": 5000,
        "maximum_total_tokens": 10000,
        "schema_valid": True,
        "has_acceptance_criteria": True,
        "has_stop_condition": True,
        "requires_deep_reasoning": True,
    }
    values.update(overrides)
    return AdmissionRequest(**values)


class TestAdmission:
    def test_primary_downroutes_for_bounded_work(self):
        decision = admit(request(requires_deep_reasoning=False))
        assert decision.status == AdmissionStatus.DOWNROUTE
        assert decision.model == ModelRole.WORKER

    def test_primary_is_kept_for_work_that_needs_it(self):
        decision = admit(request())
        assert decision.status == AdmissionStatus.ALLOW
        assert decision.model == ModelRole.PRIMARY

    def test_a_worker_request_is_not_downrouted(self):
        decision = admit(request(requested_model=ModelRole.WORKER, requires_deep_reasoning=False))
        assert decision.status == AdmissionStatus.ALLOW

    @pytest.mark.parametrize(
        "field",
        ["estimated_fresh_tokens", "estimated_total_tokens"],
    )
    def test_oversized_requests_are_rejected(self, field):
        assert admit(request(**{field: 1_000_000})).status == AdmissionStatus.REJECT

    @pytest.mark.parametrize(
        "field", ["schema_valid", "has_acceptance_criteria", "has_stop_condition"]
    )
    def test_a_malformed_request_goes_back_for_repair(self, field):
        assert admit(request(**{field: False})).status == AdmissionStatus.REQUIRE_REPAIR

    def test_repair_comes_before_a_cache_hit(self):
        decision = admit(request(schema_valid=False, cache_hit=True))
        assert decision.status == AdmissionStatus.REQUIRE_REPAIR

    def test_a_cache_hit_needs_no_model(self):
        decision = admit(request(cache_hit=True))
        assert decision.status == AdmissionStatus.CACHE_HIT
        assert decision.model is None


class TestBudget:
    BUDGET = Budget(100, 50, 4, 10, 60)

    def test_normal(self):
        decision = evaluate_budget(self.BUDGET, Usage(fresh_input_tokens=20))
        assert decision.allowed is True
        assert decision.level == "NORMAL"

    def test_soft_and_hard_warnings_stay_allowed(self):
        soft = evaluate_budget(self.BUDGET, Usage(fresh_input_tokens=70))
        hard = evaluate_budget(self.BUDGET, Usage(fresh_input_tokens=88))
        assert (soft.level, soft.allowed) == ("SOFT", True)
        assert (hard.level, hard.allowed) == ("HARD", True)

    def test_any_exhausted_dimension_stops_work(self):
        assert evaluate_budget(self.BUDGET, Usage(fresh_input_tokens=100)).allowed is False
        assert evaluate_budget(self.BUDGET, Usage(turns=4)).allowed is False
        assert evaluate_budget(self.BUDGET, Usage(elapsed_seconds=60.0)).level == "EMERGENCY"


class TestRouter:
    def test_fallback_requires_a_checkpoint(self):
        with pytest.raises(RuntimeError):
            choose_orchestrator(
                active_model=ModelRole.PRIMARY,
                fallback_activated=False,
                fallback_reason="primary_budget_exhausted",
                checkpoint_exists=False,
            )

    def test_fallback_is_one_way(self):
        decision = choose_orchestrator(
            active_model=ModelRole.PRIMARY,
            fallback_activated=True,
            fallback_reason=None,
            checkpoint_exists=True,
        )
        assert decision.model == ModelRole.FALLBACK

    def test_an_ineligible_failure_keeps_the_current_orchestrator(self):
        decision = choose_orchestrator(
            active_model=ModelRole.PRIMARY,
            fallback_activated=False,
            fallback_reason="unit_test_failed",
            checkpoint_exists=True,
        )
        assert decision.model == ModelRole.PRIMARY
        assert decision.fallback_activated is False

    def test_an_eligible_failure_activates_fallback(self):
        decision = choose_orchestrator(
            active_model=ModelRole.PRIMARY,
            fallback_activated=False,
            fallback_reason="primary_rate_limit",
            checkpoint_exists=True,
        )
        assert decision.model == ModelRole.FALLBACK
        assert decision.fallback_activated is True


class TestWatchdog:
    NOW = datetime(2026, 1, 1, tzinfo=UTC)

    def ago(self, seconds: int) -> datetime:
        return self.NOW - timedelta(seconds=seconds)

    def test_continue_warn_stop_by_time(self):
        policy = WatchdogPolicy()
        assert evaluate_watchdog(WatchdogState(self.ago(10)), policy, self.NOW) == "CONTINUE"
        assert evaluate_watchdog(WatchdogState(self.ago(60)), policy, self.NOW) == "WARN"
        stopped = evaluate_watchdog(WatchdogState(self.ago(130)), policy, self.NOW)
        assert stopped == "CHECKPOINT_AND_STOP"

    def test_activity_without_proof_stops_a_fresh_run(self):
        state = WatchdogState(self.ago(1), actions_since_progress=6)
        assert evaluate_watchdog(state, WatchdogPolicy(), self.NOW) == "CHECKPOINT_AND_STOP"
        state = WatchdogState(self.ago(1), tokens_since_progress=5000)
        assert evaluate_watchdog(state, WatchdogPolicy(), self.NOW) == "CHECKPOINT_AND_STOP"


class TestLoopFingerprint:
    def test_volatile_fields_do_not_change_the_fingerprint(self):
        a = fingerprint("grep", {"query": "x", "timestamp": 1}, "discovering")
        b = fingerprint("grep", {"timestamp": 2, "query": "x"}, "discovering")
        assert a == b

    def test_a_different_state_is_a_different_call(self):
        assert fingerprint("grep", {"q": "x"}, "one") != fingerprint("grep", {"q": "x"}, "two")

    def test_tool_name_is_normalised(self):
        assert fingerprint(" GREP ", {}, "s") == fingerprint("grep", {}, "s")

    def test_tracker_denies_after_the_limit(self):
        tracker = LoopTracker(limit=2)
        tracker.record("abc")
        tracker.record("abc")
        assert tracker.should_deny("abc") is False
        tracker.record("abc")
        assert tracker.should_deny("abc") is True


class TestCache:
    def test_key_is_stable_across_dependency_order(self):
        assert make_cache_key("n", {"a": 1, "b": 2}) == make_cache_key("n", {"b": 2, "a": 1})

    def test_only_a_validated_entry_with_equal_dependencies_matches(self):
        deps = {"source_hash": "abc"}
        good = CacheEntry("k", CacheState.VALIDATED, deps, "ref")
        stale = CacheEntry("k", CacheState.STALE, deps, "ref")
        assert dependencies_match(good, deps) is True
        assert dependencies_match(stale, deps) is False
        assert dependencies_match(good, {"source_hash": "changed"}) is False


class TestContext:
    def test_a_path_outside_the_lease_is_refused(self):
        lease = ContextLease("lease-1", "read one file", ("src/a.py",))
        assert_path_allowed("src/a.py", lease)
        with pytest.raises(PermissionError):
            assert_path_allowed("src/b.py", lease)

    def test_capsule_hash_tracks_content(self):
        base = {
            "capsule_id": "c",
            "target_model": "worker",
            "purpose": {"action": "x"},
            "maximum_fresh_tokens": 1,
            "maximum_total_tokens": 2,
            "dependency_hash": "d",
        }
        one = ContextCapsule(content={"a": 1}, **base)
        same = ContextCapsule(content={"a": 1}, **base)
        other = ContextCapsule(content={"a": 2}, **base)
        assert one.content_hash() == same.content_hash()
        assert one.content_hash() != other.content_hash()


class TestFiles:
    def test_checkpoint_round_trip_leaves_no_temporary_file(self, tmp_path: Path):
        path = tmp_path / "deep" / "checkpoint.json"
        write_checkpoint(path, {"b": 1, "a": 2})
        assert read_checkpoint(path) == {"a": 2, "b": 1}
        assert [p.name for p in path.parent.iterdir()] == ["checkpoint.json"]

    def test_ledger_appends_and_adds_an_id_and_timestamp(self, tmp_path: Path):
        ledger = EventLedger(tmp_path / "events.jsonl")
        ledger.append({"kind": "one"})
        ledger.append({"kind": "two", "event_id": "mine"})
        rows = [json.loads(line) for line in ledger.path.read_text().splitlines()]
        assert [r["kind"] for r in rows] == ["one", "two"]
        assert rows[0]["event_id"].startswith("evt-")
        assert rows[1]["event_id"] == "mine"
        assert "timestamp" in rows[0]


class TestRunPaths:
    def test_override_beats_the_environment(self, monkeypatch, tmp_path: Path):
        monkeypatch.setenv("AGENT_GATEWAY_RUN_DIR", str(tmp_path / "env"))
        assert paths.run_dir(tmp_path / "explicit") == tmp_path / "explicit"

    def test_environment_is_used(self, monkeypatch, tmp_path: Path):
        monkeypatch.setenv("AGENT_GATEWAY_RUN_DIR", str(tmp_path / "env"))
        assert paths.run_dir() == tmp_path / "env"
        assert paths.events_path() == tmp_path / "env" / "events.jsonl"
        assert paths.checkpoint_path() == tmp_path / "env" / "checkpoint.json"

    def test_default_is_under_home_not_the_working_directory(self, monkeypatch, tmp_path: Path):
        monkeypatch.delenv("AGENT_GATEWAY_RUN_DIR")
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
        assert paths.run_dir() == tmp_path / ".agent_gateway" / "runs" / "current"

    def test_two_working_directories_resolve_to_one_ledger(self, monkeypatch, tmp_path: Path):
        monkeypatch.setenv("AGENT_GATEWAY_RUN_DIR", str(tmp_path / "run"))
        first, second = tmp_path / "one", tmp_path / "two"
        first.mkdir()
        second.mkdir()
        monkeypatch.chdir(first)
        from_first = paths.resolve_run_path("events.jsonl")
        monkeypatch.chdir(second)
        assert paths.resolve_run_path("events.jsonl") == from_first

    def test_an_absolute_path_is_honoured(self, tmp_path: Path):
        absolute = (tmp_path / "elsewhere.jsonl").resolve()
        assert paths.resolve_run_path(absolute) == absolute
