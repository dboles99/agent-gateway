"""The rule engine. A missing context must never look like a clean pass."""
from __future__ import annotations

from pathlib import Path

import pytest

from agent_gateway.governance.rule_eval import (
    UNKNOWN,
    RuleSyntaxError,
    evaluate,
    evaluate_pack,
    explain,
    load_pack,
    parse_condition,
    render_rules,
)
from agent_gateway.governance.rulepack import RulepackCompileError, compile_rulepack

PACK = Path(__file__).resolve().parents[1] / "examples" / "rules" / "release-gate.yaml"


def pack(*rules: tuple[str, int, str, str]) -> dict:
    """A compiled-shape pack, already in priority order."""
    return {
        "id": "TEST-PACK",
        "version": "1",
        "rules": [
            {"id": rid, "p": prio, "if": when, "do": action, "sev": "MEDIUM"}
            for rid, prio, when, action in rules
        ],
    }


def write_pack(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "pack.yaml"
    path.write_text(body, encoding="utf-8")
    return path


class TestFailSafe:
    def test_empty_context_matches_nothing_and_says_so(self):
        decision = evaluate_pack(PACK, {})
        assert decision.matched is False
        assert decision.action is None
        assert len(decision.unknown) == 4

    def test_missing_dotted_ref_is_unknown_not_false(self):
        decision = evaluate(pack(("R1", 10, "change.tests_passed == false", "ACCEPT")), {})
        assert decision.matched is False
        assert decision.unknown == ("R1",)

    def test_bare_name_falls_back_to_its_own_text(self):
        compiled = pack(("R1", 10, "requested_model == primary", "STOP"))
        assert evaluate(compiled, {"requested_model": "primary"}).matched
        assert not evaluate(compiled, {"requested_model": "worker"}).matched

    def test_in_against_a_non_collection_is_unknown(self):
        compiled = pack(("R1", 10, "fallback.reason in permitted_reasons", "ALLOW"))
        decision = evaluate(compiled, {"fallback": {"reason": "permit"}})
        assert decision.matched is False
        assert decision.unknown == ("R1",)

    def test_type_mismatch_is_unknown(self):
        compiled = pack(("R1", 10, "change.review_confidence < 0.9", "HUMAN_REVIEW"))
        decision = evaluate(compiled, {"change": {"review_confidence": "high"}})
        assert decision.matched is False
        assert decision.unknown == ("R1",)

    def test_unknown_has_no_boolean_value(self):
        with pytest.raises(TypeError):
            bool(UNKNOWN)


class TestPriority:
    def test_first_match_wins(self):
        compiled = pack(
            ("HIGH", 10, "flag == true", "REJECT"),
            ("LOW", 20, "flag == true", "ACCEPT"),
        )
        decision = evaluate(compiled, {"flag": True})
        assert (decision.rule_id, decision.action) == ("HIGH", "REJECT")

    def test_an_unknown_rule_does_not_block_a_later_match(self):
        compiled = pack(
            ("FIRST", 10, "absent.thing == true", "REJECT"),
            ("SECOND", 20, "flag == true", "ACCEPT"),
        )
        decision = evaluate(compiled, {"flag": True})
        assert decision.action == "ACCEPT"
        assert decision.unknown == ("FIRST",)

    def test_accept_is_terminal_but_not_blocking(self):
        context = {
            "change": {"touches_credentials": False, "tests_passed": True, "review_confidence": 0.95}
        }
        decision = evaluate_pack(PACK, context)
        assert decision.action == "ACCEPT"
        assert decision.terminal is True
        assert decision.blocking is False

    def test_reject_blocks(self):
        decision = evaluate_pack(PACK, {"change": {"touches_credentials": True}})
        assert decision.action == "REJECT"
        assert decision.blocking is True

    def test_no_match_does_not_block(self):
        assert evaluate_pack(PACK, {}).blocking is False

    def test_human_review_beats_accept(self):
        context = {
            "change": {"touches_credentials": False, "tests_passed": True, "review_confidence": 0.8}
        }
        decision = evaluate_pack(PACK, context)
        assert decision.action == "HUMAN_REVIEW"
        assert decision.terminal is True


class TestGrammar:
    def test_kleene_logic(self):
        compiled = pack(("R", 10, "a == false and missing.x == true", "GO"))
        assert evaluate(compiled, {"a": True}).unknown == ()
        compiled = pack(("R", 10, "a == true or missing.x == true", "GO"))
        assert evaluate(compiled, {"a": True}).matched

    def test_in_and_not_in_with_a_list_literal(self):
        compiled = pack(("R", 10, "status not in [ALLOW, ALLOW_WITH_LIMITS]", "DENY"))
        assert evaluate(compiled, {"status": "BLOCK"}).matched
        assert not evaluate(compiled, {"status": "ALLOW"}).matched

    def test_in_with_a_referenced_collection(self):
        compiled = pack(("R", 10, "fallback.reason in fallback.permitted", "ALLOW"))
        context = {"fallback": {"reason": "rate_limit", "permitted": ["rate_limit"]}}
        assert evaluate(compiled, context).matched

    def test_numeric_comparison(self):
        compiled = pack(("R", 10, "n > 5", "GO"))
        assert evaluate(compiled, {"n": 6}).matched
        assert not evaluate(compiled, {"n": 5}).matched

    def test_quoted_strings(self):
        compiled = pack(("R", 10, 'run.s == "queued" or run.s == "in_progress"', "DENY"))
        assert evaluate(compiled, {"run": {"s": "in_progress"}}).matched

    def test_parentheses(self):
        compiled = pack(("R", 10, "(a == true or b == true) and c == true", "GO"))
        assert evaluate(compiled, {"a": True, "b": False, "c": True}).matched
        assert not evaluate(compiled, {"a": True, "b": False, "c": False}).matched

    def test_a_syntax_error_is_reported_not_raised_mid_decision(self):
        decision = evaluate(pack(("BAD", 10, "a ===== b", "GO")), {})
        assert decision.matched is False
        assert decision.unknown == ("BAD",)
        assert decision.trace[0]["result"] == "SYNTAX_ERROR"

    def test_an_empty_expression_is_rejected(self):
        with pytest.raises(RuleSyntaxError):
            parse_condition("   ")


class TestReaderOutput:
    def test_explain_names_the_rule_and_the_values_it_saw(self):
        decision = evaluate_pack(PACK, {"change": {"touches_credentials": True}})
        text = explain(decision, include_skipped=False)
        assert "DECISION REJECT" in text
        assert "RG-001" in text
        assert "change.touches_credentials" in text

    def test_explain_lists_what_the_context_failed_to_supply(self):
        text = explain(evaluate_pack(PACK, {}))
        assert "UNKNOWN" in text
        assert "change.tests_passed" in text

    def test_explain_is_deterministic(self):
        context = {"change": {"touches_credentials": True}}
        assert explain(evaluate_pack(PACK, context)) == explain(evaluate_pack(PACK, context))

    def test_render_rules_is_complete(self):
        text = render_rules(load_pack(PACK))
        assert "RELEASE-GATE" in text
        for rule in load_pack(PACK)["rules"]:
            assert rule["id"] in text


class TestLoadingAndCaching:
    def test_a_pack_is_cached_while_unchanged(self):
        assert load_pack(PACK) is load_pack(PACK)

    def test_repeated_evaluation_does_not_reparse(self):
        compiled = load_pack(PACK)
        context = {"change": {"touches_credentials": True}}
        evaluate(compiled, context)
        before = parse_condition.cache_info()
        for _ in range(500):
            evaluate(compiled, context)
        after = parse_condition.cache_info()
        assert after.misses == before.misses
        assert after.hits > before.hits

    def test_an_edited_pack_is_reloaded(self, tmp_path: Path):
        body = 'rulepack: {id: P, version: "1", rules: [{id: A, priority: 1, when: "a == 1", action: STOP}]}'
        path = write_pack(tmp_path, body)
        first = load_pack(path)
        path.write_text(body.replace("id: A", "id: BB"), encoding="utf-8")
        assert load_pack(path)["rules"][0]["id"] == "BB"
        assert first["rules"][0]["id"] == "A"

    def test_a_malformed_condition_fails_at_load(self, tmp_path: Path):
        path = write_pack(
            tmp_path,
            'rulepack: {id: P, version: "1", rules: [{id: A, priority: 1, when: "a ==== b", action: STOP}]}',
        )
        with pytest.raises(RuleSyntaxError):
            load_pack(path)


class TestCompiler:
    def test_rules_are_ordered_by_priority(self, tmp_path: Path):
        path = write_pack(
            tmp_path,
            'rulepack:\n  id: T\n  version: "1"\n  rules:\n'
            "    - {id: B, priority: 20, when: x, action: STOP}\n"
            "    - {id: A, priority: 10, when: y, action: ACCEPT}\n",
        )
        assert [rule["id"] for rule in compile_rulepack(path)["rules"]] == ["A", "B"]

    def test_the_compiled_shape(self):
        compiled = load_pack(PACK)
        assert set(compiled) == {"id", "version", "rules", "source_hash", "compiled_hash"}
        assert set(compiled["rules"][0]) == {"id", "p", "if", "do", "sev"}

    def test_the_compiled_hash_ignores_comments_but_the_source_hash_does_not(self, tmp_path: Path):
        body = 'rulepack: {id: P, version: "1", rules: [{id: A, priority: 1, when: x, action: STOP}]}'
        plain = compile_rulepack(write_pack(tmp_path, body))
        commented = compile_rulepack(write_pack(tmp_path, "# note\n" + body))
        assert plain["compiled_hash"] == commented["compiled_hash"]
        assert plain["source_hash"] != commented["source_hash"]

    @pytest.mark.parametrize(
        "body",
        [
            "not_a_pack: true",
            'rulepack: {version: "1"}',
            "rulepack: {id: P}",
            'rulepack: {id: P, version: "1", rules: [{priority: 1, when: x, action: STOP}]}',
            'rulepack: {id: P, version: "1", rules: [{id: A, when: x, action: STOP}]}',
            'rulepack: {id: P, version: "1", rules: ['
            "{id: A, priority: 1, when: x, action: STOP},"
            "{id: A, priority: 2, when: x, action: STOP}]}",
        ],
    )
    def test_a_bad_pack_is_refused(self, tmp_path: Path, body: str):
        with pytest.raises(RulepackCompileError):
            compile_rulepack(write_pack(tmp_path, body))
