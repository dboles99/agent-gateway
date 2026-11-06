"""Evaluate compiled rule packs against a context.

**Missing data never looks like a pass.** A dotted reference the context does not
supply resolves to UNKNOWN, not to False and not to its own name. A rule whose
condition is UNKNOWN does not fire, and the decision records it, so a wiring
fault cannot pass as a clean result.

**Cheap on repeat.** Conditions are parsed once per unique expression and packs
are cached on their file identity.

**Readable by a model.** ``explain()`` renders a decision as compact, deterministic
text: what fired, what each operand resolved to, and what was skipped as unknown.

There is no ``eval()``. The grammar is closed and parsed by hand.
"""
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from .rulepack import compile_rulepack

__all__ = [
    "BLOCKING_ACTIONS",
    "UNKNOWN",
    "Decision",
    "RuleSyntaxError",
    "evaluate",
    "evaluate_pack",
    "explain",
    "load_pack",
    "parse_condition",
    "render_rules",
]


class RuleSyntaxError(ValueError):
    """A `when` expression could not be parsed."""


class _Unknown:
    """Third truth value: the context did not supply what the rule asked for."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "UNKNOWN"

    def __bool__(self) -> bool:  # pragma: no cover - guards accidental truthiness
        raise TypeError("UNKNOWN has no boolean value; compare it explicitly")


UNKNOWN = _Unknown()


# --------------------------------------------------------------------------
# Tokenizer
# --------------------------------------------------------------------------

_TOKEN_RE = re.compile(
    r"""
    \s*(?:
        (?P<str>"[^"]*"|'[^']*')
      | (?P<op>==|!=|<=|>=|<|>)
      | (?P<punct>[\[\](),])
      | (?P<num>-?\d+(?:\.\d+)?)
      | (?P<name>[A-Za-z_][\w.\-]*)
    )
    """,
    re.VERBOSE,
)

_KEYWORDS = {"and", "or", "not", "in", "true", "false", "null", "none"}


def _tokenize(text: str) -> list[tuple[str, str]]:
    tokens: list[tuple[str, str]] = []
    pos = 0
    while pos < len(text):
        if text[pos].isspace():
            pos += 1
            continue
        match = _TOKEN_RE.match(text, pos)
        if not match or match.end() == pos:
            raise RuleSyntaxError(f"Cannot tokenize at {pos}: {text[pos:pos + 20]!r}")
        pos = match.end()
        kind = match.lastgroup or ""
        value = (match.group(kind) or "").strip()
        if kind == "name" and value.lower() in _KEYWORDS:
            tokens.append(("kw", value.lower()))
        else:
            tokens.append((kind, value))
    return tokens


# --------------------------------------------------------------------------
# AST
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _Literal:
    value: Any

    def resolve(self, ctx: Mapping[str, Any]) -> Any:
        return self.value

    def render(self) -> str:
        return repr(self.value)


@dataclass(frozen=True)
class _ListLiteral:
    items: tuple[Any, ...]

    def resolve(self, ctx: Mapping[str, Any]) -> Any:
        return [
            item.resolve(ctx) if hasattr(item, "resolve") else item for item in self.items
        ]

    def render(self) -> str:
        return "[" + ", ".join(i.render() for i in self.items) + "]"


@dataclass(frozen=True)
class _Ref:
    """A dotted or bare name looked up in the context.

    A *dotted* name is unambiguously a context lookup: `change.tests_passed` can
    only mean "ask the context". Missing means UNKNOWN.

    A *bare* name is ambiguous: a pack may write `requested_model == primary` or
    `status not in [ALLOW, ALLOW_WITH_LIMITS]`, where `primary` and `ALLOW` are
    enum values, not variables. So a bare name falls back to its own text when
    the context has no such key.
    """

    path: str

    @property
    def dotted(self) -> bool:
        return "." in self.path

    def resolve(self, ctx: Mapping[str, Any]) -> Any:
        cursor: Any = ctx
        for part in self.path.split("."):
            if isinstance(cursor, Mapping) and part in cursor:
                cursor = cursor[part]
                continue
            if not isinstance(cursor, Mapping) and hasattr(cursor, part):
                cursor = getattr(cursor, part)
                continue
            return UNKNOWN if self.dotted else self.path
        return cursor

    def render(self) -> str:
        return self.path


@dataclass(frozen=True)
class _Compare:
    op: str
    left: Any
    right: Any

    def evaluate(self, ctx: Mapping[str, Any]) -> Any:
        left = self.left.resolve(ctx)
        right = self.right.resolve(ctx)
        if left is UNKNOWN or right is UNKNOWN:
            return UNKNOWN
        try:
            if self.op == "==":
                return left == right
            if self.op == "!=":
                return left != right
            if self.op == "<":
                return left < right
            if self.op == ">":
                return left > right
            if self.op == "<=":
                return left <= right
            if self.op == ">=":
                return left >= right
        except TypeError:
            # A real type mismatch in the context, such as "high" < 0.9, is not
            # a false condition. Say unknown rather than invent an answer.
            return UNKNOWN
        raise RuleSyntaxError(f"Unknown operator {self.op}")

    def render(self) -> str:
        return f"{self.left.render()} {self.op} {self.right.render()}"


@dataclass(frozen=True)
class _In:
    needle: Any
    haystack: Any
    negated: bool

    def evaluate(self, ctx: Mapping[str, Any]) -> Any:
        needle = self.needle.resolve(ctx)
        haystack = self.haystack.resolve(ctx)
        if needle is UNKNOWN or haystack is UNKNOWN:
            return UNKNOWN
        # `in` always means membership of a collection. A bare name that fell
        # back to its own text is a missing context key, not a string to search;
        # substring matching there would answer confidently and wrongly.
        if isinstance(haystack, str) or not isinstance(haystack, (Sequence, set, frozenset)):
            return UNKNOWN
        present = needle in haystack
        return not present if self.negated else present

    def render(self) -> str:
        kw = "not in" if self.negated else "in"
        return f"{self.needle.render()} {kw} {self.haystack.render()}"


@dataclass(frozen=True)
class _Truthy:
    """A bare operand used as a condition."""

    operand: Any

    def evaluate(self, ctx: Mapping[str, Any]) -> Any:
        value = self.operand.resolve(ctx)
        if value is UNKNOWN:
            return UNKNOWN
        return bool(value)

    def render(self) -> str:
        return self.operand.render()


@dataclass(frozen=True)
class _BoolOp:
    op: str
    parts: tuple[Any, ...]

    def evaluate(self, ctx: Mapping[str, Any]) -> Any:
        results = [part.evaluate(ctx) for part in self.parts]
        # Kleene logic: a definite answer beats an unknown one.
        if self.op == "and":
            if any(r is False for r in results):
                return False
            return UNKNOWN if any(r is UNKNOWN for r in results) else True
        if any(r is True for r in results):
            return True
        return UNKNOWN if any(r is UNKNOWN for r in results) else False

    def render(self) -> str:
        return f" {self.op} ".join(p.render() for p in self.parts)


# --------------------------------------------------------------------------
# Parser: recursive descent over the closed grammar
# --------------------------------------------------------------------------


class _Parser:
    def __init__(self, tokens: list[tuple[str, str]], source: str) -> None:
        self.tokens = tokens
        self.pos = 0
        self.source = source

    def peek(self) -> tuple[str, str] | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def take(self) -> tuple[str, str]:
        if self.pos >= len(self.tokens):
            raise RuleSyntaxError(f"Unexpected end of expression: {self.source!r}")
        token = self.tokens[self.pos]
        self.pos += 1
        return token

    def expect(self, kind: str, value: str | None = None) -> tuple[str, str]:
        token = self.take()
        if token[0] != kind or (value is not None and token[1] != value):
            raise RuleSyntaxError(
                f"Expected {value or kind}, got {token[1]!r} in {self.source!r}"
            )
        return token

    def parse(self) -> Any:
        node = self.parse_or()
        if self.peek() is not None:
            raise RuleSyntaxError(
                f"Trailing tokens after expression: {self.source!r}"
            )
        return node

    def parse_or(self) -> Any:
        parts = [self.parse_and()]
        while (tok := self.peek()) and tok == ("kw", "or"):
            self.take()
            parts.append(self.parse_and())
        return parts[0] if len(parts) == 1 else _BoolOp("or", tuple(parts))

    def parse_and(self) -> Any:
        parts = [self.parse_condition()]
        while (tok := self.peek()) and tok == ("kw", "and"):
            self.take()
            parts.append(self.parse_condition())
        return parts[0] if len(parts) == 1 else _BoolOp("and", tuple(parts))

    def parse_condition(self) -> Any:
        if (tok := self.peek()) and tok == ("punct", "("):
            self.take()
            node = self.parse_or()
            self.expect("punct", ")")
            return node

        left = self.parse_operand()
        tok = self.peek()
        if tok is None:
            return _Truthy(left)

        if tok[0] == "op":
            self.take()
            return _Compare(tok[1], left, self.parse_operand())

        if tok == ("kw", "not"):
            self.take()
            self.expect("kw", "in")
            return _In(left, self.parse_operand(), negated=True)

        if tok == ("kw", "in"):
            self.take()
            return _In(left, self.parse_operand(), negated=False)

        return _Truthy(left)

    def parse_operand(self) -> Any:
        kind, value = self.take()
        if kind == "str":
            return _Literal(value[1:-1])
        if kind == "num":
            return _Literal(float(value) if "." in value else int(value))
        if kind == "kw":
            if value == "true":
                return _Literal(True)
            if value == "false":
                return _Literal(False)
            if value in ("null", "none"):
                return _Literal(None)
            raise RuleSyntaxError(f"Keyword {value!r} is not an operand")
        if kind == "punct" and value == "[":
            items: list[Any] = []
            if (tok := self.peek()) and tok == ("punct", "]"):
                self.take()
                return _ListLiteral(())
            while True:
                items.append(self.parse_operand())
                nxt = self.take()
                if nxt == ("punct", "]"):
                    break
                if nxt != ("punct", ","):
                    raise RuleSyntaxError(
                        f"Expected , or ] in list, got {nxt[1]!r}"
                    )
            return _ListLiteral(tuple(items))
        if kind == "name":
            return _Ref(value)
        raise RuleSyntaxError(f"Unexpected token {value!r} in {self.source!r}")


@lru_cache(maxsize=1024)
def parse_condition(expression: str) -> Any:
    """Parse a `when` expression into an evaluable tree.

    Cached: packs reuse the same expressions, and evaluation in a loop should
    not re-parse.
    """
    text = (expression or "").strip()
    if not text:
        raise RuleSyntaxError("Empty when expression")
    return _Parser(_tokenize(text), text).parse()


# --------------------------------------------------------------------------
# Decisions
# --------------------------------------------------------------------------


# Actions that mean "the thing you asked about must not proceed". Distinct from
# rulepack.TERMINAL_ACTIONS, which only means "the decision ends here": ACCEPT is
# terminal but is not a refusal.
BLOCKING_ACTIONS = frozenset(
    {
        "REJECT",
        "REJECT_DISPATCH",
        "REJECT_HANDOFF",
        "REJECT_ACCEPTANCE",
        "DENY_ACTION",
        "STOP",
        "CHECKPOINT_AND_STOP",
        "HUMAN_REVIEW",
        "REQUIRE_PRIMARY_REVIEW",
    }
)


@dataclass(frozen=True)
class Decision:
    """The outcome of evaluating a pack against one context."""

    action: str | None
    severity: str | None
    rule_id: str | None
    matched: bool
    pack_id: str
    fired: tuple[str, ...] = ()
    unknown: tuple[str, ...] = ()
    trace: tuple[dict[str, Any], ...] = field(default=())

    @property
    def terminal(self) -> bool:
        """The decision ends here. Says nothing about whether it passed."""
        from .rulepack import TERMINAL_ACTIONS

        return self.action in TERMINAL_ACTIONS

    @property
    def blocking(self) -> bool:
        """The requested action must not proceed. This is the gate signal."""
        return self.action in BLOCKING_ACTIONS


def _operand_values(node: Any, ctx: Mapping[str, Any], into: dict[str, Any]) -> None:
    """Collect what each referenced name actually resolved to, for the trace."""
    if isinstance(node, _Ref):
        value = node.resolve(ctx)
        into[node.path] = "UNKNOWN" if value is UNKNOWN else value
        return
    for attr in ("left", "right", "needle", "haystack", "operand"):
        child = getattr(node, attr, None)
        if child is not None:
            _operand_values(child, ctx, into)
    for child in getattr(node, "parts", ()) or ():
        _operand_values(child, ctx, into)
    for child in getattr(node, "items", ()) or ():
        _operand_values(child, ctx, into)


def evaluate(compiled: Mapping[str, Any], context: Mapping[str, Any]) -> Decision:
    """Return the first matching rule, in the pack's compiled priority order.

    `compile_rulepack` has already sorted by (priority, id), so the first match
    is the highest priority. Rules whose condition is UNKNOWN are recorded and
    skipped: they never fire.
    """
    trace: list[dict[str, Any]] = []
    unknown: list[str] = []
    fired: list[str] = []

    for rule in compiled.get("rules", []):
        rule_id = str(rule.get("id"))
        expression = rule.get("if", "")
        entry: dict[str, Any] = {"rule": rule_id, "when": expression}
        try:
            node = parse_condition(expression)
        except RuleSyntaxError as exc:
            entry["result"] = "SYNTAX_ERROR"
            entry["error"] = str(exc)
            trace.append(entry)
            unknown.append(rule_id)
            continue

        values: dict[str, Any] = {}
        _operand_values(node, context, values)
        entry["resolved"] = values
        result = node.evaluate(context)

        if result is UNKNOWN:
            entry["result"] = "UNKNOWN"
            trace.append(entry)
            unknown.append(rule_id)
            continue

        entry["result"] = bool(result)
        trace.append(entry)
        if result:
            fired.append(rule_id)
            return Decision(
                action=rule.get("do"),
                severity=rule.get("sev"),
                rule_id=rule_id,
                matched=True,
                pack_id=str(compiled.get("id", "")),
                fired=tuple(fired),
                unknown=tuple(unknown),
                trace=tuple(trace),
            )

    return Decision(
        action=None,
        severity=None,
        rule_id=None,
        matched=False,
        pack_id=str(compiled.get("id", "")),
        fired=(),
        unknown=tuple(unknown),
        trace=tuple(trace),
    )


# --------------------------------------------------------------------------
# Loading, cached on file identity
# --------------------------------------------------------------------------

_PACK_CACHE: dict[tuple[str, int, int], dict[str, Any]] = {}


def load_pack(path: str | Path) -> dict[str, Any]:
    """Compile a pack, reusing the result while the file is unchanged.

    Keyed on (path, mtime_ns, size) so an edited pack is picked up without a
    restart, and an unchanged one costs a stat() rather than a YAML parse.
    """
    resolved = Path(path).resolve()
    stat = resolved.stat()
    key = (str(resolved), stat.st_mtime_ns, stat.st_size)
    cached = _PACK_CACHE.get(key)
    if cached is None:
        cached = compile_rulepack(resolved)
        # Parse every condition now, so a malformed pack fails at load. A rule
        # that cannot parse can never fire, so it must not reach a live gate.
        for rule in cached.get("rules", []):
            parse_condition(rule.get("if", ""))
        _PACK_CACHE[key] = cached
    return cached


def evaluate_pack(path: str | Path, context: Mapping[str, Any]) -> Decision:
    """Load (cached) and evaluate a pack in one call."""
    return evaluate(load_pack(path), context)


# --------------------------------------------------------------------------
# Rendering for a model reader
# --------------------------------------------------------------------------


def explain(decision: Decision, *, include_skipped: bool = True) -> str:
    """Render a decision as compact deterministic text.

    Written to be read by an agent: the verdict first, then the rule that
    produced it and the values it actually saw, then anything that could not be
    judged. No colour, no tables, stable line order.
    """
    lines: list[str] = []
    if decision.matched:
        lines.append(
            f"DECISION {decision.action} "
            f"[{decision.severity}] by {decision.rule_id} of {decision.pack_id}"
        )
    else:
        lines.append(f"DECISION none: no rule matched in {decision.pack_id}")

    for entry in decision.trace:
        if entry["result"] is True:
            lines.append(f"  MATCHED {entry['rule']}: {entry['when']}")
            for name, value in sorted(entry.get("resolved", {}).items()):
                lines.append(f"    {name} = {value!r}")

    if include_skipped and decision.unknown:
        lines.append(f"  UNKNOWN ({len(decision.unknown)}): context did not supply:")
        for entry in decision.trace:
            if entry["result"] not in ("UNKNOWN", "SYNTAX_ERROR"):
                continue
            missing = sorted(
                name
                for name, value in entry.get("resolved", {}).items()
                if value == "UNKNOWN"
            )
            detail = ", ".join(missing) or entry.get("error", "")
            lines.append(f"    {entry['rule']}: {detail}")

    return "\n".join(lines)


def render_rules(compiled: Mapping[str, Any]) -> str:
    """Render a whole pack as compact text, for putting in a prompt."""
    lines = [f"RULEPACK {compiled.get('id')} v{compiled.get('version')}"]
    for rule in compiled.get("rules", []):
        lines.append(
            f"  {rule['id']} p{rule['p']} [{rule['sev']}] "
            f"IF {rule['if']} THEN {rule['do']}"
        )
    return "\n".join(lines)
