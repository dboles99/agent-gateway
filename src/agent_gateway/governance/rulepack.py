"""Compile a YAML rule pack into a compact, hashed, priority-ordered form."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


class RulepackCompileError(ValueError):
    pass


TERMINAL_ACTIONS = {
    "REJECT_DISPATCH",
    "STOP",
    "CHECKPOINT_AND_STOP",
    "HANDOFF_TO_FALLBACK",
    "REJECT",
    "ACCEPT",
    "HUMAN_REVIEW",
}


def load_rulepack(path: str | Path) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "rulepack" not in data:
        raise RulepackCompileError("Missing top-level 'rulepack' object")
    return data["rulepack"]


def compile_rulepack(path: str | Path) -> dict[str, Any]:
    source = load_rulepack(path)
    for key in ("id", "version"):
        if key not in source:
            raise RulepackCompileError(f"Rule pack requires '{key}'")

    rules = source.get("rules", [])
    seen_ids: set[str] = set()
    for rule in rules:
        rule_id = rule.get("id")
        if not rule_id:
            raise RulepackCompileError("Every rule requires an id")
        if rule_id in seen_ids:
            raise RulepackCompileError(f"Duplicate rule id: {rule_id}")
        seen_ids.add(rule_id)
        if "priority" not in rule or "when" not in rule or "action" not in rule:
            raise RulepackCompileError(f"Incomplete rule: {rule_id}")

    ordered = sorted(rules, key=lambda item: (item["priority"], item["id"]))
    compact_rules = [
        {
            "id": rule["id"],
            "p": rule["priority"],
            "if": rule["when"],
            "do": rule["action"],
            "sev": rule.get("severity", "MEDIUM"),
        }
        for rule in ordered
    ]

    canonical = json.dumps(compact_rules, sort_keys=True, separators=(",", ":"))
    return {
        "id": source["id"],
        "version": source["version"],
        "rules": compact_rules,
        "source_hash": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        "compiled_hash": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }
