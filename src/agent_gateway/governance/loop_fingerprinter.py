from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any


def normalize_args(arguments: dict[str, Any]) -> dict[str, Any]:
    ignored = {"timestamp", "request_id", "nonce"}
    return {k: arguments[k] for k in sorted(arguments) if k not in ignored}


def fingerprint(tool: str, arguments: dict[str, Any], task_state: str) -> str:
    payload = {
        "tool": tool.lower().strip(),
        "arguments": normalize_args(arguments),
        "task_state": task_state,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass
class LoopTracker:
    limit: int = 2
    counts: dict[str, int] = field(default_factory=dict)

    def record(self, value: str) -> int:
        self.counts[value] = self.counts.get(value, 0) + 1
        return self.counts[value]

    def should_deny(self, value: str) -> bool:
        return self.counts.get(value, 0) > self.limit
