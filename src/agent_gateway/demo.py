"""A small capability handler, used by the examples and the quick start."""
from __future__ import annotations

from typing import Any


def word_count(packet: dict[str, Any]) -> dict[str, Any]:
    """Count the words in ``inputs.text`` and report one named artifact."""
    text = packet["inputs"].get("text")
    if not isinstance(text, str):
        return {
            "status": "blocked",
            "unresolved": [{"reason": "input_missing", "detail": "inputs.text is required"}],
        }
    return {
        "status": "passed",
        "artifacts": [{"name": "word_count", "value": len(text.split())}],
        "validation": {"counted": True},
    }
