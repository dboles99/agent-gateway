from __future__ import annotations

from typing import Any

from .registry import CapabilityRegistry


class HandlerResultMalformed(TypeError):
    """A handler returned something other than a mapping."""


def execute_task(
    task_packet: dict[str, Any],
    registry: CapabilityRegistry,
) -> dict[str, Any]:
    """Run a handler and shape its return value into a result packet.

    This does not enforce anything. Use ``dispatch.run_task_packet`` at a boundary.
    """
    capability_id = task_packet["capability"]
    handler = registry.resolve(capability_id)
    result = handler(task_packet)
    if not isinstance(result, dict):
        raise HandlerResultMalformed(f"handler returned {type(result).__name__}, not a dict")

    return {
        "execution_id": task_packet["execution_id"],
        "task_id": task_packet["task_id"],
        "status": result.get("status", "passed"),
        "artifacts": result.get("artifacts", []),
        "validation": result.get("validation", {}),
        "unresolved": result.get("unresolved", []),
        "progress_proofs": result.get("progress_proofs", []),
    }
