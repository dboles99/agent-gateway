from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4


def progress_event(
    *,
    run_id: str,
    task_id: str,
    event_type: str,
    status: str,
    weight: int,
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "event_id": f"evt-{uuid4().hex}",
        "timestamp": datetime.now(UTC).isoformat(),
        "run_id": run_id,
        "task_id": task_id,
        "event_type": event_type,
        "status": status,
        "progress_weight": weight,
        "evidence": evidence or {},
    }
