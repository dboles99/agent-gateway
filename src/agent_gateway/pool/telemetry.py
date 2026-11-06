"""Metadata-only telemetry. Prompts, context and model output are never recorded."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .._config import POOL_TELEMETRY
from .contracts import TaskContract
from .result_normalizer import ResultEnvelope


def _default_telemetry_path() -> Path | None:
    raw = os.environ.get(POOL_TELEMETRY, "").strip()
    return Path(raw) if raw else None


@dataclass
class TelemetryRecorder:
    path: Path | None = None

    def record(self, task: TaskContract, result: ResultEnvelope) -> bool:
        target = self.path if self.path is not None else _default_telemetry_path()
        if target is None:
            return False
        event: dict[str, Any] = {
            "recorded_at": datetime.now(UTC).isoformat(),
            "task_id": task.task_id,
            "parent_task_id": task.parent_task_id,
            "orchestrator": task.orchestrator,
            "mode": task.mode,
            "role": task.agent_role,
            "model_tag": task.model_tag,
            "execution": task.execution,
            "privacy_classification": task.privacy_classification,
            "result_status": result.status,
            "requires_review": result.requires_review,
            "latency_ms": result.metrics.get("latency_ms"),
            "prompt_tokens": result.metrics.get("prompt_tokens"),
            "output_tokens": result.metrics.get("output_tokens"),
            "error_category": result.metrics.get("error_category"),
        }
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(event, sort_keys=True) + "\n")
            return True
        except OSError:
            return False
