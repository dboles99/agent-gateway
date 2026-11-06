"""Turn whatever a model returned into one envelope the caller can trust.

Model output is untrusted. It fills the advisory fields only: the ids, the review
flag and the metrics come from the caller, so a model cannot relabel its own result
or clear the review requirement.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any

_MODEL_STATUSES = {"completed", "partial", "failed"}


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _confidence(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(number):
        return 0.0
    return max(0.0, min(1.0, number))


@dataclass
class ResultEnvelope:
    task_id: str
    model_tag: str
    agent_role: str
    status: str
    summary: str
    recommendations: list[Any] = field(default_factory=list)
    findings: list[Any] = field(default_factory=list)
    risks: list[Any] = field(default_factory=list)
    assumptions: list[Any] = field(default_factory=list)
    proposed_changes: list[Any] = field(default_factory=list)
    tests_proposed: list[Any] = field(default_factory=list)
    confidence: float = 0.0
    requires_review: bool = True
    metrics: dict[str, Any] = field(
        default_factory=lambda: {"latency_ms": 0, "prompt_tokens": None, "output_tokens": None}
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "model_tag": self.model_tag,
            "agent_role": self.agent_role,
            "status": self.status,
            "summary": self.summary,
            "recommendations": list(self.recommendations),
            "findings": list(self.findings),
            "risks": list(self.risks),
            "assumptions": list(self.assumptions),
            "proposed_changes": list(self.proposed_changes),
            "tests_proposed": list(self.tests_proposed),
            "confidence": self.confidence,
            "requires_review": self.requires_review,
            "metrics": dict(self.metrics),
        }


def rejected_result(
    *,
    task_id: str,
    model_tag: str,
    agent_role: str,
    summary: str,
    status: str = "rejected",
    error_category: str | None = None,
) -> ResultEnvelope:
    metrics: dict[str, Any] = {"latency_ms": 0, "prompt_tokens": None, "output_tokens": None}
    if error_category:
        metrics["error_category"] = error_category
    return ResultEnvelope(
        task_id=task_id,
        model_tag=model_tag,
        agent_role=agent_role,
        status=status,
        summary=summary,
        metrics=metrics,
    )


def normalize_result(
    raw_output: str | dict[str, Any],
    *,
    task_id: str,
    model_tag: str,
    agent_role: str,
    metrics: dict[str, Any] | None = None,
) -> ResultEnvelope:
    parsed: dict[str, Any] | None
    if isinstance(raw_output, dict):
        parsed = raw_output
    else:
        try:
            loaded = json.loads(raw_output)
            parsed = loaded if isinstance(loaded, dict) else None
        except json.JSONDecodeError:
            parsed = None

    base_metrics = {"latency_ms": 0, "prompt_tokens": None, "output_tokens": None}
    if metrics:
        base_metrics.update(metrics)

    if parsed is None:
        summary = str(raw_output).strip().replace("\x00", "")[:500]
        return ResultEnvelope(
            task_id=task_id,
            model_tag=model_tag,
            agent_role=agent_role,
            status="partial",
            summary=summary or "Model returned non-JSON output.",
            metrics=base_metrics,
        )

    status = str(parsed.get("status") or "completed")
    return ResultEnvelope(
        task_id=task_id,
        model_tag=model_tag,
        agent_role=agent_role,
        status=status if status in _MODEL_STATUSES else "partial",
        summary=str(parsed.get("summary") or ""),
        recommendations=_as_list(parsed.get("recommendations")),
        findings=_as_list(parsed.get("findings")),
        risks=_as_list(parsed.get("risks")),
        assumptions=_as_list(parsed.get("assumptions")),
        proposed_changes=_as_list(parsed.get("proposed_changes")),
        tests_proposed=_as_list(parsed.get("tests_proposed")),
        confidence=_confidence(parsed.get("confidence")),
        requires_review=True,
        metrics=base_metrics,
    )
