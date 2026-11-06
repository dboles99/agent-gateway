from __future__ import annotations

from dataclasses import dataclass

from .models import ModelRole

ELIGIBLE_FALLBACK_REASONS = {
    "primary_rate_limit",
    "primary_unavailable",
    "primary_budget_exhausted",
    "primary_orchestration_failure",
    "operator_takeover",
}


@dataclass(frozen=True)
class RouteDecision:
    model: ModelRole
    reason: str
    fallback_activated: bool


def choose_orchestrator(
    *,
    active_model: ModelRole,
    fallback_activated: bool,
    fallback_reason: str | None,
    checkpoint_exists: bool,
) -> RouteDecision:
    """Choose who orchestrates next.

    Fallback is one-way: once activated it stays on, so a run cannot bounce
    between models. It also needs a checkpoint, because the fallback model has
    nothing else to continue from.
    """
    if fallback_activated:
        return RouteDecision(ModelRole.FALLBACK, "one-way fallback lock", True)

    if fallback_reason is None:
        return RouteDecision(active_model, "continue current orchestrator", False)

    if fallback_reason not in ELIGIBLE_FALLBACK_REASONS:
        return RouteDecision(active_model, "failure is not fallback eligible", False)

    if not checkpoint_exists:
        raise RuntimeError("Fallback blocked because no checkpoint exists")

    return RouteDecision(ModelRole.FALLBACK, fallback_reason, True)
