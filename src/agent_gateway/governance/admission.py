from __future__ import annotations

from dataclasses import dataclass

from .models import AdmissionStatus, ModelRole


@dataclass(frozen=True)
class AdmissionRequest:
    requested_model: ModelRole
    estimated_fresh_tokens: int
    estimated_total_tokens: int
    maximum_fresh_tokens: int
    maximum_total_tokens: int
    schema_valid: bool
    has_acceptance_criteria: bool
    has_stop_condition: bool
    cache_hit: bool = False
    requires_deep_reasoning: bool = False


@dataclass(frozen=True)
class AdmissionDecision:
    status: AdmissionStatus
    model: ModelRole | None
    reason: str


def admit(request: AdmissionRequest) -> AdmissionDecision:
    """Decide whether a request may run, and on which model.

    Checks run in a fixed order: a malformed request goes back for repair before
    anything else, a validated cache hit spends nothing, an oversized request is
    refused, and the primary model is kept for work that needs it.
    """
    if not request.schema_valid:
        return AdmissionDecision(AdmissionStatus.REQUIRE_REPAIR, None, "schema invalid")
    if not request.has_acceptance_criteria:
        return AdmissionDecision(
            AdmissionStatus.REQUIRE_REPAIR, None, "missing acceptance criteria"
        )
    if not request.has_stop_condition:
        return AdmissionDecision(
            AdmissionStatus.REQUIRE_REPAIR, None, "missing stop condition"
        )
    if request.cache_hit:
        return AdmissionDecision(AdmissionStatus.CACHE_HIT, None, "validated cache hit")
    if request.estimated_fresh_tokens > request.maximum_fresh_tokens:
        return AdmissionDecision(AdmissionStatus.REJECT, None, "fresh-token budget exceeded")
    if request.estimated_total_tokens > request.maximum_total_tokens:
        return AdmissionDecision(AdmissionStatus.REJECT, None, "total context budget exceeded")

    model = request.requested_model
    if model == ModelRole.PRIMARY and not request.requires_deep_reasoning:
        return AdmissionDecision(
            AdmissionStatus.DOWNROUTE,
            ModelRole.WORKER,
            "primary model not required for bounded implementation",
        )

    return AdmissionDecision(AdmissionStatus.ALLOW, model, "request admitted")
