from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime


@dataclass
class WatchdogState:
    last_progress_at: datetime
    actions_since_progress: int = 0
    tokens_since_progress: int = 0


@dataclass(frozen=True)
class WatchdogPolicy:
    warning_seconds: int = 60
    intervention_seconds: int = 120
    maximum_actions_without_proof: int = 5
    maximum_tokens_without_proof: int = 4000


def evaluate_watchdog(
    state: WatchdogState,
    policy: WatchdogPolicy,
    now: datetime | None = None,
) -> str:
    now = now or datetime.now(UTC)
    elapsed = (now - state.last_progress_at).total_seconds()

    if (
        elapsed >= policy.intervention_seconds
        or state.actions_since_progress > policy.maximum_actions_without_proof
        or state.tokens_since_progress > policy.maximum_tokens_without_proof
    ):
        return "CHECKPOINT_AND_STOP"

    if elapsed >= policy.warning_seconds:
        return "WARN"

    return "CONTINUE"
