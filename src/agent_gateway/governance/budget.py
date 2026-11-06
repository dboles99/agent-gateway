from __future__ import annotations

from dataclasses import dataclass

from .models import Budget, Usage


@dataclass(frozen=True)
class BudgetDecision:
    allowed: bool
    level: str
    reason: str


def evaluate_budget(budget: Budget, usage: Usage) -> BudgetDecision:
    checks = {
        "fresh_input_tokens": (usage.fresh_input_tokens, budget.max_fresh_input_tokens),
        "output_tokens": (usage.output_tokens, budget.max_output_tokens),
        "turns": (usage.turns, budget.max_turns),
        "tool_calls": (usage.tool_calls, budget.max_tool_calls),
        "wall_seconds": (usage.elapsed_seconds, budget.max_wall_seconds),
    }

    for name, (current, maximum) in checks.items():
        if current >= maximum:
            return BudgetDecision(False, "EMERGENCY", f"{name} hard limit reached")

    ratios = [current / maximum for current, maximum in checks.values() if maximum > 0]
    peak = max(ratios, default=0.0)

    if peak >= 0.88:
        return BudgetDecision(True, "HARD", "checkpoint before additional expensive work")
    if peak >= 0.70:
        return BudgetDecision(True, "SOFT", "compact context and reduce work")
    return BudgetDecision(True, "NORMAL", "within budget")
