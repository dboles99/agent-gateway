from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AdmissionStatus(str, Enum):
    ALLOW = "ALLOW"
    ALLOW_WITH_LIMITS = "ALLOW_WITH_LIMITS"
    DOWNROUTE = "DOWNROUTE"
    CACHE_HIT = "CACHE_HIT"
    REQUIRE_REPAIR = "REQUIRE_REPAIR"
    REQUIRE_HUMAN = "REQUIRE_HUMAN"
    REJECT = "REJECT"


class ModelRole(str, Enum):
    """A place in the model line-up, not a vendor or a model name."""

    PRIMARY = "primary"
    FALLBACK = "fallback"
    WORKER = "worker"
    LIGHT = "light"


class CircuitState(str, Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


@dataclass(frozen=True)
class Budget:
    max_fresh_input_tokens: int
    max_output_tokens: int
    max_turns: int
    max_tool_calls: int
    max_wall_seconds: int


@dataclass
class Usage:
    fresh_input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    turns: int = 0
    tool_calls: int = 0
    elapsed_seconds: float = 0.0


@dataclass(frozen=True)
class ProgressProof:
    proof_type: str
    weight: int
    evidence: dict[str, Any]


@dataclass
class RunState:
    run_id: str
    task_id: str
    model: ModelRole
    usage: Usage = field(default_factory=Usage)
    proofs: list[ProgressProof] = field(default_factory=list)
    completed: bool = False
    fallback_activated: bool = False
