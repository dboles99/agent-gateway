"""Optional round trip against a real Ollama server.

Skipped unless AGENT_GATEWAY_LIVE_MODEL names a pulled model:

    AGENT_GATEWAY_LIVE_MODEL=your-model:latest pytest tests/test_live_ollama.py
"""
from __future__ import annotations

import os

import pytest

from agent_gateway.pool import OllamaAgentPool
from agent_gateway.pool.registry import (
    AgentPoolConfig,
    ModelEntry,
    PoolDefaults,
    RoleEntry,
)

LIVE_MODEL = os.environ.get("AGENT_GATEWAY_LIVE_MODEL", "")


@pytest.mark.skipif(not LIVE_MODEL, reason="AGENT_GATEWAY_LIVE_MODEL is not set")
def test_a_real_model_round_trip():
    model = ModelEntry(
        model_id="live",
        model_tag=LIVE_MODEL,
        enabled=True,
        lifecycle="candidate",
        roles=("critic",),
        privacy_classes=("public",),
    )
    pool = OllamaAgentPool(
        config=AgentPoolConfig(
            defaults=PoolDefaults(enabled=True, mode="manual"),
            models={"live": model},
            roles={"critic": RoleEntry(role_id="critic", purpose="Find defects.")},
        )
    )
    result = pool.delegate_task(
        {
            "task_id": "live-1",
            "orchestrator": "other",
            "mode": "manual",
            "agent_role": "critic",
            "privacy_classification": "public",
            "maximum_output_tokens": 300,
            "timeout_seconds": 300,
            "objective": "def add(a, b): return a - b. Name the defect in one sentence.",
        }
    )
    assert result.status in {"completed", "partial"}, result.summary
    assert result.requires_review is True
    assert result.metrics["latency_ms"] > 0
