"""A tour of agent_gateway. Runs offline and writes only to a temporary folder.

    python examples/quickstart.py
"""
from __future__ import annotations

import json
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

from agent_gateway.gateway import run_task_packet
from agent_gateway.governance.admission import AdmissionRequest, admit
from agent_gateway.governance.budget import evaluate_budget
from agent_gateway.governance.checkpoint import read_checkpoint, write_checkpoint
from agent_gateway.governance.loop_fingerprinter import LoopTracker, fingerprint
from agent_gateway.governance.models import Budget, ModelRole, Usage
from agent_gateway.governance.router import choose_orchestrator
from agent_gateway.governance.rule_eval import evaluate_pack, explain
from agent_gateway.governance.watchdog import (
    WatchdogPolicy,
    WatchdogState,
    evaluate_watchdog,
)
from agent_gateway.pool import OllamaAgentPool
from agent_gateway.pool.local_client import OllamaResponse
from agent_gateway.pool.registry import (
    AgentPoolConfig,
    ModelEntry,
    PoolDefaults,
    RoleEntry,
)

HERE = Path(__file__).resolve().parent


class StubClient:
    """Stands in for Ollama so this example needs no model."""

    def generate(self, *, model_tag, prompt, timeout_seconds, maximum_output_tokens):
        text = json.dumps({"summary": "One missing test.", "confidence": 0.6})
        return OllamaResponse(text=text, metrics={"latency_ms": 1})


def main() -> None:
    decision = admit(
        AdmissionRequest(
            requested_model=ModelRole.PRIMARY,
            estimated_fresh_tokens=1000,
            estimated_total_tokens=2000,
            maximum_fresh_tokens=5000,
            maximum_total_tokens=10000,
            schema_valid=True,
            has_acceptance_criteria=True,
            has_stop_condition=True,
        )
    )
    print(f"admission: {decision.status.value} to {decision.model.value}: {decision.reason}")

    budget = Budget(1000, 500, 4, 10, 60)
    print("budget:", evaluate_budget(budget, Usage(fresh_input_tokens=900)).level)

    tracker = LoopTracker(limit=2)
    call = fingerprint("grep", {"pattern": "x", "timestamp": 1}, "searching")
    for _ in range(3):
        tracker.record(call)
    print("loop: deny repeat call:", tracker.should_deny(call))

    state = WatchdogState(last_progress_at=datetime.now(UTC) - timedelta(seconds=130))
    print("watchdog:", evaluate_watchdog(state, WatchdogPolicy()))

    with tempfile.TemporaryDirectory() as tmp:
        path = write_checkpoint(Path(tmp) / "checkpoint.json", {"next_actions": ["rerun tests"]})
        route = choose_orchestrator(
            active_model=ModelRole.PRIMARY,
            fallback_activated=False,
            fallback_reason="primary_rate_limit",
            checkpoint_exists=path.exists(),
        )
        print(f"fallback: {route.model.value}, resume at {read_checkpoint(path)['next_actions']}")

    context = json.loads((HERE / "rules" / "context-pass.json").read_text(encoding="utf-8"))
    print(explain(evaluate_pack(HERE / "rules" / "release-gate.yaml", context)))
    print(explain(evaluate_pack(HERE / "rules" / "release-gate.yaml", {})).splitlines()[0])

    packet = json.loads((HERE / "task_packet.json").read_text(encoding="utf-8"))
    config = HERE / "capabilities.yaml"
    result = run_task_packet(packet, config_path=config)
    print("dispatch:", result["status"], result["artifacts"])
    refused = run_task_packet({**packet, "capability": "text.summarise"}, config_path=config)
    print("dispatch:", refused["status"], refused["unresolved"][0]["reason"])

    pool = OllamaAgentPool(
        config=AgentPoolConfig(
            defaults=PoolDefaults(enabled=True, mode="manual"),
            models={
                "local": ModelEntry(
                    model_id="local",
                    model_tag="demo:latest",
                    enabled=True,
                    lifecycle="candidate",
                    roles=("critic",),
                    privacy_classes=("internal",),
                )
            },
            roles={"critic": RoleEntry(role_id="critic", purpose="Find defects.")},
        ),
        client=StubClient(),
    )
    envelope = pool.delegate_task(
        {
            "task_id": "t-3",
            "orchestrator": "other",
            "mode": "manual",
            "agent_role": "critic",
            "objective": "Review this change.",
        }
    )
    print(f"pool: {envelope.status}, {envelope.model_tag}, review required: {envelope.requires_review}")


if __name__ == "__main__":
    main()
