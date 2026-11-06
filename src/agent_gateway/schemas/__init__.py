"""The JSON Schemas that define the task, result and governance objects."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SCHEMA_DIR = Path(__file__).resolve().parent
TASK_PACKET = SCHEMA_DIR / "task_packet.schema.json"
RESULT_PACKET = SCHEMA_DIR / "result_packet.schema.json"
CONTEXT_CAPSULE = SCHEMA_DIR / "context_capsule.schema.json"
AGENT_DEFINITION = SCHEMA_DIR / "agent_definition.schema.json"


def load_schema(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
