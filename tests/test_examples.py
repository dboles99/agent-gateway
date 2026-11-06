"""The README's examples must actually run."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_quickstart_runs_and_prints_each_stage():
    result = subprocess.run(
        [sys.executable, str(ROOT / "examples" / "quickstart.py")],
        capture_output=True,
        text=True,
        check=True,
    )
    lines = result.stdout.splitlines()
    assert lines[0].startswith("admission: DOWNROUTE to worker")
    assert "budget: HARD" in lines
    assert "loop: deny repeat call: True" in lines
    assert "watchdog: CHECKPOINT_AND_STOP" in lines
    assert any(line.startswith("DECISION ACCEPT") for line in lines)
    assert any("no rule matched" in line for line in lines)
    assert any(line.startswith("dispatch: passed") for line in lines)
    assert "dispatch: blocked capability_unbuilt" in lines
    assert lines[-1].startswith("pool: completed, demo:latest")


def test_the_console_entry_point_works():
    script = Path(sys.executable).with_name("agent-gateway")
    result = subprocess.run(
        [
            str(script),
            "gateway",
            "execute",
            str(ROOT / "examples" / "task_packet.json"),
            "--config",
            str(ROOT / "examples" / "capabilities.yaml"),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout)["status"] == "passed"
