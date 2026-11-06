"""Governance commands. Most are meant to be called from agent-host hooks."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import typer
import yaml
from jsonschema import Draft202012Validator

from ..schemas import CONTEXT_CAPSULE, SCHEMA_DIR, TASK_PACKET, load_schema
from .admission import AdmissionRequest, admit
from .checkpoint import write_checkpoint
from .ledger import EventLedger
from .loop_fingerprinter import LoopTracker, fingerprint
from .models import AdmissionStatus, ModelRole
from .paths import resolve_run_path
from .rule_eval import evaluate, explain, load_pack, render_rules
from .rulepack import compile_rulepack

app = typer.Typer(no_args_is_help=True, help="Admission, loop checks, ledger, rule packs.")

_ADMITTED = {
    AdmissionStatus.ALLOW,
    AdmissionStatus.ALLOW_WITH_LIMITS,
    AdmissionStatus.DOWNROUTE,
    AdmissionStatus.CACHE_HIT,
}


def _load_json_or_yaml(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    return yaml.safe_load(text)


def _bump_loop_counter(state_path: Path, value: str, limit: int) -> LoopTracker:
    """Record one sighting of ``value`` in the persisted counts.

    An unreadable state file starts a fresh count. This runs inside a hook, and a
    corrupt counter must not take the agent's tool call down with it.
    """
    resolved = resolve_run_path(state_path)
    counts: dict[str, int] = {}
    if resolved.exists():
        try:
            loaded = json.loads(resolved.read_text(encoding="utf-8"))
            counts = loaded if isinstance(loaded, dict) else {}
        except (OSError, json.JSONDecodeError):
            counts = {}
    tracker = LoopTracker(limit=limit, counts=counts)
    tracker.record(value)
    write_checkpoint(resolved, tracker.counts)
    return tracker


@app.command("compile-rules")
def compile_rules_command(
    path: Path,
    output: Path | None = typer.Option(None, "--output"),  # noqa: B008
) -> None:
    """Compile a rule pack and print the result as JSON."""
    rendered = json.dumps(compile_rulepack(path), indent=2)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    typer.echo(rendered)


@app.command("evaluate-rules")
def evaluate_rules_command(
    path: Path,
    context: Path = typer.Option(..., "--context", help="JSON or YAML context file"),  # noqa: B008
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output"),
) -> None:
    """Evaluate a rule pack against a context and report what fired.

    Exits 1 only when the decision blocks, so it can be used as a shell gate.
    """
    decision = evaluate(load_pack(path), _load_json_or_yaml(context) or {})
    if as_json:
        typer.echo(
            json.dumps(
                {
                    "pack": decision.pack_id,
                    "matched": decision.matched,
                    "rule": decision.rule_id,
                    "action": decision.action,
                    "severity": decision.severity,
                    "terminal": decision.terminal,
                    "blocking": decision.blocking,
                    "unknown": list(decision.unknown),
                    "trace": list(decision.trace),
                },
                indent=2,
                default=str,
            )
        )
    else:
        typer.echo(explain(decision))
    if decision.blocking:
        raise typer.Exit(code=1)


@app.command("render-rules")
def render_rules_command(path: Path) -> None:
    """Print a rule pack as compact text."""
    typer.echo(render_rules(load_pack(path)))


@app.command("validate-task")
def validate_task(path: Path) -> None:
    """Check a JSON or YAML task packet against the schema."""
    instance = _load_json_or_yaml(path)
    errors = sorted(
        Draft202012Validator(load_schema(TASK_PACKET)).iter_errors(instance),
        key=lambda e: list(e.path),
    )
    if errors:
        for error in errors:
            typer.echo(f"ERROR: {error.message}")
        raise typer.Exit(code=1)
    typer.echo("Task packet valid")


@app.command("validate-schemas")
def validate_schemas() -> None:
    """Check that every schema shipped with the package is itself valid."""
    errors: list[str] = []
    paths = sorted(SCHEMA_DIR.glob("*.schema.json"))
    for schema_path in paths:
        try:
            Draft202012Validator.check_schema(load_schema(schema_path))
        except Exception as exc:  # any failure here is the finding
            errors.append(f"{schema_path.name}: {exc}")
    if errors:
        typer.echo("\n".join(errors))
        raise typer.Exit(code=1)
    typer.echo(f"{len(paths)} schemas valid")


@app.command("admit")
def admit_command(path: Path) -> None:
    """Run admission on a context capsule. Exits 0 when the work may proceed.

    The capsule is validated against its schema. ``acceptance``, ``stop_condition``
    and ``requires_deep_reasoning`` are read from the capsule itself.
    """
    capsule = _load_json_or_yaml(path)
    valid = Draft202012Validator(load_schema(CONTEXT_CAPSULE)).is_valid(capsule)
    capsule = capsule if isinstance(capsule, dict) else {}
    estimates = capsule.get("token_estimate", {}) if valid else {}
    limits = capsule.get("limits", {}) if valid else {}
    request = AdmissionRequest(
        requested_model=ModelRole(capsule.get("target_model", ModelRole.WORKER.value))
        if valid
        else ModelRole.WORKER,
        estimated_fresh_tokens=int(estimates.get("fresh_dynamic", 0)),
        estimated_total_tokens=int(estimates.get("total", 0)),
        maximum_fresh_tokens=int(limits.get("maximum_fresh_tokens", 0)),
        maximum_total_tokens=int(limits.get("maximum_total_tokens", 0)),
        schema_valid=valid,
        has_acceptance_criteria=bool(capsule.get("acceptance")),
        has_stop_condition=bool(capsule.get("stop_condition")),
        requires_deep_reasoning=bool(capsule.get("requires_deep_reasoning")),
    )
    decision = admit(request)
    typer.echo(
        json.dumps(
            {
                "status": decision.status.value,
                "model": decision.model.value if decision.model else None,
                "reason": decision.reason,
            },
            indent=2,
        )
    )
    if decision.status not in _ADMITTED:
        raise typer.Exit(code=1)


@app.command("log-event")
def log_event_command(ledger_path: Path) -> None:
    """Read a JSON event from stdin and append it to the JSONL ledger.

    A relative path resolves against the run directory, not the working directory.
    """
    payload = json.loads(sys.stdin.read())
    EventLedger(resolve_run_path(ledger_path)).append(payload)


@app.command("check-loop")
def check_loop_command(state_path: Path, limit: int = 2) -> None:
    """Exit 1 when an identical tool call repeats beyond the limit.

    Reads {"tool", "arguments", "task_state"} JSON from stdin and counts its
    fingerprint in the state file. The caller decides what exit 1 means.
    """
    payload = json.loads(sys.stdin.read())
    value = fingerprint(
        payload.get("tool", ""), payload.get("arguments", {}), payload.get("task_state", "")
    )
    tracker = _bump_loop_counter(state_path, value, limit)
    if tracker.should_deny(value):
        typer.echo(
            f"Repeated tool call detected ({tracker.counts[value]}x with identical "
            f"arguments): {payload.get('tool')}. Consider a different approach instead "
            "of repeating this call.",
            err=True,
        )
        raise typer.Exit(code=1)


@app.command("check-loop-hook")
def check_loop_hook_command(state_path: Path, limit: int = 2) -> None:
    """PreToolUse hook adapter for check-loop, in warn-only mode.

    Reads a Claude Code PreToolUse payload (tool_name, tool_input, session_id) and
    always exits 0 with hookSpecificOutput JSON: permissionDecision is always
    "allow", with additionalContext carrying a warning past the repeat limit.
    """
    payload = json.loads(sys.stdin.read())
    tool = payload.get("tool_name") or payload.get("tool", "")
    arguments = payload.get("tool_input") or payload.get("arguments", {})
    task_state = payload.get("session_id") or payload.get("task_state", "")

    value = fingerprint(tool, arguments, task_state)
    tracker = _bump_loop_counter(state_path, value, limit)

    output: dict[str, Any] = {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "allow",
        }
    }
    if tracker.should_deny(value):
        output["hookSpecificOutput"]["additionalContext"] = (
            f"Repeated tool call detected ({tracker.counts[value]}x with identical "
            f"arguments): {tool}. Consider a different approach instead of repeating "
            "this call, or checkpoint if you are stuck."
        )
    typer.echo(json.dumps(output))


@app.command("write-checkpoint")
def write_checkpoint_command(path: Path) -> None:
    """Read a JSON checkpoint from stdin and write it atomically to path."""
    payload = json.loads(sys.stdin.read())
    resolved = resolve_run_path(path)
    write_checkpoint(resolved, payload)
    typer.echo(f"Checkpoint written to {resolved}")
