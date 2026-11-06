"""The surface an external orchestrator calls.

Exit codes are for the shell. The result packet on stdout is for the caller.
``execute`` exits 0 only on a ``passed`` result, so it can be used as a gate.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import typer
from jsonschema import Draft202012Validator

from ..governance.ledger import EventLedger
from .dispatch import (
    DEFAULT_TASK_SCHEMA,
    SchemaUnavailable,
    load_packet_schema,
    run_task_packet,
)
from .registry import CapabilityConfigError, handler_status, load_declared_capabilities

app = typer.Typer(no_args_is_help=True, help="Capabilities and task packet dispatch.")

_STDIN = "-"


def _read_packet(source: str) -> Any:
    """Read a task packet from a file or stdin, or exit 2 with the reason.

    Exit 2, not 1: a packet that could not be read is a different fact from a
    packet that was read and refused.
    """
    try:
        text = sys.stdin.read() if source == _STDIN else Path(source).read_text(encoding="utf-8")
    except OSError as exc:
        typer.echo(f"cannot read task packet: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        typer.echo(f"task packet is not JSON: {exc}", err=True)
        raise typer.Exit(code=2) from exc


@app.command("capabilities")
def capabilities_command(
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output"),
    config: Path | None = typer.Option(None, "--config", help="Capability config"),  # noqa: B008
) -> None:
    """List declared capabilities and whether each one is actually built.

    Exits 1 when a declared handler does not resolve. An unbuilt capability is not
    a failure, but a declared handler that will not import is a defect.
    """
    try:
        declared = load_declared_capabilities(config)
    except CapabilityConfigError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc

    rows = []
    for capability_id, entry in sorted(declared.items()):
        state, detail = handler_status(entry)
        rows.append(
            {
                "id": capability_id,
                "executor": entry.executor,
                "handler": entry.handler,
                "state": state,
                "detail": detail,
                "required_outputs": list(entry.required_outputs),
            }
        )

    broken = [row for row in rows if row["state"] == "unresolvable"]
    if as_json:
        typer.echo(json.dumps({"capabilities": rows}, indent=2))
    elif not rows:
        typer.echo("no capabilities declared")
    else:
        for row in rows:
            typer.echo(f"{row['state']:<13} {row['id']}")
            if row["handler"]:
                typer.echo(f"              handler: {row['handler']}")
            if row["required_outputs"]:
                typer.echo(f"              required: {', '.join(row['required_outputs'])}")
            if row["state"] == "unresolvable":
                typer.echo(f"              !! {row['detail']}")
        built = sum(1 for row in rows if row["state"] == "ready")
        unbuilt = sum(1 for row in rows if row["state"] == "unbuilt")
        typer.echo(
            f"\n{len(rows)} declared, {built} built, {unbuilt} unbuilt, "
            f"{len(broken)} unresolvable"
        )
    if broken:
        raise typer.Exit(code=1)


@app.command("validate")
def validate_command(
    packet: str = typer.Argument(_STDIN, help="Task packet JSON file, or - for stdin"),
) -> None:
    """Check a task packet against the schema without dispatching it."""
    data = _read_packet(packet)
    try:
        schema = load_packet_schema(DEFAULT_TASK_SCHEMA)
    except SchemaUnavailable as exc:  # pragma: no cover - ships with the package
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc

    errors = sorted(
        f"{'/'.join(str(p) for p in error.absolute_path) or '<root>'}: {error.message}"
        for error in Draft202012Validator(schema).iter_errors(data)
    )
    if errors:
        typer.echo("task packet is INVALID:", err=True)
        for error in errors:
            typer.echo(f"  {error}", err=True)
        raise typer.Exit(code=1)
    typer.echo("task packet is valid")


@app.command("execute")
def execute_command(
    packet: str = typer.Argument(_STDIN, help="Task packet JSON file, or - for stdin"),
    config: Path | None = typer.Option(None, "--config", help="Capability config"),  # noqa: B008
    progress_ledger: Path | None = typer.Option(  # noqa: B008
        None,
        "--progress-ledger",
        help="Append the result's progress proofs to this JSONL ledger",
    ),
) -> None:
    """Dispatch a task packet and print the result packet.

    Always prints a result packet, including on a refusal. Exit 0 only when the
    status is ``passed``.
    """
    data = _read_packet(packet)
    result = run_task_packet(data, config_path=config)

    if progress_ledger is not None:
        ledger = EventLedger(progress_ledger)
        for proof in result.get("progress_proofs") or []:
            ledger.append(proof)

    typer.echo(json.dumps(result, indent=2))
    if result.get("status") != "passed":
        raise typer.Exit(code=1)
