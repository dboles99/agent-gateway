"""Commands for the advisory Ollama pool."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer
import yaml

from .delegation import OllamaAgentPool

app = typer.Typer(no_args_is_help=True, help="The advisory Ollama agent pool.")


def _load_task(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    data = json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
    if not isinstance(data, dict):
        raise typer.BadParameter(f"Task file must contain an object: {path}")
    return data


def _apply_overrides(payload: dict[str, Any], role: str, model: str) -> dict[str, Any]:
    if role:
        payload["agent_role"] = role
    if model and model != "auto":
        payload["model_tag"] = model
    return payload


@app.command("models")
def models_command() -> None:
    """List the registered models."""
    typer.echo(json.dumps({"models": OllamaAgentPool().list_models()}, indent=2, sort_keys=True))


@app.command("delegate")
def delegate_command(
    task_file: Path = typer.Option(..., "--file"),  # noqa: B008
    role: str = typer.Option("", "--role"),
    model: str = typer.Option("auto", "--model"),
) -> None:
    """Send one task to the pool and print the result envelope."""
    payload = _apply_overrides(_load_task(task_file), role, model)
    result = OllamaAgentPool().delegate_task(payload)
    typer.echo(json.dumps(result.to_dict(), indent=2, sort_keys=True))


@app.command("shadow")
def shadow_command(
    task_file: Path = typer.Option(..., "--file"),  # noqa: B008
    role: str = typer.Option("shadow-reviewer", "--role"),
    model: str = typer.Option("auto", "--model"),
    primary_summary: str = typer.Option("", "--primary-summary"),
) -> None:
    """Run an advisory review of finished work and print the result envelope."""
    payload = _apply_overrides(_load_task(task_file), role, model)
    result = OllamaAgentPool().run_shadow_review(payload, primary_result_summary=primary_summary)
    typer.echo(json.dumps(result.to_dict(), indent=2, sort_keys=True))
