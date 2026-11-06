"""The ``agent-gateway`` command."""
from __future__ import annotations

import typer

from ._config import ensure_env_loaded
from .gateway.cli import app as gateway_app
from .governance.cli import app as governance_app
from .pool.cli import app as pool_app

app = typer.Typer(
    no_args_is_help=True,
    help="Admission control, loop checks and task packet dispatch for agents.",
)
app.add_typer(governance_app, name="governance")
app.add_typer(gateway_app, name="gateway")
app.add_typer(pool_app, name="pool")


@app.callback()
def _load_env() -> None:
    # A comment, not a docstring: Typer would print a docstring as the app's help.
    ensure_env_loaded()
