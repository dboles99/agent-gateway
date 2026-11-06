from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch, tmp_path):
    """Keep tests away from the real home directory and any developer settings."""
    import os

    for name in list(os.environ):
        if name.startswith("AGENT_GATEWAY_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("AGENT_GATEWAY_RUN_DIR", str(tmp_path / "run"))
    monkeypatch.setenv("AGENT_GATEWAY_CAPABILITIES", str(tmp_path / "no-capabilities.yaml"))
    monkeypatch.setenv("AGENT_GATEWAY_POOL_REGISTRY", str(tmp_path / "no-pool.yaml"))
    monkeypatch.setenv("AGENT_GATEWAY_POOL_ROLES", str(tmp_path / "no-roles.yaml"))
