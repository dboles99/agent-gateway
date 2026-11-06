"""Declared capabilities and the gateway that dispatches task packets to them.

A capability is declared in a YAML file before it has a handler. ``run_task_packet``
enforces what the declaration says, and answers every outcome with a result packet.
"""
from .dispatch import run_task_packet
from .executor import execute_task
from .registry import CapabilityRegistry, load_declared_registry

__all__ = [
    "CapabilityRegistry",
    "execute_task",
    "load_declared_registry",
    "run_task_packet",
]
