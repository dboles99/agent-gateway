"""The capability config, and the handlers bound to it.

Two fields name an implementation and they are different fields:

``executor``
    The intended implementation, for a person to read. Free text. Nothing resolves it.
``handler``
    The machine half: a ``pkg.module:attr`` path to a callable that takes a task
    packet and returns a dict. Absent means deliberately unbuilt. Present and not
    importable is a defect. The two must not report alike.

``handler`` is imported and called, so the capability config is trusted code.
"""
from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from .._config import capabilities_path

Capability = Callable[[dict[str, Any]], dict[str, Any]]


class CapabilityConfigError(ValueError):
    """The capability config exists and could not be parsed."""


class HandlerNotDeclared(LookupError):
    """The capability declares no ``handler``, so it is deliberately unbuilt."""


class HandlerUnresolvable(LookupError):
    """A ``handler`` is declared and does not resolve to a callable."""


@dataclass(frozen=True)
class DeclaredCapability:
    """One entry from the capability config."""

    id: str
    executor: str = ""
    handler: str = ""
    input_schema: str = ""
    output_schema: str = ""
    required_outputs: tuple[str, ...] = ()
    base_dir: Path = Path()

    def schema_path(self, which: str) -> Path | None:
        """Absolute path of a declared schema, or None when none is declared.

        Relative paths resolve against the config file's directory. The path is
        returned whether or not it exists: a declared schema missing on disk is a
        defect to refuse on, not something to treat as undeclared.
        """
        declared = getattr(self, which, "")
        if not declared:
            return None
        candidate = Path(declared)
        return candidate if candidate.is_absolute() else self.base_dir / candidate


def load_declared_capabilities(
    path: str | Path | None = None,
) -> dict[str, DeclaredCapability]:
    """Read the declared capabilities, keyed by id.

    An absent file is an empty declaration. A file that exists and cannot be parsed
    raises :class:`CapabilityConfigError`. The parse is cached on the file's mtime
    and size, so a long-running process sees an edit to the config.
    """
    target = Path(path) if path is not None else capabilities_path()
    try:
        stat = target.stat()
    except OSError:
        return {}
    return dict(_parse_declared(target, (stat.st_mtime_ns, stat.st_size)))


@lru_cache(maxsize=8)
def _parse_declared(
    target: Path, _stamp: tuple[int, int]
) -> dict[str, DeclaredCapability]:
    if not target.is_file():
        return {}
    try:
        data = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise CapabilityConfigError(f"{target}: {exc}") from exc
    if not isinstance(data, dict):
        raise CapabilityConfigError(f"{target}: top level must be a mapping")

    declared: dict[str, DeclaredCapability] = {}
    for entry in data.get("capabilities", []) or []:
        if not isinstance(entry, dict) or not entry.get("id"):
            continue
        declared[str(entry["id"])] = DeclaredCapability(
            id=str(entry["id"]),
            executor=str(entry.get("executor", "")),
            handler=str(entry.get("handler", "") or ""),
            input_schema=str(entry.get("input_schema", "")),
            output_schema=str(entry.get("output_schema", "")),
            required_outputs=tuple(entry.get("required_outputs", ()) or ()),
            base_dir=target.resolve().parent,
        )
    return declared


@dataclass
class CapabilityRegistry:
    capabilities: dict[str, Capability] = field(default_factory=dict)
    config_path: str | Path | None = None

    def register(
        self, capability_id: str, handler: Capability, *, strict: bool = False
    ) -> None:
        """Register a handler.

        ``strict=True`` refuses an id the capability config never declared, which
        catches a mistyped or renamed capability at registration, not at the first
        dispatch.
        """
        if capability_id in self.capabilities:
            raise ValueError(f"Capability already registered: {capability_id}")
        if strict and capability_id not in load_declared_capabilities(self.config_path):
            raise KeyError(f"Capability not declared in the config: {capability_id}")
        self.capabilities[capability_id] = handler

    def resolve(self, capability_id: str) -> Capability:
        try:
            return self.capabilities[capability_id]
        except KeyError as exc:
            declared = load_declared_capabilities(self.config_path)
            if capability_id in declared:
                raise KeyError(
                    f"Capability '{capability_id}' is declared but has no handler "
                    f"registered (executor: {declared[capability_id].executor or 'unset'})"
                ) from exc
            raise KeyError(f"Unknown capability: {capability_id}") from exc

    def unimplemented(self) -> tuple[str, ...]:
        """Declared capability ids with no handler registered."""
        declared = load_declared_capabilities(self.config_path)
        return tuple(sorted(set(declared) - set(self.capabilities)))

    def undeclared(self) -> tuple[str, ...]:
        """Registered ids the capability config does not declare."""
        declared = load_declared_capabilities(self.config_path)
        return tuple(sorted(set(self.capabilities) - set(declared)))


def resolve_declared_handler(declared: DeclaredCapability) -> Capability:
    """Import the callable a capability declares as its ``handler``.

    Accepts ``pkg.module:attr`` or ``pkg.module.attr``. Raises
    :class:`HandlerNotDeclared` when there is no handler at all, and
    :class:`HandlerUnresolvable` when one is declared and does not import or is not
    callable, so a caller can tell an unbuilt capability from a broken one.
    """
    target = declared.handler.strip()
    if not target:
        raise HandlerNotDeclared(
            f"Capability '{declared.id}' declares no handler "
            f"(intended executor: {declared.executor or 'unset'})"
        )

    module_name, sep, attr = target.partition(":")
    if not sep:
        module_name, _, attr = target.rpartition(".")
    if not module_name or not attr:
        raise HandlerUnresolvable(
            f"Capability '{declared.id}' declares handler {target!r}, which is "
            "not a 'module:attr' or 'module.attr' path"
        )

    try:
        module = importlib.import_module(module_name)
    # Any import failure gives the same verdict: the declared handler does not
    # resolve. An ImportError inside the target must not escape as a crash.
    except Exception as exc:
        raise HandlerUnresolvable(
            f"Capability '{declared.id}' declares handler {target!r}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    try:
        candidate = getattr(module, attr)
    except AttributeError as exc:
        raise HandlerUnresolvable(
            f"Capability '{declared.id}' declares handler {target!r}: module "
            f"{module_name!r} defines no {attr!r}"
        ) from exc

    if not callable(candidate):
        raise HandlerUnresolvable(
            f"Capability '{declared.id}' declares handler {target!r}, which is "
            f"a {type(candidate).__name__} and not callable"
        )
    return candidate


def handler_status(declared: DeclaredCapability) -> tuple[str, str]:
    """(state, detail) for one capability, without raising.

    States: ``ready``, ``unbuilt``, ``unresolvable``.
    """
    try:
        resolve_declared_handler(declared)
    except HandlerNotDeclared as exc:
        return "unbuilt", str(exc)
    except HandlerUnresolvable as exc:
        return "unresolvable", str(exc)
    return "ready", declared.handler


def load_declared_registry(
    path: str | Path | None = None,
) -> tuple[CapabilityRegistry, dict[str, str]]:
    """Build a registry from the config, binding every capability that resolves.

    Returns the registry and a ``{capability_id: reason}`` map of those that did
    not bind, so a caller can report the gap up front.
    """
    registry = CapabilityRegistry(config_path=path)
    unbound: dict[str, str] = {}
    for capability_id, declared in load_declared_capabilities(path).items():
        state, detail = handler_status(declared)
        if state == "ready":
            registry.register(capability_id, resolve_declared_handler(declared))
        else:
            unbound[capability_id] = detail
    return registry, unbound
