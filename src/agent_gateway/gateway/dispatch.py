"""The boundary an out-of-process orchestrator crosses to reach a capability.

``execute_task`` shapes a handler's return value into a result packet. This module
is the layer that enforces, and it sits apart from the command line so an in-process
caller cannot get the dispatch without the checks.

The config's ``input_schema``, ``output_schema`` and ``required_outputs`` are
enforced here. A declaration no code reads is an instruction, not a mechanism.

**Every outcome is a valid result packet, including every refusal.** A caller in
another process cannot read a traceback, and one exit code cannot say what went
wrong.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from ..schemas import RESULT_PACKET, TASK_PACKET
from .executor import HandlerResultMalformed, execute_task
from .progress import progress_event
from .registry import (
    CapabilityConfigError,
    CapabilityRegistry,
    DeclaredCapability,
    HandlerNotDeclared,
    HandlerUnresolvable,
    load_declared_capabilities,
    load_declared_registry,
    resolve_declared_handler,
)

DEFAULT_TASK_SCHEMA = TASK_PACKET
DEFAULT_RESULT_SCHEMA = RESULT_PACKET

# Why a refusal was made. Each is a distinct fact and they must not collapse into
# one another: `capability_unbuilt` is a normal state, `handler_unresolvable` is a
# defect in the config.
REASON_UNREADABLE = "task_packet_unreadable"
REASON_INVALID = "task_packet_invalid"
REASON_CONFIG_UNREADABLE = "capability_config_unreadable"
REASON_UNDECLARED = "capability_undeclared"
REASON_UNBUILT = "capability_unbuilt"
REASON_UNRESOLVABLE = "handler_unresolvable"
REASON_SCHEMA_MISSING = "declared_schema_missing"
REASON_HANDLER_RAISED = "handler_raised"
REASON_RESULT_MALFORMED = "handler_result_malformed"
REASON_OUTPUTS_MISSING = "required_outputs_missing"
REASON_RESULT_INVALID = "result_packet_invalid"


class SchemaUnavailable(RuntimeError):
    """A schema the config declares is not on disk."""


def load_packet_schema(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SchemaUnavailable(f"{path}: {type(exc).__name__}: {exc}") from exc


def _refusal(
    packet: Any,
    *,
    status: str,
    reason: str,
    detail: str,
    errors: list[str] | None = None,
) -> dict[str, Any]:
    """Build a result packet describing a refusal.

    Built from literals of known-good type so it cannot itself fail validation. The
    ids are echoed from the packet when it could carry them, and empty otherwise.
    """
    ids: dict[str, Any] = packet if isinstance(packet, dict) else {}
    unresolved: dict[str, Any] = {"reason": reason, "detail": detail}
    if errors:
        unresolved["errors"] = list(errors)
    return {
        "execution_id": str(ids.get("execution_id") or ""),
        "task_id": str(ids.get("task_id") or ""),
        "status": status,
        "artifacts": [],
        "validation": {"passed": False, "reason": reason},
        "unresolved": [unresolved],
        "progress_proofs": [
            progress_event(
                run_id=str(ids.get("execution_id") or ""),
                task_id=str(ids.get("task_id") or ""),
                event_type="dispatch_refused",
                status=status,
                weight=0,
                evidence={"reason": reason},
            )
        ],
    }


def _produced(result: dict[str, Any]) -> set[str]:
    """Output names a result packet claims to have produced.

    One shape counts: a named artifact, ``{"artifacts": [{"name": "..."}]}``. An
    artifact with no name counts as nothing, otherwise one anonymous artifact would
    discharge every declared output.
    """
    return {
        str(artifact["name"])
        for artifact in result.get("artifacts") or []
        if isinstance(artifact, dict) and artifact.get("name")
    }


def _validator_for(
    declared: DeclaredCapability | None, which: str, default: Path
) -> Draft202012Validator:
    """Validator for a capability's declared schema, else the protocol default.

    A capability that declares no schema is still dispatching a task packet, so the
    protocol shape is the floor. One that declares a schema missing on disk raises:
    a check that did not happen is never reported as a pass.
    """
    path = declared.schema_path(which) if declared is not None else None
    return Draft202012Validator(load_packet_schema(path or default))


def _schema_errors(validator: Draft202012Validator, instance: Any) -> list[str]:
    return sorted(
        f"{'/'.join(str(p) for p in error.absolute_path) or '<root>'}: {error.message}"
        for error in validator.iter_errors(instance)
    )


def run_task_packet(
    packet: Any,
    registry: CapabilityRegistry | None = None,
    *,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Dispatch one task packet and return a result packet. Never raises.

    When ``registry`` is None the handlers declared in the capability config are
    bound. Pass one to dispatch against handlers registered in process.
    """
    if not isinstance(packet, dict):
        return _refusal(
            packet,
            status="blocked",
            reason=REASON_UNREADABLE,
            detail=f"A task packet must be a JSON object, got {type(packet).__name__}",
        )

    capability_id = str(packet.get("capability") or "")
    try:
        declared_all = load_declared_capabilities(config_path)
    except CapabilityConfigError as exc:
        return _refusal(
            packet, status="blocked", reason=REASON_CONFIG_UNREADABLE, detail=str(exc)
        )
    declared = declared_all.get(capability_id)

    # ---- the packet itself -------------------------------------------------
    try:
        task_validator = _validator_for(declared, "input_schema", DEFAULT_TASK_SCHEMA)
    except SchemaUnavailable as exc:
        return _refusal(
            packet,
            status="blocked",
            reason=REASON_SCHEMA_MISSING,
            detail=f"declared input_schema unreadable: {exc}",
        )

    errors = _schema_errors(task_validator, packet)
    if errors:
        declared_schema = declared.input_schema if declared else ""
        return _refusal(
            packet,
            status="blocked",
            reason=REASON_INVALID,
            detail=f"task packet fails {declared_schema or DEFAULT_TASK_SCHEMA.name}",
            errors=errors,
        )

    # ---- the capability ----------------------------------------------------
    if declared is None:
        return _refusal(
            packet,
            status="blocked",
            reason=REASON_UNDECLARED,
            detail=(
                f"'{capability_id}' is not declared in the capability config. "
                f"Declared: {', '.join(sorted(declared_all)) or 'none'}"
            ),
        )

    if registry is None:
        registry, _unbound = load_declared_registry(config_path)

    if capability_id in registry.capabilities:
        handler = registry.capabilities[capability_id]
    else:
        try:
            handler = resolve_declared_handler(declared)
        except HandlerNotDeclared as exc:
            return _refusal(
                packet, status="blocked", reason=REASON_UNBUILT, detail=str(exc)
            )
        except HandlerUnresolvable as exc:
            return _refusal(
                packet, status="blocked", reason=REASON_UNRESOLVABLE, detail=str(exc)
            )

    # ---- run it ------------------------------------------------------------
    dispatch_registry = CapabilityRegistry(capabilities={capability_id: handler})
    try:
        result = execute_task(packet, dispatch_registry)
    except HandlerResultMalformed as exc:
        return _refusal(
            packet, status="failed", reason=REASON_RESULT_MALFORMED, detail=str(exc)
        )
    # A handler fault must not crash the gateway: the caller is in another process
    # and needs a result packet, not a traceback on our stderr.
    except Exception as exc:
        return _refusal(
            packet,
            status="failed",
            reason=REASON_HANDLER_RAISED,
            detail=f"{type(exc).__name__}: {exc}",
        )

    # ---- what the capability declared it must produce ----------------------
    # Only a claimed pass is checked. A handler that reports `blocked` or `failed`
    # has already said it did not finish, and its own reason is the useful one.
    missing = sorted(set(declared.required_outputs) - _produced(result))
    if result.get("status") == "passed" and missing:
        return _refusal(
            packet,
            status="failed",
            reason=REASON_OUTPUTS_MISSING,
            detail=(
                f"'{capability_id}' declares required_outputs "
                f"{list(declared.required_outputs)}; the handler produced "
                f"{sorted(_produced(result)) or 'none'}"
            ),
            errors=missing,
        )

    result["progress_proofs"] = list(result["progress_proofs"]) + [
        progress_event(
            run_id=str(packet.get("execution_id") or ""),
            task_id=str(packet.get("task_id") or ""),
            event_type="capability_completed",
            status=str(result.get("status", "passed")),
            weight=1,
            evidence={"capability": capability_id},
        )
    ]

    # ---- the result we are about to hand back ------------------------------
    try:
        result_validator = _validator_for(declared, "output_schema", DEFAULT_RESULT_SCHEMA)
    except SchemaUnavailable as exc:
        return _refusal(
            packet,
            status="failed",
            reason=REASON_SCHEMA_MISSING,
            detail=f"declared output_schema unreadable: {exc}",
        )

    result_errors = _schema_errors(result_validator, result)
    if result_errors:
        return _refusal(
            packet,
            status="failed",
            reason=REASON_RESULT_INVALID,
            detail=f"'{capability_id}' produced a result that is not a result packet",
            errors=result_errors,
        )
    return result
