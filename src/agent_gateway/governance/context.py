from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ContextLease:
    lease_id: str
    purpose: str
    permitted_paths: tuple[str, ...]
    permitted_symbols: tuple[str, ...] = ()
    maximum_reads: int = 8
    maximum_bytes: int = 120_000


@dataclass(frozen=True)
class ContextCapsule:
    capsule_id: str
    target_model: str
    purpose: dict[str, str]
    content: dict[str, Any]
    maximum_fresh_tokens: int
    maximum_total_tokens: int
    dependency_hash: str

    def canonical_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    def content_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def assert_path_allowed(path: str, lease: ContextLease) -> None:
    if path not in lease.permitted_paths:
        raise PermissionError(f"Path outside context lease: {path}")
