from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any


class CacheState(str, Enum):
    FRESH = "FRESH"
    VALIDATED = "VALIDATED"
    STALE = "STALE"
    SUSPECT = "SUSPECT"
    INVALIDATED = "INVALIDATED"
    QUARANTINED = "QUARANTINED"


@dataclass(frozen=True)
class CacheEntry:
    key: str
    state: CacheState
    dependencies: dict[str, str]
    value_ref: str


def make_cache_key(namespace: str, dependencies: dict[str, Any]) -> str:
    raw = json.dumps(dependencies, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return f"{namespace}:{digest}"


def dependencies_match(entry: CacheEntry, current: dict[str, str]) -> bool:
    return entry.dependencies == current and entry.state == CacheState.VALIDATED
