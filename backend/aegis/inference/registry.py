"""Content-addressed local model registry.

A model is loaded only if its bytes hash to the digest recorded for that
version. Rollback is a pointer swap, not a redeploy — which matters when the
device is on a mast in a field.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.errors import ModelUnavailable


@dataclass(slots=True)
class ModelEntry:
    name: str
    version: str
    variant: str                       # fp32 | fp16 | int8-dynamic | int8-static
    path: str | None = None
    sha256: str | None = None
    dim: int = 384
    loaded: bool = False
    promoted_at: float = field(default_factory=time.time)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.name}@{self.version}/{self.variant}"


class ModelRegistry:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.entries: dict[str, ModelEntry] = {}
        self.active: dict[str, str] = {}      # role -> entry key
        self.history: list[tuple[float, str, str]] = []

    def register(self, role: str, entry: ModelEntry, activate: bool = True) -> ModelEntry:
        self.entries[entry.key] = entry
        if activate:
            self.activate(role, entry.key)
        return entry

    def activate(self, role: str, key: str) -> ModelEntry:
        if key not in self.entries:
            raise ModelUnavailable(f"unknown model {key}")
        self.active[role] = key
        self.history.append((time.time(), role, key))
        return self.entries[key]

    def rollback(self, role: str) -> ModelEntry | None:
        prior = [k for _, r, k in reversed(self.history[:-1]) if r == role]
        if not prior:
            return None
        return self.activate(role, prior[0])

    def get(self, role: str) -> ModelEntry:
        key = self.active.get(role)
        if key is None:
            raise ModelUnavailable(f"no active model for role '{role}'")
        return self.entries[key]

    @staticmethod
    def digest(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    def verify(self, entry: ModelEntry) -> bool:
        """Refuse to load a graph whose bytes do not match the recorded digest."""
        if not entry.path or not entry.sha256:
            return False
        path = Path(entry.path)
        if not path.exists():
            return False
        return self.digest(path) == entry.sha256

    def snapshot(self) -> dict[str, Any]:
        return {
            "active": dict(self.active),
            "entries": [
                {"key": e.key, "variant": e.variant, "loaded": e.loaded, "dim": e.dim}
                for e in self.entries.values()
            ],
            "switches": len(self.history),
        }
