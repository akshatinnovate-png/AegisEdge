"""Declarative policy engine.

Governance cannot be scattered through the code paths that happen to touch
data; it is one hot-reloadable document, evaluated on the ingest path, and
every decision it makes is recorded in the audit chain.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..memory.schema import MemoryPoint, Sensitivity, SyncClass

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

DEFAULT_POLICY: dict[str, Any] = {
    "version": 0,
    "default": {"sync_class": "sync_full", "ttl_days": 90, "redact": False},
    "rules": [
        {"when": {"sensitivity": "restricted"},
         "then": {"sync_class": "local_only", "ttl_days": 30,
                  "reason": "restricted-class memory may never leave the device"},
         "name": "restricted-stays-home"},
    ],
    "egress": {"max_points_per_batch": 256, "strip_payload_keys": []},
}


@dataclass(slots=True)
class Decision:
    sync_class: SyncClass
    ttl_s: float | None
    redact: bool
    pin: bool
    rule: str
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {"sync_class": self.sync_class.value, "ttl_s": self.ttl_s, "redact": self.redact,
                "pin": self.pin, "rule": self.rule, "reason": self.reason}


class PolicyEngine:
    @staticmethod
    def resolve(path: str | Path) -> Path:
        """Find the policy file regardless of the working directory.

        A relative policy path that fails to resolve used to fall back to the
        built-in default *silently* — so a node started from the wrong
        directory would quietly run without the operator's governance rules,
        which is the one failure this subsystem must never have.
        """
        candidate = Path(path)
        if candidate.is_absolute() or candidate.exists():
            return candidate
        package_root = Path(__file__).resolve().parents[2]      # backend/
        for base in (Path.cwd(), package_root, package_root.parent):
            resolved = base / candidate
            if resolved.exists():
                return resolved
        return candidate

    def __init__(self, path: str | Path) -> None:
        self.path = self.resolve(path)
        self.document: dict[str, Any] = DEFAULT_POLICY
        self.loaded_at = 0.0
        self.mtime = 0.0
        self.evaluations = 0
        self.denied_egress = 0
        self.using_default = True
        self.reload()

    # -- lifecycle --------------------------------------------------------

    def reload(self) -> bool:
        if yaml is None or not self.path.exists():
            self.document = DEFAULT_POLICY
            self.loaded_at = time.time()
            self.using_default = True
            return False
        mtime = self.path.stat().st_mtime
        if mtime == self.mtime:
            return False
        try:
            document = yaml.safe_load(self.path.read_text(encoding="utf-8"))
            if isinstance(document, dict) and document.get("rules") is not None:
                self.document = document
                self.mtime = mtime
                self.loaded_at = time.time()
                self.using_default = False
                return True
        except Exception:
            pass                       # a broken policy file keeps the last good one
        return False

    # -- evaluation -------------------------------------------------------

    def _matches(self, when: dict[str, Any], point: MemoryPoint) -> bool:
        for key, expected in (when or {}).items():
            if key == "sensitivity":
                if point.sensitivity.value != expected:
                    return False
            elif key == "collection":
                if point.collection != expected:
                    return False
            elif key == "min_confidence":
                if point.confidence < float(expected):
                    return False
            elif key == "source_prefix":
                if not (point.source or "").startswith(str(expected)):
                    return False
            else:
                if point.payload.get(key) != expected:
                    return False
        return True

    def evaluate(self, point: MemoryPoint) -> Decision:
        self.evaluations += 1
        base = dict(self.document.get("default", {}))
        rule_name = "default"
        for rule in self.document.get("rules", []):
            if self._matches(rule.get("when", {}), point):
                base.update(rule.get("then", {}))
                rule_name = rule.get("name", "unnamed")
                break
        ttl_days = base.get("ttl_days")
        return Decision(
            sync_class=SyncClass(base.get("sync_class", "sync_full")),
            ttl_s=float(ttl_days) * 86400 if ttl_days else None,
            redact=bool(base.get("redact", False)),
            pin=bool(base.get("pin", False)),
            rule=rule_name,
            reason=str(base.get("reason", "default posture")),
        )

    def may_egress(self, point: MemoryPoint) -> bool:
        allowed = point.sync_class in {SyncClass.FULL, SyncClass.REDACTED, SyncClass.METADATA_ONLY}
        if not allowed or point.sensitivity is Sensitivity.RESTRICTED:
            self.denied_egress += 1
            return False
        return True

    @property
    def egress_batch(self) -> int:
        return int(self.document.get("egress", {}).get("max_points_per_batch", 256))

    @property
    def strip_keys(self) -> list[str]:
        return list(self.document.get("egress", {}).get("strip_payload_keys", []))

    def snapshot(self) -> dict[str, Any]:
        return {
            "version": self.document.get("version"),
            "rules": [r.get("name") for r in self.document.get("rules", [])],
            "evaluations": self.evaluations,
            "denied_egress": self.denied_egress,
            "loaded_at": self.loaded_at,
            "file": str(self.path),
            "using_built_in_default": self.using_default,
            "warning": (None if not self.using_default else
                        f"policy file not found at {self.path} — running the built-in "
                        f"default, which is more restrictive but is not your policy"),
        }
