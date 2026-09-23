"""Span tracing.

"The query took 400 ms" is not a diagnosis. Spans nest the actual work —
embed, plan, recall, fuse, rerank, escalate — so a slow query names its own
culprit. Shaped like OpenTelemetry (trace id, span id, parent, attributes) so
it exports without translation, but with no dependency and no collector
required: an edge node is frequently the only thing that will ever see it.
"""
from __future__ import annotations

import contextvars
import time
from dataclasses import dataclass, field
from typing import Any, Iterator

from .ids import ulid

_current: contextvars.ContextVar["Span | None"] = contextvars.ContextVar("aegis_span", default=None)


@dataclass
class Span:
    name: str
    trace_id: str
    span_id: str = field(default_factory=lambda: ulid()[-12:])
    parent_id: str | None = None
    start: float = field(default_factory=time.perf_counter)
    end: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[tuple[float, str]] = field(default_factory=list)
    status: str = "ok"
    children: list["Span"] = field(default_factory=list)

    @property
    def ms(self) -> float:
        return ((self.end or time.perf_counter()) - self.start) * 1000

    @property
    def self_ms(self) -> float:
        """Time not accounted for by children — where the cost actually is."""
        return round(self.ms - sum(c.ms for c in self.children), 3)

    def set(self, **attributes: Any) -> "Span":
        self.attributes.update(attributes)
        return self

    def event(self, message: str) -> "Span":
        self.events.append((round(self.ms, 3), message))
        return self

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "span_id": self.span_id, "parent_id": self.parent_id,
            "ms": round(self.ms, 3), "self_ms": self.self_ms, "status": self.status,
            "attributes": self.attributes,
            "events": [{"at_ms": at, "message": m} for at, m in self.events],
            "children": [c.as_dict() for c in self.children],
        }

    def flatten(self) -> list[dict[str, Any]]:
        rows = [{"name": self.name, "ms": round(self.ms, 3), "self_ms": self.self_ms,
                 "span_id": self.span_id, "parent_id": self.parent_id}]
        for child in self.children:
            rows.extend(child.flatten())
        return rows

    def hotspot(self) -> dict[str, Any]:
        rows = self.flatten()
        return max(rows, key=lambda r: r["self_ms"]) if rows else {}


class Tracer:
    def __init__(self, keep: int = 64) -> None:
        self.completed: list[Span] = []
        self.keep = keep
        self.started = 0

    def span(self, name: str, **attributes: Any) -> "_SpanContext":
        return _SpanContext(self, name, attributes)

    def _finish(self, span: Span) -> None:
        if span.parent_id is None:
            self.completed.append(span)
            if len(self.completed) > self.keep:
                del self.completed[: len(self.completed) - self.keep]

    def recent(self, limit: int = 5) -> list[dict[str, Any]]:
        return [s.as_dict() for s in self.completed[-limit:]]

    def slowest(self, limit: int = 3) -> list[dict[str, Any]]:
        ordered = sorted(self.completed, key=lambda s: -s.ms)[:limit]
        return [{"trace": s.name, "ms": round(s.ms, 2), "hotspot": s.hotspot()} for s in ordered]

    def snapshot(self) -> dict[str, Any]:
        return {"traces_kept": len(self.completed), "started": self.started,
                "slowest": self.slowest()}


class _SpanContext:
    __slots__ = ("_tracer", "_name", "_attributes", "_span", "_token")

    def __init__(self, tracer: Tracer, name: str, attributes: dict[str, Any]) -> None:
        self._tracer = tracer
        self._name = name
        self._attributes = attributes
        self._span: Span | None = None
        self._token: Any = None

    def __enter__(self) -> Span:
        parent = _current.get()
        span = Span(
            name=self._name,
            trace_id=parent.trace_id if parent else ulid(),
            parent_id=parent.span_id if parent else None,
            attributes=dict(self._attributes),
        )
        if parent:
            parent.children.append(span)
        self._span = span
        self._token = _current.set(span)
        self._tracer.started += 1
        return span

    def __exit__(self, exc_type, exc, tb) -> None:
        span = self._span
        if span is not None:
            span.end = time.perf_counter()
            if exc_type is not None:
                span.status = f"error:{exc_type.__name__}"
            self._tracer._finish(span)
        if self._token is not None:
            _current.reset(self._token)


TRACER = Tracer()
