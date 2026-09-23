"""In-process pub/sub with bounded, drop-oldest subscriber queues.

A slow WebSocket client must never be able to stall the ingest path, so
backpressure is resolved by dropping the oldest event for that subscriber and
counting the drop, never by blocking the publisher.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator


@dataclass(slots=True)
class Event:
    channel: str
    kind: str
    payload: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)
    level: str = "info"

    def as_dict(self) -> dict[str, Any]:
        return {
            "channel": self.channel,
            "kind": self.kind,
            "ts": self.ts,
            "level": self.level,
            **self.payload,
        }


class Subscription:
    __slots__ = ("channels", "queue", "dropped", "_bus")

    def __init__(self, bus: "EventBus", channels: set[str], maxsize: int) -> None:
        self._bus = bus
        self.channels = channels
        self.queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=maxsize)
        self.dropped = 0

    def offer(self, event: Event) -> None:
        if self.channels and event.channel not in self.channels:
            return
        try:
            self.queue.put_nowait(event)
        except asyncio.QueueFull:
            try:
                self.queue.get_nowait()          # shed the oldest, keep the newest
                self.dropped += 1
                self.queue.put_nowait(event)
            except (asyncio.QueueEmpty, asyncio.QueueFull):  # pragma: no cover
                self.dropped += 1

    async def __aiter__(self) -> AsyncIterator[Event]:
        while True:
            yield await self.queue.get()

    def close(self) -> None:
        self._bus.unsubscribe(self)


class EventBus:
    CHANNELS = ("telemetry", "sync", "search", "reasoning", "alerts", "memory", "renewal", "link")

    def __init__(self, history: int = 256) -> None:
        self._subs: list[Subscription] = []
        self._history: list[Event] = []
        self._history_max = history
        self.published = 0

    def subscribe(self, channels: set[str] | None = None, maxsize: int = 256) -> Subscription:
        sub = Subscription(self, channels or set(), maxsize)
        self._subs.append(sub)
        return sub

    def unsubscribe(self, sub: Subscription) -> None:
        if sub in self._subs:
            self._subs.remove(sub)

    def publish(self, channel: str, kind: str, level: str = "info", **payload: Any) -> Event:
        event = Event(channel=channel, kind=kind, payload=payload, level=level)
        self.published += 1
        self._history.append(event)
        if len(self._history) > self._history_max:
            del self._history[: len(self._history) - self._history_max]
        for sub in list(self._subs):
            sub.offer(event)
        return event

    def replay(self, limit: int = 40) -> list[Event]:
        return self._history[-limit:]

    @property
    def subscribers(self) -> int:
        return len(self._subs)
