"""A swappable world: the clock and the coin, in one place.

Distributed systems fail on timing and ordering, and those are exactly the
things an ordinary test cannot control. A partition that only matters when it
lands between the digest and the fetch, a clock that steps backwards during a
merge, two peers pulling the same operation in the other order — these are
found by luck, reproduced never, and fixed by argument.

Unless the system has no independent access to time or randomness. If every
`now()` and every coin flip comes from one place, that place can be replaced
with a seeded, virtual one, and then an entire distributed execution becomes a
pure function of its seed. A failure found at seed 8,271,443 is a failure that
can be reproduced forever, shrunk to its shortest form, fixed, and kept fixed.

This is how FoundationDB, TigerBeetle and Antithesis test, and it is the
reason their bug reports are seeds rather than stories.

The cost is a discipline: inside the simulated layers, `time.time()` and the
`random` module are not to be called directly. `ENV.now()` and `ENV.random`
instead. `scripts/audit_determinism.py` checks that the rule holds, because a
single direct call is enough to make a run irreproducible and the failure mode
is silent.
"""
from __future__ import annotations

import asyncio
import contextlib
import random as _random
import threading
import time as _time
from typing import Any, Iterator


class Environment:
    """The real world: the system clock and an unseeded generator."""

    name = "real"
    deterministic = False

    def __init__(self) -> None:
        self.random = _random.Random()

    def now(self) -> float:
        """Wall-clock seconds since the epoch."""
        return _time.time()

    def monotonic(self) -> float:
        """A clock that never goes backwards, for measuring durations."""
        return _time.perf_counter()

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(seconds)

    def snapshot(self) -> dict[str, Any]:
        return {"environment": self.name, "deterministic": self.deterministic}


class VirtualEnvironment(Environment):
    """A seeded world where time only moves when something asks it to.

    Sleeping does not wait; it advances the clock. A simulated hour costs
    nothing, which is what makes a hundred thousand simulated days fit inside
    ninety seconds of real time.
    """

    name = "virtual"
    deterministic = True

    def __init__(self, seed: int, start: float = 1_700_000_000.0) -> None:
        self.seed = int(seed)
        self.random = _random.Random(self.seed)
        self._now = float(start)
        self._start = float(start)
        self.sleeps = 0
        self.advanced_s = 0.0
        self._lock = threading.Lock()

    def now(self) -> float:
        return self._now

    def monotonic(self) -> float:
        return self._now - self._start

    def advance(self, seconds: float) -> float:
        """Move the world forward. The only way time passes here."""
        with self._lock:
            self._now += max(float(seconds), 0.0)
            self.advanced_s += max(float(seconds), 0.0)
        return self._now

    def skew(self, seconds: float) -> float:
        """Step the clock, in either direction.

        Deliberately allowed to go backwards: NTP does it, and a system whose
        correctness depends on it never happening is a system that has not
        been asked.
        """
        with self._lock:
            self._now += float(seconds)
        return self._now

    async def sleep(self, seconds: float) -> None:
        self.sleeps += 1
        self.advance(seconds)
        # Yield so other simulated tasks interleave, without real waiting.
        await asyncio.sleep(0)

    def snapshot(self) -> dict[str, Any]:
        return {"environment": self.name, "deterministic": True, "seed": self.seed,
                "virtual_now": self._now, "simulated_seconds": round(self.advanced_s, 3),
                "sleeps": self.sleeps}


# The ambient environment. Real unless a simulation replaces it.
ENV: Environment = Environment()


def current() -> Environment:
    return ENV


@contextlib.contextmanager
def simulated(seed: int, start: float = 1_700_000_000.0) -> Iterator[VirtualEnvironment]:
    """Run a block inside a seeded virtual world, then restore reality."""
    global ENV
    previous = ENV
    world = VirtualEnvironment(seed, start)
    ENV = world
    # Identifier state is module-level and would otherwise carry across
    # simulations, making the second run of a seed differ from the first.
    # Imported here rather than at the top because `ids` imports this module.
    from . import ids
    ids._reset()
    try:
        yield world
    finally:
        ENV = previous
        ids._reset()


def now() -> float:
    return ENV.now()


def monotonic() -> float:
    return ENV.monotonic()


def rng() -> _random.Random:
    return ENV.random


async def sleep(seconds: float) -> None:
    await ENV.sleep(seconds)
