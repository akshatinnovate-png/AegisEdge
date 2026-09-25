"""Monotonic, lexicographically sortable identifiers (ULID-shaped).

These draw their timestamp and their randomness from the ambient environment
rather than from `time` and `os.urandom` directly, which is not a stylistic
preference. An operation id is what the IBLT hashes, what `sorted()` orders a
fetch by, and what a replayed history names its operations with. While ids came
from the wall clock, two sweeps over the same seeds returned 454 and 456
failures — close enough to look like noise, and a direct contradiction of the
claim that an execution is a pure function of its seed.

The per-millisecond counter state is module-level, so it also has to be reset
when a simulation starts. `determinism.simulated()` does that.
"""
from __future__ import annotations

import threading

from . import determinism

_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
_lock = threading.Lock()
_last_ms = 0
_last_rand = 0


def _encode(value: int, length: int) -> str:
    out = []
    for _ in range(length):
        out.append(_ALPHABET[value & 0x1F])
        value >>= 5
    return "".join(reversed(out))


def ulid() -> str:
    """Time-ordered 26-char id. Two calls in the same millisecond still sort."""
    global _last_ms, _last_rand
    with _lock:
        ms = int(determinism.now() * 1000)
        if ms == _last_ms:
            _last_rand += 1
        else:
            _last_ms = ms
            _last_rand = determinism.rng().getrandbits(80)
        return _encode(ms, 10) + _encode(_last_rand & ((1 << 80) - 1), 16)


def _reset() -> None:
    """Forget the per-millisecond counter. Called when a simulation begins."""
    global _last_ms, _last_rand
    with _lock:
        _last_ms, _last_rand = 0, 0


def short_id(prefix: str = "pt") -> str:
    return f"{prefix}-{ulid()[-10:].lower()}"
