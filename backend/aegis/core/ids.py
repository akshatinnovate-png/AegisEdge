"""Monotonic, lexicographically sortable identifiers (ULID-shaped)."""
from __future__ import annotations

import os
import threading
import time

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
        ms = int(time.time() * 1000)
        if ms == _last_ms:
            _last_rand += 1
        else:
            _last_ms = ms
            _last_rand = int.from_bytes(os.urandom(10), "big")
        return _encode(ms, 10) + _encode(_last_rand & ((1 << 80) - 1), 16)


def short_id(prefix: str = "pt") -> str:
    return f"{prefix}-{ulid()[-10:].lower()}"
