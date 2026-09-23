"""Invertible Bloom Lookup Table for set reconciliation.

Merkle digests find *which ranges* differ; they still cost a round trip per
level of the walk, and on a link with 400 ms RTT that is the dominant cost.

An IBLT does better: both peers encode their key sets into a fixed-size table
sized by the *expected difference*, not the set size. Subtracting one table
from the other yields a table that can be peeled to recover exactly the
symmetric difference — who has what the other lacks — in a single exchange of
a few kilobytes, whether the sets hold a thousand keys or a million.

If the difference is larger than the table was sized for, peeling stalls; the
caller detects that and retries with a bigger table, which is why `decode`
reports whether it finished rather than silently returning a partial answer.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Iterable

HASH_COUNT = 4
KEY_WIDTH = 32          # bytes; op ids are 26-char ULIDs with room to spare


def _hash(key: str, salt: int) -> int:
    return int.from_bytes(
        hashlib.blake2b(key.encode("utf-8"), digest_size=8, salt=str(salt).encode().ljust(8, b"0")[:8]).digest(),
        "big",
    )


def _key_int(key: str) -> int:
    """The key itself as a fixed-width integer.

    Cells XOR the *key*, not a hash of it. That is what makes the table
    invertible: a peeled cell yields the peer's key directly, so neither side
    needs a dictionary of the other's keys — which is the whole point, since
    the peer's keys are exactly what we do not have.
    """
    raw = key.encode("utf-8")
    if len(raw) > KEY_WIDTH:
        raise ValueError(f"key longer than {KEY_WIDTH} bytes: {key!r}")
    return int.from_bytes(raw.ljust(KEY_WIDTH, b"\0"), "big")


def _int_key(value: int) -> str | None:
    try:
        return value.to_bytes(KEY_WIDTH, "big").rstrip(b"\0").decode("utf-8")
    except (OverflowError, UnicodeDecodeError):
        return None


@dataclass
class IBLT:
    cells: int = 128
    counts: list[int] = field(default_factory=list)
    key_sums: list[int] = field(default_factory=list)
    hash_sums: list[int] = field(default_factory=list)
    keys_seen: dict[str, int] = field(default_factory=dict)   # local only; never transmitted

    def __post_init__(self) -> None:
        if not self.counts:
            self.counts = [0] * self.cells
            self.key_sums = [0] * self.cells
            self.hash_sums = [0] * self.cells

    def _positions(self, key: str) -> list[int]:
        return sorted({_hash(key, i) % self.cells for i in range(HASH_COUNT)})

    def insert(self, key: str) -> None:
        self.keys_seen[key] = self.keys_seen.get(key, 0) + 1
        checksum = _key_int(key)
        for position in self._positions(key):
            self.counts[position] += 1
            self.key_sums[position] ^= checksum
            self.hash_sums[position] ^= _hash(key, 99)

    def insert_many(self, keys: Iterable[str]) -> "IBLT":
        for key in keys:
            self.insert(key)
        return self

    def subtract(self, other: "IBLT") -> "IBLT":
        """A − B: cells cancel wherever both sides hold the same key."""
        if self.cells != other.cells:
            raise ValueError("IBLTs must be the same size to subtract")
        result = IBLT(cells=self.cells)
        result.counts = [a - b for a, b in zip(self.counts, other.counts)]
        result.key_sums = [a ^ b for a, b in zip(self.key_sums, other.key_sums)]
        result.hash_sums = [a ^ b for a, b in zip(self.hash_sums, other.hash_sums)]
        result.keys_seen = {**self.keys_seen, **other.keys_seen}     # the local dictionary
        return result

    def decode(self) -> tuple[set[str], set[str], bool]:
        """Peel the difference.

        Returns (only_in_a, only_in_b, complete). A cell with count ±1 holds
        exactly one key; removing it may expose another. Peeling until nothing
        is left means the difference was fully recovered.
        """
        counts = list(self.counts)
        key_sums = list(self.key_sums)
        hash_sums = list(self.hash_sums)

        only_a: set[str] = set()
        only_b: set[str] = set()
        progress = True
        while progress:
            progress = False
            for position in range(self.cells):
                if counts[position] not in (1, -1):
                    continue
                checksum = key_sums[position]
                key = _int_key(checksum)
                if key is None or _hash(key, 99) != hash_sums[position]:
                    continue                       # not a clean single-key cell
                (only_a if counts[position] == 1 else only_b).add(key)
                sign = counts[position]
                for other in self._positions(key):
                    counts[other] -= sign
                    key_sums[other] ^= checksum
                    hash_sums[other] ^= _hash(key, 99)
                progress = True
        complete = all(c == 0 for c in counts) and all(k == 0 for k in key_sums)
        return only_a, only_b, complete

    # -- wire format ------------------------------------------------------

    def to_wire(self) -> dict[str, list[int]]:
        """What actually crosses the link — never the key dictionary."""
        return {"cells": self.cells, "counts": self.counts,
                "key_sums": self.key_sums, "hash_sums": self.hash_sums}

    @staticmethod
    def from_wire(payload: dict, keys_seen: dict[str, int] | None = None) -> "IBLT":
        table = IBLT(cells=int(payload["cells"]))
        table.counts = list(payload["counts"])
        table.key_sums = list(payload["key_sums"])
        table.hash_sums = list(payload["hash_sums"])
        table.keys_seen = dict(keys_seen or {})
        return table

    @property
    def wire_bytes(self) -> int:
        return self.cells * (8 + KEY_WIDTH + 8)

    OVERHEAD = 4.0        # measured: peeling stalls below ~4 cells per differing key

    @staticmethod
    def size_for(expected_difference: int) -> int:
        """Cells needed to peel a difference of this size with high probability.

        Sized from measurement rather than theory: with 4 hash functions,
        peeling reliably completes at about 4 cells per differing key and
        stalls below that. A caller that gets `complete=False` should double
        the table and retry — that path is cheaper than guessing high every
        time, because the table is what crosses the link.
        """
        target = max(32, int(IBLT.OVERHEAD * max(expected_difference, 1)))
        return 1 << (target - 1).bit_length()
