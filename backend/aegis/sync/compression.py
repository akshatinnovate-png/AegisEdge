"""Wire codec.

The link is the scarce resource, not the CPU. Operations go out delta-encoded
against a shared dictionary of recent values, integers as varints, and vectors
as int8 codes with a per-vector scale — a 384-dim float32 embedding drops from
1536 bytes to 388, before any general-purpose compressor sees it.

zlib then runs over the result when it actually helps; a payload that does not
compress is sent raw with a flag rather than paying for a wasted pass.
"""
from __future__ import annotations

import json
import zlib

from .identity import jsonable
from dataclasses import dataclass, field
from typing import Any

import numpy as np


def varint(value: int) -> bytes:
    """LEB128. Op cursors and counters are small; 8 bytes each is waste."""
    if value < 0:
        raise ValueError("varint is unsigned")
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        out.append(byte | (0x80 if value else 0))
        if not value:
            return bytes(out)


def read_varint(data: bytes, offset: int = 0) -> tuple[int, int]:
    result = shift = 0
    while True:
        byte = data[offset]
        offset += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, offset
        shift += 7


def quantize_vector(vector: list[float] | np.ndarray) -> dict[str, Any]:
    array = np.asarray(vector, dtype=np.float32)
    if not array.size:
        return {"scale": 0.0, "codes": ""}
    scale = float(np.abs(array).max()) or 1.0
    codes = np.clip(np.round(array / scale * 127.0), -127, 127).astype(np.int8)
    return {"scale": round(scale, 6), "codes": codes.tobytes().hex()}


def dequantize_vector(payload: dict[str, Any]) -> list[float]:
    if not payload.get("codes"):
        return []
    codes = np.frombuffer(bytes.fromhex(payload["codes"]), dtype=np.int8)
    return (codes.astype(np.float32) / 127.0 * float(payload["scale"])).tolist()


@dataclass
class CodecStats:
    encoded: int = 0
    raw_bytes: int = 0
    wire_bytes: int = 0
    compressed_frames: int = 0
    skipped_compression: int = 0

    def as_dict(self) -> dict[str, float]:
        ratio = (self.raw_bytes / self.wire_bytes) if self.wire_bytes else 1.0
        return {"frames": self.encoded, "raw_bytes": self.raw_bytes, "wire_bytes": self.wire_bytes,
                "ratio": round(ratio, 2), "saved_bytes": max(0, self.raw_bytes - self.wire_bytes),
                "compressed_frames": self.compressed_frames,
                "skipped_compression": self.skipped_compression}


_ABSENT = object()


class WireCodec:
    """Encodes a batch of operations for transmission.

    It compresses; it does not edit. That distinction is load-bearing and was
    learned the hard way twice. This codec used to quantize an operation's
    dense vector on the way out and dequantize it on the way in — a lossy,
    non-round-tripping transform applied to content that is *signed*. The
    vector that was signed was therefore never the vector that arrived, so
    every real upsert carrying a vector failed verification at the receiver
    and was refused as a forgery. Nothing caught it, because no test had ever
    signed a body with a vector in it.

    The quantizing now happens where the operation is built, which is the only
    place it can happen without the signature and the wire disagreeing. The
    wire is the same size as before; what changed is that what a peer checks
    is what the author signed.

    The other lesson is the frame-local delta context, below.
    """

    MIN_COMPRESS_BYTES = 256

    # Fields worth eliding: low-cardinality labels that repeat across every
    # operation in a batch. `sensitivity` is on this list, which is why the
    # bug below mattered as much as it did.
    DELTA_FIELDS = ("collection", "sensitivity", "model_version", "device_id")

    def __init__(self, level: int = 6) -> None:
        self.level = level
        self.stats = CodecStats()
        self.elided = 0

    def encode(self, ops: list[dict[str, Any]]) -> dict[str, Any]:
        # Normalised first: an operation body may hold the point's float32
        # array rather than a copy of it as a list, and neither the raw-size
        # accounting nor the frame can carry an array.
        ops = [jsonable(op) for op in ops]
        raw = json.dumps(ops, separators=(",", ":"), default=str).encode("utf-8")

        # The delta context is rebuilt for every frame and never survives one.
        #
        # It used to live on the encoder and persist across calls, while the
        # decoder started from nothing on each frame. The result was silent
        # data loss: the first operation carrying `sensitivity: restricted`
        # sent the label, every later one sent a hole, and the receiver — with
        # no context to fill the hole from — stored the operation with no
        # sensitivity label at all.
        #
        # That is worse than a compression bug. The receiving node decides
        # whether it may hold an operation by reading exactly that label, so an
        # operation the policy would have refused arrived looking ordinary and
        # was accepted. The deterministic simulator surfaced it as a signature
        # mismatch — the body that was signed was not the body that arrived —
        # which is the whole argument for signing the thing.
        #
        # Cross-frame state was never sound here anyway: this transport drops,
        # reorders and duplicates frames, so "the value from the previous
        # frame" is not a thing a receiver can know. A frame that carries its
        # own context is self-describing, and the redundancy that pays for
        # compression is within a batch, not between batches.
        shaped: list[dict[str, Any]] = []
        context: dict[str, Any] = {}
        for op in ops:
            body = dict(op.get("body") or {})
            delta = []
            for key in self.DELTA_FIELDS:
                if key not in body:
                    continue
                if context.get(key, _ABSENT) == body[key]:
                    delta.append(key)                                     # already in this frame
                else:
                    context[key] = body[key]
            for key in delta:
                body.pop(key, None)
            self.elided += len(delta)
            shaped.append({**op, "body": body, "_delta": delta})

        payload = json.dumps(shaped, separators=(",", ":"), default=str).encode("utf-8")
        compressed = zlib.compress(payload, self.level) if len(payload) >= self.MIN_COMPRESS_BYTES else payload
        used_compression = len(compressed) < len(payload)
        frame = compressed if used_compression else payload

        self.stats.encoded += 1
        self.stats.raw_bytes += len(raw)
        self.stats.wire_bytes += len(frame)
        self.stats.compressed_frames += int(used_compression)
        self.stats.skipped_compression += int(not used_compression)
        return {"z": used_compression, "n": len(ops), "payload": frame.hex(),
                "raw_bytes": len(raw), "wire_bytes": len(frame)}

    def decode(self, frame: dict[str, Any]) -> list[dict[str, Any]]:
        """Reconstruct a frame using only what the frame itself carries."""
        blob = bytes.fromhex(frame["payload"])
        if frame.get("z"):
            blob = zlib.decompress(blob)
        shaped = json.loads(blob.decode("utf-8"))
        context: dict[str, Any] = {}
        out: list[dict[str, Any]] = []
        for op in shaped:
            body = dict(op.get("body") or {})
            for key in op.pop("_delta", []):
                if key in context:
                    body[key] = context[key]
                else:
                    # A hole with nothing to fill it from. Under the old
                    # cross-frame scheme this happened constantly and passed
                    # silently; now it cannot happen for a well-formed frame,
                    # so if it ever does the frame is corrupt and saying so is
                    # better than handing back an operation missing a label
                    # that access decisions are made on.
                    raise ValueError(
                        f"frame elides {key!r} with no value earlier in the same frame")
            for key in self.DELTA_FIELDS:
                if key in body:
                    context[key] = body[key]
            out.append({**op, "body": body})
        return out

    def snapshot(self) -> dict[str, float]:
        return {**self.stats.as_dict(), "fields_elided": self.elided}
