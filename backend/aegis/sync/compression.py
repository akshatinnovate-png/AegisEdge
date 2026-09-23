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


class WireCodec:
    """Encodes a batch of operations for transmission."""

    MIN_COMPRESS_BYTES = 256

    def __init__(self, level: int = 6) -> None:
        self.level = level
        self.stats = CodecStats()
        self.dictionary: dict[str, Any] = {}

    def encode(self, ops: list[dict[str, Any]]) -> dict[str, Any]:
        raw = json.dumps(ops, separators=(",", ":"), default=str).encode("utf-8")

        shaped: list[dict[str, Any]] = []
        for op in ops:
            body = dict(op.get("body") or {})
            if body.get("dense"):
                body["dense_q"] = quantize_vector(body.pop("dense"))     # 4x on the biggest field
            # delta against the previous op's shared fields
            delta = {}
            for key in ("collection", "sensitivity", "model_version", "device_id"):
                if key in body and self.dictionary.get(key) == body[key]:
                    delta[key] = None                                     # unchanged: send a hole
                elif key in body:
                    self.dictionary[key] = body[key]
            for key in delta:
                body.pop(key, None)
            shaped.append({**op, "body": body, "_delta": sorted(delta)})

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

    def decode(self, frame: dict[str, Any], dictionary: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        blob = bytes.fromhex(frame["payload"])
        if frame.get("z"):
            blob = zlib.decompress(blob)
        shaped = json.loads(blob.decode("utf-8"))
        context = dict(dictionary or {})
        out: list[dict[str, Any]] = []
        for op in shaped:
            body = dict(op.get("body") or {})
            for key in op.pop("_delta", []):
                if key in context:
                    body[key] = context[key]
            for key in ("collection", "sensitivity", "model_version", "device_id"):
                if key in body:
                    context[key] = body[key]
            if "dense_q" in body:
                body["dense"] = dequantize_vector(body.pop("dense_q"))
            out.append({**op, "body": body})
        return out

    def snapshot(self) -> dict[str, float]:
        return self.stats.as_dict()
