"""Scalar (int8) and binary quantization with full-precision rescoring.

Cold tier at 1 bit/dim is a 32x memory drop; the recall it costs is bought
back by over-fetching candidates and rescoring the survivors against the
full-precision vectors kept on disk.
"""
from __future__ import annotations

import numpy as np


class ScalarQuantizer:
    """Symmetric per-vector int8 quantization."""

    dtype = np.int8

    @staticmethod
    def encode(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        scale = np.abs(vectors).max(axis=1, keepdims=True)
        scale[scale == 0] = 1.0
        codes = np.clip(np.round(vectors / scale * 127.0), -127, 127).astype(np.int8)
        return codes, scale.astype(np.float32)

    @staticmethod
    def decode(codes: np.ndarray, scale: np.ndarray) -> np.ndarray:
        return (codes.astype(np.float32) / 127.0) * scale


class BinaryQuantizer:
    """1 bit per dimension; similarity via packed Hamming distance."""

    @staticmethod
    def encode(vectors: np.ndarray) -> np.ndarray:
        return np.packbits(vectors > 0, axis=1)

    @staticmethod
    def similarity(query_bits: np.ndarray, codes: np.ndarray, dim: int) -> np.ndarray:
        if codes.size == 0:
            return np.zeros(0, dtype=np.float32)
        xor = np.bitwise_xor(codes, query_bits)
        hamming = np.unpackbits(xor, axis=1)[:, :dim].sum(axis=1)
        # map [0, dim] Hamming onto a [-1, 1] cosine-like score
        return (1.0 - 2.0 * hamming / dim).astype(np.float32)


def compression_ratio(dim: int, tier: str) -> float:
    full = dim * 4
    if tier == "warm":
        return full / (dim + 4)
    if tier == "cold":
        return full / (dim / 8)
    return 1.0
