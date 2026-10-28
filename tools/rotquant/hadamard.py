"""128-point normalized Sylvester Hadamard rotation on the input (K) side.

A matrix W with GGUF dimensions ne0 = K (rows of K contiguous values, NumPy
shape (..., N, K)) is stored as W' = W·D·H, where D is a ±1 diagonal sign
vector and H is block diagonal with K / 128 normalized Hadamard blocks.
At run time the activation becomes x' = H·D·x, so W'·x' = W·D·H·H·D·x = W·x
because H·H = I and D·D = I.
"""
from __future__ import annotations

import zlib

import numpy as np

BLOCK_SIZE = 128


def sylvester(n: int = BLOCK_SIZE, dtype=np.float64) -> np.ndarray:
    """Normalized Sylvester Hadamard matrix of order n (a power of two)."""
    if n < 1 or n & (n - 1):
        raise ValueError(f"Hadamard order must be a power of two, got {n}")
    h = np.ones((1, 1), dtype=np.float64)
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return (h / np.sqrt(n)).astype(dtype)


_H32 = sylvester(BLOCK_SIZE, np.float32)
_H64 = sylvester(BLOCK_SIZE, np.float64)


def signs(seed: int, instance: str, k: int) -> np.ndarray:
    """Deterministic ±1 sign vector for one input-group instance.

    The vector is stored in the output file, so this derivation only has to
    be reproducible for the converter itself, not for readers.
    """
    if k % BLOCK_SIZE:
        raise ValueError(f"input dimension {k} is not a multiple of {BLOCK_SIZE}")
    sequence = np.random.SeedSequence([int(seed), zlib.crc32(instance.encode())])
    bits = np.random.Generator(np.random.PCG64(sequence)).integers(0, 2, size=k)
    return (bits * 2 - 1).astype(np.float32)


def rotate_last_axis(values: np.ndarray, sign: np.ndarray) -> np.ndarray:
    """Return values·D·H along the last axis.

    For a weight row this is W' = W·D·H. For an activation row vector the
    same expression is x'ᵀ = xᵀ·D·H, i.e. x' = H·D·x since H is symmetric,
    so one function serves both sides.
    """
    k = values.shape[-1]
    if k % BLOCK_SIZE:
        raise ValueError(f"input dimension {k} is not a multiple of {BLOCK_SIZE}")
    if sign.shape != (k,):
        raise ValueError(f"sign vector has shape {sign.shape}, expected ({k},)")
    h = _H64 if values.dtype == np.float64 else _H32
    work = values.astype(h.dtype, copy=False) * sign.astype(h.dtype)
    blocks = work.reshape(*values.shape[:-1], k // BLOCK_SIZE, BLOCK_SIZE)
    return (blocks @ h).reshape(values.shape)


def f32_to_bf16_bits(values: np.ndarray) -> np.ndarray:
    """Round float32 to BF16 bits with round-to-nearest-even (finite input)."""
    values = np.ascontiguousarray(values, dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("rotated tensor contains non-finite values")
    bits = values.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return (rounded >> np.uint32(16)).astype("<u2")


def bf16_bits_to_f32(bits: np.ndarray) -> np.ndarray:
    return (np.asarray(bits, dtype="<u2").astype(np.uint32) << np.uint32(16)).view(np.float32)
