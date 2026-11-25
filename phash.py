"""Lightweight DCT perceptual hash (64-bit) for near-dup prefilter."""

from __future__ import annotations

import cv2
import numpy as np


def dct_phash(bgr: np.ndarray, hash_size: int = 8) -> int:
    """64-bit DCT pHash (imagehash-compatible idea; brightness-robust via median)."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (hash_size * 4, hash_size * 4), interpolation=cv2.INTER_AREA)
    f = np.float32(small)
    dct = cv2.dct(f)
    block = dct[:hash_size, :hash_size].copy()
    # exclude DC for brightness invariance
    flat = block.flatten()[1:]
    med = float(np.median(flat))
    bits = (block.flatten() > med).astype(np.uint8)
    bits[0] = 0
    out = 0
    for b in bits:
        out = (out << 1) | int(b)
    return int(out)


def hamming(a: int, b: int) -> int:
    return int((a ^ b).bit_count())


def frame_phashes(frames_bgr: list, every: int = 1) -> list[int]:
    return [dct_phash(f) for i, f in enumerate(frames_bgr) if i % every == 0]


def coverage(a: list[int], b: list[int], max_dist: int = 8) -> float:
    """Fraction of hashes in a that have a near match in b."""
    if not a:
        return 0.0
    hits = 0
    for ha in a:
        if any(hamming(ha, hb) <= max_dist for hb in b):
            hits += 1
    return hits / len(a)
