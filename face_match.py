from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable
import numpy as np

@dataclass(frozen=True)
class MatchResult:
    matched: bool
    similarity: float
    distance: float
    threshold: float

def normalize(vector: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(vector), dtype=np.float32)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("Embedding must be a non-empty 1D array")
    norm = np.linalg.norm(arr)
    if not np.isfinite(norm) or norm <= 1e-8:
        raise ValueError("Invalid embedding")
    return arr / norm

def cosine_similarity(a: Iterable[float], b: Iterable[float]) -> float:
    av = normalize(a)
    bv = normalize(b)
    if av.shape != bv.shape:
        raise ValueError(f"Embedding dimensions differ: {av.size} vs {bv.size}")
    return float(np.clip(np.dot(av, bv), -1.0, 1.0))

def compare_embeddings(a: Iterable[float], b: Iterable[float], threshold: float = 0.42) -> MatchResult:
    similarity = cosine_similarity(a, b)
    return MatchResult(
        matched=similarity >= threshold,
        similarity=similarity,
        distance=1.0 - similarity,
        threshold=threshold,
    )
