"""Error, similarity, and runtime metrics for comparing spectrograms.

All two-argument metrics accept complex or real arrays of the same shape.
Magnitude-based metrics compare ``|A|`` to ``|B|``; phase-based metrics
compare wrapped ``angle(A) - angle(B)``.
"""

from __future__ import annotations

import time
import tracemalloc
from typing import Any, Callable

import numpy as np


# ---------------------------------------------------------------------------
# Scalar error metrics
# ---------------------------------------------------------------------------

def _abs(A: np.ndarray) -> np.ndarray:
    return np.abs(A)


def mse(A: np.ndarray, B: np.ndarray) -> float:
    """Mean squared error on magnitudes."""
    d = _abs(A) - _abs(B)
    return float(np.mean(d * d))


def mae(A: np.ndarray, B: np.ndarray) -> float:
    """Mean absolute error on magnitudes."""
    return float(np.mean(np.abs(_abs(A) - _abs(B))))


def frobenius(A: np.ndarray, B: np.ndarray) -> float:
    """Frobenius norm of the complex spectrogram difference."""
    return float(np.linalg.norm(A - B))


def cosine_similarity_flat(A: np.ndarray, B: np.ndarray) -> float:
    """Cosine similarity of flattened magnitude spectrograms."""
    a = _abs(A).ravel()
    b = _abs(B).ravel()
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def pearson_flat(A: np.ndarray, B: np.ndarray) -> float:
    """Pearson correlation of flattened magnitude spectrograms."""
    a = _abs(A).ravel()
    b = _abs(B).ravel()
    if a.std() == 0.0 or b.std() == 0.0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


# ---------------------------------------------------------------------------
# Phase + energy
# ---------------------------------------------------------------------------

def phase_error_stats(A: np.ndarray, B: np.ndarray, mag_threshold: float = 1e-8) -> dict[str, float]:
    """Wrapped phase-error stats, computed only where both magnitudes exceed
    ``mag_threshold`` (phases of near-zero bins are meaningless)."""
    mask = (_abs(A) > mag_threshold) & (_abs(B) > mag_threshold)
    if not mask.any():
        return {"mean": 0.0, "median": 0.0, "std": 0.0, "n_valid": 0}
    diff = np.angle(A[mask]) - np.angle(B[mask])
    diff = np.mod(diff + np.pi, 2 * np.pi) - np.pi  # wrap to (-pi, pi]
    return {
        "mean": float(np.mean(diff)),
        "median": float(np.median(diff)),
        "std": float(np.std(diff)),
        "n_valid": int(mask.sum()),
    }


def energy_preservation_error(x: np.ndarray, Z: np.ndarray) -> float:
    """Absolute relative error between time-domain energy and total spectrogram
    energy (per-frame sum of squared magnitudes, averaged per frame).

    Useful as a sanity check; strict Parseval holds only for non-overlapping
    rectangular windows, so this is a rough diagnostic, not a pass/fail test.
    """
    signal_energy = float(np.sum(np.asarray(x, dtype=np.float64) ** 2))
    if signal_energy == 0.0:
        return 0.0
    # Divide by n_fft to undo the non-unitary DFT scaling.
    spec_energy_per_frame = np.sum(np.abs(Z) ** 2, axis=1) / Z.shape[1]
    spec_energy = float(np.mean(spec_energy_per_frame) * Z.shape[0])
    return float(abs(spec_energy - signal_energy) / signal_energy)


def reconstruction_error(x: np.ndarray, x_hat: np.ndarray) -> float:
    """Root-mean-square error between a signal and its reconstruction."""
    x = np.asarray(x, dtype=np.float64)
    x_hat = np.asarray(x_hat, dtype=np.float64)
    n = min(x.size, x_hat.size)
    d = x[:n] - x_hat[:n]
    return float(np.sqrt(np.mean(d * d)))


# ---------------------------------------------------------------------------
# Matrix-view metrics
# ---------------------------------------------------------------------------

def gram_matrix(frames: np.ndarray) -> np.ndarray:
    """Frame-wise Gram matrix ``frames @ frames.conj().T`` of shape ``(nf, nf)``."""
    F = np.asarray(frames)
    return F @ F.conj().T


def covariance_matrix(frames: np.ndarray) -> np.ndarray:
    """Covariance of coefficients *across* frames, shape ``(n_bins, n_bins)``.

    ``frames`` is ``(n_frames, n_bins)``. Returns the complex covariance
    estimated with bias (divide by ``n_frames``).
    """
    F = np.asarray(frames)
    mean = F.mean(axis=0, keepdims=True)
    centered = F - mean
    return (centered.conj().T @ centered) / max(F.shape[0], 1)


def spectrogram_difference(A: np.ndarray, B: np.ndarray) -> dict[str, np.ndarray]:
    """Return magnitude and wrapped-phase difference matrices."""
    mag_diff = _abs(A) - _abs(B)
    phase_diff = np.angle(A) - np.angle(B)
    phase_diff = np.mod(phase_diff + np.pi, 2 * np.pi) - np.pi
    return {"magnitude": mag_diff, "phase": phase_diff}


def framewise_correlation(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Per-frame Pearson correlation of magnitudes, length ``n_frames``."""
    a = _abs(A)
    b = _abs(B)
    out = np.zeros(a.shape[0], dtype=np.float64)
    for i in range(a.shape[0]):
        if a[i].std() == 0.0 or b[i].std() == 0.0:
            out[i] = 0.0
        else:
            out[i] = float(np.corrcoef(a[i], b[i])[0, 1])
    return out


# ---------------------------------------------------------------------------
# Runtime + memory helpers
# ---------------------------------------------------------------------------

def timeit_call(fn: Callable[..., Any], *args: Any, repeats: int = 3, **kwargs: Any) -> float:
    """Median wall-clock runtime (seconds) of ``fn(*args, **kwargs)`` over ``repeats`` runs."""
    runs = []
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        runs.append(time.perf_counter() - t0)
    return float(np.median(runs))


def measure_memory(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> tuple[Any, int]:
    """Return ``(result, peak_bytes)`` using ``tracemalloc``.

    The peak is the tracemalloc-observed high-water mark during the call; it
    reflects Python-level allocations, not OS RSS.
    """
    tracemalloc.start()
    try:
        result = fn(*args, **kwargs)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return result, int(peak)
