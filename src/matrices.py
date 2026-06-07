"""Matrix / operator views used by the STFT comparison.

The PRD asks for "many possible matrices" — this module exposes them as
explicit operators so experiments can talk about the same objects the
write-up describes (transform, window, framing, spectral filtering,
Toeplitz convolution). Operators are returned as dense ``np.ndarray``
when size is small and as ``scipy.sparse`` matrices when they would
otherwise be wasteful (framing).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import scipy.sparse as sp
from scipy.linalg import toeplitz

WindowName = Literal["rect", "hann", "hamming", "blackman"]


# ---------------------------------------------------------------------------
# Transform matrices
# ---------------------------------------------------------------------------

def dft_matrix(N: int) -> np.ndarray:
    """Classical (unnormalized) DFT matrix ``D`` such that ``D @ x == np.fft.fft(x)``."""
    if N <= 0:
        raise ValueError("N must be positive")
    k = np.arange(N)
    # W[n, k] = exp(-2j*pi*n*k/N)
    n = k[:, None]
    return np.exp(-2j * np.pi * n * k[None, :] / N)


def unitary_dft_matrix(N: int) -> np.ndarray:
    """Unitary DFT = ``dft_matrix(N) / sqrt(N)``; QFT-equivalent operator."""
    return dft_matrix(N) / np.sqrt(N)


def unitary_dft_from_dft(D: np.ndarray) -> np.ndarray:
    """Algebraic relation helper: given a DFT matrix ``D``, return ``D / sqrt(N)``."""
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("D must be square")
    return D / np.sqrt(D.shape[0])


def verify_unitarity(U: np.ndarray, tol: float = 1e-10) -> bool:
    """Return ``True`` if ``U @ U.conj().T`` is within ``tol`` of identity."""
    if U.ndim != 2 or U.shape[0] != U.shape[1]:
        return False
    I = np.eye(U.shape[0])
    return bool(np.allclose(U @ U.conj().T, I, atol=tol))


# ---------------------------------------------------------------------------
# Window matrices
# ---------------------------------------------------------------------------

def window(name: WindowName, N: int) -> np.ndarray:
    """1-D window of length ``N``. Supported: rect, hann, hamming, blackman."""
    if N <= 0:
        raise ValueError("N must be positive")
    name = name.lower()  # type: ignore[assignment]
    if name == "rect":
        return np.ones(N, dtype=np.float64)
    if name == "hann":
        return np.hanning(N).astype(np.float64)
    if name == "hamming":
        return np.hamming(N).astype(np.float64)
    if name == "blackman":
        return np.blackman(N).astype(np.float64)
    raise ValueError(f"unknown window '{name}'")


def window_matrix(name: WindowName, N: int) -> np.ndarray:
    """Diagonal matrix form of :func:`window`."""
    return np.diag(window(name, N))


# ---------------------------------------------------------------------------
# Framing operator
# ---------------------------------------------------------------------------

def overlap_ratio_to_hop(frame_len: int, ratio: float) -> int:
    """Convert hop-size ratio (1.0 = no overlap, 0.5 = 50% overlap) to samples.

    ``ratio`` is interpreted as hop/frame_len so that the overlap is
    ``1 - ratio``. Clamped to ``[1, frame_len]``.
    """
    if not (0.0 < ratio <= 1.0):
        raise ValueError("ratio must lie in (0, 1]")
    hop = int(round(ratio * frame_len))
    return max(1, min(hop, frame_len))


def n_frames(signal_len: int, frame_len: int, hop: int) -> int:
    """Number of full frames of length ``frame_len`` at step ``hop`` in a signal."""
    if signal_len < frame_len:
        return 0
    return 1 + (signal_len - frame_len) // hop


def framing_matrix(signal_len: int, frame_len: int, hop: int) -> sp.csr_matrix:
    """Sparse framing operator ``F`` of shape ``(n_frames*frame_len, signal_len)``.

    ``(F @ x).reshape(n_frames, frame_len)`` yields the frames. Kept sparse
    because the identity-like pattern wastes memory when dense.
    """
    if frame_len <= 0 or hop <= 0:
        raise ValueError("frame_len and hop must be positive")
    nf = n_frames(signal_len, frame_len, hop)
    if nf == 0:
        return sp.csr_matrix((0, signal_len))
    rows = np.repeat(np.arange(nf * frame_len), 1)
    cols = np.concatenate(
        [np.arange(i * hop, i * hop + frame_len) for i in range(nf)]
    )
    data = np.ones_like(rows, dtype=np.float64)
    return sp.csr_matrix(
        (data, (rows, cols)), shape=(nf * frame_len, signal_len)
    )


# ---------------------------------------------------------------------------
# Spectral filtering operators (diagonal in the DFT basis)
# ---------------------------------------------------------------------------

def identity_op(N: int) -> np.ndarray:
    """Identity spectral mask."""
    return np.ones(N, dtype=np.float64)


def _bin_mask(N: int, keep: np.ndarray) -> np.ndarray:
    mask = np.zeros(N, dtype=np.float64)
    mask[keep] = 1.0
    # Mirror for real-signal symmetry so the mask is Hermitian-friendly.
    mask = np.maximum(mask, mask[::-1])
    return mask


def lowpass_op(N: int, cutoff_bin: int) -> np.ndarray:
    """Diagonal low-pass mask keeping bins ``[0, cutoff_bin]`` (and mirror)."""
    cutoff_bin = int(np.clip(cutoff_bin, 0, N // 2))
    keep = np.arange(0, cutoff_bin + 1)
    return _bin_mask(N, keep)


def highpass_op(N: int, cutoff_bin: int) -> np.ndarray:
    """Diagonal high-pass mask zeroing bins below ``cutoff_bin``."""
    cutoff_bin = int(np.clip(cutoff_bin, 0, N // 2))
    keep = np.arange(cutoff_bin, N // 2 + 1)
    return _bin_mask(N, keep)


def bandpass_op(N: int, lo: int, hi: int) -> np.ndarray:
    """Diagonal band-pass mask keeping bins ``[lo, hi]`` (and mirror)."""
    lo = int(np.clip(lo, 0, N // 2))
    hi = int(np.clip(hi, lo, N // 2))
    keep = np.arange(lo, hi + 1)
    return _bin_mask(N, keep)


def random_diag_op(N: int, seed: int = 0) -> np.ndarray:
    """Random phase-only diagonal operator ``diag(e^{i theta_k})`` (unitary)."""
    rng = np.random.default_rng(seed)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=N)
    return np.exp(1j * phases)


# ---------------------------------------------------------------------------
# Toeplitz convolution matrix
# ---------------------------------------------------------------------------

def toeplitz_conv_matrix(kernel: np.ndarray, N: int, mode: str = "full") -> np.ndarray:
    """Linear-convolution matrix ``T`` such that ``T @ x == np.convolve(x, kernel, mode)``.

    ``mode`` is one of ``"full"`` (default) or ``"same"``. For ``"same"``,
    the output is cropped to length ``N`` centered on the full convolution.
    """
    kernel = np.asarray(kernel, dtype=np.float64).ravel()
    k = kernel.size
    full_len = N + k - 1
    col = np.zeros(full_len, dtype=np.float64)
    col[:k] = kernel
    row = np.zeros(N, dtype=np.float64)
    row[0] = kernel[0]
    T = toeplitz(col, row)
    if mode == "full":
        return T
    if mode == "same":
        start = (k - 1) // 2
        return T[start : start + N, :]
    raise ValueError("mode must be 'full' or 'same'")
