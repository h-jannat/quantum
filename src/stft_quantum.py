"""Quantum-inspired STFT simulator.

Pipeline (per frame):

1. apply window,
2. zero-pad up to the next power of two,
3. amplitude-encode — divide by the frame's L2 norm so the resulting vector
   is a valid quantum state (unit L2),
4. apply a unitary DFT (``np.fft.fft(..., norm="ortho")``), which is the
   matrix form of the Quantum Fourier Transform,
5. rescale the coefficients by ``norm * sqrt(N)`` so the output matches
   the raw windowed-FFT convention used in :mod:`stft_classical`.

Differences from gate-level QFT
-------------------------------
This is a **simulation** of the mathematical pipeline. On real hardware:

* **State preparation.** Encoding the windowed frame into ``log2(N)`` qubits
  requires a state-prep circuit (e.g. Mottonen, Schuld/Park). For arbitrary
  amplitudes, known circuits have ``O(N)`` gate depth, which removes any
  would-be ``O(log N)`` advantage of QFT alone. We skip this cost entirely
  by writing the statevector directly into memory.
* **Measurement and shot noise.** Hardware gives probabilities, not
  amplitudes; recovering complex coefficients needs many shots and auxiliary
  circuits. Here we read the full statevector classically — no shot noise.
* **Decoherence and gate error.** Absent. Our operator is exactly unitary
  to double-precision.
* **Classical post-processing.** ``norm * sqrt(N)`` rescaling is bookkeeping
  that exists only because we normalise to a quantum state — it is not a
  physical operation.

So any runtime / accuracy observations in this project reflect numpy's FFT,
**not** a physical quantum advantage.
"""

from __future__ import annotations

import numpy as np

from .matrices import WindowName, n_frames, window
from .stft_classical import frame_signal


def next_pow2(n: int) -> int:
    """Smallest power of two ``>= max(1, n)``."""
    if n <= 1:
        return 1
    return 1 << (int(n - 1).bit_length())


def amplitude_encode(frame: np.ndarray) -> tuple[np.ndarray, float]:
    """Normalise ``frame`` to unit L2 and return ``(state, norm)``.

    If the input is the zero vector, returns a zero state with ``norm = 0``;
    downstream code treats this as a degenerate (silent) frame.
    """
    frame = np.asarray(frame, dtype=np.complex128)
    norm = float(np.linalg.norm(frame))
    if norm == 0.0:
        return frame.copy(), 0.0
    return frame / norm, norm


def qft_simulate(state: np.ndarray) -> np.ndarray:
    """Statevector-level QFT: orthonormal FFT of a unit-norm amplitude vector."""
    return np.fft.fft(state, norm="ortho")


def stft_quantum(
    x: np.ndarray,
    frame_len: int,
    hop: int,
    window_name: WindowName = "hann",
    n_fft: int | None = None,
    fs: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quantum-inspired STFT.

    The transform length is forced to the next power of two ``>= n_fft`` (or
    ``>= frame_len`` if ``n_fft`` is ``None``) so that the simulated QFT
    operates on ``log2(N)`` qubits. Output coefficients are rescaled so
    ``stft_quantum`` matches ``stft_custom`` up to floating-point error when
    ``n_fft`` is already a power of two.

    Returns ``(Z, freqs, times)`` with ``Z.shape == (n_frames, N)``.
    """
    if n_fft is None:
        n_fft = frame_len
    N = next_pow2(max(n_fft, frame_len))

    frames = frame_signal(x, frame_len, hop)
    nf = frames.shape[0]
    w = window(window_name, frame_len)
    Z = np.empty((nf, N), dtype=np.complex128)
    for i in range(nf):
        padded = np.zeros(N, dtype=np.complex128)
        padded[:frame_len] = frames[i] * w
        state, norm = amplitude_encode(padded)
        spec = qft_simulate(state)
        # unitary DFT = D / sqrt(N); undo the normalisation to match raw FFT.
        Z[i] = spec * (norm * np.sqrt(N))

    freqs = np.fft.fftfreq(N, d=1.0 / fs)
    times = (np.arange(nf) * hop + frame_len / 2.0) / fs
    return Z, freqs, times
