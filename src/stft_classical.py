"""Classical Short-Time Fourier Transform.

Two implementations:

* :func:`stft_custom` — framing + windowing + zero-padding + ``np.fft.fft``.
  This is the reference the quantum-inspired pipeline (Phase 4) is compared
  against. No per-frame scaling is applied; the raw windowed-FFT
  coefficients are returned.
* :func:`stft_scipy` — thin wrapper around :func:`scipy.signal.stft`, rescaled
  back to raw-FFT units so the two implementations are numerically comparable
  via :func:`validate_against_scipy`.

:func:`istft_custom` performs overlap-add reconstruction with window
normalization for the reconstruction-error metric.
"""

from __future__ import annotations

import numpy as np
import scipy.signal as sps

from .matrices import WindowName, n_frames, window


def frame_signal(x: np.ndarray, frame_len: int, hop: int) -> np.ndarray:
    """Slice ``x`` into frames of shape ``(n_frames, frame_len)``.

    Trailing samples that don't fit a full frame are dropped. Returns an
    empty ``(0, frame_len)`` array if ``x`` is shorter than one frame.
    """
    x = np.asarray(x, dtype=np.float64)
    if frame_len <= 0 or hop <= 0:
        raise ValueError("frame_len and hop must be positive")
    nf = n_frames(x.size, frame_len, hop)
    if nf == 0:
        return np.empty((0, frame_len), dtype=np.float64)
    # Strided view avoids copying until the caller needs it.
    stride = x.strides[0]
    frames = np.lib.stride_tricks.as_strided(
        x, shape=(nf, frame_len), strides=(hop * stride, stride), writeable=False
    )
    return np.ascontiguousarray(frames)


def stft_custom(
    x: np.ndarray,
    frame_len: int,
    hop: int,
    window_name: WindowName = "hann",
    n_fft: int | None = None,
    fs: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Framing + windowing + zero-padding + ``np.fft.fft``.

    Parameters
    ----------
    x : 1-D signal.
    frame_len, hop : frame length and hop size in samples.
    window_name : one of ``rect | hann | hamming | blackman``.
    n_fft : transform length (defaults to ``frame_len``; zero-padded if larger).
    fs : sample rate, only used to populate the returned frequency axis.

    Returns
    -------
    Z : complex ``(n_frames, n_fft)`` spectrogram (full FFT, not single-sided).
    freqs : length-``n_fft`` frequency axis in Hz.
    times : length-``n_frames`` frame-center time axis in seconds.
    """
    if n_fft is None:
        n_fft = frame_len
    if n_fft < frame_len:
        raise ValueError("n_fft must be >= frame_len")

    frames = frame_signal(x, frame_len, hop)
    w = window(window_name, frame_len)
    windowed = frames * w[None, :]
    if n_fft > frame_len:
        pad = np.zeros((windowed.shape[0], n_fft - frame_len), dtype=np.float64)
        windowed = np.concatenate([windowed, pad], axis=1)

    Z = np.fft.fft(windowed, axis=1)
    freqs = np.fft.fftfreq(n_fft, d=1.0 / fs)
    times = (np.arange(frames.shape[0]) * hop + frame_len / 2.0) / fs
    return Z, freqs, times


def stft_scipy(
    x: np.ndarray,
    fs: int,
    frame_len: int,
    hop: int,
    window_name: WindowName = "hann",
    n_fft: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reference STFT via ``scipy.signal.stft``, rescaled to raw-FFT units.

    scipy normalises its output (``scaling='spectrum'`` divides by
    ``win.sum()``); we multiply back so the coefficients match the
    non-normalised convention of :func:`stft_custom`. The returned
    spectrogram is single-sided ``(n_freq_bins, n_frames)`` — use
    :func:`match_scipy_shape` in tests before comparing against the
    two-sided output of :func:`stft_custom`.
    """
    if n_fft is None:
        n_fft = frame_len
    w = window(window_name, frame_len)
    freqs, times, Zxx = sps.stft(
        x,
        fs=fs,
        window=w,
        nperseg=frame_len,
        noverlap=frame_len - hop,
        nfft=n_fft,
        boundary=None,
        padded=False,
        return_onesided=True,
        scaling="spectrum",
    )
    # scipy's 'spectrum' scaling divides by win.sum(); undo it so magnitudes
    # match a raw FFT of the windowed frame.
    Zxx = Zxx * w.sum()
    return Zxx, freqs, times


def validate_against_scipy(
    x: np.ndarray,
    fs: int,
    frame_len: int,
    hop: int,
    window_name: WindowName = "hann",
    n_fft: int | None = None,
) -> float:
    """Return the max absolute error between :func:`stft_custom` (folded to
    single-sided) and :func:`stft_scipy`.
    """
    Z_custom, _, _ = stft_custom(x, frame_len, hop, window_name, n_fft=n_fft, fs=fs)
    Z_scipy, _, _ = stft_scipy(x, fs, frame_len, hop, window_name, n_fft=n_fft)
    n_bins = Z_scipy.shape[0]
    # stft_custom: (n_frames, n_fft) two-sided -> take first n_bins, transpose.
    Z_custom_one = Z_custom[:, :n_bins].T
    # Align frame count (scipy may end on a different last frame for short sigs).
    n_common = min(Z_custom_one.shape[1], Z_scipy.shape[1])
    err = np.max(np.abs(Z_custom_one[:, :n_common] - Z_scipy[:, :n_common]))
    return float(err)


def istft_custom(
    Z: np.ndarray,
    hop: int,
    window_name: WindowName,
    signal_len: int,
    frame_len: int | None = None,
) -> np.ndarray:
    """Overlap-add reconstruction from a two-sided STFT ``(n_frames, n_fft)``.

    Uses the squared-window normalisation so that, given perfect coverage,
    ``istft_custom(stft_custom(x))`` returns ``x`` up to floating-point
    error at the edges.
    """
    nf, n_fft = Z.shape
    if frame_len is None:
        frame_len = n_fft
    if frame_len > n_fft:
        raise ValueError("frame_len must be <= n_fft")
    frames = np.fft.ifft(Z, axis=1).real[:, :frame_len]
    w = window(window_name, frame_len)
    out = np.zeros(signal_len, dtype=np.float64)
    norm = np.zeros(signal_len, dtype=np.float64)
    for i in range(nf):
        start = i * hop
        end = start + frame_len
        if end > signal_len:
            end = signal_len
        seg = end - start
        out[start:end] += frames[i, :seg] * w[:seg]
        norm[start:end] += w[:seg] ** 2
    mask = norm > 1e-12
    out[mask] /= norm[mask]
    return out
