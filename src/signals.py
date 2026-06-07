"""Synthetic test-signal generators for the STFT comparison experiments.

Every generator returns a tuple ``(t, x)`` where ``t`` is a 1-D ``float64``
time axis in seconds and ``x`` is a 1-D ``float64`` signal of the same
length. Durations are interpreted at the given sample rate, so the number
of samples is ``int(round(duration * fs))``.

Use :func:`get_signal_registry` to obtain a name → callable mapping that
experiment runners can iterate over.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from scipy.signal import chirp as _scipy_chirp
from scipy.signal import square as _scipy_square

from .config import DEFAULT_FS, SEED


def _time_axis(duration: float, fs: int) -> np.ndarray:
    """Return an evenly-spaced time axis with ``round(duration * fs)`` samples."""
    n = int(round(duration * fs))
    return np.arange(n, dtype=np.float64) / float(fs)


def sine(
    freq: float = 440.0,
    duration: float = 1.0,
    fs: int = DEFAULT_FS,
    amplitude: float = 1.0,
    phase: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Single sinusoid at ``freq`` Hz."""
    t = _time_axis(duration, fs)
    x = amplitude * np.sin(2.0 * np.pi * freq * t + phase)
    return t, x


def sum_of_sines(
    freqs: Sequence[float] = (220.0, 440.0, 880.0),
    amps: Sequence[float] | None = None,
    duration: float = 1.0,
    fs: int = DEFAULT_FS,
) -> tuple[np.ndarray, np.ndarray]:
    """Sum of sinusoids at the given frequencies with optional per-tone amps."""
    freqs = np.asarray(freqs, dtype=np.float64)
    if amps is None:
        amps = np.ones_like(freqs)
    else:
        amps = np.asarray(amps, dtype=np.float64)
    if amps.shape != freqs.shape:
        raise ValueError("amps and freqs must have the same shape")
    t = _time_axis(duration, fs)
    x = np.zeros_like(t)
    for f, a in zip(freqs, amps):
        x += a * np.sin(2.0 * np.pi * f * t)
    return t, x


def chirp_signal(
    f0: float = 100.0,
    f1: float = 4_000.0,
    duration: float = 1.0,
    fs: int = DEFAULT_FS,
    method: str = "linear",
) -> tuple[np.ndarray, np.ndarray]:
    """Frequency sweep from ``f0`` to ``f1`` Hz (scipy ``chirp``)."""
    t = _time_axis(duration, fs)
    x = _scipy_chirp(t, f0=f0, f1=f1, t1=duration, method=method).astype(np.float64)
    return t, x


def impulse(
    duration: float = 1.0,
    fs: int = DEFAULT_FS,
    position: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Unit impulse at ``position`` (as a fraction of duration, 0..1)."""
    t = _time_axis(duration, fs)
    x = np.zeros_like(t)
    idx = int(np.clip(round(position * (len(t) - 1)), 0, len(t) - 1))
    x[idx] = 1.0
    return t, x


def square_wave(
    freq: float = 220.0,
    duration: float = 1.0,
    fs: int = DEFAULT_FS,
    duty: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Square wave at ``freq`` Hz with given duty cycle (0..1)."""
    t = _time_axis(duration, fs)
    x = _scipy_square(2.0 * np.pi * freq * t, duty=duty).astype(np.float64)
    return t, x


def speech_like(
    duration: float = 1.0,
    fs: int = DEFAULT_FS,
    f0: float = 140.0,
    formants: Sequence[float] = (700.0, 1_220.0, 2_600.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Toy voiced-speech surrogate: harmonic stack of ``f0`` shaped by formants
    and an amplitude envelope. Not a real vocal-tract model — just a signal
    with speech-like time-varying structure for STFT tests."""
    t = _time_axis(duration, fs)
    # Harmonics up to Nyquist.
    n_harm = int((fs / 2.0) // f0)
    x = np.zeros_like(t)
    for k in range(1, n_harm + 1):
        fk = k * f0
        # Simple formant shaping: boost harmonics near each formant.
        gain = 0.0
        for fmt in formants:
            gain += np.exp(-((fk - fmt) ** 2) / (2.0 * (150.0 ** 2)))
        x += (gain / k) * np.sin(2.0 * np.pi * fk * t)
    # Slow amplitude envelope (syllable-ish) + small vibrato in f0.
    envelope = 0.5 * (1.0 + np.sin(2.0 * np.pi * 3.0 * t))
    x = x * envelope
    peak = np.max(np.abs(x))
    if peak > 0:
        x /= peak
    return t, x


def add_noise(
    signal: np.ndarray,
    snr_db: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Add AWGN to ``signal`` at the requested SNR in dB.

    ``snr_db = inf`` returns the signal unchanged. Noise power is computed
    from the signal's mean-square amplitude.
    """
    if np.isinf(snr_db):
        return signal.astype(np.float64, copy=True)
    if rng is None:
        rng = np.random.default_rng(SEED)
    sig = np.asarray(signal, dtype=np.float64)
    sig_power = float(np.mean(sig ** 2))
    if sig_power == 0.0:
        return sig.copy()
    noise_power = sig_power / (10.0 ** (snr_db / 10.0))
    noise = rng.normal(0.0, np.sqrt(noise_power), size=sig.shape)
    return sig + noise


def load_audio(
    path: str | Path,
    mono: bool = True,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Load a WAV/FLAC/OGG file as ``(t, x, fs)`` using ``soundfile``.

    ``soundfile`` is an optional dependency — install it if you need this.
    Returns the time axis, signal (downmixed to mono if requested), and the
    file's native sample rate.
    """
    try:
        import soundfile as sf  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - optional dep
        raise ImportError(
            "load_audio requires the optional 'soundfile' package "
            "(pip install soundfile)"
        ) from exc
    data, fs = sf.read(str(path), always_2d=False)
    x = np.asarray(data, dtype=np.float64)
    if x.ndim == 2 and mono:
        x = x.mean(axis=1)
    t = np.arange(x.shape[0], dtype=np.float64) / float(fs)
    return t, x, int(fs)


def noisy_sine(
    freq: float = 440.0,
    duration: float = 1.0,
    fs: int = DEFAULT_FS,
    snr_db: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Convenience: sine tone with AWGN at ``snr_db``."""
    t, x = sine(freq=freq, duration=duration, fs=fs)
    return t, add_noise(x, snr_db=snr_db)


SignalFn = Callable[..., tuple[np.ndarray, np.ndarray]]


def get_signal_registry() -> dict[str, SignalFn]:
    """Return a name → generator mapping used by the experiment runner."""
    return {
        "sine": sine,
        "sum_of_sines": sum_of_sines,
        "chirp": chirp_signal,
        "impulse": impulse,
        "square": square_wave,
        "speech_like": speech_like,
        "noisy_sine": noisy_sine,
    }
