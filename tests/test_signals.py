import numpy as np
import pytest

from src.signals import add_noise, get_signal_registry, sine


def test_registry_contains_expected_signals():
    reg = get_signal_registry()
    for name in ("sine", "sum_of_sines", "chirp", "impulse", "square",
                 "speech_like", "noisy_sine"):
        assert name in reg


@pytest.mark.parametrize("name", list(get_signal_registry().keys()))
def test_each_signal_returns_matching_time_and_amplitude(name):
    fn = get_signal_registry()[name]
    t, x = fn(duration=0.1, fs=8000)
    assert t.shape == x.shape
    assert t.ndim == 1
    assert np.isfinite(x).all()
    assert t.dtype == np.float64 and x.dtype == np.float64


def test_add_noise_matches_target_snr_within_tolerance():
    _, s = sine(freq=200.0, duration=1.0, fs=8000)
    rng = np.random.default_rng(0)
    y = add_noise(s, snr_db=10.0, rng=rng)
    noise = y - s
    measured = 10.0 * np.log10(np.mean(s ** 2) / np.mean(noise ** 2))
    assert abs(measured - 10.0) < 0.3


def test_add_noise_inf_snr_is_noop():
    _, s = sine(duration=0.1, fs=8000)
    y = add_noise(s, snr_db=float("inf"))
    assert np.array_equal(y, s)
