import numpy as np
import pytest

from src.signals import chirp_signal, sine
from src.stft_classical import stft_custom
from src.stft_quantum import amplitude_encode, next_pow2, qft_simulate, stft_quantum


def test_next_pow2():
    assert next_pow2(1) == 1
    assert next_pow2(2) == 2
    assert next_pow2(3) == 4
    assert next_pow2(1023) == 1024
    assert next_pow2(1024) == 1024


def test_amplitude_encode_unit_norm_and_roundtrip():
    v = np.array([3.0, 4.0])
    state, norm = amplitude_encode(v)
    assert abs(norm - 5.0) < 1e-12
    assert np.allclose(np.linalg.norm(state), 1.0)
    assert np.allclose(state * norm, v)


def test_amplitude_encode_zero_input_returns_zero_norm():
    state, norm = amplitude_encode(np.zeros(8))
    assert norm == 0.0


def test_qft_simulate_is_unitary():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(16) + 1j * rng.standard_normal(16)
    x /= np.linalg.norm(x)
    y = qft_simulate(x)
    assert abs(np.linalg.norm(y) - 1.0) < 1e-12


@pytest.mark.parametrize("frame_len", [64, 128, 256])
def test_quantum_matches_classical_when_n_fft_is_pow2(frame_len):
    _, s = sine(freq=500.0, duration=0.2, fs=8000)
    Zc, _, _ = stft_custom(s, frame_len, frame_len // 2, "hann", fs=8000)
    Zq, _, _ = stft_quantum(s, frame_len, frame_len // 2, "hann", fs=8000)
    assert np.max(np.abs(Zc - Zq)) < 1e-10


def test_quantum_pads_to_next_pow2():
    _, s = chirp_signal(duration=0.1, fs=8000)
    Zq, _, _ = stft_quantum(s, frame_len=200, hop=100, window_name="hann",
                            n_fft=200, fs=8000)
    assert Zq.shape[1] == 256
