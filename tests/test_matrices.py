import numpy as np
import pytest

from src.matrices import (
    bandpass_op, dft_matrix, highpass_op, lowpass_op, overlap_ratio_to_hop,
    random_diag_op, toeplitz_conv_matrix, unitary_dft_from_dft,
    unitary_dft_matrix, verify_unitarity, window,
)


@pytest.mark.parametrize("N", [4, 8, 16, 32, 64])
def test_dft_matrix_matches_numpy_fft(N):
    rng = np.random.default_rng(N)
    x = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    assert np.allclose(dft_matrix(N) @ x, np.fft.fft(x))


@pytest.mark.parametrize("N", [4, 8, 16, 32])
def test_unitary_dft_is_unitary(N):
    U = unitary_dft_matrix(N)
    assert verify_unitarity(U)
    # Algebraic relation.
    assert np.allclose(U, unitary_dft_from_dft(dft_matrix(N)))


@pytest.mark.parametrize("name", ["rect", "hann", "hamming", "blackman"])
def test_windows_have_correct_length_and_range(name):
    w = window(name, 64)
    assert w.shape == (64,)
    assert w.dtype == np.float64
    assert np.all(w >= -1e-12)  # blackman has tiny-negative endpoints
    assert np.all(w <= 1.0 + 1e-12)


def test_overlap_ratio_to_hop_bounds():
    assert overlap_ratio_to_hop(128, 1.0) == 128
    assert overlap_ratio_to_hop(128, 0.5) == 64
    assert overlap_ratio_to_hop(128, 0.01) == 1  # clamped to 1


def test_spectral_ops_shape_and_symmetry():
    for op in (lowpass_op(32, 8), highpass_op(32, 8), bandpass_op(32, 4, 12)):
        assert op.shape == (32,)
        assert np.array_equal(op, op[::-1])  # Hermitian-symmetric mask
    rd = random_diag_op(16, seed=0)
    assert np.allclose(np.abs(rd), 1.0)


def test_toeplitz_matches_numpy_convolve():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(20)
    kern = np.array([1.0, -0.5, 0.25])
    T_full = toeplitz_conv_matrix(kern, 20, mode="full")
    T_same = toeplitz_conv_matrix(kern, 20, mode="same")
    assert np.allclose(T_full @ x, np.convolve(x, kern, "full"))
    assert np.allclose(T_same @ x, np.convolve(x, kern, "same"))
