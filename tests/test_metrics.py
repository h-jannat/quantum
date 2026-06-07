import numpy as np

from src.metrics import (
    cosine_similarity_flat, frobenius, mae, mse, pearson_flat,
    phase_error_stats, reconstruction_error, spectrogram_difference,
)


def test_identical_inputs_yield_zero_error_and_unit_similarity():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((4, 8)) + 1j * rng.standard_normal((4, 8))
    assert mse(A, A) == 0.0
    assert mae(A, A) == 0.0
    assert frobenius(A, A) == 0.0
    assert abs(cosine_similarity_flat(A, A) - 1.0) < 1e-12
    assert abs(pearson_flat(A, A) - 1.0) < 1e-12


def test_mse_mae_known_values():
    A = np.array([[1.0, 0.0]])
    B = np.array([[0.0, 0.0]])
    assert mse(A, B) == 0.5   # (1^2 + 0^2) / 2
    assert mae(A, B) == 0.5   # (|1| + |0|) / 2


def test_phase_error_zero_for_aligned_spectra():
    rng = np.random.default_rng(1)
    A = rng.standard_normal((4, 8)) + 1j * rng.standard_normal((4, 8))
    stats = phase_error_stats(A, 2.0 * A)  # scaling preserves phase
    assert abs(stats["mean"]) < 1e-12
    assert stats["std"] < 1e-12


def test_spectrogram_difference_shapes_match():
    A = np.ones((3, 5), dtype=complex)
    B = 0.5 * np.ones((3, 5), dtype=complex)
    d = spectrogram_difference(A, B)
    assert d["magnitude"].shape == A.shape
    assert np.allclose(d["magnitude"], 0.5)
    assert d["phase"].shape == A.shape


def test_reconstruction_error_zero_for_identical():
    x = np.arange(10, dtype=float)
    assert reconstruction_error(x, x) == 0.0
