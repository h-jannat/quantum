import numpy as np

from src.matrices import framing_matrix, n_frames
from src.stft_classical import frame_signal


def test_frame_signal_matches_manual_slicing():
    x = np.arange(20, dtype=float)
    F = frame_signal(x, frame_len=8, hop=4)
    assert F.shape == (4, 8)
    for i in range(4):
        assert np.array_equal(F[i], x[i * 4 : i * 4 + 8])


def test_frame_signal_short_signal_returns_empty():
    x = np.arange(5, dtype=float)
    F = frame_signal(x, frame_len=8, hop=4)
    assert F.shape == (0, 8)


def test_framing_matrix_equals_manual_frames():
    x = np.arange(24, dtype=float)
    F = framing_matrix(24, 8, 4)
    frames = (F @ x).reshape(-1, 8)
    expected = np.stack([x[i * 4 : i * 4 + 8] for i in range(n_frames(24, 8, 4))])
    assert np.array_equal(frames, expected)
