import pytest

from src.signals import chirp_signal, sine
from src.stft_classical import validate_against_scipy


@pytest.mark.parametrize("window_name", ["rect", "hann", "hamming", "blackman"])
@pytest.mark.parametrize("frame_len,hop", [(128, 64), (256, 128), (512, 256)])
def test_custom_stft_matches_scipy(window_name, frame_len, hop):
    _, s = sine(freq=500.0, duration=0.2, fs=8000)
    err = validate_against_scipy(s, fs=8000, frame_len=frame_len, hop=hop,
                                 window_name=window_name)
    assert err < 1e-8


def test_matches_scipy_on_chirp():
    _, s = chirp_signal(duration=0.3, fs=8000)
    err = validate_against_scipy(s, fs=8000, frame_len=256, hop=128, window_name="hann")
    assert err < 1e-8
