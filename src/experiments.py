"""Grid-based experiment runner comparing classical vs quantum-inspired STFT.

The public entry points are :func:`run_grid` (sweeps signals × frame × hop ×
window × zero-pad × SNR and returns a metrics DataFrame),
:func:`run_scaling` (runtime + error vs ``n_fft``), :func:`run_noise_sweep`
(error vs SNR), and :func:`run_matrix_view` (DFT / unitary-DFT / filter-op /
Gram comparisons). :func:`run_all` ties them together and produces the CSV
tables + plots referenced by the report.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .config import DEFAULT_FS, PLOTS_DIR, SEED, TABLES_DIR, ensure_output_dirs, set_global_seed
from .matrices import (
    WindowName, bandpass_op, dft_matrix, highpass_op, identity_op,
    lowpass_op, overlap_ratio_to_hop, random_diag_op, toeplitz_conv_matrix,
    unitary_dft_matrix, verify_unitarity,
)
from .metrics import (
    cosine_similarity_flat, covariance_matrix, energy_preservation_error,
    frobenius, framewise_correlation, gram_matrix, mae, mse, pearson_flat,
    phase_error_stats, reconstruction_error, spectrogram_difference,
    timeit_call,
)
from .plots import (
    plot_difference_heatmap, plot_error_vs_frame, plot_error_vs_snr,
    plot_matrix, plot_runtime_vs_frame, plot_spectrogram, plot_waveform,
    plot_window_bars,
)
from .signals import add_noise, get_signal_registry
from .stft_classical import istft_custom, stft_custom
from .stft_quantum import stft_quantum

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config + grid
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExperimentConfig:
    signal: str
    frame_len: int
    hop_ratio: float
    window: WindowName
    zero_pad_factor: int
    snr_db: float
    fs: int = DEFAULT_FS
    duration: float = 0.5

    @property
    def hop(self) -> int:
        return overlap_ratio_to_hop(self.frame_len, self.hop_ratio)

    @property
    def n_fft(self) -> int:
        return self.frame_len * self.zero_pad_factor


@dataclass
class GridSpec:
    signals: list[str]
    frame_lens: list[int]
    hop_ratios: list[float]
    windows: list[WindowName]
    zero_pad_factors: list[int]
    snr_levels: list[float] = field(default_factory=lambda: [float("inf")])
    fs: int = DEFAULT_FS
    duration: float = 0.5

    def enumerate(self) -> list[ExperimentConfig]:
        combos = itertools.product(
            self.signals, self.frame_lens, self.hop_ratios,
            self.windows, self.zero_pad_factors, self.snr_levels,
        )
        return [
            ExperimentConfig(
                signal=s, frame_len=f, hop_ratio=h, window=w,
                zero_pad_factor=z, snr_db=n, fs=self.fs, duration=self.duration,
            )
            for (s, f, h, w, z, n) in combos
        ]


QUICK_GRID = GridSpec(
    signals=["sine", "chirp", "square"],
    frame_lens=[128, 256],
    hop_ratios=[0.5],
    windows=["hann", "hamming"],
    zero_pad_factors=[1, 2],
    snr_levels=[float("inf"), 10.0],
)

FULL_GRID = GridSpec(
    signals=["sine", "sum_of_sines", "chirp", "impulse", "square", "speech_like"],
    frame_lens=[64, 128, 256, 512, 1024],
    hop_ratios=[1.0, 0.75, 0.5, 0.25],
    windows=["rect", "hann", "hamming", "blackman"],
    zero_pad_factors=[1, 2],
    snr_levels=[float("inf"), 20.0, 10.0, 0.0],
)


# ---------------------------------------------------------------------------
# Single-run driver
# ---------------------------------------------------------------------------

def _generate_signal(cfg: ExperimentConfig) -> np.ndarray:
    reg = get_signal_registry()
    if cfg.signal not in reg:
        raise ValueError(f"unknown signal '{cfg.signal}'")
    fn = reg[cfg.signal]
    _, x = fn(duration=cfg.duration, fs=cfg.fs)
    if np.isfinite(cfg.snr_db):
        x = add_noise(x, snr_db=cfg.snr_db, rng=np.random.default_rng(SEED))
    return x


def run_single(cfg: ExperimentConfig) -> dict:
    """Run both STFTs for one config and return a flat metrics row."""
    x = _generate_signal(cfg)

    Zc, _, _ = stft_custom(x, cfg.frame_len, cfg.hop, cfg.window, n_fft=cfg.n_fft, fs=cfg.fs)
    Zq, _, _ = stft_quantum(x, cfg.frame_len, cfg.hop, cfg.window, n_fft=cfg.n_fft, fs=cfg.fs)

    # Align shapes (quantum forces pow2 on n_fft — classical also uses pow2
    # here because zero_pad_factor*frame_len is a pow2).
    n_bins = min(Zc.shape[1], Zq.shape[1])
    n_fr = min(Zc.shape[0], Zq.shape[0])
    Zc = Zc[:n_fr, :n_bins]
    Zq = Zq[:n_fr, :n_bins]

    phase = phase_error_stats(Zc, Zq)
    x_hat = istft_custom(Zc, hop=cfg.hop, window_name=cfg.window,
                         signal_len=x.size, frame_len=cfg.frame_len)
    row = {
        **asdict(cfg),
        "hop": cfg.hop,
        "n_fft": cfg.n_fft,
        "signal_len": int(x.size),
        "n_frames": int(Zc.shape[0]),
        "mse": mse(Zc, Zq),
        "mae": mae(Zc, Zq),
        "frobenius": frobenius(Zc, Zq),
        "cosine": cosine_similarity_flat(Zc, Zq),
        "pearson": pearson_flat(Zc, Zq),
        "phase_mean": phase["mean"],
        "phase_std": phase["std"],
        "energy_err_classical": energy_preservation_error(x, Zc),
        "energy_err_quantum": energy_preservation_error(x, Zq),
        "recon_rmse_classical": reconstruction_error(x, x_hat),
        "runtime_classical_s": timeit_call(
            stft_custom, x, cfg.frame_len, cfg.hop, cfg.window,
            repeats=3, n_fft=cfg.n_fft, fs=cfg.fs,
        ),
        "runtime_quantum_s": timeit_call(
            stft_quantum, x, cfg.frame_len, cfg.hop, cfg.window,
            repeats=3, n_fft=cfg.n_fft, fs=cfg.fs,
        ),
    }
    return row


def run_grid(configs: Iterable[ExperimentConfig]) -> pd.DataFrame:
    rows = []
    configs = list(configs)
    for i, cfg in enumerate(configs, 1):
        log.info("[%d/%d] %s", i, len(configs), cfg)
        rows.append(run_single(cfg))
    df = pd.DataFrame(rows)
    return df


def best_worst(df: pd.DataFrame, by: str = "mse") -> pd.DataFrame:
    """Return best/worst rows for the given metric (ascending = best for error-style)."""
    if df.empty:
        return df
    best = df.nsmallest(5, by).assign(rank="best")
    worst = df.nlargest(5, by).assign(rank="worst")
    return pd.concat([best, worst], ignore_index=True)


# ---------------------------------------------------------------------------
# Scaling / noise / matrix-view experiments
# ---------------------------------------------------------------------------

def run_scaling(signal: str = "chirp", fs: int = DEFAULT_FS, duration: float = 1.0,
                frame_lens: list[int] | None = None) -> pd.DataFrame:
    frame_lens = frame_lens or [64, 128, 256, 512, 1024, 2048]
    rows = []
    for fl in frame_lens:
        cfg = ExperimentConfig(
            signal=signal, frame_len=fl, hop_ratio=0.5, window="hann",
            zero_pad_factor=1, snr_db=float("inf"), fs=fs, duration=duration,
        )
        r = run_single(cfg)
        rows.append({"frame_len": fl, "method": "classical",
                     "runtime_s": r["runtime_classical_s"], "mse": r["mse"]})
        rows.append({"frame_len": fl, "method": "quantum",
                     "runtime_s": r["runtime_quantum_s"], "mse": r["mse"]})
    return pd.DataFrame(rows)


def run_noise_sweep(signal: str = "sine", fs: int = DEFAULT_FS, duration: float = 0.5,
                    snr_levels: list[float] | None = None) -> pd.DataFrame:
    snr_levels = snr_levels or [float("inf"), 30, 20, 10, 5, 0, -5]
    rows = []
    for snr in snr_levels:
        cfg = ExperimentConfig(
            signal=signal, frame_len=256, hop_ratio=0.5, window="hann",
            zero_pad_factor=1, snr_db=snr, fs=fs, duration=duration,
        )
        r = run_single(cfg)
        rows.append({"snr_db": snr if np.isfinite(snr) else 1e9,
                     "mse": r["mse"], "cosine": r["cosine"]})
    return pd.DataFrame(rows)


def run_matrix_view(N: int = 32) -> dict:
    """Compare DFT vs unitary DFT and apply a few filter operators."""
    D = dft_matrix(N)
    U = unitary_dft_matrix(N)
    x = np.random.default_rng(SEED).standard_normal(N).astype(np.complex128)

    out = {
        "N": N,
        "dft_vs_udft_err": float(np.max(np.abs(D / np.sqrt(N) - U))),
        "udft_unitary": verify_unitarity(U),
        "identity_mae": float(np.max(np.abs(D @ x - np.fft.fft(x)))),
    }

    # Filter-op effect on the unitary-DFT spectrum.
    spec = U @ x
    for name, op in [
        ("identity", identity_op(N)),
        ("lowpass",  lowpass_op(N, N // 8)),
        ("highpass", highpass_op(N, N // 4)),
        ("bandpass", bandpass_op(N, N // 8, N // 4)),
        ("random_diag", random_diag_op(N, seed=SEED)),
    ]:
        filtered = op * spec
        out[f"energy_{name}"] = float(np.sum(np.abs(filtered) ** 2))

    # Toeplitz kernel example.
    kernel = np.array([0.25, 0.5, 0.25])  # simple low-pass
    T = toeplitz_conv_matrix(kernel, N, mode="same")
    out["toeplitz_shape"] = T.shape
    out["toeplitz_rank"] = int(np.linalg.matrix_rank(T))

    return out


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_all(
    grid: GridSpec = QUICK_GRID,
    out_tables: Path = TABLES_DIR,
    out_plots: Path = PLOTS_DIR,
    skip_plots: bool = False,
) -> pd.DataFrame:
    """Run the full pipeline: grid, scaling, noise, matrix view → CSVs + plots."""
    ensure_output_dirs()
    set_global_seed(SEED)

    configs = grid.enumerate()
    log.info("Running %d configs", len(configs))
    df = run_grid(configs)
    df.to_csv(out_tables / "summary.csv", index=False)

    bw = best_worst(df)
    bw.to_csv(out_tables / "best_worst.csv", index=False)

    runtime = df.melt(
        id_vars=["frame_len", "window", "signal"],
        value_vars=["runtime_classical_s", "runtime_quantum_s"],
        var_name="method", value_name="runtime_s",
    )
    runtime["method"] = runtime["method"].str.replace("runtime_", "").str.replace("_s", "")
    runtime.to_csv(out_tables / "runtime.csv", index=False)

    scaling = run_scaling()
    scaling.to_csv(out_tables / "scaling.csv", index=False)

    noise = run_noise_sweep()
    noise.to_csv(out_tables / "noise_sweep.csv", index=False)

    mv = run_matrix_view()
    pd.DataFrame([mv]).to_csv(out_tables / "matrix_view.csv", index=False)

    if not skip_plots:
        _emit_plots(df, scaling, noise, out_plots, grid)

    log.info("Wrote tables to %s", out_tables)
    return df


def _emit_plots(df: pd.DataFrame, scaling: pd.DataFrame, noise: pd.DataFrame,
                out_plots: Path, grid: GridSpec) -> None:
    plot_runtime_vs_frame(scaling, out_plots / "runtime_vs_frame.png")
    plot_error_vs_frame(df, out_plots / "error_vs_frame.png")
    plot_error_vs_snr(noise.rename(columns={"snr_db": "snr_db"}),
                      out_plots / "error_vs_snr.png")
    plot_window_bars(df, out_plots / "window_bars.png")

    # Matrix viz.
    for N in (16, 32):
        plot_matrix(dft_matrix(N), out_plots / f"dft_{N}.png",
                    title=f"DFT matrix N={N}", mode="real_imag")
        plot_matrix(unitary_dft_matrix(N), out_plots / f"udft_{N}.png",
                    title=f"Unitary DFT N={N}", mode="real_imag")

    # Example per-signal spectrograms + diff.
    reg = get_signal_registry()
    for sig_name in grid.signals[:3]:
        t, x = reg[sig_name](duration=0.5, fs=grid.fs)
        plot_waveform(t, x, out_plots / f"wave_{sig_name}.png", title=sig_name)
        Zc, freqs, times = stft_custom(x, 256, 128, "hann", fs=grid.fs)
        Zq, _, _ = stft_quantum(x, 256, 128, "hann", fs=grid.fs)
        plot_spectrogram(Zc, out_plots / f"spec_classical_{sig_name}.png",
                         title=f"classical {sig_name}", freqs=freqs, times=times)
        plot_spectrogram(Zq, out_plots / f"spec_quantum_{sig_name}.png",
                         title=f"quantum {sig_name}", freqs=freqs, times=times)
        diff = spectrogram_difference(Zc, Zq)
        plot_difference_heatmap(diff["magnitude"],
                                out_plots / f"diff_mag_{sig_name}.png",
                                title=f"|Δ| {sig_name}", kind="magnitude")
        plot_difference_heatmap(diff["phase"],
                                out_plots / f"diff_phase_{sig_name}.png",
                                title=f"∠Δ {sig_name}", kind="phase")

    # Gram of quantum-frame coefficients for the first signal.
    _, x0 = reg[grid.signals[0]](duration=0.5, fs=grid.fs)
    Zq0, _, _ = stft_quantum(x0, 256, 128, "hann", fs=grid.fs)
    plot_matrix(gram_matrix(Zq0), out_plots / "gram_quantum.png",
                title="Gram of quantum frames", mode="magnitude")
    plot_matrix(covariance_matrix(Zq0), out_plots / "cov_quantum.png",
                title="Covariance of quantum coefficients", mode="magnitude")
