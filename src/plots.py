"""Matplotlib plotting helpers for the STFT comparison.

All functions save a PNG to ``path`` at ≥150 dpi and close the figure to
keep memory flat when called in a loop. DataFrame-based sweep plots
expect columns produced by :mod:`experiments`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import matplotlib
matplotlib.use("Agg")  # headless-safe default
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_DPI = 150


def _save(fig: plt.Figure, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_waveform(t: np.ndarray, x: np.ndarray, path: str | Path, title: str = "") -> None:
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(t, x, lw=0.8)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("amplitude")
    if title:
        ax.set_title(title)
    _save(fig, path)


def plot_spectrogram(
    S: np.ndarray,
    path: str | Path,
    title: str = "",
    freqs: np.ndarray | None = None,
    times: np.ndarray | None = None,
    db: bool = True,
) -> None:
    """Plot ``|S|`` as a heatmap; ``S`` is ``(n_frames, n_fft)``.

    Only the positive-frequency half is shown. ``freqs`` is expected to be
    ``np.fft.fftfreq``-style (two-sided); it is cropped to the first
    ``n_fft // 2 + 1`` bins.
    """
    mag = np.abs(S).T  # -> (n_fft, n_frames) for imshow
    half = mag.shape[0] // 2 + 1
    mag = mag[:half]
    if db:
        mag = 20.0 * np.log10(mag + 1e-12)
    fig, ax = plt.subplots(figsize=(8, 4))
    extent = None
    if freqs is not None and times is not None:
        extent = [times[0], times[-1], freqs[0], freqs[half - 1]]
    im = ax.imshow(
        mag, origin="lower", aspect="auto", cmap="magma", extent=extent,
    )
    ax.set_xlabel("time (s)" if times is not None else "frame")
    ax.set_ylabel("frequency (Hz)" if freqs is not None else "bin")
    if title:
        ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("dB" if db else "|S|")
    _save(fig, path)


def plot_difference_heatmap(
    D: np.ndarray,
    path: str | Path,
    title: str = "",
    kind: Literal["magnitude", "phase"] = "magnitude",
) -> None:
    data = D.T
    half = data.shape[0] // 2 + 1
    data = data[:half]
    fig, ax = plt.subplots(figsize=(8, 4))
    if kind == "phase":
        im = ax.imshow(data, origin="lower", aspect="auto", cmap="twilight",
                       vmin=-np.pi, vmax=np.pi)
        label = "phase diff (rad)"
    else:
        vmax = float(np.max(np.abs(data))) or 1.0
        im = ax.imshow(data, origin="lower", aspect="auto", cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax)
        label = "|A| - |B|"
    ax.set_xlabel("frame")
    ax.set_ylabel("bin")
    if title:
        ax.set_title(title)
    fig.colorbar(im, ax=ax, label=label)
    _save(fig, path)


def _sweep_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    path: str | Path,
    title: str,
    group_col: str | None = None,
    log_x: bool = False,
    log_y: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    if group_col is None:
        agg = df.groupby(x_col)[y_col].mean().reset_index()
        ax.plot(agg[x_col], agg[y_col], "o-")
    else:
        for key, sub in df.groupby(group_col):
            agg = sub.groupby(x_col)[y_col].mean().reset_index()
            ax.plot(agg[x_col], agg[y_col], "o-", label=str(key))
        ax.legend(title=group_col, fontsize=8)
    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    _save(fig, path)


def plot_runtime_vs_frame(df: pd.DataFrame, path: str | Path) -> None:
    _sweep_plot(
        df, "frame_len", "runtime_s", path,
        title="Runtime vs frame length",
        group_col="method",
        log_x=True, log_y=True,
    )


def plot_error_vs_frame(df: pd.DataFrame, path: str | Path, y_col: str = "mse") -> None:
    _sweep_plot(
        df, "frame_len", y_col, path,
        title=f"{y_col} vs frame length",
        log_x=True,
    )


def plot_error_vs_snr(df: pd.DataFrame, path: str | Path, y_col: str = "mse") -> None:
    _sweep_plot(
        df, "snr_db", y_col, path,
        title=f"{y_col} vs SNR",
    )


def plot_window_bars(df: pd.DataFrame, path: str | Path, y_col: str = "mse") -> None:
    agg = df.groupby("window")[y_col].mean().reset_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(agg["window"], agg[y_col], color="steelblue")
    ax.set_ylabel(y_col)
    ax.set_title(f"{y_col} by window (mean over other axes)")
    ax.grid(True, axis="y", alpha=0.3)
    _save(fig, path)


def plot_matrix(
    M: np.ndarray,
    path: str | Path,
    title: str = "",
    mode: Literal["magnitude", "real", "imag", "real_imag"] = "magnitude",
) -> None:
    """Visualise a 2-D matrix. ``real_imag`` produces side-by-side subplots."""
    if mode == "real_imag":
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, data, lbl in zip(axes, [M.real, M.imag], ["Re", "Im"]):
            vmax = float(np.max(np.abs(data))) or 1.0
            im = ax.imshow(data, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            ax.set_title(f"{title} ({lbl})" if title else lbl)
            fig.colorbar(im, ax=ax)
    else:
        if mode == "magnitude":
            data = np.abs(M); cmap = "viridis"
        elif mode == "real":
            data = M.real; cmap = "RdBu_r"
        else:
            data = M.imag; cmap = "RdBu_r"
        fig, ax = plt.subplots(figsize=(5, 4))
        if cmap == "RdBu_r":
            vmax = float(np.max(np.abs(data))) or 1.0
            im = ax.imshow(data, cmap=cmap, vmin=-vmax, vmax=vmax)
        else:
            im = ax.imshow(data, cmap=cmap)
        if title:
            ax.set_title(title)
        fig.colorbar(im, ax=ax)
    _save(fig, path)
