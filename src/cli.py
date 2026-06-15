"""Command-line interface for the Quantum vs Classical STFT comparison."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import typer

from src.config import PLOTS_DIR, SEED, TABLES_DIR, ensure_output_dirs, set_global_seed
from src.experiments import FULL_GRID, QUICK_GRID, run_all


def _print_summary(df) -> None:
    if df.empty:
        print("No experiments were run.")
        return

    cols = ["signal", "frame_len", "window", "zero_pad_factor", "snr_db", "mse", "cosine"]
    best = df.nsmallest(3, "mse")[cols]
    worst = df.nlargest(3, "mse")[cols]

    print("\n=== Best agreement (lowest MSE) ===")
    print(best.to_string(index=False))
    print("\n=== Worst agreement (highest MSE) ===")
    print(worst.to_string(index=False))

    mean_runtime = df[["runtime_classical_s", "runtime_quantum_s"]].mean()
    print("\nMean runtime per STFT call (seconds):")
    print(mean_runtime.to_string())


def main(
    quick: Annotated[
        bool,
        typer.Option("--quick", help="Run the small quick grid. This is the default."),
    ] = False,
    full: Annotated[
        bool,
        typer.Option("--full", help="Run the full PRD grid. This is slower."),
    ] = False,
    skip_plots: Annotated[
        bool,
        typer.Option("--skip-plots", help="Skip figure generation."),
    ] = False,
    seed: Annotated[int, typer.Option("--seed", help="Global RNG seed.")] = SEED,
    tables: Annotated[
        Path,
        typer.Option("--tables", help="CSV output directory."),
    ] = TABLES_DIR,
    plots: Annotated[
        Path,
        typer.Option("--plots", help="Plot output directory."),
    ] = PLOTS_DIR,
    verbose: Annotated[
        bool,
        typer.Option("-v", "--verbose", help="Enable verbose logging."),
    ] = False,
) -> None:
    if quick and full:
        raise typer.BadParameter("Use either --quick or --full, not both.")

    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(message)s",
    )

    set_global_seed(seed)
    ensure_output_dirs()
    tables.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)

    grid = FULL_GRID if full else QUICK_GRID
    df = run_all(grid, out_tables=tables, out_plots=plots, skip_plots=skip_plots)
    _print_summary(df)
    print(f"\nTables: {tables}\nPlots:  {plots}")


def run_cli() -> None:
    typer.run(main)
