"""CLI entry point for the Quantum vs Classical STFT comparison.

Examples
--------
    python main.py --quick             # small grid, fast smoke run
    python main.py --full              # full grid from the PRD
    python main.py --quick --skip-plots  # CSVs only
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.config import PLOTS_DIR, SEED, TABLES_DIR, ensure_output_dirs, set_global_seed
from src.experiments import FULL_GRID, QUICK_GRID, run_all


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    grp = p.add_mutually_exclusive_group()
    grp.add_argument("--quick", action="store_true", help="run the small quick grid (default)")
    grp.add_argument("--full", action="store_true", help="run the full PRD grid (slower)")
    p.add_argument("--skip-plots", action="store_true", help="skip figure generation")
    p.add_argument("--seed", type=int, default=SEED, help="global RNG seed")
    p.add_argument("--tables", type=Path, default=TABLES_DIR, help="CSV output directory")
    p.add_argument("--plots", type=Path, default=PLOTS_DIR, help="plot output directory")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def _print_summary(df) -> None:
    if df.empty:
        print("No experiments were run.")
        return
    best = df.nsmallest(3, "mse")[["signal", "frame_len", "window", "zero_pad_factor", "snr_db", "mse", "cosine"]]
    worst = df.nlargest(3, "mse")[["signal", "frame_len", "window", "zero_pad_factor", "snr_db", "mse", "cosine"]]
    print("\n=== Best agreement (lowest MSE) ===")
    print(best.to_string(index=False))
    print("\n=== Worst agreement (highest MSE) ===")
    print(worst.to_string(index=False))
    mean_runtime = df[["runtime_classical_s", "runtime_quantum_s"]].mean()
    print("\nMean runtime per STFT call (seconds):")
    print(mean_runtime.to_string())


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose or True else logging.WARNING,
        format="%(message)s",
    )

    set_global_seed(args.seed)
    ensure_output_dirs()
    args.tables.mkdir(parents=True, exist_ok=True)
    args.plots.mkdir(parents=True, exist_ok=True)

    grid = FULL_GRID if args.full else QUICK_GRID
    df = run_all(grid, out_tables=args.tables, out_plots=args.plots, skip_plots=args.skip_plots)
    _print_summary(df)
    print(f"\nTables: {args.tables}\nPlots:  {args.plots}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
