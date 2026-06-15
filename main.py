"""CLI entry point for the Quantum vs Classical STFT comparison.

Examples
--------
    python main.py --quick             # small grid, fast smoke run
    python main.py --full              # full grid from the PRD
    python main.py --quick --skip-plots  # CSVs only
"""

from __future__ import annotations

from src.cli import run_cli


if __name__ == "__main__":
    run_cli()
