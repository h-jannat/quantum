"""Project-wide configuration: seeds, default parameters, and output paths.

Import `set_global_seed()` at the start of any script that consumes RNGs to
keep runs reproducible. Import `ensure_output_dirs()` before writing plots or
tables so the target directories exist.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

SEED: int = 1234
"""Default RNG seed used across the project."""

DEFAULT_FS: int = 16_000
"""Default sampling rate (Hz) for synthetic test signals."""

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
RESULTS_DIR: Path = REPO_ROOT / "results"
PLOTS_DIR: Path = RESULTS_DIR / "plots"
TABLES_DIR: Path = RESULTS_DIR / "tables"


def set_global_seed(seed: int = SEED) -> None:
    """Seed Python `random`, NumPy, and `PYTHONHASHSEED` for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def ensure_output_dirs() -> None:
    """Create `results/plots` and `results/tables` if they don't exist."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
