# Quantum vs Classical STFT

A reproducible research-style Python project that implements a classical Short-Time Fourier Transform (STFT) alongside a **quantum-inspired** STFT simulator (amplitude encoding + unitary DFT / QFT-equivalent), and compares the two across multiple signals, windows, frame/hop configurations, spectral operators, and noise conditions. The quantum pipeline is a *mathematical simulation* — see the Limitations section of the generated report.

## Install

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Quickstart

```bash
python main.py --quick         # small grid, ~10 s on a laptop
python main.py --full          # full PRD grid, ~5–10 min
python main.py --quick --skip-plots   # CSVs only
pytest                         # run unit tests (~1 s, 55 tests)
```

Outputs land in `results/tables/` (CSVs) and `results/plots/` (PNGs). `results/REPORT.md` is the write-up.

## Usage

```bash
python main.py [--quick | --full] [--skip-plots] [--seed N]
               [--tables DIR] [--plots DIR] [-v]
```

Programmatic use:

```python
from src.experiments import run_all, QUICK_GRID, FULL_GRID
df = run_all(QUICK_GRID)      # returns a pandas DataFrame
```

Individual pieces are in `src/`: `signals.get_signal_registry()`,
`stft_classical.stft_custom`, `stft_quantum.stft_quantum`,
`metrics.*`, `matrices.*`.

## Project layout

```
quantum/
├── src/                  # core modules
│   ├── config.py         # seeds, paths, reproducibility
│   ├── signals.py        # test-signal generators          (Phase 1)
│   ├── matrices.py       # DFT / window / framing / filter ops (Phase 2)
│   ├── stft_classical.py # custom + scipy STFT            (Phase 3)
│   ├── stft_quantum.py   # quantum-inspired STFT           (Phase 4)
│   ├── metrics.py        # error + runtime metrics        (Phase 5)
│   ├── plots.py          # plotting helpers               (Phase 6)
│   └── experiments.py    # grid runner                    (Phase 7)
├── tests/                # pytest unit tests
├── notebooks/            # exploratory notebooks
├── results/
│   ├── plots/            # generated figures
│   ├── tables/           # generated CSVs
│   └── REPORT.md         # final write-up + Limitations
├── main.py               # CLI entry point                 (Phase 8)
├── requirements.txt
├── PRD.md                # product requirements
└── TASKS.md              # task tracker (source of truth)
```

## Troubleshooting

* **`ModuleNotFoundError: pandas` / `matplotlib`** — install the core requirements: `pip install -r requirements.txt`.
* **Plots don't appear in a GUI** — `src/plots.py` uses the headless `Agg` backend by design; images are written to disk, not shown interactively. Open them from `results/plots/`.
* **`qiskit` import errors** — qiskit is optional. The main pipeline works without it; only `src/stft_quantum_qiskit.py` needs it.
* **Notebook can't find `src`** — run Jupyter from the repo root so the first notebook cell's `sys.path` insert resolves.

## Limitations

This project is **not** a real quantum hardware benchmark. The "quantum STFT" is a statevector-level mathematical simulation using unitary DFT. State preparation and measurement costs are not modeled, and no physical quantum advantage is claimed. See `results/REPORT.md` → *Limitations* for the full disclaimer.
# quantum
