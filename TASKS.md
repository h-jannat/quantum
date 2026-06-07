# TASKS.md

Source of truth for what's done and what's next. Claude reads this at session start and updates it on task completion.

## Active

- **Phase:** ✅ All 12 phases complete
- **Epic:** —
- **Task:** —
- **Next:** extension / follow-ups per PRD §Expected deliverables

Update this block whenever the active task changes. It's the first thing a new session reads.

## How this file works

- Tasks are checked off when they meet the "Task completion checklist" in `CLAUDE.md`.
- Update this file *before* committing — check the box and advance the Active block in the same commit that completes the work. Don't record commit SHAs here; `git log --grep 'Refs: T-NNN'` is the lookup.
- Task IDs are stable — never renumber an existing task. Mark it `~~T-042~~ Dropped (reason)` if abandoned.
- When scope changes, add or renumber tasks *before* writing code. Don't let `TASKS.md` drift from reality.
- Phase 0 and Phase 1 are planned to task level. Phase 2+ stays at epic level until the prior phase is done.
- At phase end: re-plan the next phase to task level with what you learned, then archive the completed phase's task list to `docs/completed/phase-N.md`. Leave a one-line summary here pointing at the archive.

---

## Phase 0 — Project Scaffolding

- [x] **T-001 Create directory structure**
  - `src/`, `results/plots/`, `results/tables/`, `notebooks/`, `tests/`
  - Empty `__init__.py` in `src/` and `tests/`
- [x] **T-002 `requirements.txt`**
  - Core: `numpy`, `scipy`, `matplotlib`, `pandas`, `pytest`
  - Optional (commented): `seaborn`, `qiskit`, `qiskit-aer`, `psutil`, `soundfile`
- [x] **T-003 `README.md` skeleton**
  - Overview, install steps, quickstart (`python main.py`), project layout, limitations pointer
- [x] **T-004 `.gitignore`**
  - pyc, venv, `results/` artifacts optional
- [x] **T-005 Reproducibility config**
  - `src/config.py` with global `SEED = 1234`, default sample rate, default output paths
  - Helper `set_global_seed()` seeding `numpy` and `random`

---

## Phase 1 — Signals Module (`src/signals.py`)

- [x] **T-101** `sine(freq, duration, fs)` — single tone
- [x] **T-102** `sum_of_sines(freqs, amps, duration, fs)`
- [x] **T-103** `chirp_signal(f0, f1, duration, fs, method="linear")`
- [x] **T-104** `impulse(duration, fs, position=0.5)`
- [x] **T-105** `square_wave(freq, duration, fs)`
- [x] **T-106** `speech_like(duration, fs)` — formant-style: sum of modulated sinusoids + envelope
- [x] **T-107** `add_noise(signal, snr_db)` — AWGN at target SNR
- [x] **T-108** `load_audio(path)` — optional, guarded by `soundfile` import
- [x] **T-109** `get_signal_registry()` — dict mapping name → callable, used by experiments
- [x] **T-110** Acceptance: each function returns `(t, x)` with consistent shapes; docstrings present

---

## Phase 2 — Matrices Module (`src/matrices.py`)

- [x] **T-201** `dft_matrix(N)` — unnormalized DFT matrix, matches `np.fft.fft`
- [x] **T-202** `unitary_dft_matrix(N)` — `1/sqrt(N)` scaled, QFT-equivalent
- [x] **T-203** `verify_unitarity(U, tol=1e-10)` — returns bool
- [x] **T-204** `window(name, N)` — rect/hann/hamming/blackman (1-D)
- [x] **T-205** `window_matrix(name, N)` — diagonal form of the window
- [x] **T-206** `overlap_ratio_to_hop(frame_len, ratio)` — 0..1 → hop samples
- [x] **T-207** `framing_matrix(signal_len, frame_len, hop)` — sparse `(n_frames*frame_len, signal_len)`
- [x] **T-208** `identity_op(N)`, `lowpass_op(N, cutoff_bin)`, `highpass_op`, `bandpass_op(N, lo, hi)`
- [x] **T-209** `random_diag_op(N, seed)` — diagonal unitary-ish spectral mask
- [x] **T-210** `toeplitz_conv_matrix(kernel, N)` — circular/linear convolution operator
- [x] **T-211** Algebraic relation helper `unitary_dft_from_dft(D)` / sanity check `U = D / sqrt(N)`

---

## Phase 3 — Classical STFT (`src/stft_classical.py`)

- [x] **T-301** `frame_signal(x, frame_len, hop)` → `(n_frames, frame_len)` array
- [x] **T-302** `stft_custom(x, frame_len, hop, window_name, n_fft=None)` — framing + windowing + zero-pad + `np.fft.fft`; returns `(Z, freqs, times)`
- [x] **T-303** `stft_scipy(x, fs, frame_len, hop, window_name)` — wrapper around `scipy.signal.stft` returning a comparable shape
- [x] **T-304** `istft_custom(Z, hop, window_name, signal_len)` — overlap-add reconstruction with window normalization
- [x] **T-305** `validate_against_scipy(x, fs, ...)` — max-abs-error between custom and scipy STFT outputs

---

## Phase 4 — Quantum-Inspired STFT (`src/stft_quantum.py`)

- [x] **T-401** `next_pow2(n)` helper
- [x] **T-402** `amplitude_encode(frame)` → `(state, norm)` unit-L2 + scalar
- [x] **T-403** `qft_simulate(state)` — `np.fft.fft(state, norm="ortho")`
- [x] **T-404** `stft_quantum(x, frame_len, hop, window_name, n_fft=None)` — window → pow2 pad → encode → unitary DFT → rescale
- [x] **T-405** Module docstring block "Differences from gate-level QFT"
- [x] **T-406** Optional `src/stft_quantum_qiskit.py` single-frame demo (guarded import, N ∈ {4,8,16})

---

## Phase 5 — Metrics (`src/metrics.py`)

- [x] **T-501** `mse`, `mae`, `frobenius` on complex/real arrays
- [x] **T-502** `cosine_similarity_flat`, `pearson_flat` on magnitudes
- [x] **T-503** `phase_error_stats` — wrapped phase-diff mean/median/std
- [x] **T-504** `energy_preservation_error(x, Z)` (Parseval check)
- [x] **T-505** `reconstruction_error(x, x_hat)`
- [x] **T-506** `gram_matrix(frames)`, `covariance_matrix(frames)`
- [x] **T-507** `spectrogram_difference(A, B)` → magnitude + phase diff
- [x] **T-508** `framewise_correlation(A, B)`
- [x] **T-509** `timeit_call(fn, *args, repeats=3)` → median seconds
- [ ] **T-510** `measure_memory(fn, *args)` via `tracemalloc` (optional/guarded)

---

## Phase 6 — Plots (`src/plots.py`)

- [x] **T-601** `plot_waveform(t, x, path, title)`
- [x] **T-602** `plot_spectrogram(S, path, title, freqs, times)` with dB scale + colorbar
- [x] **T-603** `plot_difference_heatmap(D, path, title, kind="magnitude"|"phase")`
- [x] **T-604** `plot_runtime_vs_frame`, `plot_error_vs_frame`, `plot_error_vs_snr` (DataFrame in, PNG out)
- [x] **T-605** `plot_window_bars(df, path)`
- [x] **T-606** `plot_matrix(M, path, title, mode)` for DFT / unitary / Gram / Toeplitz
- [x] **T-607** All figures ≥150 dpi under `results/plots/` via `save_figure`

---

## Phase 7 — Experiments (`src/experiments.py`)

- [x] **T-701** `ExperimentConfig` dataclass + grid enumerator
- [x] **T-702** `run_single(config)` executes both STFTs and computes metrics
- [x] **T-703** `run_grid(configs)` → DataFrame + `best_worst` summary
- [x] **T-704** Matrix-view experiment (DFT vs unitary DFT, filter ops, Gram)
- [x] **T-705** Scaling (runtime vs `n_fft`) + noise sweep
- [x] **T-706** Persist CSVs + plots under `results/`

---

## Phase 8 — Entry Point (`main.py`)

- [x] **T-801** `main.py` CLI (`--quick`/`--full`/`--skip-plots`/`--seed`) orchestrating `run_all`

---

## Phase 9 — Tests (`tests/`)

- [x] **T-901** `test_signals.py` — shapes, SNR accuracy, registry
- [x] **T-902** `test_framing.py` — `frame_signal` vs manual slicing
- [x] **T-903** `test_matrices.py` — DFT / unitary-DFT equivalence, unitarity
- [x] **T-904** `test_quantum_stft.py` — normalization round-trip, matches classical at pow2
- [x] **T-905** `test_classical_vs_scipy.py` — max-abs-err < 1e-8
- [x] **T-906** `test_metrics.py` — known-value checks

---

## Phase 10 — Notebook (`notebooks/exploratory.ipynb`)

- [x] **T-1001** Minimal exploratory notebook: signal load → both STFTs → diff + matrix viz + narration

---

## Phase 11 — Report (`results/REPORT.md`)

- [x] **T-1101** Methods, methodology, closest-match/divergence analysis, conclusions, Limitations

---

## Phase 12 — Polish

- [x] **T-1201** Expand README (usage, expected runtime, troubleshooting, link to REPORT)
- [x] **T-1202** Final smoke test of `python main.py --quick` + `pytest` end-to-end

---

## Deliverables checklist (maps to PRD §Expected deliverables)

- [x] Full source code under `src/`
- [x] `requirements.txt`
- [x] `README.md` with setup + usage
- [x] Generated plots in `results/plots/`
- [x] Generated tables in `results/tables/`
- [x] `results/REPORT.md` with Limitations section
