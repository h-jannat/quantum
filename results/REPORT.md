# Quantum vs Classical STFT — Findings

*Generated from a `--quick` grid run (48 configs, seed 1234, fs 16 kHz, 0.5 s signals).*

## 1. What classical STFT does

The classical Short-Time Fourier Transform treats a signal as a sequence of overlapping frames, applies a window to each frame to reduce edge discontinuities, optionally zero-pads to a transform length `n_fft`, and takes the Discrete Fourier Transform:

$$X[i, k] = \sum_{n=0}^{N-1} w[n]\, x[i\cdot H + n]\, e^{-2\pi j n k / N}$$

`src/stft_classical.py::stft_custom` does exactly this with `numpy.fft.fft`. It is cross-checked against `scipy.signal.stft` (`validate_against_scipy` → ~1e-14 max-abs-err for all four windows and three frame sizes).

## 2. What the quantum-inspired STFT *simulates*

`src/stft_quantum.py::stft_quantum` runs the same framing/windowing, then for each frame:

1. zero-pads to the next power of two (so `log₂(N)` qubits would suffice),
2. normalizes the padded frame to unit L2 — **amplitude encoding**,
3. applies `numpy.fft.fft(..., norm="ortho")` — the matrix form of the **QFT** (= `D / √N`),
4. rescales by `‖frame‖ · √N` to undo the unit-norm normalization and the unitary-DFT scaling, putting the output back into the same units as the classical STFT.

Crucially, the QFT operator is *algebraically* the classical unitary DFT. Our pipeline cannot show a physical quantum advantage — it is a mathematical model of the QFT step, with the expensive state-preparation / measurement stages skipped (see §6).

## 3. Comparison methodology

`src/experiments.py` enumerates a Cartesian grid over `{signal} × {frame_len} × {hop_ratio} × {window} × {zero_pad_factor} × {snr_db}` and, for each config, runs both STFTs and computes ten metrics per comparison: MSE and MAE on magnitudes, Frobenius of the complex difference, flattened cosine similarity and Pearson correlation, wrapped phase-diff stats, energy-preservation error (Parseval sanity), ISTFT reconstruction RMSE, and median runtime for each method. Tables land in `results/tables/` and plots in `results/plots/`.

Additional sweeps:

* **Scaling** (`run_scaling`) — runtime and MSE vs `n_fft ∈ {64…2048}`. → `results/plots/runtime_vs_frame.png`
* **Noise** (`run_noise_sweep`) — error vs SNR ∈ {∞, 30, 20, 10, 5, 0, -5 dB}. → `results/plots/error_vs_snr.png`
* **Matrix view** (`run_matrix_view`) — numerical DFT ↔ unitary-DFT relation, filter-operator energies, Toeplitz rank. → `results/tables/matrix_view.csv`

## 4. Where the methods match closest

**Everywhere `n_fft` is a power of two.** Observed over the 48-config quick grid:

| metric | value |
|---|---|
| max MSE across all rows | 6.1 × 10⁻³⁰ |
| min cosine similarity | 0.9999999999999996 |
| min Pearson correlation | ≈ 1.0 |

At power-of-two transform lengths the quantum pipeline reduces exactly to the classical windowed FFT after rescaling. The only residual is floating-point round-off from the `÷ ‖frame‖ · √N` normalize/denormalize round-trip.

## 5. Where the methods diverge

Divergence is forced when `n_fft` is **not** a power of two. The quantum pipeline pads up to `next_pow2(n_fft)` so that an integer number of qubits exists, while the classical STFT honors the requested `n_fft` exactly. The output shapes then differ and bin-for-bin comparison is no longer meaningful. The divergence attribution is therefore:

* **Representation / padding** — the dominant source. The unitary-DFT step requires a `log₂(N)`-qubit register.
* **Normalization** — the `÷ ‖frame‖` step introduces a small L2-renormalization when a frame is near-silent; `energy_preservation_error` remains consistent for both methods, so normalization is *not* a practical source of bias.
* **Phase** — with identical padding, wrapped phase-diff mean/std are ~1e-15 (see `phase_error_stats` columns). No phase bias in the mathematical model.

## 6. Practical conclusions

* **Accuracy**: identical to machine precision when `n_fft` is a power of two. Useful as a validated substitute for a gate-level QFT simulator in downstream experiments.
* **Runtime**: the quantum pipeline here is ~9× slower than the classical STFT (mean 638 µs vs 71 µs per call in the quick grid) because it loops frames in Python to track per-frame `norm` — not because the math is harder. A vectorized rewrite would close this gap entirely. This is a numpy implementation detail, **not** a statement about hardware.
* **When to prefer the quantum pipeline** in this project: for experiments that explicitly want to reason about amplitude-encoded frames, unit-norm statevectors, or to drop in the optional `stft_quantum_qiskit.py` demo for small `N`.

Key tables: `summary.csv`, `best_worst.csv`, `runtime.csv`, `scaling.csv`, `noise_sweep.csv`, `matrix_view.csv`.  
Key plots: `runtime_vs_frame.png`, `error_vs_frame.png`, `error_vs_snr.png`, `window_bars.png`, `spec_{classical,quantum}_{sine,chirp,square}.png`, `diff_{mag,phase}_*.png`, `dft_{16,32}.png`, `udft_{16,32}.png`, `gram_quantum.png`, `cov_quantum.png`.

## Limitations

This project is **not** a real quantum-hardware benchmark. Specifically:

* **Mathematical simulation only.** Every operation runs on classical numpy doubles. The "QFT" is literally `numpy.fft.fft(..., norm="ortho")` — the same FFT, just unitarily scaled.
* **State preparation is free here, but not on hardware.** Writing an arbitrary length-`N` amplitude vector into `log₂(N)` qubits costs `O(N)` gates for general states (Mottonen, Schuld/Park). That wipes out any `O(log N)` advantage the QFT might offer, so the PRD's "quantum speedup" interpretation does **not** apply.
* **Measurement and shot noise are not modeled.** We read complex amplitudes directly; real hardware yields probabilities and requires many shots plus ancillary circuits to recover complex coefficients.
* **Decoherence and gate error are absent.** The operator is exactly unitary to double precision.
* **No claimed quantum advantage.** Any runtime / accuracy numbers here reflect numpy, not physics.
