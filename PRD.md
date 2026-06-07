Create a complete Python research-style project that simulates a “quantum STFT” and compares it against the regular classical STFT.

Project goal:
Build a reproducible Python project that:

1. implements a regular STFT,
2. implements a quantum-inspired STFT simulator based on windowing + amplitude encoding + unitary DFT/QFT-style transform,
3. compares both methods on multiple signals, parameters, and matrix/operator formulations,
4. produces clear plots, tables, and a short report-style summary of findings.

Important interpretation:
Do not claim physical quantum speedup. This is a simulation of the mathematical pipeline in Python. The “quantum STFT” should be implemented as a quantum-inspired/statevector-level model using unitary DFT (QFT-equivalent behavior), with optional Qiskit integration only as an extra module if useful.

Technical requirements:

- Use Python.
- Main libraries: numpy, scipy, matplotlib, pandas.
- Optional: seaborn only if really useful for heatmaps, but prefer matplotlib.
- Optional: qiskit for an additional demonstration, but the core project must work without qiskit.
- Organize the project cleanly into modules, scripts, and notebooks if needed.
- Add comments and docstrings.
- Make the code easy to run.

Core implementation:

1. Classical STFT
   - Implement regular STFT from scratch using framing, windowing, and FFT.
   - Also provide a reference implementation using scipy.signal.stft for validation.
   - Support multiple window types and hop sizes.

2. Quantum-inspired STFT
   - For each frame:
     - apply a window,
     - pad to a power-of-two length when needed,
     - normalize the frame for amplitude encoding,
     - apply a unitary DFT using numpy.fft.fft(..., norm="ortho") as a QFT simulator,
     - preserve complex coefficients,
     - rescale correctly after normalization so results remain comparable to classical STFT.
   - Return a spectrogram-like representation.
   - Make the implementation mathematically clear and explicitly explain where it differs from real gate-level quantum execution.

3. Optional advanced version
   - Add an optional module that demonstrates one small frame using qiskit QFT for tiny dimensions like 4, 8, or 16, only for illustration.
   - Keep this optional because the main comparison must remain fast and practical.

Comparison design:
Compare the two methods using many meaningful matrices/operators, not just one signal.
Interpret “many possible matrices” as several matrix-based viewpoints, including:

A. Transform matrices

- Classical DFT matrix
- Unitary DFT / QFT-equivalent matrix
- Compare their algebraic relationship and numerical outputs

B. Window matrices

- Rectangular
- Hann
- Hamming
- Blackman
- Show how the choice of window affects both methods

C. Framing/overlap matrices

- Different frame lengths
- Different hop sizes
- Overlap ratios such as 0%, 25%, 50%, 75%

D. Filtering / operator matrices

- Identity
- Low-pass
- High-pass
- Band-pass
- Random diagonal spectral operators
- Toeplitz convolution matrix view for at least one experiment

E. Signal similarity / error matrices

- Spectrogram difference matrix
- Magnitude difference matrix
- Phase difference matrix
- Frame-wise correlation matrix
- Gram matrix of frame embeddings
- Optional covariance matrices of coefficients across frames

F. Noise-condition matrices

- Compare under multiple SNR levels
- Clean, mildly noisy, and heavily noisy cases

Signals to test:
Include multiple types of signals:

- single sine wave
- sum of sinusoids
- chirp
- impulse
- square wave
- synthetic speech-like signal
- noisy signal
- optional real audio file if available

Metrics:
Compute and report several metrics, such as:

- MSE between magnitude spectrograms
- MAE
- Frobenius norm of spectrogram difference
- cosine similarity
- Pearson correlation of flattened spectrogram magnitudes
- phase error statistics
- energy preservation error
- reconstruction error if ISTFT-style reconstruction is included
- runtime
- memory usage if practical
- scaling with frame length / transform length

Visual outputs:
Generate high-quality plots for the comparison:

1. waveform plots of test signals
2. classical STFT spectrogram
3. quantum-inspired STFT spectrogram
4. absolute difference heatmap
5. phase difference heatmap
6. runtime vs frame size plot
7. error vs frame size plot
8. error vs SNR plot
9. bar charts comparing window choices
10. optional matrix visualizations for DFT, unitary DFT, Gram, covariance, and Toeplitz matrices

Tabular outputs:
Create clean tables in CSV and optionally Markdown:

- metric summary for each signal
- metric summary for each window type
- metric summary for each frame/hop configuration
- runtime summary
- best and worst configurations

Project structure:
Build something like:

- src/
  - stft_classical.py
  - stft_quantum.py
  - matrices.py
  - signals.py
  - metrics.py
  - plots.py
  - experiments.py
- results/
  - plots/
  - tables/
- notebooks/
  - exploratory notebook
- README.md
- requirements.txt
- main.py

Experimental workflow:

- Create a reproducible experiment runner.
- Use fixed random seeds.
- Run a grid of experiments over:
  - signal type
  - frame length
  - hop length
  - window type
  - zero-padding length
  - noise level
- Save all outputs automatically.

Analysis/reporting:
At the end, generate a concise report in Markdown that explains:

- what classical STFT does,
- what the quantum-inspired STFT simulation does,
- how the comparison was performed,
- which configurations made the two methods closest,
- where they differed,
- whether the differences are mainly due to normalization, padding, phase, or representation choices,
- practical conclusions.

Validation:

- Verify the classical custom STFT against scipy.signal.stft.
- Add small unit tests for core pieces like:
  - frame extraction,
  - unitary DFT equivalence,
  - normalization/rescaling,
  - metric calculations.

Coding style:

- Write clean, readable, modular code.
- Include docstrings and comments.
- Prefer clarity over cleverness.
- Do not leave placeholders; provide runnable code.

Expected deliverables:

1. full source code,
2. requirements.txt,
3. README with setup and usage,
4. generated plots,
5. generated tables,
6. short Markdown summary of results.

Also include a section titled “Limitations” clearly stating:

- this is not a real quantum hardware benchmark,
- the quantum STFT here is a mathematical simulation,
- state preparation and measurement costs are not fully modeled,
- any apparent performance advantage in this project should not be presented as physical quantum advantage.

Make the final result look like a small research project that I can run, inspect, and extend.
