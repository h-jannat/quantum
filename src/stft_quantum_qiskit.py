"""Optional qiskit-backed QFT demo for a single short frame.

Gated behind a guarded import so the core project runs without qiskit.
Intended for illustration only (N ∈ {4, 8, 16}) — the main experimental
pipeline uses :mod:`stft_quantum` which is faster and does not depend on
qiskit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    pass

_ALLOWED_N = (4, 8, 16)


def qft_frame_qiskit(frame: np.ndarray) -> np.ndarray:
    """Run a single frame through a qiskit ``QFT`` circuit on a statevector simulator.

    Parameters
    ----------
    frame : 1-D array of length 4, 8, or 16.

    Returns
    -------
    np.ndarray of the same length with the QFT'd amplitudes (complex).
    """
    try:
        from qiskit import QuantumCircuit  # type: ignore[import-not-found]
        from qiskit.circuit.library import QFT  # type: ignore[import-not-found]
        from qiskit.quantum_info import Statevector  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - optional dep
        raise ImportError(
            "qft_frame_qiskit requires the optional 'qiskit' package "
            "(pip install qiskit)"
        ) from exc

    frame = np.asarray(frame, dtype=np.complex128).ravel()
    N = frame.size
    if N not in _ALLOWED_N:
        raise ValueError(f"N must be one of {_ALLOWED_N}, got {N}")
    norm = np.linalg.norm(frame)
    if norm == 0.0:
        return frame.copy()
    state = frame / norm

    n_qubits = int(np.log2(N))
    circuit = QuantumCircuit(n_qubits)
    circuit.initialize(state, range(n_qubits))
    circuit.append(QFT(n_qubits, do_swaps=True), range(n_qubits))

    out = Statevector.from_instruction(circuit).data
    return np.asarray(out, dtype=np.complex128) * norm
