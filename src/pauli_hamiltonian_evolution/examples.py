"""Small reference model used to introduce Trotterization."""

from numbers import Real

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli

from .evolution import pauli_evolution

_X = Pauli("X").to_matrix()
_Z = Pauli("Z").to_matrix()


def exact_ax_bz(a: Real, b: Real, time: Real) -> np.ndarray:
    """Return exact evolution for H = a X + b Z."""
    a = float(a)
    b = float(b)
    time = float(time)
    omega = np.sqrt(a**2 + b**2)

    if np.isclose(omega, 0.0):
        return np.eye(2, dtype=complex)

    hamiltonian = a * _X + b * _Z
    return (
        np.cos(omega * time) * np.eye(2, dtype=complex)
        - 1j * np.sin(omega * time) * hamiltonian / omega
    )


def trotter_ax_bz_circuit(
    a: Real,
    b: Real,
    time: Real,
    steps: int,
) -> QuantumCircuit:
    """Build first-order Trotter evolution for H = a X + b Z."""
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
        raise ValueError("steps must be a positive integer.")

    circuit = QuantumCircuit(1)
    dt = float(time) / steps

    for _ in range(steps):
        circuit.compose(pauli_evolution("X", float(a) * dt), inplace=True)
        circuit.compose(pauli_evolution("Z", float(b) * dt), inplace=True)

    return circuit
