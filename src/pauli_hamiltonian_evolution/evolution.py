"""Circuit synthesis for exponentials of Pauli strings."""

from numbers import Real

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli


def _validate_pauli_string(pauli_string: str) -> None:
    """Validate a nonempty Pauli string."""
    if not isinstance(pauli_string, str) or not pauli_string:
        raise ValueError("Provide a nonempty Pauli string.")
    if any(letter not in "IXYZ" for letter in pauli_string):
        raise ValueError("Pauli strings may contain only I, X, Y, or Z.")


def _validate_real(value: Real, name: str) -> float:
    """Return a finite real value as float."""
    if not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite real number.")
    return float(value)


def _pauli_evolution_with_rz_shift(
    pauli_string: str,
    theta: Real,
    rz_shift: Real = 0.0,
) -> QuantumCircuit:
    """Build exp(-i theta P), optionally shifting the physical RZ angle.

    Qiskit convention is used: the rightmost Pauli character acts on qubit 0.
    For a non-identity Pauli string, the ideal circuit applies RZ(2*theta) to
    the parity target. ``rz_shift`` is added directly to that physical RZ angle.

    The all-identity string is implemented as the exact global phase exp(-i theta)
    and therefore has no RZ gate to perturb.
    """
    _validate_pauli_string(pauli_string)
    theta = _validate_real(theta, "theta")
    rz_shift = _validate_real(rz_shift, "rz_shift")

    circuit = QuantumCircuit(len(pauli_string))
    active_qubits: list[int] = []

    # Rotate every active Pauli into the Z basis.
    for qubit, letter in enumerate(reversed(pauli_string)):
        if letter == "X":
            circuit.h(qubit)
        elif letter == "Y":
            circuit.sdg(qubit)
            circuit.h(qubit)

        if letter != "I":
            active_qubits.append(qubit)

    if not active_qubits:
        circuit.global_phase = -theta
        return circuit

    # Store the parity of all active qubits on the last active qubit.
    target = active_qubits[-1]
    for control in active_qubits[:-1]:
        circuit.cx(control, target)

    circuit.rz(2 * theta + rz_shift, target)

    # Uncompute parity.
    for control in reversed(active_qubits[:-1]):
        circuit.cx(control, target)

    # Undo the basis changes.
    for qubit, letter in enumerate(reversed(pauli_string)):
        if letter == "X":
            circuit.h(qubit)
        elif letter == "Y":
            circuit.h(qubit)
            circuit.s(qubit)

    return circuit


def pauli_evolution(pauli_string: str, theta: Real) -> QuantumCircuit:
    """Return a circuit implementing exp(-i theta P)."""
    return _pauli_evolution_with_rz_shift(pauli_string, theta)


def exact_pauli_evolution(pauli_string: str, theta: Real) -> np.ndarray:
    """Return the exact dense matrix exp(-i theta P) for validation."""
    _validate_pauli_string(pauli_string)
    theta = _validate_real(theta, "theta")
    pauli_matrix = Pauli(pauli_string).to_matrix()
    dimension = pauli_matrix.shape[0]
    return (
        np.cos(theta) * np.eye(dimension, dtype=complex)
        - 1j * np.sin(theta) * pauli_matrix
    )


def pauli_evolution_coherent(
    pauli_string: str,
    theta: Real,
    epsilon: Real,
) -> QuantumCircuit:
    """Apply a fixed additive error epsilon to the physical RZ angle."""
    return _pauli_evolution_with_rz_shift(
        pauli_string,
        theta,
        rz_shift=epsilon,
    )


def pauli_evolution_random(
    pauli_string: str,
    theta: Real,
    sigma: Real,
    rng: np.random.Generator,
) -> QuantumCircuit:
    """Apply an independent Gaussian perturbation to the physical RZ angle."""
    sigma = _validate_real(sigma, "sigma")
    if sigma < 0:
        raise ValueError("sigma must be nonnegative.")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator.")

    delta = float(rng.normal(0.0, sigma))
    return _pauli_evolution_with_rz_shift(
        pauli_string,
        theta,
        rz_shift=delta,
    )
