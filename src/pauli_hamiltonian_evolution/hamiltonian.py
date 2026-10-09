"""Exact references and first-order product formulas for Pauli Hamiltonians."""

from collections.abc import Callable, Iterable, Mapping
from numbers import Real
from typing import TypeAlias

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Pauli
from scipy.linalg import expm

from .evolution import pauli_evolution

PauliTerm: TypeAlias = tuple[str, Real]
PauliTerms: TypeAlias = Mapping[str, Real] | Iterable[PauliTerm]
EvolutionBuilder: TypeAlias = Callable[..., QuantumCircuit]


def _validated_pauli_terms(terms: PauliTerms) -> list[tuple[str, float]]:
    """Return ordered ``(Pauli string, coefficient)`` pairs after validation."""
    items = list(terms.items()) if isinstance(terms, Mapping) else list(terms)
    if not items:
        raise ValueError("Provide at least one Pauli term.")

    first_string = items[0][0]
    if not isinstance(first_string, str) or not first_string:
        raise ValueError("Pauli strings must be nonempty strings.")

    n_qubits = len(first_string)
    validated: list[tuple[str, float]] = []

    for pauli_string, coefficient in items:
        if (
            not isinstance(pauli_string, str)
            or len(pauli_string) != n_qubits
            or any(letter not in "IXYZ" for letter in pauli_string)
        ):
            raise ValueError(
                "Terms must use equal-length strings containing only I, X, Y, or Z."
            )
        if not isinstance(coefficient, Real) or not np.isfinite(coefficient):
            raise ValueError("Hamiltonian coefficients must be finite real numbers.")
        validated.append((pauli_string, float(coefficient)))

    return validated


def pauli_sum_matrix(terms: PauliTerms) -> np.ndarray:
    """Build the dense matrix H = sum_alpha c_alpha P_alpha."""
    validated = _validated_pauli_terms(terms)
    dimension = 2 ** len(validated[0][0])
    hamiltonian = np.zeros((dimension, dimension), dtype=complex)

    for pauli_string, coefficient in validated:
        hamiltonian += coefficient * Pauli(pauli_string).to_matrix()

    return hamiltonian


def pauli_sum_exact(terms: PauliTerms, time: Real) -> np.ndarray:
    """Return the exact dense reference exp(-i H time)."""
    if not isinstance(time, Real) or not np.isfinite(time):
        raise ValueError("time must be a finite real number.")
    return expm(-1j * float(time) * pauli_sum_matrix(terms))


def pauli_sum_trotter_circuit(
    terms: PauliTerms,
    time: Real,
    steps: int,
    evolution: EvolutionBuilder | None = None,
    **evolution_kwargs: object,
) -> QuantumCircuit:
    """Build the first-order product-formula circuit for a Pauli Hamiltonian.

    The term order is the insertion order for mappings or the supplied order for
    iterables of ``(Pauli string, coefficient)`` pairs.
    """
    validated = _validated_pauli_terms(terms)

    if not isinstance(time, Real) or not np.isfinite(time):
        raise ValueError("time must be a finite real number.")
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
        raise ValueError("steps must be a positive integer.")

    if evolution is None:
        evolution = pauli_evolution

    circuit = QuantumCircuit(len(validated[0][0]))
    dt = float(time) / steps

    for _ in range(steps):
        for pauli_string, coefficient in validated:
            circuit.compose(
                evolution(
                    pauli_string,
                    coefficient * dt,
                    **evolution_kwargs,
                ),
                inplace=True,
            )

    return circuit


def pauli_sum_trotter(
    terms: PauliTerms,
    time: Real,
    steps: int,
    evolution: EvolutionBuilder | None = None,
    **evolution_kwargs: object,
) -> np.ndarray:
    """Return the dense unitary of the first-order product-formula circuit."""
    circuit = pauli_sum_trotter_circuit(
        terms,
        time,
        steps,
        evolution=evolution,
        **evolution_kwargs,
    )
    return Operator(circuit).data
