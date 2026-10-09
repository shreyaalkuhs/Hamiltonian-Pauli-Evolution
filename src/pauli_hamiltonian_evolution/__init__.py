"""Tools for Pauli-string evolution and first-order Trotter simulation."""

from .evolution import (
    exact_pauli_evolution,
    pauli_evolution,
    pauli_evolution_coherent,
    pauli_evolution_random,
)
from .examples import exact_ax_bz, trotter_ax_bz_circuit
from .hamiltonian import (
    pauli_sum_exact,
    pauli_sum_matrix,
    pauli_sum_trotter,
    pauli_sum_trotter_circuit,
)
from .metrics import normalized_frobenius_error

__all__ = [
    "exact_ax_bz",
    "exact_pauli_evolution",
    "normalized_frobenius_error",
    "pauli_evolution",
    "pauli_evolution_coherent",
    "pauli_evolution_random",
    "pauli_sum_exact",
    "pauli_sum_matrix",
    "pauli_sum_trotter",
    "pauli_sum_trotter_circuit",
    "trotter_ax_bz_circuit",
]
