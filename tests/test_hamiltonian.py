"""Tests for exact and Trotterized Pauli-sum Hamiltonian evolution."""

import numpy as np
import pytest

from pauli_hamiltonian_evolution import (
    normalized_frobenius_error,
    pauli_sum_exact,
    pauli_sum_matrix,
    pauli_sum_trotter,
)


def test_single_term_is_exact_for_any_step_count() -> None:
    terms = {"X": 0.7}
    exact = pauli_sum_exact(terms, time=1.0)

    for steps in (1, 2, 5, 10):
        approximate = pauli_sum_trotter(terms, time=1.0, steps=steps)
        assert normalized_frobenius_error(approximate, exact) < 1e-12


def test_commuting_terms_are_exact_for_any_step_count() -> None:
    terms = {"ZI": 0.7, "ZZ": 0.4}
    exact = pauli_sum_exact(terms, time=1.0)

    for steps in (1, 2, 5, 10):
        approximate = pauli_sum_trotter(terms, time=1.0, steps=steps)
        assert normalized_frobenius_error(approximate, exact) < 1e-12


def test_noncommuting_trotter_error_decreases() -> None:
    terms = {"XI": 0.7, "ZZ": 0.4, "IY": -0.2}
    exact = pauli_sum_exact(terms, time=1.0)

    error_one = normalized_frobenius_error(
        pauli_sum_trotter(terms, time=1.0, steps=1),
        exact,
    )
    error_twenty = normalized_frobenius_error(
        pauli_sum_trotter(terms, time=1.0, steps=20),
        exact,
    )

    assert error_twenty < error_one


def test_pauli_sum_matrix_is_hermitian_for_real_coefficients() -> None:
    matrix = pauli_sum_matrix({"XI": 0.7, "ZZ": 0.4, "IY": -0.2})
    np.testing.assert_allclose(matrix, matrix.conj().T, atol=1e-12)


def test_pauli_sum_rejects_mismatched_string_lengths() -> None:
    with pytest.raises(ValueError):
        pauli_sum_matrix([("X", 1.0), ("ZZ", 0.5)])


def test_trotter_rejects_nonpositive_step_count() -> None:
    with pytest.raises(ValueError):
        pauli_sum_trotter({"X": 1.0}, time=1.0, steps=0)
