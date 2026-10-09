"""Tests for numerical error metrics."""

import numpy as np
import pytest

from pauli_hamiltonian_evolution import normalized_frobenius_error


def test_normalized_frobenius_error_zero_for_identical_matrices() -> None:
    identity = np.eye(4, dtype=complex)
    assert normalized_frobenius_error(identity, identity) == 0.0


def test_normalized_frobenius_error_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError):
        normalized_frobenius_error(np.eye(2), np.eye(4))
