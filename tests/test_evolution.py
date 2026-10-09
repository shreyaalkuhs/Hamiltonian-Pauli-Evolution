"""Tests for explicit Pauli-string circuit synthesis."""

import numpy as np
import pytest
from qiskit.quantum_info import Operator

from pauli_hamiltonian_evolution import (
    exact_pauli_evolution,
    pauli_evolution,
    pauli_evolution_coherent,
    pauli_evolution_random,
)


@pytest.mark.parametrize(
    ("pauli_string", "theta"),
    [
        ("Y", 0.3),
        ("XIY", 0.3),
        ("XIZYI", 0.2),
        ("IIZII", 0.4),
        ("III", 0.3),
        ("X", 0.0),
        ("YZ", -0.4),
    ],
)
def test_pauli_evolution_matches_exact(pauli_string: str, theta: float) -> None:
    actual = Operator(pauli_evolution(pauli_string, theta)).data
    expected = exact_pauli_evolution(pauli_string, theta)
    np.testing.assert_allclose(actual, expected, atol=1e-12)


@pytest.mark.parametrize("bad_string", ["", "AB", "XQZ"])
def test_pauli_evolution_rejects_invalid_strings(bad_string: str) -> None:
    with pytest.raises(ValueError):
        pauli_evolution(bad_string, 0.2)


def test_cx_count_scales_with_pauli_weight() -> None:
    for weight in range(1, 8):
        circuit = pauli_evolution("X" * weight, 0.3)
        assert circuit.count_ops().get("cx", 0) == 2 * (weight - 1)


def test_zero_control_errors_recover_ideal_circuit() -> None:
    ideal = Operator(pauli_evolution("XIY", 0.3)).data
    coherent = Operator(pauli_evolution_coherent("XIY", 0.3, epsilon=0.0)).data
    random = Operator(
        pauli_evolution_random(
            "XIY",
            0.3,
            sigma=0.0,
            rng=np.random.default_rng(42),
        )
    ).data

    np.testing.assert_allclose(coherent, ideal, atol=1e-12)
    np.testing.assert_allclose(random, ideal, atol=1e-12)
