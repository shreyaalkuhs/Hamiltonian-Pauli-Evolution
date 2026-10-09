"""Numerical error metrics used in the project."""

import numpy as np


def normalized_frobenius_error(
    approximate: np.ndarray,
    exact: np.ndarray,
) -> float:
    """Return ||U_approx - U_exact||_F / sqrt(d) for square d x d matrices."""
    if approximate.shape != exact.shape:
        raise ValueError("Matrices must have the same shape.")
    if approximate.ndim != 2 or approximate.shape[0] != approximate.shape[1]:
        raise ValueError("Matrices must be square.")

    dimension = approximate.shape[0]
    return float(np.linalg.norm(approximate - exact, ord="fro") / np.sqrt(dimension))
