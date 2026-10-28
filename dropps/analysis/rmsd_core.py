"""Numerical kernels for independently fitted molecular RMSD."""

from __future__ import annotations

import numpy as np


def _coordinates_and_weights(mobile, reference, weights):
    """Validate batched coordinate arrays and expand their weights."""

    mobile_array = np.asarray(mobile, dtype=float)
    reference_array = np.asarray(reference, dtype=float)
    if mobile_array.ndim != 3 or mobile_array.shape[2] != 3:
        raise ValueError(
            "Mobile coordinates must have shape (n_molecules, n_atoms, 3)."
        )
    if reference_array.shape != mobile_array.shape:
        raise ValueError("Reference coordinates must match mobile coordinates.")
    if mobile_array.shape[0] == 0 or mobile_array.shape[1] == 0:
        raise ValueError("At least one molecule and one selected atom are required.")
    if not np.all(np.isfinite(mobile_array)) or not np.all(
        np.isfinite(reference_array)
    ):
        raise ValueError("RMSD coordinates contain non-finite values.")

    molecule_count, atom_count, _ = mobile_array.shape
    if weights is None:
        weight_array = np.ones((molecule_count, atom_count), dtype=float)
    else:
        weight_array = np.asarray(weights, dtype=float)
        if weight_array.shape == (atom_count,):
            weight_array = np.broadcast_to(
                weight_array,
                (molecule_count, atom_count),
            )
        elif weight_array.shape != (molecule_count, atom_count):
            raise ValueError(
                "Weights must have shape (n_atoms,) or (n_molecules, n_atoms)."
            )
    if not np.all(np.isfinite(weight_array)) or np.any(weight_array < 0.0):
        raise ValueError("RMSD weights must be non-negative and finite.")

    weight_sums = np.sum(weight_array, axis=1)
    if np.any(weight_sums <= 0.0):
        raise ValueError("Every molecule must have a positive total RMSD weight.")
    return mobile_array, reference_array, weight_array, weight_sums


def batched_residual_rmsd(residuals, weights=None):
    """Calculate one RMSD per batch item from already aligned residuals."""

    residuals = np.asarray(residuals, dtype=float)
    zeros = np.zeros_like(residuals)
    residuals, _, weights, weight_sums = _coordinates_and_weights(
        residuals,
        zeros,
        weights,
    )
    squared_distances = np.einsum("mai,mai->ma", residuals, residuals)
    mean_squared = np.einsum("ma,ma->m", squared_distances, weights) / weight_sums
    return np.sqrt(np.maximum(mean_squared, 0.0))


def batched_unfitted_rmsd(mobile, reference, weights=None):
    """Return coordinate RMSD without removing translation or rotation."""

    mobile, reference, weights, _ = _coordinates_and_weights(
        mobile,
        reference,
        weights,
    )
    return batched_residual_rmsd(mobile - reference, weights)


def batched_kabsch_residuals(mobile, reference, weights=None):
    """Return residual coordinates after an independent Kabsch fit per item.

    Coordinates use row-vector convention. The fit removes translation and
    permits only a proper rotation; mirror reflection is not allowed.
    """

    mobile, reference, weights, weight_sums = _coordinates_and_weights(
        mobile,
        reference,
        weights,
    )

    mobile_centers = (
        np.einsum("mai,ma->mi", mobile, weights) / weight_sums[:, np.newaxis]
    )
    reference_centers = (
        np.einsum("mai,ma->mi", reference, weights) / weight_sums[:, np.newaxis]
    )
    centered_mobile = mobile - mobile_centers[:, np.newaxis, :]
    centered_reference = reference - reference_centers[:, np.newaxis, :]

    covariance = np.einsum(
        "mai,maj,ma->mij",
        centered_mobile,
        centered_reference,
        weights,
    )
    left, _, right_transpose = np.linalg.svd(covariance)
    rotations = left @ right_transpose
    reflected = np.linalg.det(rotations) < 0.0
    if np.any(reflected):
        left = left.copy()
        left[reflected, :, -1] *= -1.0
        rotations = left @ right_transpose

    fitted = np.einsum("mai,mij->maj", centered_mobile, rotations)
    return fitted - centered_reference


def batched_kabsch_rmsd(mobile, reference, weights=None):
    """Return one independently Kabsch-fitted RMSD per batch item."""

    residuals = batched_kabsch_residuals(mobile, reference, weights)
    return batched_residual_rmsd(residuals, weights)
