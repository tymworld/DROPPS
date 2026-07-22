"""Numerical helpers for planar-interface surface-tension analysis."""

from __future__ import annotations

import numpy as np


KJ_PER_MOL_NM2_TO_MN_PER_M = 1.6605390671738467
AXIS_TO_INDEX = {"x": 0, "y": 1, "z": 2}


def volume_preserving_deformation(axis: str, strain: float) -> np.ndarray:
    """Return the affine, volume-preserving slab deformation matrix.

    The selected (interface-normal) axis is scaled by ``exp(strain)`` and
    the two tangential axes by ``exp(-strain/2)``.
    """

    if axis not in AXIS_TO_INDEX:
        raise ValueError(f"Unknown interface-normal axis '{axis}'.")
    if not np.isfinite(strain):
        raise ValueError("Strain must be finite.")

    factors = np.full(3, np.exp(-0.5 * strain), dtype=float)
    factors[AXIS_TO_INDEX[axis]] = np.exp(strain)
    return np.diag(factors)


def deform_configuration(
    positions_nm: np.ndarray,
    box_vectors_nm: np.ndarray,
    axis: str,
    strain: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Affinely deform Cartesian positions and row-wise box vectors."""

    positions = np.asarray(positions_nm, dtype=float)
    box = np.asarray(box_vectors_nm, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("Positions must have shape (n_particles, 3).")
    if box.shape != (3, 3):
        raise ValueError("Box vectors must have shape (3, 3).")

    deformation = volume_preserving_deformation(axis, strain)
    return positions @ deformation.T, box @ deformation.T


def box_volume_nm3(box_vectors_nm: np.ndarray) -> float:
    """Return the absolute periodic-cell volume in nm^3."""

    box = np.asarray(box_vectors_nm, dtype=float)
    if box.shape != (3, 3):
        raise ValueError("Box vectors must have shape (3, 3).")
    volume = float(abs(np.linalg.det(box)))
    if not np.isfinite(volume) or volume <= 0:
        raise ValueError("Periodic box must have a positive finite volume.")
    return volume


def require_orthorhombic_box(
    box_vectors_nm: np.ndarray,
    relative_tolerance: float = 1.0e-6,
) -> None:
    """Reject tilted cells for which the Cartesian slab formula is ambiguous."""

    box = np.asarray(box_vectors_nm, dtype=float)
    box_volume_nm3(box)
    lengths = np.linalg.norm(box, axis=1)
    gram = box @ box.T
    scale = np.outer(lengths, lengths)
    off_diagonal = gram - np.diag(np.diag(gram))
    if np.any(np.abs(off_diagonal) > relative_tolerance * scale):
        raise ValueError(
            "Surface-tension analysis currently requires an orthorhombic box."
        )


def surface_tension_from_energy_pair(
    energy_plus_kj_per_mol: float,
    energy_minus_kj_per_mol: float,
    strain: float,
    box_vectors_nm: np.ndarray,
    axis: str = "z",
    interfaces: int = 2,
) -> float:
    """Convert a central finite-difference energy derivative to mN/m."""

    if axis not in AXIS_TO_INDEX:
        raise ValueError(f"Unknown interface-normal axis '{axis}'.")
    if not np.isfinite(strain) or strain <= 0:
        raise ValueError("Finite-difference strain must be positive and finite.")
    if interfaces <= 0:
        raise ValueError("The number of interfaces must be positive.")

    box = np.asarray(box_vectors_nm, dtype=float)
    require_orthorhombic_box(box)
    volume_nm3 = box_volume_nm3(box)
    normal_length_nm = float(np.linalg.norm(box[AXIS_TO_INDEX[axis]]))
    energy_derivative = (
        float(energy_plus_kj_per_mol) - float(energy_minus_kj_per_mol)
    ) / (2.0 * strain)

    gamma_kj_per_mol_nm2 = (
        -normal_length_nm * energy_derivative / (interfaces * volume_nm3)
    )
    return gamma_kj_per_mol_nm2 * KJ_PER_MOL_NM2_TO_MN_PER_M


def running_average(values: np.ndarray) -> np.ndarray:
    """Return the cumulative mean of a one-dimensional series."""

    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("Values must be a non-empty one-dimensional array.")
    return np.cumsum(array) / np.arange(1, array.size + 1)


def block_statistics(values: np.ndarray, blocks: int) -> dict[str, object]:
    """Calculate a mean and SEM from contiguous block means."""

    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("Values must be a non-empty one-dimensional array.")
    if not np.all(np.isfinite(array)):
        raise ValueError("Values contain non-finite entries.")
    if blocks <= 0:
        raise ValueError("Number of blocks must be positive.")

    blocks_used = min(int(blocks), array.size)
    block_means = np.asarray(
        [chunk.mean() for chunk in np.array_split(array, blocks_used)],
        dtype=float,
    )
    sem = (
        float(block_means.std(ddof=1) / np.sqrt(blocks_used))
        if blocks_used > 1
        else float("nan")
    )
    return {
        "mean": float(array.mean()),
        "sem": sem,
        "blocks": blocks_used,
        "block_means": block_means,
    }
