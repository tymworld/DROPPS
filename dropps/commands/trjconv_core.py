"""Numerical kernels for trajectory selection and coordinate conversion.

The functions in this module deliberately do not depend on a DROPPS TPR or an
MDAnalysis ``Universe``.  Keeping time selection and coordinate transforms
separate from the command-line/file-I/O layer makes the scientific semantics
testable with small synthetic systems.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dropps.share.time_selection import (
    TIME_UNIT_TO_PS as TIME_UNIT_TO_PS,
    TimeSelection as TimeSelection,
    select_time_indices as select_time_indices,
)


@dataclass(frozen=True)
class WholePlan:
    """Static bonded traversal used to reconstruct complete molecules."""

    roots: np.ndarray
    levels: tuple[tuple[np.ndarray, np.ndarray], ...]


def _as_finite_vector(values, name, *, length=None):
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or (length is not None and array.size != length):
        expected = f" with length {length}" if length is not None else ""
        raise ValueError(f"{name} must be a one-dimensional array{expected}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values.")
    return array


def validate_cell(cell):
    """Return a validated 3x3 row-vector periodic-cell matrix."""

    matrix = np.asarray(cell, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("Periodic cell must be a finite 3x3 matrix.")
    determinant = float(np.linalg.det(matrix))
    if determinant <= 0.0:
        raise ValueError("Periodic cell must have positive volume.")
    return matrix


def minimum_image(vectors, cell):
    """Map displacement vectors to the nearest fractional cell image."""

    matrix = validate_cell(cell)
    vectors = np.asarray(vectors, dtype=np.float64)
    if vectors.shape[-1:] != (3,) or not np.all(np.isfinite(vectors)):
        raise ValueError("Displacement vectors must be finite Cartesian 3-vectors.")
    fractional = vectors @ np.linalg.inv(matrix)
    fractional -= np.round(fractional)
    return fractional @ matrix


def wrap_atoms(positions, cell):
    """Put every atom independently into the primary periodic cell."""

    matrix = validate_cell(cell)
    coordinates = np.asarray(positions, dtype=np.float64)
    fractional = coordinates @ np.linalg.inv(matrix)
    return np.mod(fractional, 1.0) @ matrix


def _validate_groups(groups, atom_count, *, require_partition=False):
    normalized = []
    seen = set()
    for group_id, group in enumerate(groups):
        indices = np.asarray(group, dtype=np.int64)
        if indices.ndim != 1 or indices.size == 0:
            raise ValueError(f"Group {group_id} must be a non-empty index vector.")
        if np.unique(indices).size != indices.size:
            raise ValueError(f"Group {group_id} contains duplicate atom indices.")
        if np.min(indices) < 0 or np.max(indices) >= atom_count:
            raise IndexError(f"Group {group_id} contains an out-of-range atom index.")
        overlap = seen.intersection(int(index) for index in indices)
        if overlap:
            raise ValueError("Coordinate groups must not overlap.")
        seen.update(int(index) for index in indices)
        normalized.append(indices)
    if require_partition and seen != set(range(atom_count)):
        raise ValueError("Coordinate groups must partition all atoms.")
    return normalized


def build_whole_plan(groups, bonds, atom_count):
    """Build a breadth-first bonded traversal for every topology molecule."""

    groups = _validate_groups(groups, atom_count, require_partition=True)
    bond_array = np.asarray(bonds, dtype=np.int64)
    if bond_array.size == 0:
        bond_array = np.empty((0, 2), dtype=np.int64)
    if bond_array.ndim != 2 or bond_array.shape[1] != 2:
        raise ValueError("Bonds must have shape (n_bonds, 2).")

    adjacency = [[] for _ in range(atom_count)]
    for atom_a, atom_b in bond_array:
        atom_a = int(atom_a)
        atom_b = int(atom_b)
        if not (0 <= atom_a < atom_count and 0 <= atom_b < atom_count):
            raise IndexError("A bond contains an out-of-range atom index.")
        adjacency[atom_a].append(atom_b)
        adjacency[atom_b].append(atom_a)

    roots = []
    edges_by_depth: dict[int, list[tuple[int, int]]] = {}
    for group_id, group in enumerate(groups):
        allowed = set(int(index) for index in group)
        root = int(group[0])
        roots.append(root)
        visited = {root}
        frontier = [root]
        depth = 0
        while frontier:
            next_frontier = []
            for parent in frontier:
                for child in adjacency[parent]:
                    if child not in allowed or child in visited:
                        continue
                    visited.add(child)
                    next_frontier.append(child)
                    edges_by_depth.setdefault(depth, []).append((parent, child))
            frontier = next_frontier
            depth += 1
        if len(visited) != group.size:
            raise ValueError(
                f"Molecule {group_id} is not connected by topology bonds; "
                "it cannot be made whole unambiguously."
            )

    levels = []
    for depth in sorted(edges_by_depth):
        edges = np.asarray(edges_by_depth[depth], dtype=np.int64)
        levels.append((edges[:, 0], edges[:, 1]))
    return WholePlan(np.asarray(roots, dtype=np.int64), tuple(levels))


def make_whole(positions, cell, plan):
    """Reconstruct bonded molecules through periodic boundaries."""

    coordinates = np.asarray(positions, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("Positions must have shape (n_atoms, 3).")
    result = coordinates.copy()
    result[plan.roots] = coordinates[plan.roots]
    for parents, children in plan.levels:
        displacements = minimum_image(
            coordinates[children] - coordinates[parents], cell
        )
        result[children] = result[parents] + displacements
    return result


def wrap_groups(positions, cell, groups, weights):
    """Shift intact groups so that each weighted center lies in the box."""

    matrix = validate_cell(cell)
    coordinates = np.asarray(positions, dtype=np.float64)
    atom_count = coordinates.shape[0]
    groups = _validate_groups(groups, atom_count)
    weights = _as_finite_vector(weights, "Atom weights", length=atom_count)
    if np.any(weights < 0.0):
        raise ValueError("Atom weights must be non-negative.")
    inverse = np.linalg.inv(matrix)
    result = coordinates.copy()
    for group_id, group in enumerate(groups):
        group_weights = weights[group]
        total = float(group_weights.sum())
        if total <= 0.0:
            raise ValueError(f"Group {group_id} has no positive center weight.")
        center = np.einsum("ai,a->i", result[group], group_weights) / total
        fractional_center = center @ inverse
        fractional_shift = np.mod(fractional_center, 1.0) - fractional_center
        result[group] += fractional_shift @ matrix
    return result


class NoJumpState:
    """Stateful temporal unwrapping in fractional box coordinates."""

    def __init__(self, reference_positions, reference_cell):
        matrix = validate_cell(reference_cell)
        reference = np.asarray(reference_positions, dtype=np.float64)
        if reference.ndim != 2 or reference.shape[1] != 3:
            raise ValueError(
                "No-jump reference positions must have shape (n_atoms, 3)."
            )
        self._atom_count = reference.shape[0]
        self._previous_fractional = reference @ np.linalg.inv(matrix)

    def apply(self, positions, cell):
        matrix = validate_cell(cell)
        coordinates = np.asarray(positions, dtype=np.float64)
        if coordinates.shape != (self._atom_count, 3):
            raise ValueError("No-jump frame atom count differs from its reference.")
        fractional = coordinates @ np.linalg.inv(matrix)
        unwrapped = fractional - np.round(fractional - self._previous_fractional)
        self._previous_fractional = unwrapped
        return unwrapped @ matrix


def center_coordinates(positions, cell, selection, axes, weights=None):
    """Translate selected Cartesian axes of a group to the cell center."""

    matrix = validate_cell(cell)
    coordinates = np.asarray(positions, dtype=np.float64)
    indices = np.asarray(selection, dtype=np.int64)
    if indices.ndim != 1 or indices.size == 0:
        raise ValueError("Centering selection must be a non-empty index vector.")
    if weights is None:
        selected_weights = np.ones(indices.size, dtype=np.float64)
    else:
        all_weights = _as_finite_vector(
            weights, "Center weights", length=coordinates.shape[0]
        )
        selected_weights = all_weights[indices]
    total = float(selected_weights.sum())
    if total <= 0.0:
        raise ValueError("Centering selection has no positive total weight.")
    current_center = (
        np.einsum("ai,a->i", coordinates[indices], selected_weights) / total
    )
    target = 0.5 * np.sum(matrix, axis=0)
    shift = target - current_center
    axis_indices = _axis_indices(axes, allow_xyz=True)
    mask = np.zeros(3, dtype=np.float64)
    mask[axis_indices] = 1.0
    return coordinates + shift * mask


def _axis_indices(axes, *, allow_xyz):
    if axes == "xyz" and allow_xyz:
        return np.asarray((0, 1, 2), dtype=np.int64)
    mapping = {"x": 0, "y": 1, "z": 2}
    try:
        return np.asarray((mapping[axes],), dtype=np.int64)
    except KeyError as exc:
        choices = "x, y, z, or xyz" if allow_xyz else "x, y, or z"
        raise ValueError(f"Center axis must be {choices}.") from exc


def _largest_periodic_true_run(mask):
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 1 or mask.size == 0:
        raise ValueError("Dense-bin mask must be a non-empty vector.")
    if not np.any(mask):
        raise ValueError("No density bin exceeds the dense-phase threshold.")
    if np.all(mask):
        raise ValueError(
            "Every density bin exceeds the threshold; no localized dense phase exists."
        )
    false_index = int(np.flatnonzero(~mask)[0])
    start = (false_index + 1) % mask.size
    rolled = np.roll(mask, -start)
    padded = np.concatenate(([False], rolled, [False])).astype(np.int8)
    transitions = np.diff(padded)
    run_starts = np.flatnonzero(transitions == 1)
    run_stops = np.flatnonzero(transitions == -1)
    lengths = run_stops - run_starts
    winner = int(np.argmax(lengths))
    return (run_starts[winner] + np.arange(lengths[winner]) + start) % mask.size


def center_dense_phase(
    positions,
    cell,
    selection,
    masses,
    axis="z",
    threshold=0.5,
    bin_width=0.5,
):
    """Center the largest periodic high-density slab on one lattice axis.

    Coordinates and ``bin_width`` use the same length unit (Angstrom in the
    command layer).  Relative density is sufficient for thresholding, so the
    histogram uses masses without dividing by the common slab volume.
    """

    matrix = validate_cell(cell)
    coordinates = np.asarray(positions, dtype=np.float64)
    indices = np.asarray(selection, dtype=np.int64)
    if indices.ndim != 1 or indices.size == 0:
        raise ValueError("Dense-phase selection must be non-empty.")
    if not np.isfinite(threshold) or not 0.0 < threshold < 1.0:
        raise ValueError("Dense-phase threshold must be between zero and one.")
    if not np.isfinite(bin_width) or bin_width <= 0.0:
        raise ValueError("Density bin width must be greater than zero.")
    axis_index = int(_axis_indices(axis, allow_xyz=False)[0])
    masses = _as_finite_vector(masses, "Atom masses", length=coordinates.shape[0])
    selected_masses = masses[indices]
    if np.any(selected_masses < 0.0) or float(selected_masses.sum()) <= 0.0:
        raise ValueError("Dense-phase selection must have positive finite mass.")

    axis_length = float(np.linalg.norm(matrix[axis_index]))
    bins = max(2, int(np.ceil(axis_length / bin_width)))
    inverse = np.linalg.inv(matrix)
    fractions = np.mod(coordinates[indices] @ inverse, 1.0)[:, axis_index]
    histogram, _ = np.histogram(
        fractions,
        bins=bins,
        range=(0.0, 1.0),
        weights=selected_masses,
    )
    maximum = float(np.max(histogram))
    if maximum <= 0.0:
        raise ValueError("Dense-phase histogram has no positive mass.")
    dense_bins = histogram > threshold * maximum
    dense_run = _largest_periodic_true_run(dense_bins)

    # Unwrap the winning run relative to its first bin before averaging the
    # bin centers, then map the result back onto the periodic interval.
    unwrapped_bins = dense_run.astype(np.float64)
    for index in range(1, unwrapped_bins.size):
        while unwrapped_bins[index] < unwrapped_bins[index - 1]:
            unwrapped_bins[index] += bins
    center_fraction = float(np.mean(unwrapped_bins + 0.5) / bins) % 1.0
    fractional_shift = np.zeros(3, dtype=np.float64)
    fractional_shift[axis_index] = 0.5 - center_fraction
    return coordinates + fractional_shift @ matrix


def _fit_weights(weights, count):
    if weights is None:
        return np.ones(count, dtype=np.float64)
    values = _as_finite_vector(weights, "Fit weights", length=count)
    if np.any(values < 0.0) or float(values.sum()) <= 0.0:
        raise ValueError("Fit weights must be non-negative with a positive sum.")
    return values


def _weighted_center(coordinates, weights):
    return np.einsum("ai,a->i", coordinates, weights) / float(weights.sum())


def _kabsch_rotation(mobile, reference, weights):
    covariance = np.einsum("ai,aj,a->ij", mobile, reference, weights)
    left, _, right_transpose = np.linalg.svd(covariance)
    rotation = left @ right_transpose
    if np.linalg.det(rotation) < 0.0:
        left[:, -1] *= -1.0
        rotation = left @ right_transpose
    return rotation


def fit_coordinates(positions, fit_indices, reference, mode, weights=None):
    """Fit a complete coordinate set using a selected atom group."""

    coordinates = np.asarray(positions, dtype=np.float64)
    indices = np.asarray(fit_indices, dtype=np.int64)
    target = np.asarray(reference, dtype=np.float64)
    if indices.ndim != 1 or indices.size == 0:
        raise ValueError("Fit selection must be a non-empty index vector.")
    if target.shape != (indices.size, 3):
        raise ValueError("Fit reference must match the selected atom coordinates.")
    selected = coordinates[indices]
    fit_weights = _fit_weights(weights, indices.size)

    if mode in {"translation", "transxy"}:
        shift = _weighted_center(target, fit_weights) - _weighted_center(
            selected, fit_weights
        )
        if mode == "transxy":
            shift[2] = 0.0
        return coordinates + shift

    if mode in {"rot+trans", "progressive"}:
        mobile_center = _weighted_center(selected, fit_weights)
        target_center = _weighted_center(target, fit_weights)
        rotation = _kabsch_rotation(
            selected - mobile_center,
            target - target_center,
            fit_weights,
        )
        return (coordinates - mobile_center) @ rotation + target_center

    if mode == "rotxy+transxy":
        mobile_xy = selected[:, :2]
        target_xy = target[:, :2]
        mobile_center = _weighted_center(mobile_xy, fit_weights)
        target_center = _weighted_center(target_xy, fit_weights)
        rotation = _kabsch_rotation_2d(
            mobile_xy - mobile_center,
            target_xy - target_center,
            fit_weights,
        )
        result = coordinates.copy()
        result[:, :2] = (coordinates[:, :2] - mobile_center) @ rotation + target_center
        return result

    raise ValueError(f"Unknown fit mode '{mode}'.")


def _kabsch_rotation_2d(mobile, reference, weights):
    covariance = np.einsum("ai,aj,a->ij", mobile, reference, weights)
    left, _, right_transpose = np.linalg.svd(covariance)
    rotation = left @ right_transpose
    if np.linalg.det(rotation) < 0.0:
        left[:, -1] *= -1.0
        rotation = left @ right_transpose
    return rotation


class FitState:
    """Reference and state management for ordinary/progressive fitting."""

    def __init__(self, fit_indices, reference, mode, weights=None):
        self.indices = np.asarray(fit_indices, dtype=np.int64)
        self.reference = np.asarray(reference, dtype=np.float64).copy()
        self.mode = mode
        self.weights = (
            None if weights is None else np.asarray(weights, dtype=np.float64)
        )

    def apply(self, positions):
        fitted = fit_coordinates(
            positions,
            self.indices,
            self.reference,
            self.mode,
            self.weights,
        )
        if self.mode == "progressive":
            self.reference = fitted[self.indices].copy()
        return fitted
