"""Numerical helpers for coarse-grained solvent-accessible surface area.

The trajectory-facing command supplies one minimum-image displacement for
every potentially overlapping pair.  Keeping the surface sampling independent
from trajectory I/O makes the geometry testable with small synthetic systems.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.spatial import cKDTree


RADIUS_CONVENTION_FACTORS = {
    "sigma-half": 0.5,
    "lj-minimum-half": 0.5 * 2.0 ** (1.0 / 6.0),
}


def fibonacci_sphere_points(count: int) -> np.ndarray:
    """Return deterministic, approximately uniform unit-sphere directions."""

    count = int(count)
    if count < 4 or count % 2:
        raise ValueError("Sphere point count must be an even integer of at least four.")
    half_count = count // 2
    indices = np.arange(half_count, dtype=np.float64) + 0.5
    z = 1.0 - 2.0 * indices / half_count
    radius_xy = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    azimuth = golden_angle * indices
    half_sphere = np.column_stack(
        (radius_xy * np.cos(azimuth), radius_xy * np.sin(azimuth), z)
    )
    return np.concatenate((half_sphere, -half_sphere), axis=0)


def _validate_positions_and_radii(positions, expanded_radii):
    coordinates = np.asarray(positions, dtype=np.float64)
    radii = np.asarray(expanded_radii, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("Surface positions must have shape (n_atoms, 3).")
    if radii.shape != (coordinates.shape[0],):
        raise ValueError("Expanded radii must contain one value per surface atom.")
    if coordinates.shape[0] == 0:
        raise ValueError("The SASA surface must contain at least one atom.")
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("Surface positions contain non-finite values.")
    if not np.all(np.isfinite(radii)) or np.any(radii <= 0.0):
        raise ValueError("Expanded radii must be positive and finite.")
    return coordinates, radii


def neighbor_vectors_from_pairs(
    atom_count: int,
    expanded_radii,
    pairs,
    pair_vectors,
):
    """Build symmetric neighbor lists from nearest-image pair displacements.

    ``pair_vectors[k]`` points from ``pairs[k, 0]`` to ``pairs[k, 1]``.
    Pairs too far apart to overlap their expanded spheres are discarded.
    """

    atom_count = int(atom_count)
    radii = np.asarray(expanded_radii, dtype=np.float64)
    pair_array = np.asarray(pairs, dtype=np.int64)
    vectors = np.asarray(pair_vectors, dtype=np.float64)
    if radii.shape != (atom_count,):
        raise ValueError("Expanded radii do not match the neighbor atom count.")
    if pair_array.size == 0:
        pair_array = np.empty((0, 2), dtype=np.int64)
        vectors = np.empty((0, 3), dtype=np.float64)
    if pair_array.ndim != 2 or pair_array.shape[1] != 2:
        raise ValueError("Neighbor pairs must have shape (n_pairs, 2).")
    if vectors.shape != (pair_array.shape[0], 3):
        raise ValueError("Pair vectors must have shape (n_pairs, 3).")
    if pair_array.size and (
        np.min(pair_array) < 0 or np.max(pair_array) >= atom_count
    ):
        raise IndexError("A SASA neighbor pair contains an invalid atom index.")
    if not np.all(np.isfinite(vectors)):
        raise ValueError("SASA neighbor vectors contain non-finite values.")

    neighbor_indices = [[] for _ in range(atom_count)]
    neighbor_vectors = [[] for _ in range(atom_count)]
    if pair_array.shape[0] == 0:
        return (
            tuple(np.empty(0, dtype=np.int64) for _ in range(atom_count)),
            tuple(np.empty((0, 3), dtype=np.float64) for _ in range(atom_count)),
        )

    squared_distances = np.einsum("ij,ij->i", vectors, vectors)
    coincidence_tolerance = np.finfo(np.float64).eps * 64.0
    if np.any(squared_distances <= coincidence_tolerance):
        pair = pair_array[int(np.flatnonzero(squared_distances <= coincidence_tolerance)[0])]
        raise ValueError(
            f"Surface atoms {int(pair[0])} and {int(pair[1])} have coincident centers."
        )
    pair_cutoffs = radii[pair_array[:, 0]] + radii[pair_array[:, 1]]
    keep = squared_distances < pair_cutoffs * pair_cutoffs
    for (atom_a, atom_b), vector in zip(pair_array[keep], vectors[keep]):
        atom_a = int(atom_a)
        atom_b = int(atom_b)
        if atom_a == atom_b:
            continue
        neighbor_indices[atom_a].append(atom_b)
        neighbor_vectors[atom_a].append(vector)
        neighbor_indices[atom_b].append(atom_a)
        neighbor_vectors[atom_b].append(-vector)

    return (
        tuple(np.asarray(values, dtype=np.int64) for values in neighbor_indices),
        tuple(
            np.asarray(values, dtype=np.float64).reshape(-1, 3)
            for values in neighbor_vectors
        ),
    )


def nonperiodic_neighbor_vectors(positions, expanded_radii):
    """Return candidate sphere-overlap vectors without periodic boundaries."""

    coordinates, radii = _validate_positions_and_radii(positions, expanded_radii)
    maximum_cutoff = 2.0 * float(np.max(radii))
    tree = cKDTree(coordinates)
    pairs = np.asarray(list(tree.query_pairs(maximum_cutoff)), dtype=np.int64)
    if pairs.size == 0:
        pairs = np.empty((0, 2), dtype=np.int64)
    else:
        pairs = pairs.reshape(-1, 2)
    vectors = (
        coordinates[pairs[:, 1]] - coordinates[pairs[:, 0]]
        if pairs.shape[0]
        else np.empty((0, 3), dtype=np.float64)
    )
    return neighbor_vectors_from_pairs(
        coordinates.shape[0],
        radii,
        pairs,
        vectors,
    )


def orthorhombic_periodic_neighbor_vectors(
    positions,
    expanded_radii,
    box_lengths,
):
    """Return nearest-image overlap vectors for an orthorhombic cell.

    Positions may be arbitrarily many box lengths from the primary cell.  They
    are reduced in float64 before constructing SciPy's periodic search tree.
    """

    coordinates, radii = _validate_positions_and_radii(positions, expanded_radii)
    lengths = np.asarray(box_lengths, dtype=np.float64)
    if lengths.shape != (3,) or not np.all(np.isfinite(lengths)):
        raise ValueError("Orthorhombic box lengths must be three finite values.")
    if np.any(lengths <= 0.0):
        raise ValueError("Orthorhombic box lengths must be positive.")
    maximum_cutoff = 2.0 * float(np.max(radii))
    if maximum_cutoff >= 0.5 * float(np.min(lengths)):
        raise ValueError(
            "The maximum SASA overlap cutoff must be less than half the "
            "shortest periodic box length."
        )

    wrapped = np.mod(coordinates, lengths)
    tree = cKDTree(wrapped, boxsize=lengths)
    pairs = np.asarray(list(tree.query_pairs(maximum_cutoff)), dtype=np.int64)
    if pairs.size == 0:
        pairs = np.empty((0, 2), dtype=np.int64)
    else:
        pairs = pairs.reshape(-1, 2)
    vectors = (
        wrapped[pairs[:, 1]] - wrapped[pairs[:, 0]]
        if pairs.shape[0]
        else np.empty((0, 3), dtype=np.float64)
    )
    if vectors.size:
        vectors -= lengths * np.round(vectors / lengths)
    return neighbor_vectors_from_pairs(
        coordinates.shape[0],
        radii,
        pairs,
        vectors,
    )


def shrake_rupley_areas(
    expanded_radii,
    neighbor_indices: Sequence[np.ndarray],
    neighbor_vectors: Sequence[np.ndarray],
    sphere_points,
    *,
    point_chunk_size: int = 256,
) -> np.ndarray:
    """Calculate one SASA value per surface atom from local neighbor geometry."""

    radii = np.asarray(expanded_radii, dtype=np.float64)
    directions = np.asarray(sphere_points, dtype=np.float64)
    if radii.ndim != 1 or radii.size == 0:
        raise ValueError("Expanded radii must be a non-empty vector.")
    if not np.all(np.isfinite(radii)) or np.any(radii <= 0.0):
        raise ValueError("Expanded radii must be positive and finite.")
    if directions.ndim != 2 or directions.shape[1] != 3 or directions.shape[0] < 4:
        raise ValueError("Sphere points must have shape (n_points, 3), with n_points >= 4.")
    if not np.all(np.isfinite(directions)):
        raise ValueError("Sphere points contain non-finite values.")
    norms = np.linalg.norm(directions, axis=1)
    if not np.allclose(norms, 1.0, rtol=1.0e-10, atol=1.0e-12):
        raise ValueError("Sphere points must lie on the unit sphere.")
    if len(neighbor_indices) != radii.size or len(neighbor_vectors) != radii.size:
        raise ValueError("Neighbor lists must contain one entry per surface atom.")
    point_chunk_size = int(point_chunk_size)
    if point_chunk_size <= 0:
        raise ValueError("Point chunk size must be greater than zero.")

    areas = np.empty(radii.size, dtype=np.float64)
    sphere_point_count = directions.shape[0]
    for atom_index, radius in enumerate(radii):
        indices = np.asarray(neighbor_indices[atom_index], dtype=np.int64)
        vectors = np.asarray(neighbor_vectors[atom_index], dtype=np.float64)
        if indices.ndim != 1 or vectors.shape != (indices.size, 3):
            raise ValueError(f"Invalid neighbor list for surface atom {atom_index}.")
        if indices.size and (np.min(indices) < 0 or np.max(indices) >= radii.size):
            raise IndexError(f"Invalid neighbor index for surface atom {atom_index}.")

        if indices.size == 0:
            accessible_count = sphere_point_count
        else:
            neighbor_radii_squared = radii[indices] ** 2
            accessible_count = 0
            for start in range(0, sphere_point_count, point_chunk_size):
                stop = min(start + point_chunk_size, sphere_point_count)
                local_points = radius * directions[start:stop]
                displacements = local_points[:, np.newaxis, :] - vectors[np.newaxis, :, :]
                squared_distances = np.einsum(
                    "pni,pni->pn",
                    displacements,
                    displacements,
                )
                occluded = np.any(
                    squared_distances < neighbor_radii_squared[np.newaxis, :],
                    axis=1,
                )
                accessible_count += int(np.count_nonzero(~occluded))
        areas[atom_index] = (
            4.0
            * np.pi
            * radius
            * radius
            * accessible_count
            / sphere_point_count
        )
    return areas


def shrake_rupley_nonperiodic(
    positions,
    radii,
    *,
    probe_radius: float = 0.14,
    sphere_point_count: int = 960,
) -> np.ndarray:
    """Convenience wrapper for non-periodic Shrake--Rupley SASA in nm^2."""

    coordinates = np.asarray(positions, dtype=np.float64)
    atomic_radii = np.asarray(radii, dtype=np.float64)
    probe_radius = float(probe_radius)
    if not np.isfinite(probe_radius) or probe_radius < 0.0:
        raise ValueError("Probe radius must be finite and non-negative.")
    expanded = atomic_radii + probe_radius
    coordinates, expanded = _validate_positions_and_radii(coordinates, expanded)
    neighbor_indices, neighbor_vectors = nonperiodic_neighbor_vectors(
        coordinates,
        expanded,
    )
    return shrake_rupley_areas(
        expanded,
        neighbor_indices,
        neighbor_vectors,
        fibonacci_sphere_points(sphere_point_count),
    )


def forcefield_radii_nm(
    itp_list,
    *,
    convention: str = "sigma-half",
    scale: float = 1.0,
) -> np.ndarray:
    """Resolve one coarse-grained bead radius from every topology atom."""

    if convention not in RADIUS_CONVENTION_FACTORS:
        choices = ", ".join(sorted(RADIUS_CONVENTION_FACTORS))
        raise ValueError(f"Unknown radius convention {convention!r}; choose {choices}.")
    scale = float(scale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("Radius scale must be positive and finite.")
    factor = RADIUS_CONVENTION_FACTORS[convention] * scale

    radii = []
    for molecule_id, topology in enumerate(itp_list):
        type_sigmas = getattr(topology, "type2sigma", None)
        pair_sigmas = getattr(topology, "types2sigma", None)
        for atom in topology.atoms:
            abbreviation = str(atom.abbr)
            if type_sigmas is not None:
                if abbreviation not in type_sigmas:
                    raise ValueError(
                        f"Molecule {molecule_id} has no sigma for bead type "
                        f"{abbreviation!r}."
                    )
                sigma = type_sigmas[abbreviation]
            elif pair_sigmas is not None:
                key = f"{abbreviation}-{abbreviation}"
                if key not in pair_sigmas:
                    raise ValueError(
                        f"Molecule {molecule_id} has no self sigma for bead type "
                        f"{abbreviation!r}."
                    )
                sigma = pair_sigmas[key]
            else:
                raise ValueError(
                    f"Molecule {molecule_id} does not contain SASA-compatible "
                    "sigma parameters."
                )
            try:
                sigma = float(sigma)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Sigma for bead type {abbreviation!r} is not a scalar in nm."
                ) from exc
            if not np.isfinite(sigma) or sigma <= 0.0:
                raise ValueError(
                    f"Sigma for bead type {abbreviation!r} must be positive and finite."
                )
            radii.append(factor * sigma)

    if not radii:
        raise ValueError("The topology does not contain atoms with SASA radii.")
    return np.asarray(radii, dtype=np.float64)
