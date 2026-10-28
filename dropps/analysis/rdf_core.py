"""Numerical kernels for radial distribution function analysis."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np

try:  # Production DROPPS installs MDAnalysis; NumPy fallback keeps core testable.
    from MDAnalysis.lib.distances import capped_distance as _mda_capped_distance
except ImportError:  # pragma: no cover - exercised in dependency-light installs.
    _mda_capped_distance = None


DEFAULT_RDF_BIN_WIDTH_NM = 0.002
_MAX_LATTICE_ENUMERATION = 2_000_000


@dataclass(frozen=True)
class RadialGrid:
    """GROMACS-compatible RDF centers and half-width first-bin edges."""

    bin_width: float
    radii: np.ndarray
    edges: np.ndarray
    shell_measures: np.ndarray
    geometry: str
    requested_limit: float

    @property
    def bin_count(self):
        return int(self.radii.size)

    @property
    def actual_rmax(self):
        return float(self.edges[-1])

    @property
    def cumulative_radii(self):
        return self.edges[1:].copy()


@dataclass(frozen=True)
class RdfPositionSet:
    """Frame positions with atom and molecule provenance."""

    coordinates: np.ndarray
    atom_indices: np.ndarray
    chain_memberships: tuple[frozenset[int], ...]

    def __post_init__(self):
        coordinates = np.asarray(self.coordinates, dtype=np.float64)
        atom_indices = np.asarray(self.atom_indices, dtype=np.int64)
        if coordinates.ndim != 2 or coordinates.shape[1] != 3:
            raise ValueError("RDF positions must have shape (n_positions, 3).")
        if coordinates.shape[0] == 0:
            raise ValueError("An RDF position set must not be empty.")
        if not np.all(np.isfinite(coordinates)):
            raise ValueError("RDF positions contain non-finite coordinates.")
        if atom_indices.shape != (coordinates.shape[0],):
            raise ValueError("RDF atom provenance must match the position count.")
        if len(self.chain_memberships) != coordinates.shape[0]:
            raise ValueError("RDF molecule provenance must match the position count.")
        if any(not membership for membership in self.chain_memberships):
            raise ValueError("Every RDF position must have molecule provenance.")
        object.__setattr__(self, "coordinates", coordinates)
        object.__setattr__(self, "atom_indices", atom_indices)

    @property
    def count(self):
        return int(self.coordinates.shape[0])


def _validate_geometry(geometry):
    if geometry not in {"3d", "xy"}:
        raise ValueError("RDF geometry must be '3d' or 'xy'.")


def build_radial_grid(bin_width, radius_limit, geometry="3d"):
    """Build integer-centered RDF bins with a half-width first shell."""

    _validate_geometry(geometry)
    width = float(bin_width)
    limit = float(radius_limit)
    if not np.isfinite(width) or width <= 0.0:
        raise ValueError("RDF bin width must be positive and finite.")
    if not np.isfinite(limit) or limit <= 0.0:
        raise ValueError("RDF maximum radius must be positive and finite.")

    # For n bins, the final complete edge is (n - 1/2) * width.
    bin_count = int(np.floor(limit / width + 0.5 + 1.0e-12))
    if bin_count < 1:
        raise ValueError(
            "RDF maximum radius is smaller than the first half-width bin."
        )
    radii = np.arange(bin_count, dtype=np.float64) * width
    edges = np.empty(bin_count + 1, dtype=np.float64)
    edges[0] = 0.0
    edges[1:] = (np.arange(1, bin_count + 1, dtype=np.float64) - 0.5) * width

    if geometry == "3d":
        cumulative_measure = (4.0 * np.pi / 3.0) * edges**3
    else:
        cumulative_measure = np.pi * edges**2
    shell_measures = np.diff(cumulative_measure)
    return RadialGrid(
        width,
        radii,
        edges,
        shell_measures,
        geometry,
        limit,
    )


def cell_vectors(dimensions):
    """Convert MDAnalysis dimensions to a validated Cartesian cell matrix."""

    values = np.asarray(dimensions, dtype=np.float64)
    if values.shape != (6,) or not np.all(np.isfinite(values)):
        raise ValueError("Periodic box dimensions must be a finite six-vector.")
    lengths = values[:3]
    if np.any(lengths <= 0.0):
        raise ValueError("Periodic box lengths must be positive.")
    alpha, beta, gamma = np.deg2rad(values[3:])
    if not (
        0.0 < alpha < np.pi
        and 0.0 < beta < np.pi
        and 0.0 < gamma < np.pi
    ):
        raise ValueError("Periodic box angles must lie strictly between 0 and 180 degrees.")

    sin_gamma = float(np.sin(gamma))
    if abs(sin_gamma) <= 1.0e-12:
        raise ValueError("Periodic box has a singular gamma angle.")
    a = np.array((lengths[0], 0.0, 0.0), dtype=np.float64)
    b = np.array(
        (lengths[1] * np.cos(gamma), lengths[1] * sin_gamma, 0.0),
        dtype=np.float64,
    )
    c_x = lengths[2] * np.cos(beta)
    c_y = lengths[2] * (
        np.cos(alpha) - np.cos(beta) * np.cos(gamma)
    ) / sin_gamma
    c_z_squared = lengths[2] ** 2 - c_x**2 - c_y**2
    tolerance = np.finfo(np.float64).eps * max(lengths[2] ** 2, 1.0) * 64.0
    if c_z_squared < -tolerance:
        raise ValueError("Periodic box angles do not define a real cell.")
    c = np.array((c_x, c_y, np.sqrt(max(c_z_squared, 0.0))), dtype=np.float64)
    cell = np.vstack((a, b, c))
    determinant = float(np.linalg.det(cell))
    if not np.isfinite(determinant) or determinant <= 0.0:
        raise ValueError("Periodic box must have positive finite volume.")
    return cell


def _geometry_basis(dimensions, geometry):
    _validate_geometry(geometry)
    cell = cell_vectors(dimensions)
    if geometry == "3d":
        return cell
    basis = cell[:2, :2]
    if float(np.linalg.det(basis)) <= 0.0:
        raise ValueError("The first two box vectors do not define a positive x-y area.")
    return basis


def box_measure(dimensions, geometry="3d"):
    """Return box volume (3D) or projected periodic area (x-y)."""

    basis = _geometry_basis(dimensions, geometry)
    return float(abs(np.linalg.det(basis)))


def _enumeration_bounds(basis, upper_bound, extra=0.0):
    inverse = np.linalg.inv(basis)
    dual_norms = np.linalg.norm(inverse, axis=0)
    bounds = np.ceil(float(upper_bound) * dual_norms + float(extra)).astype(int)
    bounds = np.maximum(bounds, 1)
    count = int(np.prod(2 * bounds + 1, dtype=np.int64))
    if count > _MAX_LATTICE_ENUMERATION:
        raise ValueError(
            "Periodic box is too ill-conditioned for a reliable minimum-image search."
        )
    return bounds, inverse


def shortest_lattice_vector(dimensions, geometry="3d"):
    """Return the exact shortest non-zero vector of the selected box lattice."""

    basis = _geometry_basis(dimensions, geometry)
    best = float(np.min(np.linalg.norm(basis, axis=1)))
    bounds, _ = _enumeration_bounds(basis, best)
    ranges = [range(-int(bound), int(bound) + 1) for bound in bounds]
    for coefficients in product(*ranges):
        if not any(coefficients):
            continue
        vector = np.asarray(coefficients, dtype=np.float64) @ basis
        norm = float(np.linalg.norm(vector))
        if 0.0 < norm < best:
            best = norm
    if not np.isfinite(best) or best <= 0.0:
        raise ValueError("Could not determine a positive periodic lattice vector.")
    return best


def safe_periodic_radius(dimensions, geometry="3d"):
    """Largest radius with a unique nearest image in the periodic lattice."""

    return 0.5 * shortest_lattice_vector(dimensions, geometry)


def minimum_image_vectors(vectors, dimensions, geometry="3d"):
    """Return exact minimum-image vectors for 3D or x-y periodic geometry."""

    _validate_geometry(geometry)
    input_vectors = np.asarray(vectors, dtype=np.float64)
    if input_vectors.shape[-1:] != (3,) or not np.all(np.isfinite(input_vectors)):
        raise ValueError("Distance vectors must be finite Cartesian 3-vectors.")
    original_shape = input_vectors.shape
    flat = input_vectors.reshape((-1, 3))
    basis = _geometry_basis(dimensions, geometry)
    components = flat if geometry == "3d" else flat[:, :2]
    inverse = np.linalg.inv(basis)
    fractional = components @ inverse
    residual_fractional = fractional - np.rint(fractional)
    best = residual_fractional @ basis
    best_squared = np.einsum("ij,ij->i", best, best)

    # Any vector improving on the centered-parallelepiped candidate must obey
    # these reciprocal-basis coefficient bounds.  Enumerating them is exact for
    # skew cells while remaining tiny for normal simulation boxes.
    upper = 0.5 * float(np.sum(np.linalg.norm(basis, axis=1)))
    bounds, _ = _enumeration_bounds(basis, upper, extra=0.5)
    ranges = [range(-int(bound), int(bound) + 1) for bound in bounds]
    zero = (0,) * basis.shape[0]
    for offset in product(*ranges):
        if offset == zero:
            continue
        candidate = (residual_fractional - np.asarray(offset)) @ basis
        squared = np.einsum("ij,ij->i", candidate, candidate)
        improve = squared < best_squared
        if np.any(improve):
            best[improve] = candidate[improve]
            best_squared[improve] = squared[improve]

    if geometry == "xy":
        result = np.zeros_like(flat)
        result[:, :2] = best
    else:
        result = best
    return result.reshape(original_shape)


def topology_exclusion_keys(exclusion_pairs, atom_count):
    """Encode undirected global atom pairs as sorted integer lookup keys."""

    atom_count = int(atom_count)
    if atom_count <= 0:
        raise ValueError("Atom count must be positive for topology exclusions.")
    pairs = np.asarray(exclusion_pairs, dtype=np.int64)
    if pairs.size == 0:
        return np.empty(0, dtype=np.int64)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("Topology exclusions must have shape (n_pairs, 2).")
    if np.min(pairs) < 0 or np.max(pairs) >= atom_count:
        raise IndexError("A topology exclusion contains an invalid atom index.")
    low = np.minimum(pairs[:, 0], pairs[:, 1])
    high = np.maximum(pairs[:, 0], pairs[:, 1])
    return np.unique(low * atom_count + high)


def _topology_pair_keys(reference_atoms, target_atoms, atom_count):
    low = np.minimum(reference_atoms, target_atoms)
    high = np.maximum(reference_atoms, target_atoms)
    return low * int(atom_count) + high


def _single_chain_ids(position_set):
    values = np.full(position_set.count, -1, dtype=np.int64)
    all_single = True
    for index, membership in enumerate(position_set.chain_memberships):
        if len(membership) == 1:
            values[index] = next(iter(membership))
        else:
            all_single = False
    return values, all_single


def _exclusion_mask(
    reference,
    target,
    reference_indices,
    target_indices,
    exclusion,
    topology_keys,
    atom_count,
    reference_chain_ids,
    target_chain_ids,
    all_single_chain,
):
    if exclusion == "none":
        return np.ones(reference_indices.size, dtype=bool)
    if exclusion == "topology":
        if np.any(reference.atom_indices[reference_indices] < 0) or np.any(
            target.atom_indices[target_indices] < 0
        ):
            raise ValueError("Topology exclusions require atom positions on both sides.")
        keys = _topology_pair_keys(
            reference.atom_indices[reference_indices],
            target.atom_indices[target_indices],
            atom_count,
        )
        return ~np.isin(keys, topology_keys, assume_unique=False)
    if exclusion != "same-molecule":
        raise ValueError(f"Unknown RDF exclusion policy: {exclusion}")
    if all_single_chain:
        return (
            reference_chain_ids[reference_indices]
            != target_chain_ids[target_indices]
        )
    return np.asarray(
        [
            reference.chain_memberships[int(ref_index)].isdisjoint(
                target.chain_memberships[int(target_index)]
            )
            for ref_index, target_index in zip(reference_indices, target_indices)
        ],
        dtype=bool,
    )


def _candidate_pairs_for_target_block(
    reference_coordinates,
    target_coordinates,
    target_start,
    grid,
    dimensions,
    pbc,
    geometry,
    pair_block_size,
):
    """Return all candidate pairs in one target tile below actual-rmax."""

    maximum = float(np.nextafter(grid.actual_rmax, 0.0))
    if _mda_capped_distance is not None:
        search_reference = reference_coordinates
        search_target = target_coordinates
        search_box = np.asarray(dimensions, dtype=np.float64) if pbc else None
        if geometry == "xy":
            search_reference = np.asarray(reference_coordinates, dtype=np.float64).copy()
            search_target = np.asarray(target_coordinates, dtype=np.float64).copy()
            search_reference[:, 2] = 0.0
            search_target[:, 2] = 0.0
            if pbc:
                search_box = np.asarray(dimensions, dtype=np.float64).copy()
                # Embed the x-y lattice in a sufficiently tall orthogonal third
                # dimension so MDAnalysis can use its cutoff-aware 3D search
                # without introducing a z-periodic neighbor.
                search_box[2] = max(2.1 * grid.actual_rmax, 1.0)
                search_box[3] = 90.0
                search_box[4] = 90.0
        pairs, distances = _mda_capped_distance(
            search_reference,
            search_target,
            max_cutoff=maximum,
            box=search_box,
            return_distances=True,
        )
        if pairs.size == 0:
            return (
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float64),
            )
        return (
            np.asarray(pairs[:, 0], dtype=np.int64),
            np.asarray(pairs[:, 1], dtype=np.int64) + int(target_start),
            np.asarray(distances, dtype=np.float64),
        )

    target_count = max(1, target_coordinates.shape[0])
    reference_block_size = max(1, int(pair_block_size) // target_count)
    reference_parts = []
    target_parts = []
    distance_parts = []
    for reference_start in range(0, len(reference_coordinates), reference_block_size):
        reference_stop = min(
            reference_start + reference_block_size,
            len(reference_coordinates),
        )
        difference = (
            target_coordinates[np.newaxis, :, :]
            - reference_coordinates[reference_start:reference_stop, np.newaxis, :]
        )
        if pbc:
            difference = minimum_image_vectors(difference, dimensions, geometry)
        elif geometry == "xy":
            difference = difference.copy()
            difference[..., 2] = 0.0
        distances = np.linalg.norm(difference, axis=-1)
        local_reference, local_target = np.nonzero(distances < maximum)
        if local_reference.size:
            reference_parts.append(local_reference + reference_start)
            target_parts.append(local_target + int(target_start))
            distance_parts.append(distances[local_reference, local_target])

    if not reference_parts:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64),
        )
    return (
        np.concatenate(reference_parts).astype(np.int64, copy=False),
        np.concatenate(target_parts).astype(np.int64, copy=False),
        np.concatenate(distance_parts).astype(np.float64, copy=False),
    )


def frame_pair_histogram(
    reference,
    target,
    grid,
    dimensions,
    *,
    pbc=True,
    geometry="3d",
    cut=0.0,
    exclusion="none",
    topology_keys=None,
    atom_count=None,
    target_block_size=512,
    pair_block_size=131072,
):
    """Histogram one frame of standard reference-target distances."""

    _validate_geometry(geometry)
    if grid.geometry != geometry:
        raise ValueError("RDF grid geometry does not match the pair geometry.")
    cut = float(cut)
    if not np.isfinite(cut) or cut < 0.0:
        raise ValueError("RDF lower cutoff must be finite and non-negative.")
    if cut >= grid.actual_rmax:
        raise ValueError("RDF lower cutoff must be below actual-rmax.")
    if pbc:
        _geometry_basis(dimensions, geometry)
    if target_block_size <= 0 or pair_block_size <= 0:
        raise ValueError("RDF pair block sizes must be positive.")

    topology_keys = (
        np.asarray(topology_keys, dtype=np.int64)
        if topology_keys is not None
        else np.empty(0, dtype=np.int64)
    )
    if exclusion == "topology" and atom_count is None:
        raise ValueError("Topology exclusions require the total atom count.")
    ref_chain_ids, ref_all_single = _single_chain_ids(reference)
    target_chain_ids, target_all_single = _single_chain_ids(target)
    all_single_chain = ref_all_single and target_all_single
    counts = np.zeros(grid.bin_count, dtype=np.int64)

    for target_start in range(0, target.count, int(target_block_size)):
        target_stop = min(target_start + int(target_block_size), target.count)
        ref_indices, target_indices, distances = _candidate_pairs_for_target_block(
            reference.coordinates,
            target.coordinates[target_start:target_stop],
            target_start,
            grid,
            dimensions,
            pbc,
            geometry,
            pair_block_size,
        )
        if distances.size == 0:
            continue
        keep = (distances > cut) & (distances < grid.actual_rmax)
        if np.any(keep):
            exclusion_keep = _exclusion_mask(
                reference,
                target,
                ref_indices,
                target_indices,
                exclusion,
                topology_keys,
                atom_count,
                ref_chain_ids,
                target_chain_ids,
                all_single_chain,
            )
            keep &= exclusion_keep
        if np.any(keep):
            counts += np.histogram(distances[keep], bins=grid.edges)[0]
    return counts


def frame_surface_histogram(
    reference_atoms,
    surface_group_ids,
    target,
    grid,
    dimensions,
    *,
    pbc=True,
    geometry="3d",
    cut=0.0,
    exclusion="none",
    target_block_size=256,
    pair_block_size=131072,
):
    """Histogram closest-reference-atom distances per target and surface entity."""

    if exclusion == "topology":
        raise ValueError("Topology exclusions cannot be combined with surface RDF.")
    groups = np.asarray(surface_group_ids, dtype=np.int64)
    if groups.shape != (reference_atoms.count,) or np.any(groups < 0):
        raise ValueError("Surface group IDs must match the reference atom count.")
    unique_groups = np.unique(groups)
    if not np.array_equal(unique_groups, np.arange(unique_groups.size)):
        raise ValueError("Surface group IDs must be contiguous from zero.")
    cut = float(cut)
    if not np.isfinite(cut) or cut < 0.0 or cut >= grid.actual_rmax:
        raise ValueError("Surface RDF cutoff must lie in [0, actual-rmax).")

    ref_chain_ids, ref_all_single = _single_chain_ids(reference_atoms)
    target_chain_ids, target_all_single = _single_chain_ids(target)
    all_single_chain = ref_all_single and target_all_single
    counts = np.zeros(grid.bin_count, dtype=np.int64)
    group_count = int(unique_groups.size)

    for target_start in range(0, target.count, int(target_block_size)):
        target_stop = min(target_start + int(target_block_size), target.count)
        ref_indices, target_indices, distances = _candidate_pairs_for_target_block(
            reference_atoms.coordinates,
            target.coordinates[target_start:target_stop],
            target_start,
            grid,
            dimensions,
            pbc,
            geometry,
            pair_block_size,
        )
        if distances.size == 0:
            continue
        keep = (distances > cut) & (distances < grid.actual_rmax)
        if np.any(keep):
            keep &= _exclusion_mask(
                reference_atoms,
                target,
                ref_indices,
                target_indices,
                exclusion,
                np.empty(0, dtype=np.int64),
                None,
                ref_chain_ids,
                target_chain_ids,
                all_single_chain,
            )
        if not np.any(keep):
            continue
        ref_indices = ref_indices[keep]
        target_indices = target_indices[keep]
        distances = distances[keep]
        keys = (
            (target_indices - target_start) * group_count
            + groups[ref_indices]
        )
        order = np.argsort(keys, kind="stable")
        ordered_keys = keys[order]
        starts = np.r_[0, np.flatnonzero(ordered_keys[1:] != ordered_keys[:-1]) + 1]
        minima = np.minimum.reduceat(distances[order], starts)
        counts += np.histogram(minima, bins=grid.edges)[0]
    return counts


class RdfAccumulator:
    """Streaming raw counts and GROMACS-compatible normalization factors."""

    def __init__(self, target_count, grid):
        if int(target_count) <= 0:
            raise ValueError("At least one RDF target is required.")
        self.grid = grid
        self.raw_counts = np.zeros(
            (int(target_count), grid.bin_count),
            dtype=np.float64,
        )
        self.reference_count_sum = 0.0
        self.target_count_sums = np.zeros(int(target_count), dtype=np.float64)
        self.target_density_sums = np.zeros(int(target_count), dtype=np.float64)
        self.measure_sum = 0.0
        self.frame_count = 0

    def add_frame(self, histograms, reference_count, target_counts, measure):
        histograms = np.asarray(histograms, dtype=np.float64)
        target_counts = np.asarray(target_counts, dtype=np.float64)
        if histograms.shape != self.raw_counts.shape:
            raise ValueError("Frame RDF histograms have an unexpected shape.")
        if target_counts.shape != (self.raw_counts.shape[0],):
            raise ValueError("Frame RDF target counts have an unexpected shape.")
        reference_count = float(reference_count)
        measure = float(measure)
        if reference_count <= 0.0 or not np.isfinite(reference_count):
            raise ValueError("RDF reference position count must be positive and finite.")
        if np.any(target_counts <= 0.0) or not np.all(np.isfinite(target_counts)):
            raise ValueError("RDF target position counts must be positive and finite.")
        if measure <= 0.0 or not np.isfinite(measure):
            raise ValueError("RDF box measure must be positive and finite.")
        if np.any(histograms < 0.0) or not np.all(np.isfinite(histograms)):
            raise ValueError("RDF histogram counts must be finite and non-negative.")

        self.raw_counts += histograms
        self.reference_count_sum += reference_count
        self.target_count_sums += target_counts
        self.target_density_sums += target_counts / measure
        self.measure_sum += measure
        self.frame_count += 1

    def _means(self):
        if self.frame_count <= 0:
            raise ValueError("No RDF trajectory frames were accumulated.")
        return (
            self.raw_counts / self.frame_count,
            self.reference_count_sum / self.frame_count,
            self.target_density_sums / self.frame_count,
        )

    def normalized(self, normalization="rdf"):
        normalization = str(normalization).replace("_", "-")
        if normalization not in {"rdf", "number-density", "none"}:
            raise ValueError(f"Unknown RDF normalization: {normalization}")
        mean_counts, mean_reference_count, mean_densities = self._means()
        result = mean_counts / mean_reference_count
        if normalization == "none":
            return result / self.grid.bin_width
        result = result / self.grid.shell_measures[np.newaxis, :]
        if normalization == "rdf":
            result = result / mean_densities[:, np.newaxis]
        return result

    def cumulative(self):
        mean_counts, mean_reference_count, _ = self._means()
        return np.cumsum(mean_counts, axis=1) / mean_reference_count

    @property
    def mean_reference_count(self):
        return self._means()[1]

    @property
    def mean_target_densities(self):
        return self._means()[2].copy()

    @property
    def mean_target_counts(self):
        if self.frame_count <= 0:
            raise ValueError("No RDF trajectory frames were accumulated.")
        return self.target_count_sums / self.frame_count

    @property
    def mean_measure(self):
        if self.frame_count <= 0:
            raise ValueError("No RDF trajectory frames were accumulated.")
        return self.measure_sum / self.frame_count
