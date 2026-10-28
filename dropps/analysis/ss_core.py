"""Geometry-based secondary-structure inference for residue-level CG chains.

The routines in this module deliberately use only coordinates and explicit
contact pairs.  They do not claim to reproduce hydrogen-bond DSSP.  The
default geometric windows follow the C-alpha-only P-SEA assignment published
by Labesse et al. (Bioinformatics 13, 291-295, 1997).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


STATE_COIL = np.uint8(0)
STATE_HELIX = np.uint8(1)
STATE_STRAND = np.uint8(2)
STATE_MIXED = np.uint8(3)
STATE_INVALID = np.uint8(4)

STATE_CODES = {
    int(STATE_COIL): "C",
    int(STATE_HELIX): "H",
    int(STATE_STRAND): "E",
    int(STATE_MIXED): "M",
    int(STATE_INVALID): "X",
}


@dataclass(frozen=True)
class PSEAProfile:
    """Centers and half-widths for P-SEA-like C-alpha geometry windows."""

    helix_angle: tuple[float, float] = (89.0, 12.0)
    strand_angle: tuple[float, float] = (124.0, 14.0)
    helix_dihedral: tuple[float, float] = (50.0, 20.0)
    strand_dihedral: tuple[float, float] = (-170.0, 45.0)
    helix_distances: tuple[tuple[float, float], ...] = (
        (0.55, 0.05),
        (0.53, 0.05),
        (0.64, 0.06),
    )
    strand_distances: tuple[tuple[float, float], ...] = (
        (0.67, 0.06),
        (0.99, 0.09),
        (1.24, 0.11),
    )
    reference_bond_nm: float = 0.38


DEFAULT_PSEA_PROFILE = PSEAProfile()


@dataclass(frozen=True)
class BackboneGeometry:
    """Per-residue and per-window geometry for one or more chains."""

    angle_degrees: np.ndarray
    dihedral_degrees: np.ndarray
    distance2_nm: np.ndarray
    distance3_nm: np.ndarray
    distance4_nm: np.ndarray
    bond_length_nm: np.ndarray


@dataclass(frozen=True)
class SecondaryStructureResult:
    """Local three-state assignment before optional inter-chain filtering."""

    states: np.ndarray
    helix: np.ndarray
    strand: np.ndarray
    geometry: BackboneGeometry


@dataclass(frozen=True)
class StrandPairRecord:
    """One accepted registered pairing between two local strand segments."""

    chain_a: int
    start_a: int
    stop_a: int
    chain_b: int
    start_b: int
    stop_b: int
    orientation: str
    registry: int
    paired_residues: int
    mean_distance_nm: float
    minimum_distance_nm: float


def _as_chain_coordinates(coordinates) -> np.ndarray:
    values = np.asarray(coordinates, dtype=np.float64)
    if values.ndim < 2 or values.shape[-1] != 3:
        raise ValueError("Coordinates must have shape (..., n_residues, 3).")
    if values.shape[-2] < 2:
        raise ValueError("At least two residue coordinates are required.")
    if not np.all(np.isfinite(values)):
        raise ValueError("Coordinates contain NaN or infinite values.")
    return values


def _safe_norm(vectors: np.ndarray) -> np.ndarray:
    return np.linalg.norm(vectors, axis=-1)


def _angles_degrees(coordinates: np.ndarray) -> np.ndarray:
    residue_count = coordinates.shape[-2]
    output = np.full(coordinates.shape[:-1], np.nan, dtype=np.float64)
    if residue_count < 3:
        return output
    left = coordinates[..., :-2, :] - coordinates[..., 1:-1, :]
    right = coordinates[..., 2:, :] - coordinates[..., 1:-1, :]
    denominator = _safe_norm(left) * _safe_norm(right)
    if np.any(denominator <= 0.0):
        raise ValueError("Consecutive residue coordinates contain a zero-length bond.")
    cosine = np.sum(left * right, axis=-1) / denominator
    output[..., 1:-1] = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return output


def _dihedrals_degrees(coordinates: np.ndarray) -> np.ndarray:
    """Return signed virtual dihedrals, associated with their first center.

    The value stored at residue ``i`` uses points ``i-1, i, i+1, i+2``.
    With this convention an ideal right-handed alpha-helix is about +50 deg.
    """

    residue_count = coordinates.shape[-2]
    output = np.full(coordinates.shape[:-1], np.nan, dtype=np.float64)
    if residue_count < 4:
        return output

    p0 = coordinates[..., :-3, :]
    p1 = coordinates[..., 1:-2, :]
    p2 = coordinates[..., 2:-1, :]
    p3 = coordinates[..., 3:, :]
    b0 = -(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2
    b1_norm = _safe_norm(b1)
    if np.any(b1_norm <= 0.0):
        raise ValueError("Consecutive residue coordinates contain a zero-length bond.")
    b1_unit = b1 / b1_norm[..., np.newaxis]
    v = b0 - np.sum(b0 * b1_unit, axis=-1)[..., np.newaxis] * b1_unit
    w = b2 - np.sum(b2 * b1_unit, axis=-1)[..., np.newaxis] * b1_unit
    v_norm = _safe_norm(v)
    w_norm = _safe_norm(w)
    valid = (v_norm > 0.0) & (w_norm > 0.0)
    values = np.full(v_norm.shape, np.nan, dtype=np.float64)
    x = np.sum(v * w, axis=-1)
    y = np.sum(np.cross(b1_unit, v) * w, axis=-1)
    values[valid] = np.degrees(np.arctan2(y[valid], x[valid]))
    output[..., 1:-2] = values
    return output


def _span_distances(
    coordinates: np.ndarray,
    bond_lengths: np.ndarray,
    separation: int,
    normalize_bonds: bool,
    reference_bond_nm: float,
) -> np.ndarray:
    residue_count = coordinates.shape[-2]
    output = np.full(coordinates.shape[:-1], np.nan, dtype=np.float64)
    if residue_count <= separation:
        return output
    distances = _safe_norm(
        coordinates[..., separation:, :] - coordinates[..., :-separation, :]
    )
    if normalize_bonds:
        contour = np.zeros_like(distances)
        for offset in range(separation):
            contour += bond_lengths[
                ..., offset : offset + residue_count - separation
            ]
        mean_bond = contour / float(separation)
        if np.any(mean_bond <= 0.0):
            raise ValueError("Backbone contour contains a zero-length bond.")
        distances = distances * (reference_bond_nm / mean_bond)
    output[..., : residue_count - separation] = distances
    return output


def calculate_backbone_geometry(
    coordinates,
    *,
    normalize_bonds: bool = True,
    reference_bond_nm: float = 0.38,
) -> BackboneGeometry:
    """Calculate virtual angles, dihedrals, and i-to-i+k distances in nm."""

    values = _as_chain_coordinates(coordinates)
    bonds = _safe_norm(values[..., 1:, :] - values[..., :-1, :])
    if np.any(bonds <= 0.0):
        raise ValueError("Consecutive residue coordinates contain a zero-length bond.")
    return BackboneGeometry(
        angle_degrees=_angles_degrees(values),
        dihedral_degrees=_dihedrals_degrees(values),
        distance2_nm=_span_distances(
            values, bonds, 2, normalize_bonds, reference_bond_nm
        ),
        distance3_nm=_span_distances(
            values, bonds, 3, normalize_bonds, reference_bond_nm
        ),
        distance4_nm=_span_distances(
            values, bonds, 4, normalize_bonds, reference_bond_nm
        ),
        bond_length_nm=bonds,
    )


def _linear_window(values: np.ndarray, center_width: tuple[float, float]) -> np.ndarray:
    center, half_width = center_width
    return np.isfinite(values) & (np.abs(values - center) <= half_width)


def _circular_window(
    values: np.ndarray,
    center_width: tuple[float, float],
) -> np.ndarray:
    center, half_width = center_width
    delta = (values - center + 180.0) % 360.0 - 180.0
    return np.isfinite(values) & (np.abs(delta) <= half_width)


def _dihedral_residue_hits(window_hits: np.ndarray) -> np.ndarray:
    """Map a four-point torsion hit to both central residues."""

    output = np.zeros_like(window_hits, dtype=bool)
    if window_hits.shape[-1] < 4:
        return output
    torsion_hits = window_hits[..., 1:-2]
    output[..., 1:-2] |= torsion_hits
    output[..., 2:-1] |= torsion_hits
    return output


def _filter_short_runs(mask: np.ndarray, minimum: int) -> np.ndarray:
    if minimum < 1:
        raise ValueError("Minimum segment length must be at least one.")
    values = np.asarray(mask, dtype=bool)
    length = values.shape[-1]
    flat = values.reshape(-1, length)
    output = np.zeros_like(flat)
    for row_index, row in enumerate(flat):
        padded = np.concatenate(([False], row, [False])).astype(np.int8)
        changes = np.diff(padded)
        starts = np.flatnonzero(changes == 1)
        stops = np.flatnonzero(changes == -1)
        for start, stop in zip(starts, stops):
            if stop - start >= minimum:
                output[row_index, start:stop] = True
    return output.reshape(values.shape)


def _regularize_center_hits(mask: np.ndarray, minimum_length: int) -> np.ndarray:
    """Convert consecutive angle-center hits into residue segments."""

    minimum_centers = max(1, int(minimum_length) - 2)
    centers = _filter_short_runs(mask, minimum_centers)
    output = centers.copy()
    output[..., 1:] |= centers[..., :-1]
    output[..., :-1] |= centers[..., 1:]
    return _filter_short_runs(output, int(minimum_length))


def _paint_windows(window_hits: np.ndarray, width: int) -> np.ndarray:
    output = np.zeros_like(window_hits, dtype=bool)
    valid_count = max(0, window_hits.shape[-1] - width + 1)
    if valid_count == 0:
        return output
    seeds = window_hits[..., :valid_count]
    for offset in range(width):
        output[..., offset : offset + valid_count] |= seeds
    return output


def _distance_candidates(
    geometry: BackboneGeometry,
    windows: tuple[tuple[float, float], ...],
    minimum_length: int,
) -> np.ndarray:
    d2_window, d3_window, d4_window = windows
    starts = (
        _linear_window(geometry.distance2_nm, d2_window)
        & _linear_window(geometry.distance3_nm, d3_window)
        & _linear_window(geometry.distance4_nm, d4_window)
    )
    painted = _paint_windows(starts, 5)
    return _filter_short_runs(painted, minimum_length)


def classify_secondary_structure(
    coordinates,
    *,
    method: str = "psea",
    min_helix_length: int = 4,
    min_strand_length: int = 3,
    normalize_bonds: bool = True,
    profile: PSEAProfile = DEFAULT_PSEA_PROFILE,
) -> SecondaryStructureResult:
    """Infer local H/E/C states from a residue-level coordinate trace.

    ``angle`` uses only the virtual bond angle and is intentionally permissive.
    ``psea`` requires an angle/dihedral match or a complete d2/d3/d4 match.
    """

    if method not in {"angle", "psea"}:
        raise ValueError("Secondary-structure method must be 'angle' or 'psea'.")
    if min_helix_length < 1 or min_strand_length < 1:
        raise ValueError("Minimum secondary-structure lengths must be positive.")

    geometry = calculate_backbone_geometry(
        coordinates,
        normalize_bonds=normalize_bonds,
        reference_bond_nm=profile.reference_bond_nm,
    )
    helix_angle = _linear_window(geometry.angle_degrees, profile.helix_angle)
    strand_angle = _linear_window(geometry.angle_degrees, profile.strand_angle)

    if method == "angle":
        helix = _regularize_center_hits(helix_angle, min_helix_length)
        strand = _regularize_center_hits(strand_angle, min_strand_length)
    else:
        helix_dihedral = _dihedral_residue_hits(
            _circular_window(
                geometry.dihedral_degrees,
                profile.helix_dihedral,
            )
        )
        strand_dihedral = _dihedral_residue_hits(
            _circular_window(
                geometry.dihedral_degrees,
                profile.strand_dihedral,
            )
        )
        helix_angles = _regularize_center_hits(
            helix_angle & helix_dihedral,
            min_helix_length,
        )
        strand_angles = _regularize_center_hits(
            strand_angle & strand_dihedral,
            min_strand_length,
        )
        # Distance windows alone cannot distinguish a right-handed trace from
        # its mirror image.  Keep the P-SEA distance rescue for noisy angles,
        # but require a locally coherent positive helical torsion as a weak
        # chirality prior for residue-level protein models.
        helix_torsions = _regularize_center_hits(
            helix_dihedral,
            min_helix_length,
        )
        helix = helix_angles | (
            _distance_candidates(
                geometry,
                profile.helix_distances,
                min_helix_length,
            )
            & helix_torsions
        )
        strand = strand_angles | _distance_candidates(
            geometry,
            profile.strand_distances,
            min_strand_length,
        )

    conflicts = helix & strand
    if np.any(conflicts):
        # A transition painted by two overlapping windows is not a confident
        # secondary-structure assignment.  Keep a conservative coil gap.
        helix = helix & ~conflicts
        strand = strand & ~conflicts

    states = np.full(helix.shape, STATE_COIL, dtype=np.uint8)
    states[helix] = STATE_HELIX
    states[strand] = STATE_STRAND
    return SecondaryStructureResult(
        states=states,
        helix=helix,
        strand=strand,
        geometry=geometry,
    )


def _strand_segments(mask: np.ndarray):
    values = np.asarray(mask, dtype=bool)
    if values.ndim != 2:
        raise ValueError("Strand masks must have shape (n_chains, n_residues).")
    labels = np.full(values.shape, -1, dtype=np.int64)
    segments = []
    for chain, row in enumerate(values):
        padded = np.concatenate(([False], row, [False])).astype(np.int8)
        changes = np.diff(padded)
        starts = np.flatnonzero(changes == 1)
        stops = np.flatnonzero(changes == -1)
        for start, stop in zip(starts, stops):
            segment_id = len(segments)
            labels[chain, start:stop] = segment_id
            segments.append((chain, int(start), int(stop)))
    return labels, segments


def _registered_run(
    rows,
    orientation: str,
    registry: int,
    slack: int,
):
    candidates = [
        row
        for row in rows
        if abs(
            (row[3] - row[1] if orientation == "parallel" else row[3] + row[1])
            - registry
        )
        <= slack
    ]
    candidates.sort(
        key=lambda row: (
            abs(
                (
                    row[3] - row[1]
                    if orientation == "parallel"
                    else row[3] + row[1]
                )
                - registry
            ),
            row[4],
        )
    )
    used_a = set()
    used_b = set()
    matched = []
    for row in candidates:
        if row[1] in used_a or row[3] in used_b:
            continue
        used_a.add(row[1])
        used_b.add(row[3])
        matched.append(row)

    matched.sort(key=lambda row: row[1])
    if not matched:
        return []
    maximum_gap = slack + 1
    runs = [[matched[0]]]
    direction = 1 if orientation == "parallel" else -1
    for row in matched[1:]:
        previous = runs[-1][-1]
        delta_a = row[1] - previous[1]
        delta_b = direction * (row[3] - previous[3])
        if 0 < delta_a <= maximum_gap and 0 < delta_b <= maximum_gap:
            runs[-1].append(row)
        else:
            runs.append([row])
    return max(
        runs,
        key=lambda run: (len(run), -float(np.mean([row[4] for row in run]))),
    )


def detect_interchain_strand_pairing(
    coordinates,
    local_strand,
    contact_pairs,
    contact_vectors,
    *,
    max_axis_angle_degrees: float = 30.0,
    max_lateral_deviation_degrees: float = 35.0,
    min_pairs: int = 2,
    registry_slack: int = 1,
    extension: int = 1,
    min_strand_length: int = 3,
):
    """Confirm local strand segments with registered inter-chain contacts.

    ``contact_pairs`` rows are ``chain_a, residue_a, chain_b, residue_b``.
    ``contact_vectors`` point from the first residue to the second and must use
    the minimum-image displacement when periodic boundaries are active.
    """

    values = _as_chain_coordinates(coordinates)
    if values.ndim != 3:
        raise ValueError("Pairing coordinates must have shape (n_chains, n_residues, 3).")
    strand = np.asarray(local_strand, dtype=bool)
    if strand.shape != values.shape[:2]:
        raise ValueError("Local-strand mask does not match the coordinate array.")
    pairs = np.asarray(contact_pairs, dtype=np.int64)
    vectors = np.asarray(contact_vectors, dtype=np.float64)
    if pairs.size == 0:
        pairs = np.empty((0, 4), dtype=np.int64)
    if vectors.size == 0:
        vectors = np.empty((0, 3), dtype=np.float64)
    if pairs.ndim != 2 or pairs.shape[1] != 4:
        raise ValueError("Contact pairs must have shape (n_pairs, 4).")
    if vectors.shape != (pairs.shape[0], 3):
        raise ValueError("Contact vectors must have shape (n_pairs, 3).")
    if min_pairs < 1:
        raise ValueError("Minimum strand-pair contacts must be positive.")
    if registry_slack < 0 or extension < 0:
        raise ValueError("Registry slack and pairing extension cannot be negative.")
    if not 0.0 <= max_axis_angle_degrees < 90.0:
        raise ValueError("Maximum strand-axis angle must be in [0, 90).")
    if not 0.0 <= max_lateral_deviation_degrees < 90.0:
        raise ValueError("Maximum lateral deviation must be in [0, 90).")

    labels, segments = _strand_segments(strand)
    paired = np.zeros_like(strand)
    if not segments or not pairs.size:
        return paired, []

    axes = []
    for chain, start, stop in segments:
        axis = values[chain, stop - 1] - values[chain, start]
        length = float(np.linalg.norm(axis))
        axes.append(axis / length if length > 0.0 else np.full(3, np.nan))
    axes = np.asarray(axes, dtype=np.float64)

    grouped = {}
    chain_count, residue_count = strand.shape
    for pair, vector in zip(pairs, vectors):
        chain_a, residue_a, chain_b, residue_b = (int(value) for value in pair)
        if not (
            0 <= chain_a < chain_count
            and 0 <= chain_b < chain_count
            and 0 <= residue_a < residue_count
            and 0 <= residue_b < residue_count
        ):
            raise ValueError("A strand contact refers to an out-of-range residue.")
        if chain_a == chain_b or not strand[chain_a, residue_a] or not strand[chain_b, residue_b]:
            continue
        segment_a = int(labels[chain_a, residue_a])
        segment_b = int(labels[chain_b, residue_b])
        if segment_a < 0 or segment_b < 0 or segment_a == segment_b:
            continue
        if segment_a > segment_b:
            segment_a, segment_b = segment_b, segment_a
            chain_a, chain_b = chain_b, chain_a
            residue_a, residue_b = residue_b, residue_a
            vector = -vector
        distance = float(np.linalg.norm(vector))
        if not np.isfinite(distance) or distance <= 0.0:
            continue
        grouped.setdefault((segment_a, segment_b), []).append(
            (chain_a, residue_a, chain_b, residue_b, distance, vector / distance)
        )

    cosine_axis = float(np.cos(np.deg2rad(max_axis_angle_degrees)))
    sine_lateral = float(np.sin(np.deg2rad(max_lateral_deviation_degrees)))
    records = []
    for (segment_a, segment_b), rows in grouped.items():
        axis_a = axes[segment_a]
        axis_b = axes[segment_b]
        if not np.all(np.isfinite(axis_a)) or not np.all(np.isfinite(axis_b)):
            continue
        alignment = float(np.dot(axis_a, axis_b))
        if alignment >= cosine_axis:
            orientation = "parallel"
        elif alignment <= -cosine_axis:
            orientation = "antiparallel"
        else:
            continue

        lateral_rows = [
            row
            for row in rows
            if abs(float(np.dot(axis_a, row[5]))) <= sine_lateral
            and abs(float(np.dot(axis_b, row[5]))) <= sine_lateral
        ]
        if not lateral_rows:
            continue
        registries = np.asarray(
            [
                row[3] - row[1]
                if orientation == "parallel"
                else row[3] + row[1]
                for row in lateral_rows
            ],
            dtype=np.int64,
        )
        best = None
        best_registry = None
        for registry in np.unique(registries):
            run = _registered_run(
                lateral_rows,
                orientation,
                int(registry),
                registry_slack,
            )
            rank = (len(run), -float(np.mean([row[4] for row in run]))) if run else (0, -np.inf)
            if best is None or rank > best[0]:
                best = (rank, run)
                best_registry = int(registry)
        if best is None or len(best[1]) < min_pairs:
            continue
        run = best[1]

        chain_a, segment_start_a, segment_stop_a = segments[segment_a]
        chain_b, segment_start_b, segment_stop_b = segments[segment_b]
        residues_a = [row[1] for row in run]
        residues_b = [row[3] for row in run]
        paired_start_a = max(segment_start_a, min(residues_a) - extension)
        paired_stop_a = min(segment_stop_a, max(residues_a) + extension + 1)
        paired_start_b = max(segment_start_b, min(residues_b) - extension)
        paired_stop_b = min(segment_stop_b, max(residues_b) + extension + 1)
        paired[chain_a, paired_start_a:paired_stop_a] = True
        paired[chain_b, paired_start_b:paired_stop_b] = True
        distances = np.asarray([row[4] for row in run], dtype=np.float64)
        records.append(
            StrandPairRecord(
                chain_a=chain_a,
                start_a=paired_start_a,
                stop_a=paired_stop_a,
                chain_b=chain_b,
                start_b=paired_start_b,
                stop_b=paired_stop_b,
                orientation=orientation,
                registry=int(best_registry),
                paired_residues=len(run),
                mean_distance_nm=float(np.mean(distances)),
                minimum_distance_nm=float(np.min(distances)),
            )
        )

    paired &= strand
    paired = _filter_short_runs(paired, min_strand_length)
    return paired, records


def apply_strand_pairing(states, paired_strand) -> np.ndarray:
    """Replace unpaired local E assignments by coil."""

    values = np.asarray(states, dtype=np.uint8)
    paired = np.asarray(paired_strand, dtype=bool)
    if values.shape != paired.shape:
        raise ValueError("Paired-strand mask does not match secondary-structure states.")
    output = values.copy()
    output[(output == STATE_STRAND) & ~paired] = STATE_COIL
    return output


def state_fractions(states) -> np.ndarray:
    """Return empirical C/H/E fractions across the leading copy axis."""

    values = np.asarray(states, dtype=np.uint8)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("States must have shape (n_copies, n_residues).")
    return np.stack(
        [
            np.mean(values == STATE_COIL, axis=0),
            np.mean(values == STATE_HELIX, axis=0),
            np.mean(values == STATE_STRAND, axis=0),
        ],
        axis=-1,
    )


def consensus_states(fractions, threshold: float = 0.5) -> np.ndarray:
    """Convert per-residue C/H/E fractions into a categorical display state."""

    values = np.asarray(fractions, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("Fractions must have shape (n_residues, 3).")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Consensus threshold must be between zero and one.")
    if not np.all(np.isfinite(values)):
        raise ValueError("State fractions contain NaN or infinite values.")
    if not np.allclose(np.sum(values, axis=1), 1.0, atol=1.0e-7, rtol=0.0):
        raise ValueError("C/H/E fractions must sum to one for every residue.")
    dominant = np.argmax(values, axis=1).astype(np.uint8)
    confidence = np.max(values, axis=1)
    tied = np.sum(
        np.isclose(values, confidence[:, np.newaxis], atol=1.0e-12, rtol=0.0),
        axis=1,
    ) > 1
    dominant[(confidence < threshold) | tied] = STATE_MIXED
    return dominant
