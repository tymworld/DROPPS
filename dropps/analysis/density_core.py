"""Shared geometry, frame-selection, and centering helpers for density analyses."""

from __future__ import annotations

import numpy as np

from dropps.share.time_selection import select_time_indices


ANGSTROM_TO_NM = 0.1
PS_TO_NS = 0.001
DALTON_PER_NM3_TO_MG_PER_ML = 1.66053906660
CHARGE_PER_NM3_TO_E_MOL_PER_ML = DALTON_PER_NM3_TO_MG_PER_ML / 1000.0
DEFAULT_DENSITY_BIN_WIDTH_NM = 0.05
AXIS_TO_INDEX = {"x": 0, "y": 1, "z": 2}


def select_frame_indices(
    trajectory,
    start_ns,
    end_ns,
    delta_ns,
    *,
    minimum_frames=1,
    require_uniform=False,
):
    """Select trajectory frames from their actual timestamps.

    The returned interval is the median selected-frame interval in ns.  Callers
    that require a constant lag grid can request strict uniformity.
    """

    frame_times_ps = np.asarray(
        [float(timestep.time) for timestep in trajectory],
        dtype=np.float64,
    )
    selection = select_time_indices(
        frame_times_ps,
        start_time=start_ns,
        end_time=end_ns,
        delta_time=delta_ns,
        time_unit="ns",
    )
    if selection.indices.size < int(minimum_frames):
        raise ValueError(
            f"At least {int(minimum_frames)} trajectory frame(s) are required."
        )

    selected_times_ns = selection.times_ps * PS_TO_NS
    if selected_times_ns.size > 1:
        intervals = np.diff(selected_times_ns)
        if np.any(intervals <= 0.0):
            raise ValueError(
                "Selected trajectory timestamps must be strictly increasing."
            )
        actual_interval_ns = float(np.median(intervals))
        tolerance = max(
            np.finfo(np.float32).eps
            * max(float(np.max(np.abs(selected_times_ns))), 1.0)
            * 8.0,
            actual_interval_ns * 1.0e-6,
        )
        if require_uniform and not np.allclose(
            intervals,
            actual_interval_ns,
            rtol=1.0e-6,
            atol=tolerance,
        ):
            raise ValueError(
                "This analysis requires uniformly spaced selected trajectory "
                "timestamps."
            )
    else:
        actual_interval_ns = float("nan")

    return selection.indices.astype(int).tolist(), actual_interval_ns


def box_geometry_nm(timestep):
    """Return validated orthorhombic box vectors, lengths, and volume in nm."""

    box_angstrom = timestep.triclinic_dimensions
    if box_angstrom is None:
        raise ValueError("Trajectory frame does not contain periodic box vectors.")
    box = np.asarray(box_angstrom, dtype=float) * ANGSTROM_TO_NM
    if box.shape != (3, 3) or not np.all(np.isfinite(box)):
        raise ValueError("Periodic box vectors must form a finite 3x3 matrix.")

    lengths = np.linalg.norm(box, axis=1)
    volume = float(abs(np.linalg.det(box)))
    if np.any(lengths <= 0.0) or volume <= 0.0:
        raise ValueError("Periodic box must have positive lengths and volume.")
    gram = box @ box.T
    off_diagonal = gram - np.diag(np.diag(gram))
    scale = np.outer(lengths, lengths)
    if np.any(np.abs(off_diagonal) > 1.0e-6 * scale):
        raise ValueError("Density analysis currently requires an orthorhombic box.")
    return box, lengths, volume


def fractional_positions(positions_nm, box_vectors_nm):
    """Map Cartesian positions into periodic fractional coordinates."""

    positions = np.asarray(positions_nm, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("Positions must have shape (n_particles, 3).")
    return np.mod(positions @ np.linalg.inv(box_vectors_nm), 1.0)


def threshold_profile_center(density_profile, threshold):
    """Locate the largest thresholded dense region on a periodic profile.

    Returns its circular center as a box fraction and the fraction of total
    profile weight contained in that region.
    """

    profile = np.asarray(density_profile, dtype=float)
    if profile.ndim != 1 or profile.size == 0:
        raise ValueError("Density profile must be a non-empty 1D array.")
    if not np.all(np.isfinite(profile)) or np.any(profile < 0.0):
        raise ValueError("Centering density profile must be finite and non-negative.")
    if not np.isfinite(threshold) or not 0.0 < threshold < 1.0:
        raise ValueError("Dense-phase threshold must be finite and between 0 and 1.")

    maximum = float(profile.max())
    total = float(profile.sum())
    if maximum <= 0.0 or total <= 0.0:
        raise ValueError("Cannot locate a dense phase in an empty density profile.")
    dense = profile > float(threshold) * maximum
    if not np.any(dense):
        raise ValueError("No density bins exceed the dense-phase threshold.")
    if np.all(dense):
        raise ValueError(
            "The density profile is uniform and has no unique dense center."
        )

    starts = np.flatnonzero(dense & ~np.roll(dense, 1))
    regions = []
    bin_count = profile.size
    for start in starts:
        length = 0
        while length < bin_count and dense[(start + length) % bin_count]:
            length += 1
        indices = (start + np.arange(length, dtype=int)) % bin_count
        weight = float(profile[indices].sum())
        regions.append((length, weight, int(start), indices))

    length, weight, start, _ = max(
        regions,
        key=lambda region: (region[0], region[1], -region[2]),
    )
    unwrapped_bin_centers = start + np.arange(length, dtype=float) + 0.5
    center_fraction = float(unwrapped_bin_centers.mean() / bin_count) % 1.0
    return center_fraction, weight / total


def profile_center_shift(bin_count, center_fraction, target_fraction=0.5):
    """Return the integer bin roll that moves a periodic center to a target."""

    if int(bin_count) != bin_count or bin_count <= 0:
        raise ValueError("Bin count must be a positive integer.")
    if not np.isfinite(center_fraction) or not np.isfinite(target_fraction):
        raise ValueError("Center and target fractions must be finite.")
    return int(np.rint((float(target_fraction) - float(center_fraction)) * bin_count))
