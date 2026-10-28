"""NumPy-only helpers for planar two-dimensional density maps."""

from __future__ import annotations

import numpy as np

from dropps.analysis.coexistence_core import periodic_center_fraction


PLANE_AXES = {
    "x": (1, 2),
    "y": (0, 2),
    "z": (0, 1),
}
DEFAULT_DENSMAP_BIN_WIDTH_NM = 0.02


def plane_axes(average_axis):
    """Return the two lattice axes retained after averaging over one axis."""

    try:
        return PLANE_AXES[str(average_axis)]
    except KeyError as exc:
        raise ValueError(
            "Average axis must be one of 'x', 'y', or 'z'."
        ) from exc


def density_grid_shape(
    plane_lengths_nm,
    bin_width_nm,
    bins_first=0,
    bins_second=0,
):
    """Choose a fixed grid from the first-frame plane lengths.

    Explicit bin counts take precedence.  Otherwise, the requested physical
    spacing is rounded to the nearest integer number of cells, matching the
    planar-grid convention used by ``gmx densmap``.
    """

    lengths = np.asarray(plane_lengths_nm, dtype=float)
    if lengths.shape != (2,) or not np.all(np.isfinite(lengths)):
        raise ValueError("Plane lengths must contain two finite values.")
    if np.any(lengths <= 0.0):
        raise ValueError("Plane lengths must be positive.")
    if not np.isfinite(bin_width_nm) or bin_width_nm <= 0.0:
        raise ValueError("Bin width must be positive and finite.")

    explicit = (bins_first, bins_second)
    result = []
    for length, count in zip(lengths, explicit):
        if int(count) != count or count < 0:
            raise ValueError("Explicit grid sizes must be non-negative integers.")
        if count:
            result.append(int(count))
        else:
            result.append(max(1, int(np.floor(length / bin_width_nm + 0.5))))
    return tuple(result)


def periodic_plane_center(fractions, weights=None):
    """Return circular centers and strengths for a compact periodic 2D group."""

    fractions = np.asarray(fractions, dtype=float)
    if fractions.ndim != 2 or fractions.shape[1] != 2 or fractions.shape[0] == 0:
        raise ValueError("Plane fractions must have shape (n_particles, 2).")
    if not np.all(np.isfinite(fractions)):
        raise ValueError("Plane fractions must be finite.")

    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        if weights.shape != (fractions.shape[0],):
            raise ValueError("Centering weights must match the number of particles.")

    centers = []
    strengths = []
    for dimension in range(2):
        center, strength = periodic_center_fraction(
            fractions[:, dimension],
            weights,
        )
        centers.append(center)
        strengths.append(strength)
    return np.asarray(centers, dtype=float), np.asarray(strengths, dtype=float)


def histogram_density_map(
    fractions,
    weights,
    plane_lengths_nm,
    averaging_thickness_nm,
    bins,
    normalization="volume",
    conversion=1.0,
):
    """Histogram weighted particles into a fractional planar grid.

    ``volume`` divides by the physical 3D cell volume, ``area`` omits the
    averaging-direction normalization, and ``count`` returns the mean weight
    in each grid cell.  The caller averages the resulting per-frame maps so
    that fluctuating box sizes are normalized correctly.
    """

    fractions = np.asarray(fractions, dtype=float)
    weights = np.asarray(weights, dtype=float)
    lengths = np.asarray(plane_lengths_nm, dtype=float)
    if fractions.ndim != 2 or fractions.shape[1] != 2:
        raise ValueError("Plane fractions must have shape (n_particles, 2).")
    if weights.shape != (fractions.shape[0],):
        raise ValueError("Weights must match the number of particle positions.")
    if not np.all(np.isfinite(fractions)) or not np.all(np.isfinite(weights)):
        raise ValueError("Fractions and weights must be finite.")
    if lengths.shape != (2,) or not np.all(np.isfinite(lengths)):
        raise ValueError("Plane lengths must contain two finite values.")
    if np.any(lengths <= 0.0):
        raise ValueError("Plane lengths must be positive.")
    if (
        not np.isfinite(averaging_thickness_nm)
        or averaging_thickness_nm <= 0.0
    ):
        raise ValueError("Averaging thickness must be positive and finite.")
    if len(bins) != 2 or any(int(count) != count or count <= 0 for count in bins):
        raise ValueError("Grid shape must contain two positive integers.")
    if normalization not in {"volume", "area", "count"}:
        raise ValueError(
            "Normalization must be 'volume', 'area', or 'count'."
        )
    if not np.isfinite(conversion):
        raise ValueError("Density conversion factor must be finite.")

    shape = (int(bins[0]), int(bins[1]))
    histogram = np.histogram2d(
        np.mod(fractions[:, 0], 1.0),
        np.mod(fractions[:, 1], 1.0),
        bins=shape,
        range=((0.0, 1.0), (0.0, 1.0)),
        weights=weights,
    )[0]
    if normalization == "count":
        divisor = 1.0
    else:
        cell_area_nm2 = float(np.prod(lengths) / np.prod(shape))
        divisor = cell_area_nm2
        if normalization == "volume":
            divisor *= float(averaging_thickness_nm)
    return histogram / divisor * float(conversion)


def slice_mask_and_thickness(
    averaging_fractions,
    averaging_length_nm,
    minimum_nm=None,
    maximum_nm=None,
):
    """Select a non-periodic interval along the averaging direction."""

    fractions = np.asarray(averaging_fractions, dtype=float)
    if fractions.ndim != 1 or not np.all(np.isfinite(fractions)):
        raise ValueError("Averaging fractions must be a finite 1D array.")
    if not np.isfinite(averaging_length_nm) or averaging_length_nm <= 0.0:
        raise ValueError("Averaging-axis length must be positive and finite.")

    lower = 0.0 if minimum_nm is None else float(minimum_nm)
    upper = float(averaging_length_nm) if maximum_nm is None else float(maximum_nm)
    if not np.isfinite(lower) or not np.isfinite(upper):
        raise ValueError("Averaging bounds must be finite when supplied.")
    lower = max(0.0, lower)
    upper = min(float(averaging_length_nm), upper)
    if upper <= lower:
        raise ValueError(
            "The averaging interval does not overlap the periodic box."
        )

    coordinates_nm = np.mod(fractions, 1.0) * float(averaging_length_nm)
    mask = (coordinates_nm >= lower) & (coordinates_nm < upper)
    return mask, upper - lower
