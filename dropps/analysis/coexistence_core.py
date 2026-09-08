"""NumPy-only helpers for planar coexistence-profile analysis."""

from __future__ import annotations

import numpy as np


TANH_10_90_FACTOR = float(2.0 * np.arctanh(0.8))


def periodic_center_fraction(fractions, weights=None):
    """Return the circular center and first-harmonic strength on ``[0, 1)``.

    The circular center is appropriate for a single periodic dense slab.  The
    dimensionless strength is zero for a perfectly uniform distribution and
    approaches one for a compact distribution.
    """

    fractions = np.asarray(fractions, dtype=float)
    if fractions.ndim != 1 or fractions.size == 0:
        raise ValueError("Fractions must be a non-empty one-dimensional array.")
    if not np.all(np.isfinite(fractions)):
        raise ValueError("Fractions contain non-finite values.")

    if weights is None:
        weights = np.ones(fractions.size, dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)
        if weights.shape != fractions.shape:
            raise ValueError("Weights must have the same shape as fractions.")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("Weights must be finite and non-negative.")
    total_weight = float(weights.sum())
    if total_weight <= 0.0:
        raise ValueError("At least one centering weight must be positive.")

    phase = np.sum(weights * np.exp(2.0j * np.pi * np.mod(fractions, 1.0)))
    strength = float(abs(phase) / total_weight)
    if abs(phase) <= np.finfo(float).eps * total_weight:
        raise ValueError(
            "The reference group has no resolvable first-harmonic center. "
            "It may be spatially uniform or contain multiple symmetric domains."
        )
    center = float(np.angle(phase) / (2.0 * np.pi)) % 1.0
    return center, strength


def center_periodic_fractions(fractions, center_fraction, target_fraction=0.5):
    """Shift fractional coordinates so that ``center_fraction`` reaches target."""

    fractions = np.asarray(fractions, dtype=float)
    if not np.all(np.isfinite(fractions)):
        raise ValueError("Fractions contain non-finite values.")
    if not np.isfinite(center_fraction) or not np.isfinite(target_fraction):
        raise ValueError("Center and target fractions must be finite.")
    return np.mod(fractions + target_fraction - center_fraction, 1.0)


def centering_schedule(frame_centers, frame_strengths, mode="frame", blocks=5):
    """Return the center applied to each frame for a requested centering mode.

    In ``block`` mode, circular centers are averaged within contiguous
    trajectory blocks and weighted by their per-frame confidence strengths.
    """

    centers = np.asarray(frame_centers, dtype=float)
    strengths = np.asarray(frame_strengths, dtype=float)
    if centers.ndim != 1 or strengths.shape != centers.shape or centers.size == 0:
        raise ValueError(
            "Frame centers and strengths must be matching non-empty arrays."
        )
    if mode not in {"frame", "block", "none"}:
        raise ValueError(f"Unknown centering mode '{mode}'.")
    if int(blocks) != blocks or blocks <= 0:
        raise ValueError("Number of blocks must be a positive integer.")

    if mode == "none":
        return {
            "centers": np.full(centers.size, np.nan, dtype=float),
            "order_values": np.asarray([], dtype=float),
            "groups": 0,
        }

    if not np.all(np.isfinite(centers)):
        raise ValueError("Frame centers contain non-finite values.")
    if (
        not np.all(np.isfinite(strengths))
        or np.any(strengths < 0.0)
        or np.any(strengths > 1.0 + 1.0e-12)
    ):
        raise ValueError("Frame centering strengths must be finite and in [0, 1].")

    if mode == "frame":
        return {
            "centers": np.mod(centers, 1.0),
            "order_values": strengths.copy(),
            "groups": centers.size,
        }

    blocks_used = min(int(blocks), centers.size)
    applied_centers = np.empty(centers.size, dtype=float)
    block_orders = []
    phasors = strengths * np.exp(2.0j * np.pi * centers)
    for indices in np.array_split(np.arange(centers.size), blocks_used):
        block_phase = np.mean(phasors[indices])
        if abs(block_phase) <= np.finfo(float).eps:
            raise ValueError(
                "A centering block has no resolvable circular center. Increase "
                "--blocks, use --center-mode frame, or inspect slab stability."
            )
        applied_centers[indices] = float(np.angle(block_phase) / (2.0 * np.pi)) % 1.0
        block_orders.append(float(abs(block_phase)))
    return {
        "centers": applied_centers,
        "order_values": np.asarray(block_orders, dtype=float),
        "groups": blocks_used,
    }


def histogram_density(
    fractions,
    weights,
    cell_volume_nm3,
    bins,
    conversion=1.0,
):
    """Histogram particles into equal-volume fractional slabs.

    ``weights`` may be particle counts or masses.  The raw histogram is divided
    by the physical volume of one slab and then multiplied by ``conversion``.
    """

    fractions = np.asarray(fractions, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if fractions.ndim != 1 or weights.shape != fractions.shape:
        raise ValueError("Fractions and weights must be matching 1D arrays.")
    if fractions.size == 0:
        raise ValueError("Cannot calculate a density for an empty selection.")
    if not np.all(np.isfinite(fractions)) or not np.all(np.isfinite(weights)):
        raise ValueError("Fractions and weights must be finite.")
    if not np.isfinite(cell_volume_nm3) or cell_volume_nm3 <= 0.0:
        raise ValueError("Cell volume must be positive and finite.")
    if int(bins) != bins or bins <= 0:
        raise ValueError("Number of bins must be a positive integer.")
    if not np.isfinite(conversion):
        raise ValueError("Density conversion factor must be finite.")

    histogram = np.histogram(
        np.mod(fractions, 1.0),
        bins=int(bins),
        range=(0.0, 1.0),
        weights=weights,
    )[0]
    bin_volume_nm3 = float(cell_volume_nm3) / int(bins)
    return histogram / bin_volume_nm3 * float(conversion)


def double_tanh_profile(
    coordinates_nm,
    dilute_density,
    dense_density,
    half_width_nm,
    interface_parameter_nm,
):
    """Return a symmetric two-interface slab density profile.

    ``interface_parameter_nm`` is the scale in each hyperbolic tangent.  The
    corresponding 10--90% interfacial width is
    ``TANH_10_90_FACTOR * interface_parameter_nm``.
    """

    coordinates = np.asarray(coordinates_nm, dtype=float)
    if interface_parameter_nm <= 0.0:
        raise ValueError("Interface parameter must be positive.")
    if half_width_nm <= 0.0:
        raise ValueError("Slab half-width must be positive.")
    shape = 0.5 * (
        np.tanh((coordinates + half_width_nm) / interface_parameter_nm)
        - np.tanh((coordinates - half_width_nm) / interface_parameter_nm)
    )
    return dilute_density + (dense_density - dilute_density) * shape


def _linear_profile_levels(profile, shape):
    """Fit ``profile = dilute + contrast * shape`` by linear least squares."""

    shape_mean = float(shape.mean())
    profile_mean = float(profile.mean())
    centered_shape = shape - shape_mean
    denominator = float(np.dot(centered_shape, centered_shape))
    if denominator <= np.finfo(float).eps:
        return None
    contrast = float(np.dot(centered_shape, profile - profile_mean) / denominator)
    if contrast <= 0.0:
        return None
    dilute = profile_mean - contrast * shape_mean
    if dilute < 0.0:
        shape_norm = float(np.dot(shape, shape))
        if shape_norm <= np.finfo(float).eps:
            return None
        dilute = 0.0
        contrast = float(np.dot(shape, profile) / shape_norm)
        if contrast <= 0.0:
            return None
    fitted = dilute + contrast * shape
    residual = profile - fitted
    sse = float(np.dot(residual, residual))
    return dilute, dilute + contrast, fitted, sse


def fit_slab_profile(coordinates_nm, density_profile):
    """Fit a centered planar slab with a deterministic NumPy grid search."""

    coordinates = np.asarray(coordinates_nm, dtype=float)
    profile = np.asarray(density_profile, dtype=float)
    if coordinates.ndim != 1 or profile.shape != coordinates.shape:
        raise ValueError("Coordinates and density profile must be matching 1D arrays.")
    if coordinates.size < 12:
        raise ValueError("At least 12 density bins are required for slab fitting.")
    if not np.all(np.isfinite(coordinates)) or not np.all(np.isfinite(profile)):
        raise ValueError("Coordinates and density profile must be finite.")
    if np.any(profile < 0.0):
        raise ValueError("Coexistence fitting requires a non-negative density profile.")

    differences = np.diff(coordinates)
    spacing = float(np.median(differences))
    if spacing <= 0.0 or not np.allclose(
        differences, spacing, rtol=1.0e-5, atol=1.0e-10
    ):
        raise ValueError("Density coordinates must be uniformly increasing.")
    cell_length = spacing * coordinates.size
    profile_variance = float(np.dot(profile - profile.mean(), profile - profile.mean()))
    if profile_variance <= np.finfo(float).eps * max(float(profile.size), 1.0):
        raise ValueError(
            "Density profile is uniform and has no coexisting slab to fit."
        )

    half_min = max(2.0 * spacing, 0.05 * cell_length)
    half_max = min(0.46 * cell_length, 0.5 * cell_length - 2.0 * spacing)
    interface_min = max(0.1 * spacing, 1.0e-8)
    interface_max = min(0.20 * cell_length, 0.8 * half_max)
    if half_max <= half_min or interface_max <= interface_min:
        raise ValueError("Simulation cell is too short for a two-interface slab fit.")

    best = None

    def consider(half_width, interface_parameter):
        nonlocal best
        shape = 0.5 * (
            np.tanh((coordinates + half_width) / interface_parameter)
            - np.tanh((coordinates - half_width) / interface_parameter)
        )
        levels = _linear_profile_levels(profile, shape)
        if levels is None:
            return
        dilute, dense, fitted, sse = levels
        if best is None or sse < best[0]:
            best = (sse, half_width, interface_parameter, dilute, dense, fitted)

    half_values = np.linspace(half_min, half_max, 72)
    interface_values = np.geomspace(interface_min, interface_max, 64)
    for half_width in half_values:
        for interface_parameter in interface_values:
            consider(float(half_width), float(interface_parameter))

    if best is None:
        raise ValueError(
            "Could not fit a dense-centered slab. Check the reference group and "
            "whether the trajectory contains two coexisting phases."
        )

    half_step = float(half_values[1] - half_values[0])
    interface_step = max(
        float(best[2]) * (float(interface_values[1] / interface_values[0]) - 1.0),
        interface_min,
    )
    for _ in range(5):
        half_candidates = np.linspace(
            max(half_min, best[1] - 2.0 * half_step),
            min(half_max, best[1] + 2.0 * half_step),
            17,
        )
        interface_candidates = np.linspace(
            max(interface_min, best[2] - 2.0 * interface_step),
            min(interface_max, best[2] + 2.0 * interface_step),
            17,
        )
        for half_width in half_candidates:
            for interface_parameter in interface_candidates:
                consider(float(half_width), float(interface_parameter))
        half_step *= 0.25
        interface_step *= 0.25

    sse, half_width, interface_parameter, dilute, dense, fitted = best
    r_squared = 1.0 - sse / profile_variance
    rmse = float(np.sqrt(sse / profile.size))
    return {
        "dilute_density": float(dilute),
        "dense_density": float(dense),
        "contrast": float(dense - dilute),
        "half_width_nm": float(half_width),
        "slab_width_nm": float(2.0 * half_width),
        "interface_parameter_nm": float(interface_parameter),
        "interface_width_10_90_nm": float(TANH_10_90_FACTOR * interface_parameter),
        "r_squared": float(r_squared),
        "rmse": rmse,
        "fitted_profile": np.asarray(fitted, dtype=float),
    }


def contiguous_block_statistics(values, blocks):
    """Return full mean and SEM calculated from contiguous block means."""

    values = np.asarray(values, dtype=float)
    if values.ndim < 1 or values.shape[0] == 0:
        raise ValueError("Values must contain at least one frame.")
    if not np.all(np.isfinite(values)):
        raise ValueError("Values contain non-finite entries.")
    if int(blocks) != blocks or blocks <= 0:
        raise ValueError("Number of blocks must be a positive integer.")

    blocks_used = min(int(blocks), values.shape[0])
    block_means = np.stack(
        [chunk.mean(axis=0) for chunk in np.array_split(values, blocks_used)],
        axis=0,
    )
    sem = (
        block_means.std(axis=0, ddof=1) / np.sqrt(blocks_used)
        if blocks_used > 1
        else np.full(values.shape[1:], np.nan, dtype=float)
    )
    return {
        "mean": values.mean(axis=0),
        "sem": sem,
        "block_means": block_means,
        "blocks": blocks_used,
    }


def fit_slab_profile_blocks(coordinates_nm, frame_profiles, blocks):
    """Fit the full profile and every contiguous trajectory block."""

    frame_profiles = np.asarray(frame_profiles, dtype=float)
    if frame_profiles.ndim != 2 or frame_profiles.shape[0] == 0:
        raise ValueError("Frame profiles must have shape (n_frames, n_bins).")
    if int(blocks) != blocks or blocks <= 0:
        raise ValueError("Number of blocks must be a positive integer.")

    full_fit = fit_slab_profile(coordinates_nm, frame_profiles.mean(axis=0))
    blocks_used = min(int(blocks), frame_profiles.shape[0])
    block_fits = [
        fit_slab_profile(coordinates_nm, chunk.mean(axis=0))
        for chunk in np.array_split(frame_profiles, blocks_used)
    ]
    scalar_keys = (
        "dilute_density",
        "dense_density",
        "contrast",
        "half_width_nm",
        "slab_width_nm",
        "interface_parameter_nm",
        "interface_width_10_90_nm",
        "r_squared",
        "rmse",
    )
    block_values = {
        key: np.asarray([fit[key] for fit in block_fits], dtype=float)
        for key in scalar_keys
    }
    sem = {
        key: (
            float(values.std(ddof=1) / np.sqrt(blocks_used))
            if blocks_used > 1
            else float("nan")
        )
        for key, values in block_values.items()
    }
    return {
        "fit": full_fit,
        "sem": sem,
        "block_values": block_values,
        "blocks": blocks_used,
    }


def plateau_masks(
    coordinates_nm,
    half_width_nm,
    interface_parameter_nm,
    buffer_factor=2.0,
):
    """Return dense- and dilute-plateau masks away from both interfaces."""

    coordinates = np.asarray(coordinates_nm, dtype=float)
    if coordinates.ndim != 1 or coordinates.size == 0:
        raise ValueError("Coordinates must be a non-empty one-dimensional array.")
    if not np.isfinite(buffer_factor) or buffer_factor < 0.0:
        raise ValueError("Plateau buffer factor must be finite and non-negative.")
    dense_limit = float(half_width_nm - buffer_factor * interface_parameter_nm)
    dilute_limit = float(half_width_nm + buffer_factor * interface_parameter_nm)
    absolute_coordinate = np.abs(coordinates)
    dense_mask = absolute_coordinate <= dense_limit
    dilute_mask = absolute_coordinate >= dilute_limit
    if np.count_nonzero(dense_mask) < 2 or np.count_nonzero(dilute_mask) < 2:
        raise ValueError(
            "The fitted slab leaves fewer than two bins in a phase plateau. "
            "Reduce --plateau-buffer, use finer bins, or simulate a longer slab."
        )
    return dense_mask, dilute_mask
