"""NumPy-only helpers for phase-resolved molecular gyration analysis."""

from __future__ import annotations

import numpy as np


def periodic_radius_of_gyration(positions_nm, masses, box_vectors_nm):
    """Return a mass-weighted molecular Rg after minimum-image reconstruction.

    The first selected particle anchors the reconstruction.  This matches the
    periodic molecular-center convention used by the phase exchange analyses
    and is valid when the selected part of one molecule spans less than half a
    box length along each periodic direction.
    """

    positions = np.asarray(positions_nm, dtype=float)
    weights = np.asarray(masses, dtype=float)
    box = np.asarray(box_vectors_nm, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3 or positions.shape[0] == 0:
        raise ValueError(
            "Positions must have shape (n_particles, 3) with at least one row."
        )
    if weights.shape != (positions.shape[0],):
        raise ValueError("Masses must match the number of positions.")
    if not np.all(np.isfinite(positions)):
        raise ValueError("Positions must be finite.")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("Masses must be finite and non-negative.")
    if float(weights.sum()) <= 0.0:
        raise ValueError("At least one molecular mass must be positive.")
    if box.shape != (3, 3) or not np.all(np.isfinite(box)):
        raise ValueError("Box vectors must form a finite 3x3 matrix.")
    determinant = float(np.linalg.det(box))
    if not np.isfinite(determinant) or abs(determinant) <= np.finfo(float).eps:
        raise ValueError("Box vectors must define a non-singular periodic cell.")

    fractional = positions @ np.linalg.inv(box)
    anchor = fractional[0]
    displacement = fractional - anchor
    displacement -= np.floor(displacement + 0.5)
    reconstructed = (anchor + displacement) @ box
    center = np.average(reconstructed, axis=0, weights=weights)
    squared_distances = np.sum((reconstructed - center) ** 2, axis=1)
    return float(np.sqrt(np.average(squared_distances, weights=weights)))


def _phase_inputs(rg_nm, states, molecule_groups, group_count, phases):
    rg = np.asarray(rg_nm, dtype=float)
    phase_states = np.asarray(states)
    groups = np.asarray(molecule_groups, dtype=int)
    phase_codes = tuple(int(phase) for phase in phases)
    if rg.ndim != 2 or rg.shape[0] == 0 or rg.shape[1] == 0:
        raise ValueError("Rg values must have shape (n_frames, n_molecules).")
    if phase_states.shape != rg.shape:
        raise ValueError("Phase states must have the same shape as Rg values.")
    if groups.shape != (rg.shape[1],):
        raise ValueError("Molecule groups must contain one group index per molecule.")
    if not np.all(np.isfinite(rg)) or np.any(rg < 0.0):
        raise ValueError("Rg values must be finite and non-negative.")
    if int(group_count) != group_count or group_count <= 0:
        raise ValueError("Group count must be a positive integer.")
    if np.any(groups < 0) or np.any(groups >= int(group_count)):
        raise ValueError(
            "Molecule group indices are outside the requested group count."
        )
    if not phase_codes or len(set(phase_codes)) != len(phase_codes):
        raise ValueError("Phase codes must be non-empty and unique.")
    return rg, phase_states, groups, int(group_count), phase_codes


def phase_rg_time_series(rg_nm, states, molecule_groups, group_count, phases):
    """Return per-frame mean, standard deviation, and count for every group/phase."""

    rg, phase_states, groups, group_count, phase_codes = _phase_inputs(
        rg_nm, states, molecule_groups, group_count, phases
    )
    series_count = group_count * len(phase_codes)
    means = np.full((rg.shape[0], series_count), np.nan, dtype=float)
    standard_deviations = np.full_like(means, np.nan)
    counts = np.zeros(means.shape, dtype=int)
    series = []
    series_index = 0
    for group_index in range(group_count):
        in_group = groups == group_index
        for phase in phase_codes:
            series.append((group_index, phase))
            for frame_index in range(rg.shape[0]):
                selected = in_group & (phase_states[frame_index] == phase)
                values = rg[frame_index, selected]
                counts[frame_index, series_index] = values.size
                if values.size:
                    means[frame_index, series_index] = float(values.mean())
                    standard_deviations[frame_index, series_index] = float(values.std())
            series_index += 1
    return {
        "series": tuple(series),
        "mean_rg_nm": means,
        "std_rg_nm": standard_deviations,
        "counts": counts,
    }


def phase_rg_summary(
    rg_nm,
    states,
    molecule_groups,
    group_count,
    phases,
    blocks=5,
):
    """Summarize pooled frame-molecule Rg observations by group and phase."""

    rg, phase_states, groups, group_count, phase_codes = _phase_inputs(
        rg_nm, states, molecule_groups, group_count, phases
    )
    if int(blocks) != blocks or blocks <= 0:
        raise ValueError("Number of blocks must be a positive integer.")
    frame_blocks = np.array_split(np.arange(rg.shape[0]), min(int(blocks), rg.shape[0]))
    rows = []
    for group_index in range(group_count):
        group_mask = groups == group_index
        molecule_count = int(np.count_nonzero(group_mask))
        for phase in phase_codes:
            selected = (phase_states == phase) & group_mask[np.newaxis, :]
            values = rg[selected]
            block_means = []
            for frame_indices in frame_blocks:
                block_selected = selected[frame_indices]
                block_values = rg[frame_indices][block_selected]
                if block_values.size:
                    block_means.append(float(block_values.mean()))
            blocks_used = len(block_means)
            block_sem = (
                float(np.std(block_means, ddof=1) / np.sqrt(blocks_used))
                if blocks_used > 1
                else float("nan")
            )
            observation_count = int(values.size)
            rows.append(
                {
                    "group_index": group_index,
                    "phase": phase,
                    "molecule_count": molecule_count,
                    "observations": observation_count,
                    "occupancy_fraction": observation_count
                    / (rg.shape[0] * molecule_count),
                    "occupied_frames": int(np.count_nonzero(np.any(selected, axis=1))),
                    "observed_molecules": int(
                        np.count_nonzero(np.any(selected, axis=0))
                    ),
                    "mean_rg_nm": float(values.mean()) if values.size else float("nan"),
                    "std_rg_nm": float(values.std()) if values.size else float("nan"),
                    "median_rg_nm": float(np.median(values))
                    if values.size
                    else float("nan"),
                    "min_rg_nm": float(values.min()) if values.size else float("nan"),
                    "max_rg_nm": float(values.max()) if values.size else float("nan"),
                    "block_sem_rg_nm": block_sem,
                    "blocks_used": blocks_used,
                }
            )
    return rows


def phase_rg_histogram(
    rg_nm,
    states,
    molecule_groups,
    group_count,
    phases,
    bin_width_nm,
):
    """Return phase-conditioned PDFs on one shared Rg grid."""

    rg, phase_states, groups, group_count, phase_codes = _phase_inputs(
        rg_nm, states, molecule_groups, group_count, phases
    )
    if not np.isfinite(bin_width_nm) or bin_width_nm <= 0.0:
        raise ValueError("Histogram bin width must be positive and finite.")
    selected_values = []
    for group_index in range(group_count):
        group_mask = groups == group_index
        for phase in phase_codes:
            values = rg[(phase_states == phase) & group_mask[np.newaxis, :]]
            if values.size:
                selected_values.append(values)
    pooled = np.concatenate(selected_values) if selected_values else rg.reshape(-1)
    lower = np.floor(float(pooled.min()) / bin_width_nm) * bin_width_nm
    upper = np.ceil(float(pooled.max()) / bin_width_nm) * bin_width_nm
    if upper <= lower:
        upper = lower + bin_width_nm
    edges = np.arange(lower, upper + 0.5 * bin_width_nm, bin_width_nm)
    if edges.size < 2:
        edges = np.asarray([lower, lower + bin_width_nm], dtype=float)

    densities = []
    series = []
    for group_index in range(group_count):
        group_mask = groups == group_index
        for phase in phase_codes:
            values = rg[(phase_states == phase) & group_mask[np.newaxis, :]]
            series.append((group_index, phase))
            if values.size:
                counts, _ = np.histogram(values, bins=edges)
                densities.append(counts.astype(float) / (values.size * np.diff(edges)))
            else:
                densities.append(np.full(edges.size - 1, np.nan, dtype=float))
    return {
        "series": tuple(series),
        "bin_centers_nm": 0.5 * (edges[:-1] + edges[1:]),
        "density_per_nm": np.stack(densities, axis=1),
        "edges_nm": edges,
    }
