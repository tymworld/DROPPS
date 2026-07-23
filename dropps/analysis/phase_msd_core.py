"""NumPy-only helpers for phase-conditioned molecular MSD analysis."""

from __future__ import annotations

import numpy as np


def unwrap_fractional_trajectories(fractions, box_lengths_nm):
    """Unwrap periodic fractional trajectories into continuous nm coordinates.

    Frame-to-frame fractional displacements use the minimum-image convention
    and are converted with the mean of the two neighboring box lengths.  Pure
    affine box rescaling is therefore excluded from the molecular displacement.
    """

    fractions = np.asarray(fractions, dtype=float)
    lengths = np.asarray(box_lengths_nm, dtype=float)
    if fractions.ndim != 3 or fractions.shape[2] != 3:
        raise ValueError("Fractions must have shape (n_frames, n_molecules, 3).")
    if lengths.shape != (fractions.shape[0], 3):
        raise ValueError("Box lengths must have shape (n_frames, 3).")
    if fractions.shape[0] < 2 or fractions.shape[1] == 0:
        raise ValueError("At least two frames and one molecule are required.")
    if not np.all(np.isfinite(fractions)):
        raise ValueError("Fractions contain non-finite values.")
    if not np.all(np.isfinite(lengths)) or np.any(lengths <= 0.0):
        raise ValueError("Box lengths must be positive and finite.")

    wrapped = np.mod(fractions, 1.0)
    fractional_steps = np.mod(np.diff(wrapped, axis=0) + 0.5, 1.0) - 0.5
    step_lengths = 0.5 * (lengths[:-1] + lengths[1:])
    cartesian_steps = fractional_steps * step_lengths[:, np.newaxis, :]
    unwrapped = np.empty_like(wrapped, dtype=float)
    unwrapped[0] = wrapped[0] * lengths[0, np.newaxis, :]
    unwrapped[1:] = unwrapped[0] + np.cumsum(cartesian_steps, axis=0)
    return unwrapped


def consecutive_phase_lengths(states, phase):
    """Return consecutive run lengths in ``phase`` beginning at every frame."""

    states = np.asarray(states)
    if states.ndim != 2 or states.shape[0] == 0 or states.shape[1] == 0:
        raise ValueError("States must have shape (n_frames, n_molecules).")
    if int(phase) != phase:
        raise ValueError("Phase code must be an integer.")

    run_lengths = np.zeros(states.shape, dtype=np.int64)
    in_phase = states == int(phase)
    run_lengths[-1] = in_phase[-1]
    for frame_index in range(states.shape[0] - 2, -1, -1):
        run_lengths[frame_index] = in_phase[frame_index] * (
            run_lengths[frame_index + 1] + 1
        )
    return run_lengths


def _weighted_block_sem(
    values,
    weights,
    blocks,
    origin_indices=None,
    total_origins=None,
):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if values.ndim != 1 or weights.shape != values.shape or values.size == 0:
        return float("nan"), 0
    if origin_indices is None:
        origin_indices = np.arange(values.size)
        total_origins = values.size
    else:
        origin_indices = np.asarray(origin_indices, dtype=int)
        if origin_indices.shape != values.shape:
            raise ValueError("Origin indices must match block-statistic values.")
        if total_origins is None or total_origins <= 0:
            raise ValueError("Total origins must be positive when indices are given.")
    blocks_used = min(int(blocks), int(total_origins))
    block_means = []
    for origin_block in np.array_split(np.arange(total_origins), blocks_used):
        selected = np.isin(origin_indices, origin_block)
        block_weight = float(weights[selected].sum())
        if block_weight > 0.0:
            block_means.append(
                float(np.sum(values[selected] * weights[selected]) / block_weight)
            )
    if len(block_means) < 2:
        return float("nan"), len(block_means)
    block_means = np.asarray(block_means, dtype=float)
    return (
        float(block_means.std(ddof=1) / np.sqrt(block_means.size)),
        int(block_means.size),
    )


def phase_conditioned_msd(
    positions_nm,
    states,
    molecule_weights,
    phase,
    dimensions=(0, 1, 2),
    conditioning="continuous",
    max_lag_frames=None,
    min_particles=2,
    blocks=5,
):
    """Calculate raw and phase-translation-corrected molecular MSD.

    ``continuous`` requires a molecule to remain in the requested phase for the
    entire interval. ``intermittent`` requires phase membership only at both
    endpoints.  At each time origin and lag, the mass-weighted mean displacement
    of the eligible cohort is subtracted before calculating the intrinsic MSD.
    Consequently ``raw_msd = intrinsic_msd + translation_msd`` up to rounding.
    """

    positions = np.asarray(positions_nm, dtype=float)
    states = np.asarray(states)
    weights = np.asarray(molecule_weights, dtype=float)
    if positions.ndim != 3 or positions.shape[2] != 3:
        raise ValueError("Positions must have shape (n_frames, n_molecules, 3).")
    if states.shape != positions.shape[:2]:
        raise ValueError("States must match the frame and molecule dimensions.")
    if weights.shape != (positions.shape[1],):
        raise ValueError("Molecule weights must contain one value per molecule.")
    if positions.shape[0] < 2 or positions.shape[1] == 0:
        raise ValueError("At least two frames and one molecule are required.")
    if not np.all(np.isfinite(positions)):
        raise ValueError("Positions contain non-finite values.")
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError("Molecule weights must be positive and finite.")
    if conditioning not in {"continuous", "intermittent"}:
        raise ValueError("Conditioning must be 'continuous' or 'intermittent'.")
    if int(min_particles) != min_particles or min_particles <= 0:
        raise ValueError("Minimum particles must be a positive integer.")
    if int(blocks) != blocks or blocks <= 0:
        raise ValueError("Number of blocks must be a positive integer.")

    dimensions = tuple(int(dimension) for dimension in dimensions)
    if not dimensions or len(set(dimensions)) != len(dimensions):
        raise ValueError("At least one unique MSD dimension is required.")
    if any(dimension not in (0, 1, 2) for dimension in dimensions):
        raise ValueError("MSD dimensions must be chosen from 0, 1, and 2.")

    frame_count = positions.shape[0]
    if max_lag_frames is None:
        max_lag_frames = frame_count // 2
    if (
        int(max_lag_frames) != max_lag_frames
        or max_lag_frames < 0
        or max_lag_frames >= frame_count
    ):
        raise ValueError("Maximum lag must be an integer from 0 to n_frames - 1.")

    run_lengths = (
        consecutive_phase_lengths(states, phase)
        if conditioning == "continuous"
        else None
    )
    selected_positions = positions[:, :, dimensions]
    rows = []
    for lag_frames in range(int(max_lag_frames) + 1):
        origin_count = frame_count - lag_frames
        if lag_frames == 0:
            displacements = np.zeros_like(selected_positions)
        else:
            displacements = (
                selected_positions[lag_frames:] - selected_positions[:-lag_frames]
            )

        if conditioning == "continuous":
            eligible = run_lengths[:origin_count] > lag_frames
        else:
            eligible = (states[:origin_count] == phase) & (
                states[lag_frames:] == phase
            )

        particles_per_origin = np.count_nonzero(eligible, axis=1)
        valid_origins = particles_per_origin >= int(min_particles)
        weighted_eligible = eligible * weights[np.newaxis, :]
        origin_weights = weighted_eligible.sum(axis=1)
        valid_origins &= origin_weights > 0.0

        if not np.any(valid_origins):
            rows.append(
                {
                    "lag_frames": lag_frames,
                    "raw_msd_nm2": float("nan"),
                    "intrinsic_msd_nm2": float("nan"),
                    "translation_msd_nm2": float("nan"),
                    "raw_block_sem_nm2": float("nan"),
                    "intrinsic_block_sem_nm2": float("nan"),
                    "translation_block_sem_nm2": float("nan"),
                    "contributing_pairs": 0,
                    "time_origins": 0,
                    "mean_particles_per_origin": float("nan"),
                    "blocks_used": 0,
                }
            )
            continue

        valid_origin_indices = np.flatnonzero(valid_origins)
        displacements = displacements[valid_origins]
        eligible = eligible[valid_origins]
        weighted_eligible = weighted_eligible[valid_origins]
        origin_weights = origin_weights[valid_origins]
        particles_per_origin = particles_per_origin[valid_origins]

        mean_displacements = np.einsum(
            "omd,om->od", displacements, weighted_eligible
        ) / origin_weights[:, np.newaxis]
        raw_squared = np.sum(displacements * displacements, axis=2)
        corrected = displacements - mean_displacements[:, np.newaxis, :]
        intrinsic_squared = np.sum(corrected * corrected, axis=2)
        raw_origin = np.sum(raw_squared * weighted_eligible, axis=1) / origin_weights
        intrinsic_origin = (
            np.sum(intrinsic_squared * weighted_eligible, axis=1) / origin_weights
        )
        translation_origin = np.sum(mean_displacements * mean_displacements, axis=1)

        total_weight = float(origin_weights.sum())
        raw_msd = float(np.sum(raw_origin * origin_weights) / total_weight)
        intrinsic_msd = float(
            np.sum(intrinsic_origin * origin_weights) / total_weight
        )
        translation_msd = float(
            np.sum(translation_origin * origin_weights) / total_weight
        )
        raw_sem, raw_blocks = _weighted_block_sem(
            raw_origin,
            origin_weights,
            blocks,
            valid_origin_indices,
            origin_count,
        )
        intrinsic_sem, intrinsic_blocks = _weighted_block_sem(
            intrinsic_origin,
            origin_weights,
            blocks,
            valid_origin_indices,
            origin_count,
        )
        translation_sem, translation_blocks = _weighted_block_sem(
            translation_origin,
            origin_weights,
            blocks,
            valid_origin_indices,
            origin_count,
        )
        rows.append(
            {
                "lag_frames": lag_frames,
                "raw_msd_nm2": raw_msd,
                "intrinsic_msd_nm2": intrinsic_msd,
                "translation_msd_nm2": translation_msd,
                "raw_block_sem_nm2": raw_sem,
                "intrinsic_block_sem_nm2": intrinsic_sem,
                "translation_block_sem_nm2": translation_sem,
                "contributing_pairs": int(np.sum(eligible)),
                "time_origins": int(origin_weights.size),
                "mean_particles_per_origin": float(particles_per_origin.mean()),
                "blocks_used": min(raw_blocks, intrinsic_blocks, translation_blocks),
            }
        )
    return rows
