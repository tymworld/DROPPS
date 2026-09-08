"""NumPy-only helpers for phase-exchange and residence-time analysis."""

from __future__ import annotations

import numpy as np


DENSE = 0
INTERFACE = 1
DILUTE = 2
STATE_NAMES = {DENSE: "dense", INTERFACE: "interface", DILUTE: "dilute"}
_BULK_STATES = {DENSE, DILUTE}


def periodic_weighted_mean_fraction(fractions, weights=None):
    """Return a molecule center on ``[0, 1)`` using minimum-image unwrapping.

    Coordinates are unwrapped relative to the first selected atom.  This is
    robust for molecules crossing a periodic boundary, provided that the
    selected part of one molecule spans less than half the box length.
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
    if float(weights.sum()) <= 0.0:
        raise ValueError("At least one molecular weight must be positive.")

    anchor = float(np.mod(fractions[0], 1.0))
    displacement = np.mod(fractions - anchor + 0.5, 1.0) - 0.5
    return float(np.mod(anchor + np.average(displacement, weights=weights), 1.0))


def phase_cutoffs(
    half_width_nm,
    interface_parameter_nm,
    threshold=0.1,
    cell_length_nm=None,
):
    """Return dense and dilute distance cutoffs around a fitted slab.

    ``threshold`` is the normalized dilute-side density threshold.  For the
    default 0.1, the interface state is the fitted 10--90% transition region.
    """

    values = (half_width_nm, interface_parameter_nm, threshold)
    if not all(np.isfinite(value) for value in values):
        raise ValueError("Fit dimensions and interface threshold must be finite.")
    if half_width_nm <= 0.0 or interface_parameter_nm <= 0.0:
        raise ValueError("Slab half-width and interface parameter must be positive.")
    if not 0.0 < threshold < 0.5:
        raise ValueError("Interface threshold must lie strictly between 0 and 0.5.")

    offset = float(np.arctanh(1.0 - 2.0 * threshold) * interface_parameter_nm)
    dense_cutoff = float(half_width_nm - offset)
    dilute_cutoff = float(half_width_nm + offset)
    if dense_cutoff <= 0.0:
        raise ValueError(
            "The fitted dense plateau is narrower than the requested interface "
            "region. Increase --interface-threshold or inspect the slab fit."
        )
    if cell_length_nm is not None:
        if not np.isfinite(cell_length_nm) or cell_length_nm <= 0.0:
            raise ValueError("Cell length must be positive and finite.")
        if dilute_cutoff >= 0.5 * cell_length_nm:
            raise ValueError(
                "The fitted dilute plateau is narrower than the requested interface "
                "region. Increase --interface-threshold or inspect the slab fit."
            )
    return dense_cutoff, dilute_cutoff, offset


def classify_phase_positions(coordinates_nm, dense_cutoff_nm, dilute_cutoff_nm):
    """Classify signed centered coordinates as dense, interface, or dilute."""

    coordinates = np.asarray(coordinates_nm, dtype=float)
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("Molecular coordinates contain non-finite values.")
    if (
        not np.isfinite(dense_cutoff_nm)
        or not np.isfinite(dilute_cutoff_nm)
        or dense_cutoff_nm < 0.0
        or dilute_cutoff_nm <= dense_cutoff_nm
    ):
        raise ValueError("Phase cutoffs must satisfy 0 <= dense < dilute.")

    absolute_coordinate = np.abs(coordinates)
    states = np.full(coordinates.shape, INTERFACE, dtype=np.int8)
    states[absolute_coordinate <= dense_cutoff_nm] = DENSE
    states[absolute_coordinate >= dilute_cutoff_nm] = DILUTE
    return states


def sampling_edges(times_ns):
    """Return observation-cell edges around strictly increasing sample times."""

    times = np.asarray(times_ns, dtype=float)
    if times.ndim != 1 or times.size < 2:
        raise ValueError("At least two one-dimensional sample times are required.")
    if not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0.0):
        raise ValueError("Sample times must be finite and strictly increasing.")
    edges = np.empty(times.size + 1, dtype=float)
    edges[0] = times[0]
    edges[-1] = times[-1]
    edges[1:-1] = 0.5 * (times[:-1] + times[1:])
    return edges


def _state_matrix(states, frame_count):
    matrix = np.asarray(states)
    if matrix.ndim == 1:
        matrix = matrix[:, np.newaxis]
    if matrix.ndim != 2 or matrix.shape[0] != frame_count:
        raise ValueError("States must have shape (n_frames, n_molecules).")
    if not np.all(np.isin(matrix, tuple(STATE_NAMES))):
        raise ValueError("States contain an unknown phase code.")
    return matrix.astype(np.int8, copy=False)


def extract_phase_episodes(times_ns, states):
    """Extract continuous phase-occupancy episodes and censoring flags."""

    times = np.asarray(times_ns, dtype=float)
    edges = sampling_edges(times)
    state_matrix = _state_matrix(states, times.size)
    episodes = []

    for molecule_index in range(state_matrix.shape[1]):
        molecule_states = state_matrix[:, molecule_index]
        start_frame = 0
        for frame_index in range(1, times.size + 1):
            segment_ends = (
                frame_index == times.size
                or molecule_states[frame_index] != molecule_states[start_frame]
            )
            if not segment_ends:
                continue
            end_frame = frame_index - 1
            start_time = float(edges[start_frame])
            end_time = float(edges[frame_index])
            episodes.append(
                {
                    "molecule_index": molecule_index,
                    "phase": int(molecule_states[start_frame]),
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "start_time_ns": start_time,
                    "end_time_ns": end_time,
                    "duration_ns": end_time - start_time,
                    "left_censored": start_frame == 0,
                    "right_censored": end_frame == times.size - 1,
                }
            )
            start_frame = frame_index
    return episodes


def extract_exchange_events(times_ns, states):
    """Extract confirmed dense-to-dilute and dilute-to-dense passages.

    An interface excursion that returns to its source phase is not an exchange.
    Event waiting time starts at the last confirmed arrival in the source bulk;
    it therefore includes unsuccessful interface excursions.
    """

    times = np.asarray(times_ns, dtype=float)
    edges = sampling_edges(times)
    state_matrix = _state_matrix(states, times.size)
    events = []

    for molecule_index in range(state_matrix.shape[1]):
        current_bulk = None
        current_entry_time = None
        current_entry_left_censored = False
        pending_departure_time = None

        for frame_index, phase_value in enumerate(state_matrix[:, molecule_index]):
            phase = int(phase_value)
            boundary_time = float(edges[frame_index])
            if current_bulk is None:
                if phase in _BULK_STATES:
                    current_bulk = phase
                    current_entry_time = boundary_time
                    current_entry_left_censored = frame_index == 0
                continue

            if phase == INTERFACE:
                if pending_departure_time is None:
                    pending_departure_time = boundary_time
                continue

            if phase == current_bulk:
                pending_departure_time = None
                continue

            departure_time = (
                boundary_time
                if pending_departure_time is None
                else pending_departure_time
            )
            arrival_time = boundary_time
            events.append(
                {
                    "molecule_index": molecule_index,
                    "source_phase": current_bulk,
                    "destination_phase": phase,
                    "departure_time_ns": departure_time,
                    "arrival_time_ns": arrival_time,
                    "transition_time_ns": arrival_time - departure_time,
                    "waiting_time_since_last_exchange_ns": (
                        departure_time - current_entry_time
                    ),
                    "waiting_left_censored": current_entry_left_censored,
                }
            )
            current_bulk = phase
            current_entry_time = arrival_time
            current_entry_left_censored = False
            pending_departure_time = None
    return events


def kaplan_meier(durations, event_observed):
    """Return Kaplan--Meier rows for right-censored residence durations."""

    durations = np.asarray(durations, dtype=float)
    observed = np.asarray(event_observed)
    if durations.ndim != 1 or observed.shape != durations.shape:
        raise ValueError("Durations and event flags must be matching 1D arrays.")
    if durations.size == 0:
        return []
    if not np.all(np.isfinite(durations)) or np.any(durations < 0.0):
        raise ValueError("Durations must be finite and non-negative.")
    if not np.all(np.isin(observed, (False, True, 0, 1))):
        raise ValueError("Event flags must be boolean values.")
    observed = observed.astype(bool)

    rows = [
        {
            "time_ns": 0.0,
            "survival": 1.0,
            "at_risk": int(durations.size),
            "events": 0,
            "censored": 0,
        }
    ]
    survival = 1.0
    for time_value in np.unique(durations):
        at_risk = int(np.count_nonzero(durations >= time_value))
        at_time = durations == time_value
        events = int(np.count_nonzero(at_time & observed))
        censored = int(np.count_nonzero(at_time & ~observed))
        if events:
            survival *= 1.0 - events / at_risk
        rows.append(
            {
                "time_ns": float(time_value),
                "survival": float(survival),
                "at_risk": at_risk,
                "events": events,
                "censored": censored,
            }
        )
    return rows
