"""Shared physical-time trajectory frame selection helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


TIME_UNIT_TO_PS = {
    "fs": 1.0e-3,
    "ps": 1.0,
    "ns": 1.0e3,
    "us": 1.0e6,
    "ms": 1.0e9,
    "s": 1.0e12,
}


@dataclass(frozen=True)
class TimeSelection:
    """Selected input-frame indices and the corresponding times in ps."""

    indices: np.ndarray
    times_ps: np.ndarray
    single_time: bool


def _as_finite_vector(values, name, *, length=None):
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or (length is not None and array.size != length):
        expected = f" with length {length}" if length is not None else ""
        raise ValueError(f"{name} must be a one-dimensional array{expected}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values.")
    return array


def _to_ps(value, time_unit, name):
    if value is None:
        return None
    try:
        factor = TIME_UNIT_TO_PS[time_unit]
    except KeyError as exc:
        choices = ", ".join(TIME_UNIT_TO_PS)
        raise ValueError(
            f"Unknown time unit '{time_unit}'; choose from {choices}."
        ) from exc
    result = float(value) * factor
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def select_time_indices(
    frame_times_ps,
    start_time=None,
    end_time=None,
    delta_time=None,
    time_unit="ns",
):
    """Select frames by physical time.

    An equal explicit start and end time requests exactly one frame: the saved
    frame nearest that time is selected, with an earlier frame winning a tie.
    For a range with ``delta_time``, frames nearest the regularly spaced target
    times are selected. Duplicate nearest frames are removed while preserving
    order.
    """

    times = _as_finite_vector(frame_times_ps, "Trajectory times")
    if times.size == 0:
        raise ValueError("The trajectory contains no frames.")
    if np.any(np.diff(times) < 0.0):
        raise ValueError("Trajectory times must be monotonically non-decreasing.")

    requested_start = _to_ps(start_time, time_unit, "Start time")
    requested_end = _to_ps(end_time, time_unit, "End time")
    requested_delta = _to_ps(delta_time, time_unit, "Delta time")
    if requested_delta is not None and requested_delta <= 0.0:
        raise ValueError("Delta time must be greater than zero.")

    lower = times[0] if requested_start is None else requested_start
    upper = times[-1] if requested_end is None else requested_end
    scale = max(abs(lower), abs(upper), abs(times[0]), abs(times[-1]), 1.0)
    equality_tolerance = np.finfo(np.float64).eps * scale * 16.0
    boundary_tolerance = max(
        equality_tolerance,
        np.finfo(np.float32).eps * scale * 4.0,
    )

    if lower < times[0] - boundary_tolerance or upper > times[-1] + boundary_tolerance:
        raise ValueError(
            f"Requested time range {lower:g}-{upper:g} ps lies outside the "
            f"trajectory range {times[0]:g}-{times[-1]:g} ps."
        )
    if upper < lower - equality_tolerance:
        raise ValueError("End time must not be earlier than start time.")

    single_time = (
        requested_start is not None
        and requested_end is not None
        and abs(requested_start - requested_end) <= equality_tolerance
    )
    if single_time:
        if requested_delta is not None:
            raise ValueError("Delta time cannot be combined with an equal -b and -e.")
        index = int(np.argmin(np.abs(times - requested_start)))
        indices = np.asarray([index], dtype=np.int64)
        return TimeSelection(indices, times[indices].copy(), True)

    in_range = np.flatnonzero(
        (times >= lower - boundary_tolerance) & (times <= upper + boundary_tolerance)
    )
    if in_range.size == 0:
        raise ValueError("No saved trajectory frame falls inside the requested range.")

    if requested_delta is None:
        indices = in_range.astype(np.int64, copy=False)
        return TimeSelection(indices, times[indices].copy(), False)

    anchor = lower if requested_start is not None else float(times[in_range[0]])
    ratio_tolerance = equality_tolerance / requested_delta
    target_count = (
        int(np.floor((upper - anchor) / requested_delta + ratio_tolerance)) + 1
    )
    targets = anchor + requested_delta * np.arange(target_count, dtype=np.float64)
    candidate_times = times[in_range]
    selected = []
    for target in targets:
        nearest = int(in_range[int(np.argmin(np.abs(candidate_times - target)))])
        if not selected or nearest != selected[-1]:
            selected.append(nearest)
    if not selected:
        raise ValueError("The requested time interval selected no trajectory frames.")
    indices = np.asarray(selected, dtype=np.int64)
    return TimeSelection(indices, times[indices].copy(), False)
