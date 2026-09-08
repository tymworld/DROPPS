"""Shared validation for analyses that require uniformly sampled frames."""

from __future__ import annotations

import numpy as np


def uniform_frame_interval_ns(frame_times_ns):
    """Return the sampling interval, rejecting irregular or invalid timestamps."""
    times = np.asarray(frame_times_ns, dtype=float)
    if times.ndim != 1 or times.size < 2:
        raise ValueError("at least two trajectory frames are required")
    if not np.all(np.isfinite(times)):
        raise ValueError("trajectory frame times must be finite")

    intervals = np.diff(times)
    if np.any(intervals <= 0.0):
        raise ValueError("trajectory frame times must be strictly increasing")
    interval = float(np.median(intervals))
    tolerance = max(
        np.finfo(np.float32).eps * max(float(np.max(np.abs(times))), 1.0) * 8.0,
        abs(interval) * 1.0e-6,
    )
    if not np.allclose(intervals, interval, rtol=1.0e-6, atol=tolerance):
        raise ValueError(
            "this analysis requires uniformly sampled frames; "
            "the selected trajectory timestamps are irregular"
        )
    return interval
