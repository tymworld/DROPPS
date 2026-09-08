"""Fixed-schema, restart-friendly progress reporting for ``mdrun``."""

from __future__ import annotations

import os
import time

from openmm import unit


class ProgressReporter:
    """Write step/time/progress/performance fields in a fixed column order."""

    def __init__(
        self,
        file,
        report_interval,
        total_steps,
        append=False,
        separator=",",
    ):
        if int(report_interval) <= 0:
            raise ValueError("Progress report interval must be positive.")
        if int(total_steps) < 0:
            raise ValueError("Total simulation steps must not be negative.")
        self._report_interval = int(report_interval)
        self._total_steps = int(total_steps)
        self._separator = separator
        self._opened_file = isinstance(file, (str, os.PathLike))
        self._append = bool(append)
        if self._opened_file:
            self._stream = open(
                os.fspath(file), "a" if self._append else "w", encoding="utf-8"
            )
        else:
            self._stream = file
        self._start_step = None
        self._start_clock = None
        self._start_time_ps = None
        if not self._append:
            self._stream.write(
                "# step"
                + self._separator
                + self._separator.join(
                    (
                        "time_ps",
                        "progress_percent",
                        "speed_ns_per_day",
                        "elapsed_s",
                        "remaining_s",
                    )
                )
                + "\n"
            )
            self._stream.flush()

    def describeNextReport(self, simulation):
        if self._start_step is None:
            self._start_step = int(simulation.currentStep)
            self._start_clock = time.perf_counter()
        steps = self._report_interval - simulation.currentStep % self._report_interval
        return steps, False, False, False, False

    def report(self, simulation, state):
        current_clock = time.perf_counter()
        elapsed_s = max(0.0, current_clock - self._start_clock)
        time_ps = state.getTime().value_in_unit(unit.picosecond)
        completed_since_start = simulation.currentStep - self._start_step
        if self._start_time_ps is None:
            step_size_ps = simulation.integrator.getStepSize().value_in_unit(
                unit.picosecond
            )
            self._start_time_ps = time_ps - completed_since_start * step_size_ps
        elapsed_ns = max(0.0, (time_ps - self._start_time_ps) / 1000.0)
        speed = elapsed_ns / (elapsed_s / 86400.0) if elapsed_s > 0.0 else 0.0
        progress = (
            100.0 * simulation.currentStep / self._total_steps
            if self._total_steps > 0
            else 100.0
        )
        if completed_since_start > 0 and elapsed_s > 0.0:
            seconds_per_step = elapsed_s / completed_since_start
            remaining_s = max(
                0.0, (self._total_steps - simulation.currentStep) * seconds_per_step
            )
        else:
            remaining_s = 0.0
        values = (
            str(int(simulation.currentStep)),
            f"{time_ps:.12g}",
            f"{progress:.8g}",
            f"{speed:.8g}",
            f"{elapsed_s:.8g}",
            f"{remaining_s:.8g}",
        )
        self._stream.write(self._separator.join(values) + "\n")
        self._stream.flush()

    def close(self):
        if self._opened_file and self._stream is not None:
            self._stream.close()
            self._stream = None

    def __del__(self):
        self.close()
