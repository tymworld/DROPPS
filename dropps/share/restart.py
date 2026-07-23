"""Helpers for producing restart-safe simulation output."""

from __future__ import annotations

import os
import tempfile

from openmm.app import XTCFile
from openmm.app.internal.xtc_utils import get_xtc_nframes


class RestartableXTCReporter:
    """Write an XTC trajectory whose step numbers remain valid after appending."""

    def __init__(
        self,
        file,
        report_interval,
        append=False,
        enforce_periodic_box=None,
    ):
        if int(report_interval) <= 0:
            raise ValueError("XTC report interval must be positive.")
        self._file_name = os.fspath(file)
        self._report_interval = int(report_interval)
        self._append = bool(append)
        self._enforce_periodic_box = enforce_periodic_box
        self._xtc = None

    def describeNextReport(self, simulation):
        steps = self._report_interval - simulation.currentStep % self._report_interval
        return steps, True, False, False, False, self._enforce_periodic_box

    def report(self, simulation, state):
        if self._xtc is None:
            first_step = simulation.currentStep
            if self._append:
                frame_count = get_xtc_nframes(self._file_name.encode("utf-8"))
                first_step -= frame_count * self._report_interval
                if first_step < 0:
                    raise ValueError(
                        "Existing XTC trajectory is inconsistent with the restart step."
                    )
            self._xtc = XTCFile(
                self._file_name,
                simulation.topology,
                simulation.integrator.getStepSize(),
                first_step,
                self._report_interval,
                self._append,
            )
        self._xtc.writeModel(
            state.getPositions(),
            periodicBoxVectors=state.getPeriodicBoxVectors(),
        )


def truncate_delimited_report(file, checkpoint_step, separator=None):
    """Remove data rows newer than a checkpoint from a text reporter file."""

    path = os.fspath(file)
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return False

    with open(path, "r", encoding="utf-8") as stream:
        lines = stream.readlines()

    kept_lines = []
    changed = False
    for line in lines:
        stripped = line.lstrip()
        if not stripped or stripped.startswith(("#", "@")):
            kept_lines.append(line)
            continue
        if separator is not None and separator not in stripped:
            kept_lines.append(line)
            continue
        token = stripped.split(separator, 1)[0] if separator else stripped.split()[0]
        try:
            step = int(token)
        except ValueError:
            kept_lines.append(line)
            continue
        if step <= checkpoint_step:
            kept_lines.append(line)
        else:
            changed = True

    if changed:
        _atomic_write_text(path, kept_lines)
    return changed


def truncate_xtc(file, checkpoint_step):
    """Atomically discard XTC frames whose stored step is after a checkpoint."""

    path = os.fspath(file)
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return False

    # Import lazily: this path is used only while resuming an existing run.
    from MDAnalysis.lib.formats.libmdaxdr import XTCFile as XTCReaderWriter

    with XTCReaderWriter(path, "r") as source:
        offsets = source.calc_offsets()
        if len(offsets) == 0:
            return False
        source.seek(len(offsets) - 1)
        last_frame = source.read()
        if last_frame.step <= checkpoint_step:
            return False

    directory = os.path.dirname(os.path.abspath(path))
    descriptor, temporary_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(path)}.", suffix=".restart", dir=directory
    )
    os.close(descriptor)
    frames_written = 0
    try:
        with XTCReaderWriter(path, "r") as source, XTCReaderWriter(
            temporary_path, "w"
        ) as target:
            for frame in source:
                if frame.step > checkpoint_step:
                    break
                target.write(
                    frame.x,
                    frame.box,
                    frame.step,
                    frame.time,
                    frame.prec,
                )
                frames_written += 1
        if frames_written == 0:
            os.unlink(temporary_path)
            os.unlink(path)
        else:
            os.replace(temporary_path, path)
    except BaseException:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)
        raise
    return True


def save_checkpoint_safely(simulation, file):
    """Write a checkpoint atomically without destroying the previous one."""

    path = os.fspath(file)
    directory = os.path.dirname(os.path.abspath(path))
    descriptor, temporary_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(path)}.", suffix=".tmp", dir=directory
    )
    os.close(descriptor)
    try:
        simulation.saveCheckpoint(temporary_path)
        os.replace(temporary_path, path)
    except BaseException:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)
        raise


def _atomic_write_text(path, lines):
    directory = os.path.dirname(os.path.abspath(path))
    descriptor, temporary_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(path)}.", suffix=".tmp", dir=directory
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.writelines(lines)
        os.replace(temporary_path, path)
    except BaseException:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)
        raise
