"""Concatenate DROPPS XTC trajectory segments."""

from __future__ import annotations

import os
import tempfile
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dropps.fileio.filename_control import validate_extension
from dropps.share.argument_parser import ArgumentParser
from dropps.share.command_class import single_command
from dropps.share.time_selection import TIME_UNIT_TO_PS, select_time_indices


prog = "trjcat"
desc = "Concatenate DROPPS XTC trajectory segments in time order."


@dataclass(frozen=True)
class _FrameRecord:
    source_index: int
    frame_index: int
    source_priority: int
    time_ps: float
    step: int


@dataclass(frozen=True)
class _TrajectorySource:
    path: str
    source_index: int
    first_time_ps: float
    records: tuple[_FrameRecord, ...]
    atom_count: int


def _scan_source(path, source_index):
    from MDAnalysis.lib.formats.libmdaxdr import XTCFile

    records = []
    atom_count = None
    previous_time = None
    with XTCFile(path, "r") as reader:
        for frame_index, frame in enumerate(reader):
            time_ps = float(frame.time)
            if not np.isfinite(time_ps):
                raise ValueError(f"Trajectory {path} contains a non-finite time.")
            if previous_time is not None and time_ps < previous_time:
                raise ValueError(
                    f"Trajectory {path} is not ordered by increasing frame time."
                )
            previous_time = time_ps

            current_atom_count = int(np.asarray(frame.x).shape[0])
            if atom_count is None:
                atom_count = current_atom_count
            elif current_atom_count != atom_count:
                raise ValueError(
                    f"Trajectory {path} changes atom count between frames."
                )
            records.append(
                _FrameRecord(
                    source_index=source_index,
                    frame_index=frame_index,
                    source_priority=0,
                    time_ps=time_ps,
                    step=int(frame.step),
                )
            )

    if not records:
        raise ValueError(f"Input trajectory {path} contains no frames.")
    return _TrajectorySource(
        path=os.fspath(path),
        source_index=source_index,
        first_time_ps=records[0].time_ps,
        records=tuple(records),
        atom_count=int(atom_count),
    )


def _ordered_records(sources, *, sort_sources=True, keep_duplicates=False):
    """Return chronological records with later input segments taking priority."""

    ordered_sources = list(sources)
    if sort_sources:
        ordered_sources.sort(key=lambda item: (item.first_time_ps, item.source_index))

    records = []
    for priority, source in enumerate(ordered_sources):
        records.extend(
            _FrameRecord(
                source_index=record.source_index,
                frame_index=record.frame_index,
                source_priority=priority,
                time_ps=record.time_ps,
                step=record.step,
            )
            for record in source.records
        )

    if sort_sources:
        records.sort(
            key=lambda item: (
                item.time_ps,
                item.source_priority,
                item.frame_index,
            )
        )

    if not keep_duplicates:
        # XTC stores time in single precision. Repeated boundary frames therefore
        # have exactly equal decoded times; retaining the highest source priority
        # implements the usual "later segment wins" continuation rule.
        last_for_time = {}
        for index, record in enumerate(records):
            previous = last_for_time.get(record.time_ps)
            if (
                previous is None
                or record.source_priority >= records[previous].source_priority
            ):
                last_for_time[record.time_ps] = index
        records = [
            record
            for index, record in enumerate(records)
            if last_for_time[record.time_ps] == index
        ]

    times = np.asarray([record.time_ps for record in records], dtype=float)
    if times.size > 1 and np.any(np.diff(times) < 0.0):
        raise ValueError(
            "Concatenated frame times are not monotonic. Remove --no-sort or "
            "provide the input files in chronological order."
        )
    return records


def _select_records(records, args):
    if not records:
        raise ValueError("No trajectory frames are available for concatenation.")
    if args.start_time is None and args.end_time is None and args.delta_time is None:
        return records
    selection = select_time_indices(
        [record.time_ps for record in records],
        start_time=args.start_time,
        end_time=args.end_time,
        delta_time=args.delta_time,
        time_unit=args.time_unit,
    )
    return [records[int(index)] for index in selection.indices]


def _write_records(output, sources, records):
    from MDAnalysis.lib.formats.libmdaxdr import XTCFile

    output_path = os.path.abspath(os.fspath(output))
    directory = os.path.dirname(output_path)
    descriptor, temporary_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(output_path)}.", suffix=".tmp", dir=directory
    )
    os.close(descriptor)

    source_by_index = {source.source_index: source for source in sources}
    try:
        with ExitStack() as stack:
            readers = {}
            for source in sources:
                reader = stack.enter_context(XTCFile(source.path, "r"))
                reader.calc_offsets()
                readers[source.source_index] = reader
            writer = stack.enter_context(XTCFile(temporary_path, "w"))

            for record in records:
                reader = readers[record.source_index]
                reader.seek(record.frame_index)
                frame = reader.read()
                if (
                    float(frame.time) != record.time_ps
                    or int(frame.step) != record.step
                ):
                    source = source_by_index[record.source_index]
                    raise ValueError(
                        f"Trajectory {source.path} changed while it was being read."
                    )
                writer.write(
                    frame.x,
                    frame.box,
                    frame.step,
                    frame.time,
                    frame.prec,
                )
        os.replace(temporary_path, output_path)
    except BaseException:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)
        raise


def trjcat(args):
    inputs = [os.fspath(path) for path in args.input]
    output = validate_extension(args.output, "xtc")
    output_resolved = Path(output).resolve()
    if any(Path(path).resolve() == output_resolved for path in inputs):
        raise ValueError("Output trajectory must not also be an input trajectory.")
    for path in inputs:
        if Path(path).suffix.lower() != ".xtc":
            raise ValueError(f"Input trajectory must be XTC: {path}")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Input trajectory not found: {path}")

    sources = [_scan_source(path, index) for index, path in enumerate(inputs)]
    atom_counts = {source.atom_count for source in sources}
    if len(atom_counts) != 1:
        details = ", ".join(f"{source.path}={source.atom_count}" for source in sources)
        raise ValueError(f"Input trajectories have different atom counts: {details}.")

    records = _ordered_records(
        sources,
        sort_sources=args.sort,
        keep_duplicates=args.keep_duplicates,
    )
    records = _select_records(records, args)
    _write_records(output, sources, records)
    print(
        f"## Wrote {len(records)} frames from {len(sources)} trajectories to "
        f"{os.path.abspath(output)}."
    )


def getargs_trjcat(argv):
    parser = ArgumentParser(prog=prog, description=desc)
    parser.add_argument(
        "-f",
        "--input",
        nargs="+",
        required=True,
        help="Input DROPPS XTC trajectory segments.",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output concatenated XTC trajectory.",
    )
    parser.add_argument(
        "-b", "--start-time", type=float, help="First trajectory time to write."
    )
    parser.add_argument(
        "-e", "--end-time", type=float, help="Last trajectory time to write."
    )
    parser.add_argument(
        "-dt",
        "--delta-time",
        type=float,
        help="Approximate physical interval between output frames.",
    )
    parser.add_argument(
        "-tu",
        "--time-unit",
        choices=tuple(TIME_UNIT_TO_PS),
        default="ps",
        help="Unit used by -b, -e, and -dt.",
    )
    sort_group = parser.add_mutually_exclusive_group(section="parameters")
    sort_group.add_argument(
        "--sort",
        dest="sort",
        action="store_true",
        default=True,
        help="Sort input segments by their first frame time before merging.",
    )
    sort_group.add_argument(
        "--no-sort",
        dest="sort",
        action="store_false",
        help=(
            "Preserve command-line input order; frame times must remain monotonic. "
            "The default is to sort."
        ),
    )
    parser.add_argument(
        "-cat",
        "--keep-duplicates",
        action="store_true",
        default=False,
        help="Keep frames with duplicate times instead of using the later segment.",
    )
    return parser.parse_args(argv)


trjcat_commands = single_command(prog, getargs_trjcat, trjcat, desc)
