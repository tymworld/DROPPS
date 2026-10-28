"""Concatenate and time-filter DROPPS CSV-EDR files."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dropps.fileio.filename_control import validate_extension
from dropps.share.argument_parser import ArgumentParser
from dropps.share.command_class import single_command
from dropps.share.energy_reporter import EDRWriter, EnergyField, read_edr
from dropps.share.time_selection import TIME_UNIT_TO_PS, select_time_indices


prog = "eneconv"
desc = "Concatenate and time-filter DROPPS CSV-format energy files."


_SCIENTIFIC_METADATA_KEYS = (
    "contains_velocities",
    "kinetic_method",
    "pressure_coupling",
    "pressure_kind",
    "pressure_method",
    "pressure_strain",
    "production_temperature_K",
)


@dataclass(frozen=True)
class _EnergyRecord:
    source_index: int
    row_index: int
    step: int
    time_ps: float


def _check_compatible(reference, candidate, path, *, allow_incompatible=False):
    if candidate.header.fields != reference.header.fields:
        raise ValueError(
            f"EDR field schema differs in {path}: "
            f"{candidate.header.fields} != {reference.header.fields}."
        )
    reference_units = {
        field: reference.header.units.get(field, "unknown")
        for field in reference.header.fields
    }
    candidate_units = {
        field: candidate.header.units.get(field, "unknown")
        for field in candidate.header.fields
    }
    if candidate_units != reference_units:
        raise ValueError(f"EDR field units differ in {path}.")

    mismatches = []
    for key in _SCIENTIFIC_METADATA_KEYS:
        first = reference.header.metadata.get(key)
        second = candidate.header.metadata.get(key)
        if first is not None and second is not None and first != second:
            mismatches.append(f"{key}: {first!r} != {second!r}")
    if mismatches and not allow_incompatible:
        raise ValueError(
            f"EDR scientific metadata differs in {path}: "
            + "; ".join(mismatches)
            + ". Use --allow-incompatible only when this combination is intentional."
        )


def _merge_records(datasets):
    """Merge records by step, allowing a later file to replace a duplicate."""

    by_step = {}
    for source_index, data in enumerate(datasets):
        for row_index, (step, time_ps) in enumerate(zip(data.steps, data.times_ps)):
            record = _EnergyRecord(
                source_index=source_index,
                row_index=row_index,
                step=int(step),
                time_ps=float(time_ps),
            )
            previous = by_step.get(record.step)
            if previous is not None and previous.time_ps != record.time_ps:
                raise ValueError(
                    f"EDR step {record.step} occurs at conflicting times "
                    f"{previous.time_ps:g} and {record.time_ps:g} ps."
                )
            by_step[record.step] = record

    records = sorted(by_step.values(), key=lambda item: (item.time_ps, item.step))
    times = np.asarray([record.time_ps for record in records], dtype=float)
    steps = np.asarray([record.step for record in records], dtype=np.int64)
    if times.size > 1 and np.any(np.diff(times) <= 0.0):
        raise ValueError(
            "Merged EDR contains one time associated with more than one step."
        )
    if steps.size > 1 and np.any(np.diff(steps) <= 0):
        raise ValueError("Merged EDR steps are not strictly increasing with time.")
    return records


def _select_records(records, args):
    if not records:
        raise ValueError("The input EDR files contain no samples.")
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


def _output_metadata(reference, inputs):
    metadata = {
        key: value for key, value in reference.header.metadata.items() if key != "units"
    }
    metadata["eneconv_sources"] = [os.path.abspath(path) for path in inputs]
    return metadata


def _write_records(output, inputs, datasets, records):
    reference = datasets[0]
    fields = reference.header.fields
    specifications = {
        field: EnergyField(
            field,
            reference.header.units.get(field, "unknown"),
            "Concatenated DROPPS EDR observable",
        )
        for field in fields
    }

    output_path = os.path.abspath(os.fspath(output))
    directory = os.path.dirname(output_path)
    descriptor, temporary_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(output_path)}.", suffix=".tmp", dir=directory
    )
    os.close(descriptor)
    writer = None
    try:
        writer = EDRWriter(
            temporary_path,
            fields,
            specifications,
            metadata=_output_metadata(reference, inputs),
        )
        for record in records:
            data = datasets[record.source_index]
            values = {
                field: float(data.values[field][record.row_index]) for field in fields
            }
            writer.write(record.step, record.time_ps, values)
        writer.close()
        writer = None
        os.replace(temporary_path, output_path)
    except BaseException:
        if writer is not None:
            writer.close()
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)
        raise


def eneconv(args):
    inputs = [os.fspath(path) for path in args.input]
    output = validate_extension(args.output, "edr")
    output_resolved = Path(output).resolve()
    if any(Path(path).resolve() == output_resolved for path in inputs):
        raise ValueError("Output EDR must not also be an input EDR.")
    for path in inputs:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Input EDR not found: {path}")

    datasets = [read_edr(path) for path in inputs]
    reference = datasets[0]
    for path, data in zip(inputs[1:], datasets[1:]):
        _check_compatible(
            reference,
            data,
            path,
            allow_incompatible=args.allow_incompatible,
        )

    records = _merge_records(datasets)
    records = _select_records(records, args)
    _write_records(output, inputs, datasets, records)
    print(
        f"## Wrote {len(records)} samples from {len(inputs)} EDR files to "
        f"{os.path.abspath(output)}."
    )


def getargs_eneconv(argv):
    parser = ArgumentParser(prog=prog, description=desc)
    parser.add_argument(
        "-f",
        "--input",
        nargs="+",
        required=True,
        help="Input DROPPS CSV-format energy files (.edr).",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output concatenated DROPPS energy file (.edr).",
    )
    parser.add_argument(
        "-b", "--start-time", type=float, help="First energy-sample time to write."
    )
    parser.add_argument(
        "-e", "--end-time", type=float, help="Last energy-sample time to write."
    )
    parser.add_argument(
        "-dt",
        "--delta-time",
        type=float,
        help="Approximate physical interval between output samples.",
    )
    parser.add_argument(
        "-tu",
        "--time-unit",
        choices=tuple(TIME_UNIT_TO_PS),
        default="ps",
        help="Unit used by -b, -e, and -dt.",
    )
    parser.add_argument(
        "--allow-incompatible",
        action="store_true",
        default=False,
        help=(
            "Allow differing scientific metadata while still requiring identical "
            "fields and units."
        ),
    )
    return parser.parse_args(argv)


eneconv_commands = single_command(prog, getargs_eneconv, eneconv, desc)
