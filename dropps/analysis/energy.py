"""Inspect a DROPPS EDR and export selected observables to XVG."""

from __future__ import annotations

import os
import shlex
import sys

import numpy as np

from dropps.fileio.filename_control import validate_extension
from dropps.share.argument_parser import ArgumentParser
from dropps.share.command_class import single_command
from dropps.share.energy_reporter import read_edr


prog = "energy"
desc = "Interactively select observables from a DROPPS EDR and export XVG."


def _print_terms(header, stream=sys.stdout):
    print("Available energy terms:\n", file=stream)
    for index, name in enumerate(header.fields, start=1):
        unit_name = header.units.get(name, "unknown")
        print(f"{index:4d}  {name:<24} [{unit_name}]", file=stream)


def _match_term(token, fields):
    lowered = token.lower()
    exact = [name for name in fields if name.lower() == lowered]
    if exact:
        return exact[0]
    prefixes = [name for name in fields if name.lower().startswith(lowered)]
    if len(prefixes) == 1:
        return prefixes[0]
    if len(prefixes) > 1:
        raise ValueError(f"Selection '{token}' is ambiguous: {', '.join(prefixes)}.")
    raise ValueError(f"Unknown energy term '{token}'.")


def resolve_terms(tokens, fields):
    """Resolve names, unique prefixes, 1-based indices, and ``all``."""

    selected = []
    for token in tokens:
        token = str(token).strip()
        if not token:
            continue
        if token.lower() == "all":
            matches = list(fields)
        elif token.isdigit():
            index = int(token)
            if not 1 <= index <= len(fields):
                raise ValueError(
                    f"Energy term number {index} is outside 1-{len(fields)}."
                )
            matches = [fields[index - 1]]
        else:
            matches = [_match_term(token, fields)]
        for match in matches:
            if match not in selected:
                selected.append(match)
    return selected


def interactive_select(header, input_func=input, stream=sys.stdout):
    """Collect a gmx-energy-style multi-line term selection."""

    _print_terms(header, stream=stream)
    print(
        "\nSelect terms by number or name. Use 'all' for every term; "
        "finish with an empty line or 0.",
        file=stream,
    )
    selected = []
    while True:
        try:
            line = input_func("> ")
        except EOFError:
            break
        if not line.strip():
            break
        try:
            tokens = shlex.split(line)
        except ValueError as exc:
            print(f"ERROR: {exc}", file=stream)
            continue
        stop = False
        if "0" in tokens:
            tokens = tokens[: tokens.index("0")]
            stop = True
        try:
            additions = resolve_terms(tokens, header.fields)
        except ValueError as exc:
            print(f"ERROR: {exc}", file=stream)
            continue
        for name in additions:
            if name not in selected:
                selected.append(name)
        if selected:
            print(f"Selected: {', '.join(selected)}", file=stream)
        if stop:
            break
    return selected


def _select_rows(data, start_time, end_time, delta_time):
    if start_time is not None and end_time is not None and end_time < start_time:
        raise ValueError("End time must not precede start time.")
    selected = np.ones(len(data.steps), dtype=bool)
    if start_time is not None:
        selected &= data.times_ps >= float(start_time)
    if end_time is not None:
        selected &= data.times_ps <= float(end_time)
    indices = np.flatnonzero(selected)
    if delta_time is not None:
        if not np.isfinite(delta_time) or delta_time <= 0.0:
            raise ValueError("Output interval must be positive and finite.")
        thinned = []
        next_time = None
        tolerance = max(abs(float(delta_time)), 1.0) * 1.0e-10
        for index in indices:
            time_ps = data.times_ps[index]
            if next_time is None or time_ps >= next_time - tolerance:
                thinned.append(index)
                next_time = time_ps + float(delta_time)
        indices = np.asarray(thinned, dtype=int)
    if len(indices) == 0:
        raise ValueError("No EDR samples fall within the requested time window.")
    return indices


def _write_xvg(path, source, data, indices, terms, x_axis):
    if x_axis == "time":
        x_values = data.times_ps[indices]
        x_name = "time_ps"
        x_label = "Time (ps)"
    else:
        x_values = data.steps[indices]
        x_name = "step"
        x_label = "Step"

    selected_units = [data.header.units.get(name, "unknown") for name in terms]
    known_units = {unit_name for unit_name in selected_units if unit_name != "unknown"}
    y_label = (
        next(iter(known_units)) if len(known_units) == 1 else "Value (see legends)"
    )
    with open(path, "w", encoding="utf-8") as stream:
        stream.write("# DROPPS energy terms extracted from CSV-EDR\n")
        stream.write(f"# source={os.path.abspath(source)}\n")
        stream.write(f"# columns: {x_name} {' '.join(terms)}\n")
        stream.write(
            "# units: "
            + " ".join(
                f"{name}={data.header.units.get(name, 'unknown')}" for name in terms
            )
            + "\n"
        )
        stream.write('@ title "DROPPS energy terms"\n')
        stream.write(f'@ xaxis label "{x_label}"\n')
        stream.write(f'@ yaxis label "{y_label}"\n')
        stream.write("@TYPE xy\n")
        for series, (name, unit_name) in enumerate(zip(terms, selected_units)):
            stream.write(f'@ s{series} legend "{name} ({unit_name})"\n')
        arrays = [data.values[name][indices] for name in terms]
        for row_index, x_value in enumerate(x_values):
            fields = [f"{float(x_value):.12g}"]
            fields.extend(f"{float(values[row_index]):.12g}" for values in arrays)
            stream.write("\t".join(fields) + "\n")


def energy(args):
    data = read_edr(args.input)
    if args.list:
        _print_terms(data.header)
        return
    if args.output is None:
        raise ValueError("--output is required unless --list is used.")
    if not data.header.fields:
        raise ValueError("The EDR contains no selectable energy terms.")

    if args.terms:
        terms = resolve_terms(args.terms, data.header.fields)
    else:
        terms = interactive_select(data.header)
    if not terms:
        print("## No energy terms selected; no XVG file was written.")
        return

    indices = _select_rows(data, args.start_time, args.end_time, args.delta_time)
    output = validate_extension(args.output, "xvg")
    _write_xvg(output, args.input, data, indices, terms, args.x_axis)
    print(
        f"## Wrote {len(indices)} samples and {len(terms)} energy terms to "
        f"{os.path.abspath(output)}."
    )


def getargs_energy(argv):
    parser = ArgumentParser(prog=prog, description=desc)
    parser.add_argument(
        "-f",
        "--input",
        required=True,
        help="Input DROPPS CSV-format energy file (.edr).",
    )
    parser.add_argument(
        "-o", "--output", help="Output selected energy time series (.xvg)."
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available terms and exit without writing XVG.",
    )
    parser.add_argument(
        "--terms",
        nargs="+",
        help=(
            "Terms to export by name, unique prefix, or 1-based number. If omitted, "
            "select terms interactively."
        ),
    )
    parser.add_argument(
        "-b", "--start-time", type=float, help="First sample time to export, in ps."
    )
    parser.add_argument(
        "-e", "--end-time", type=float, help="Last sample time to export, in ps."
    )
    parser.add_argument(
        "-dt", "--delta-time", type=float, help="Approximate output interval, in ps."
    )
    parser.add_argument(
        "--x-axis",
        choices=("time", "step"),
        default="time",
        help="XVG x-axis quantity.",
    )
    return parser.parse_args(argv)


energy_commands = single_command(prog, getargs_energy, energy, desc)
