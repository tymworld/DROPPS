# gsd2xtc tool in DROPPS package by Yiming Tang @ Fudan
# Development started on Nov 17 2025

from dropps.share.argument_parser import ArgumentParser
from dropps.share.command_class import single_command

import MDAnalysis as mda
from MDAnalysis.coordinates import XTC
import numpy as np
from pathlib import Path

from dropps.fileio.filename_control import validate_extension

from tqdm import tqdm

prog = "gsd2xtc"
desc = "Convert a GSD trajectory to XTC format."


def _has_valid_box(dimensions):
    if dimensions is None:
        return False
    dimensions = np.asarray(dimensions, dtype=float)
    return (
        dimensions.size >= 6
        and np.all(np.isfinite(dimensions[:6]))
        and np.all(dimensions[:3] > 0.0)
    )


def _frame_time_ps(frame_index, time_step_ns):
    if not np.isfinite(time_step_ns) or time_step_ns <= 0.0:
        raise ValueError("time step must be a positive finite number")
    if frame_index < 0:
        raise ValueError("frame index must be non-negative")
    return frame_index * time_step_ns * 1000.0


def getargs_gsd2xtc(argv):
    # Command line argument parser

    parser = ArgumentParser(prog=prog, description=desc)

    parser.add_argument(
        "-f", "--input", type=str, required=True, help="Input trajectory file (.gsd)."
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Output trajectory file (.xtc); the extension is added if omitted.",
    )

    parser.add_argument(
        "-ts",
        "--time-step",
        type=float,
        default=0.1,
        help="Time between consecutive GSD frames, in ns.",
    )

    parser.add_argument(
        "-box",
        "--box",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="Box lengths in nm; required when the GSD file has no periodic box.",
    )

    args = parser.parse_args(argv)

    return args


def gsd2xtc(args):
    if not np.isfinite(args.time_step) or args.time_step <= 0.0:
        raise ValueError("time step must be a positive finite number")

    output_filename = validate_extension(args.output, "xtc")
    if Path(args.input).resolve() == Path(output_filename).resolve():
        raise ValueError("input and output trajectory paths must be different")

    u = mda.Universe(args.input)

    box_override = None
    if args.box is not None:
        if any(length <= 0 for length in args.box):
            raise ValueError("All box lengths must be positive.")
        box_override = np.array([*args.box, 90.0, 90.0, 90.0])
    elif not _has_valid_box(u.dimensions):
        raise ValueError(
            "The GSD trajectory has no periodic box. Supply box lengths with --box X Y Z."
        )

    factor = 10.0

    print(
        f"## Will treat {len(u.trajectory)} frames in {args.input} with {u.trajectory.n_atoms} atoms."
    )

    with XTC.XTCWriter(output_filename, u.trajectory.n_atoms) as xtc_writer:
        for i, ts in tqdm(enumerate(u.trajectory)):
            # Multiply the coordinates by the factor
            ts.positions *= factor

            if box_override is not None:
                ts.dimensions = box_override
            elif not _has_valid_box(ts.dimensions):
                raise ValueError(
                    f"GSD frame {i} has no periodic box. Supply box lengths with --box X Y Z."
                )

            # Multiply the box vectors by the factor
            ts.dimensions = [
                ts.dimensions[0] * 10.0,
                ts.dimensions[1] * 10.0,
                ts.dimensions[2] * 10.0,
                ts.dimensions[3],
                ts.dimensions[4],
                ts.dimensions[5],
            ]

            # Set the time for the frame
            ts.time = _frame_time_ps(i, args.time_step)

            # Write the modified frame to the XTC file
            xtc_writer.write(u)
        print(f"## XTC file written to {output_filename}")


gsd2xtc_commands = single_command("gsd2xtc", getargs_gsd2xtc, gsd2xtc, desc)
