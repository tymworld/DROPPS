# Mean square deviation (MSD) tool in DROPPS package by Yiming Tang @ Fudan
# Development started on September 16 2025

from dropps.share.argument_parser import ArgumentParser
from dropps.share.command_class import single_command
from dropps.share.trajectory import trajectory_class
from dropps.fileio.filename_control import validate_extension
from dropps.fileio.xvg_reader import write_xvg
from dropps.analysis.time_core import uniform_frame_interval_ns

import warnings
from Bio import BiopythonDeprecationWarning

warnings.filterwarnings("ignore", category=BiopythonDeprecationWarning)

from MDAnalysis.transformations import nojump  # noqa: E402 - suppress import warning above

import numpy as np  # noqa: E402 - paired with guarded MDAnalysis import

import MDAnalysis.analysis.msd  # noqa: E402 - suppress import warning above

prog = "msd"
desc = "Calculate the mean-square displacement of a selected atom group."


def getargs_msd(argv):
    parser = ArgumentParser(prog=prog, description=desc)

    parser.add_argument(
        "-s",
        "--run-input",
        type=str,
        required=True,
        help="Input DROPPS run file (.tpr) containing the system and simulation settings.",
    )

    parser.add_argument(
        "-f", "--input", type=str, required=True, help="Input trajectory file (.xtc)."
    )

    parser.add_argument(
        "-n",
        "--index",
        type=str,
        required=False,
        help="Optional index file (.ndx) defining additional atom groups.",
    )

    parser.add_argument(
        "-sel",
        "--selection",
        type=int,
        help="Index group whose mean-square displacement is calculated; if omitted, prompt interactively.",
    )

    parser.add_argument(
        "-t",
        "--msd-type",
        choices=["xyz", "xy", "yz", "xz", "x", "y", "z"],
        default="xyz",
        type=str,
        help="Cartesian dimensions included in the MSD.",
    )

    parser.add_argument(
        "-b",
        "--start-time",
        type=float,
        help="First trajectory time to analyze, in ns.",
    )

    parser.add_argument(
        "-e", "--end-time", type=float, help="Last trajectory time to analyze, in ns."
    )

    parser.add_argument(
        "-dt",
        "--delta-time",
        type=float,
        help="Approximate interval between analyzed frames, in ns.",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Output mean-square-displacement time series (.xvg); the extension is added if omitted.",
    )

    args = parser.parse_args(argv)

    return args


def msd(args):
    if not args.output:
        print("ERROR: No output file specified.")
        quit()

    # load trajectory into memory

    try:
        trajectory = trajectory_class(args.run_input, args.index, args.input)
    except Exception as exc:
        print(
            "## An exception occurred when trying to open trajectory file %s."
            % args.input
        )
        print(f"## Root cause: {exc}")
        quit()

    # We treat time for analysis and generate frame for analysis'

    start_frame, end_frame, interval_frame = trajectory.time2frame(
        args.start_time, args.end_time, args.delta_time
    )
    frame_indices = list(range(start_frame, end_frame + 1, interval_frame))
    frame_times_ns = [
        float(trajectory.Universe.trajectory[index].time) / 1000.0
        for index in frame_indices
    ]
    frame_dt_ns = uniform_frame_interval_ns(frame_times_ns)

    if args.selection is not None:
        print(f"## Will use group {args.selection} for Rg calculations.")
        selection, selection_name = trajectory.getSelection(f"group {args.selection}")
    else:
        trajectory.index.print_all()
        selection, selection_name = trajectory.getSelection_interactive()

    print(
        f"## Will calculate MSD for group {selection_name} at {args.msd_type} dimensions."
    )

    # ``frame_times_ns`` above leaves file-backed readers positioned at the last
    # sampled frame.  MDAnalysis 2.10 applies transformations immediately to the
    # current timestep, so NoJump would otherwise start without a previous cell
    # and fail.  Rewind before registering this stateful transformation.
    trajectory.Universe.trajectory[0]
    trajectory.Universe.trajectory.add_transformations(nojump.NoJump())

    try:
        import tidynamics  # noqa: F401 - enables MDAnalysis' FFT implementation

        use_fft = True
    except ImportError:
        use_fft = False
        warnings.warn(
            "tidynamics is unavailable; falling back to the slower direct MSD algorithm.",
            RuntimeWarning,
        )

    msd_run = MDAnalysis.analysis.msd.EinsteinMSD(
        selection, msd_type=args.msd_type, fft=use_fft
    )
    msd_run.run(start=start_frame, stop=end_frame + 1, step=interval_frame)

    # MDAnalysis consumes coordinates in Angstrom, so its MSD is in Angstrom^2.
    msd_timeseries = np.asarray(msd_run.results.timeseries, dtype=float) / 100.0
    lagtimes = np.arange(msd_run.n_frames, dtype=float) * frame_dt_ns

    output_filename = validate_extension(args.output, "xvg")

    xlabel = "Lag time (ns)"
    ylabel = "Mean-square displacement (nm^2)"
    title = f"Mean-square displacement of group {selection_name}"

    write_xvg(
        output_filename,
        lagtimes,
        msd_timeseries,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
    )


msd_commands = single_command(prog, getargs_msd, msd, desc)
