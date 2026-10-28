# Mean square deviation (MSD) tool in DROPPS package by Yiming Tang @ Fudan
# Development started on September 16 2025

from dropps.share.argument_parser import ArgumentParser
from dropps.share.trajectory import trajectory_class
from dropps.fileio.filename_control import validate_extension
from dropps.fileio.xvg_reader import write_xvg
from dropps.analysis.time_core import uniform_frame_interval_ns

import warnings
from Bio import BiopythonDeprecationWarning
warnings.filterwarnings("ignore", category=BiopythonDeprecationWarning)

from dropps.commands.trjconv_core import NoJumpState

import numpy as np

from tqdm import tqdm

prog = "msd"
desc = "Calculate the mean-square displacement of a selected atom group."


def getargs_msd(argv):

    parser = ArgumentParser(prog=prog, description=desc)

    parser.add_argument('-s', '--run-input', type=str, required=True, 
                        help="Input DROPPS run file (.tpr) containing the system and simulation settings.")

    parser.add_argument('-f', '--input', type=str, required=True, 
                    help="Input trajectory file (.xtc).")

    parser.add_argument('-n', '--index', type=str, required=False,
                        help="Optional index file (.ndx) defining additional atom groups.")
    
    parser.add_argument('-sel', '--selection', type=int,
                        help="Index group whose mean-square displacement is calculated; if omitted, prompt interactively.")
    
    parser.add_argument('-t', '--msd-type', choices=['xyz', 'xy', 'yz', 'xz', 'x', 'y', 'z'],
                        default='xyz', type=str,
                        help="Cartesian dimensions included in the MSD.")
    
    parser.add_argument('-b', '--start-time', type=float,
                        help="First trajectory time to analyze, in ns.")

    parser.add_argument('-e', '--end-time', type=float,
                        help="Last trajectory time to analyze, in ns.")

    parser.add_argument('-dt', '--delta-time', type=float,
                        help="Approximate interval between analyzed frames, in ns.")

    parser.add_argument('-o', '--output', type=str, required=True,
                        help="Output mean-square-displacement time series (.xvg); the extension is added if omitted.")

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
        print("## An exception occurred when trying to open trajectory file %s." % args.input)
        print(f"## Root cause: {exc}")
        quit()

    # We treat time for analysis and generate frame for analysis'


    start_frame, end_frame, interval_frame = trajectory.time2frame(args.start_time, args.end_time, args.delta_time)
    frame_indices = list(range(start_frame, end_frame + 1, interval_frame))
    frame_times_ns = [
        float(trajectory.Universe.trajectory[index].time) / 1000.0
        for index in frame_indices
    ]
    frame_dt_ns = uniform_frame_interval_ns(frame_times_ns)

    if args.selection is not None:
        print(f"## Will use group {args.selection} for MSD calculations.")
        selection, selection_name = trajectory.getSelection(f"group {args.selection}")
    else:
        trajectory.index.print_all()
        selection, selection_name = trajectory.getSelection_interactive()

    print(f"## Will calculate MSD for group {selection_name} at {args.msd_type} dimensions.")

    if len(selection) == 0:
        raise ValueError("MSD requires a non-empty atom selection.")
    # Temporal unwrapping must visit EVERY saved frame before subsampling.
    # Otherwise --delta-time can turn a displacement larger than half a box
    # into the wrong periodic image, even when the saved trajectory is valid.
    reader = trajectory.Universe.trajectory
    first = reader[0]
    state = NoJumpState(selection.positions, first.triclinic_dimensions)
    positions = np.empty((len(frame_indices), len(selection), 3), dtype=float)
    output_index = 0
    for frame_index in tqdm(range(end_frame + 1), desc="## Unwrapping MSD trajectory"):
        frame = reader[frame_index]
        whole = state.apply(selection.positions, frame.triclinic_dimensions)
        if frame_index == frame_indices[output_index]:
            positions[output_index] = whole / 10.0  # Angstrom -> nm
            output_index += 1
            if output_index == len(frame_indices):
                break
    axes = ["xyz".index(axis) for axis in args.msd_type]
    positions = positions[:, :, axes]
    positions -= positions[0]
    try:
        import tidynamics
    except ImportError:
        warnings.warn("tidynamics is unavailable; using direct MSD averaging.", RuntimeWarning)
        msd_timeseries = np.zeros(len(frame_indices), dtype=float)
        for lag in range(1, len(frame_indices)):
            displacement = positions[lag:] - positions[:-lag]
            msd_timeseries[lag] = np.mean(np.sum(displacement ** 2, axis=-1))
    else:
        msd_timeseries = np.mean(
            [tidynamics.msd(positions[:, atom]) for atom in range(len(selection))],
            axis=0,
        )
        msd_timeseries[0] = 0.0
    lagtimes = np.arange(len(frame_indices), dtype=float) * frame_dt_ns

    output_filename = validate_extension(args.output, "xvg")

    xlabel = "Lag time (ns)"
    ylabel = "Mean-square displacement (nm^2)"
    title = f"Mean-square displacement of group {selection_name}"

    write_xvg(output_filename, lagtimes, msd_timeseries, title=title,
              xlabel=xlabel, ylabel=ylabel)


from dropps.share.command_class import single_command
msd_commands = single_command(prog, getargs_msd, msd, desc)
