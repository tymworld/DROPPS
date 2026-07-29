# Mean square deviation (MSD) tool in DROPPS package by Yiming Tang @ Fudan
# Development started on September 16 2025

from dropps.share.argument_parser import ArgumentParser
from dropps.share.trajectory import trajectory_class
from dropps.fileio.filename_control import validate_extension
from dropps.fileio.xvg_reader import write_xvg

import warnings
from Bio import BiopythonDeprecationWarning
warnings.filterwarnings("ignore", category=BiopythonDeprecationWarning)

from MDAnalysis.transformations import nojump

from openmm.unit import nanosecond

import numpy as np

import MDAnalysis.analysis.msd

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
    
    parser.add_argument('-b', '--start-time', type=int,
                        help="First trajectory time to analyze, in ns.")

    parser.add_argument('-e', '--end-time', type=int,
                        help="Last trajectory time to analyze, in ns.")

    parser.add_argument('-dt', '--delta-time', type=int,
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

    if args.selection is not None:
        print(f"## Will use group {args.selection} for Rg calculations.")
        selection, selection_name = trajectory.getSelection(f"group {args.selection}")
    else:
        trajectory.index.print_all()
        selection, selection_name = trajectory.getSelection_interactive()

    print(f"## Will calculate MSD for group {selection_name} at {args.msd_type} dimensions.")

    trajectory.Universe.trajectory.add_transformations(nojump.NoJump())
    
    msd_run = MDAnalysis.analysis.msd.EinsteinMSD(selection, msd_type=args.msd_type) 
    msd_run.run(start=start_frame, stop=end_frame + 1, step=interval_frame)

    msd_timeseries = np.array(msd_run.results.timeseries)
    lagtimes = np.arange(msd_run.n_frames) * interval_frame * trajectory.time_step().value_in_unit(nanosecond)

    output_filename = validate_extension(args.output, "xvg")

    xlabel = "Lag time (ns)"
    ylabel = "Mean square deviation"
    title = f"Mean square deviation of group {selection_name}" 

    write_xvg(output_filename, lagtimes, msd_timeseries, title=title,
              xlabel=xlabel, ylabel=ylabel)


from dropps.share.command_class import single_command
msd_commands = single_command(prog, getargs_msd, msd, desc)
