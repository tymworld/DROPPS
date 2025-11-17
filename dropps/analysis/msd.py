# Mean square deviation (MSD) tool in DROPPS package by Yiming Tang @ Fudan
# Development started on September 16 2025

from argparse import ArgumentParser
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
desc = '''This program calculate mean square deviation of molecules within a trajectory.'''


def getargs_msd(argv):

    parser = ArgumentParser(prog=prog, description=desc)

    parser.add_argument('-s', '--run-input', type=str, required=True, 
                        help="TPR file containing all information for a simulation run.")

    parser.add_argument('-f', '--input', type=str, required=True, 
                    help="XTC file which is taken as input trajectory.")

    parser.add_argument('-n', '--index', type=str, required=False,
                        help="Index file containing non-default groups.")
    
    parser.add_argument('-sel', '--selection', type=int, nargs='+',
                        help="Groups of atoms on which the mean square deviation will be printed")
    
    parser.add_argument('-t', '--msd-type', choices=['xyz', 'xy', 'yz', 'xz', 'x', 'y', 'z'],
                        default='xyz', type=str,
                        help="Desired dimensions to be included in the MSD")
    
    parser.add_argument('-b', '--start-time', type=int,
                        help="Time (ns) of the first frame to calculate.")

    parser.add_argument('-e', '--end-time', type=int,
                        help="Time (ns) of the last frame to calculate.")

    parser.add_argument('-dt', '--delta-time', type=int,
                        help="Intervals (ns) between two calculated frames.")

    parser.add_argument('-o', '--output', type=str, required=False, 
                        help="XVG file to write verbose msd for each group as a function of time.")

    args = parser.parse_args(argv)

    return args


def msd(args):

    if not args.output:
        print("ERROR: No output file specified.")
        quit()
    
    # load trajectory into memory

    try:
        trajectory = trajectory_class(args.run_input, args.index, args.input)
    except:
        print("## An exception occurred when trying to open trajectory file %s." % args.input)
        quit()

    # We treat time for analysis and generate frame for analysis'


    start_frame, end_frame, interval_frame = trajectory.time2frame(args.start_time, args.end_time, args.delta_time)

    if args.selection is not None:
        print(f"## Will use group {args.selection} for Rg calculations.")
        selection = trajectory.getSelection(f"group {args.selection}")
        selection_name = f"group{args.selection}" 
    else:
        trajectory.index.print_all()
        selection, selection_name = trajectory.getSelection_interactive()

    print(f"## Will calculate MSD for group {selection_name} at {args.msd_type} dimensions.")

    trajectory.Universe.trajectory.add_transformations(nojump.NoJump())
    
    msd_run = MDAnalysis.analysis.msd.EinsteinMSD(selection, msd_type=args.msd_type) 
    msd_run.run(start=start_frame, stop=end_frame, step=interval_frame)

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

