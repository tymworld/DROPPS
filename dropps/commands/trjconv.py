# trjconv tool in DROPPS package by Yiming Tang @ Fudan
# Development started on Nov 17 2025

from argparse import ArgumentParser
from dropps.share.trajectory import trajectory_class
from dropps.fileio.filename_control import validate_extension
from dropps.analysis.density import shift_density_center

import MDAnalysis.transformations as transformations
from MDAnalysis.analysis import lineardensity as lin

from MDAnalysis import Writer

from os.path import splitext
import numpy as np

prog = "trjconv"
desc = '''This program read and convert trajectory files in many ways.'''

def getargs_trjconv(argv):

    # Command line argument parser

    parser = ArgumentParser(prog=prog, description=desc)

    parser.add_argument('-s', '--run-input', type=str, required=True, 
                        help="TPR file containing all information for a simulation run.")
    
    parser.add_argument('-f', '--input', type=str, required=True, 
                    help="XTC file which is taken as input trajectory.")
    
    parser.add_argument('-n', '--index', type=str, required=False,
                        help="Index file containing non-default groups.")
    
    parser.add_argument('-o', '--output', type=str,
                        help="Output file path for trajectory. Allowed filetype: pdb, xtc")
    
    #parser.add_argument('-dpc', '--dense-phase-center', type=bool, action='store_true', default=False,
    #                    help="Whether to center dense phase in the z direction")
    
    #parser.add_argument('-dpt', '--dense-phase-threshold', default=0.5, type=float,
    #                    help="Threshold for dense phase which is a multiplier of the highest density.")
    
    parser.add_argument('-b', '--start-time', type=int,
                        help="Time (ns) of the first frame to calculate.")

    parser.add_argument('-e', '--end-time', type=int,
                        help="Time (ns) of the last frame to calculate.")

    parser.add_argument('-dt', '--delta-time', type=int,
                        help="Intervals (ns) between two calculated frames.")
    
    parser.add_argument('-pbc', '--pbc', type=str, choices=["none", "atom", "mol"],
                        default="none", required=True,
                        help="Translation performed on each beads to treat against perodic boundary conditions.")

    args = parser.parse_args(argv)

    return args

def trjconv(args):

    # load trajectory into memory

    try:
        trajectory = trajectory_class(args.run_input, args.index, args.input)
    except:
        print("## An exception occurred when trying to open trajectory file %s." % args.input)
        quit()

    start_frame, end_frame, interval_frame = trajectory.time2frame(args.start_time, args.end_time, args.delta_time)

    print(trajectory.Universe.trajectory[0])
    print(trajectory.Universe.trajectory[0][0])

    # We now set workflow for transformations

    trajectory.index.print_all()
    if args.pbc == "atom":
        selection_wrap, _ = trajectory.getSelection_interactive("Group for treating pbc: atom wrapping")
        workflow = [transformations.wrap(selection_wrap)]
        trajectory.Universe.trajectory.add_transformations(*workflow)
    elif args.pbc == "mol":
        selection_wrap, _ = trajectory.getSelection_interactive("Group for treating pbc: mol wrapping")
        workflow = [transformations.wrap(selection_wrap),
                    transformations.unwrap(selection_wrap)]
        trajectory.Universe.trajectory.add_transformations(*workflow)

    file_extention = splitext(args.output)[1].lower()
    if file_extention not in [".pdb", ".xtc"]:
        print(f"ERROR: Cannot write trajectory to {args.output} with extention {file_extention}")
        quit()

    output_trajectory = args.output

    selection_output, selection_output_name = trajectory.getSelection_interactive(f"Group for output to {output_trajectory}")

    # Now we output

    with Writer(output_trajectory, trajectory.Universe.atoms.n_atoms) as writer:
        # Loop over the frames in the specified range and write selected frames
        for ts in trajectory.Universe.trajectory[start_frame:end_frame+1:interval_frame]:
            writer.write(selection_output)  # Write the selected atoms to the trajectory
    print(f"## Trajectory written to file {output_trajectory}")

    
from dropps.share.command_class import single_command
trjconv_commands = single_command("trjconv", getargs_trjconv, trjconv, desc)