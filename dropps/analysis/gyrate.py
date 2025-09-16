# gyrate tool in DROPPS package by Yiming Tang @ Fudan
# Development started on July 18 2025

from argparse import ArgumentParser
from dropps.share.trajectory import trajectory_class
from openmm.unit import nanosecond, picosecond
from tqdm import tqdm
import numpy as np

from dropps.fileio.xvg_reader import write_xvg
from dropps.fileio.filename_control import validate_extension

import warnings
from Bio import BiopythonDeprecationWarning

warnings.filterwarnings("ignore", category=BiopythonDeprecationWarning)

from MDAnalysis.transformations import unwrap
prog = "gyrate"
desc = '''This program calculate the radius of gyration of all chains in a HPS trajectory.'''

def radius_of_gyration(positions, masses):

    com = np.average(positions, axis=0, weights=masses)
    squared_distances = np.sum((positions - com)**2, axis=1)
    return np.sqrt(np.average(squared_distances, weights=masses))

def getargs_gyrate(argv):

    parser = ArgumentParser(prog=prog, description=desc)

    parser.add_argument('-s', '--run-input', type=str, required=True, 
                        help="TPR file containing all information for a simulation run.")

    parser.add_argument('-f', '--input', type=str, required=True, 
                    help="XTC file which is taken as input trajectory.")

    parser.add_argument('-n', '--index', type=str, required=False,
                        help="Index file containing non-default groups.")
    
    parser.add_argument('-sel', '--selection-calculate', type=int, nargs='+',
                        help="Groups of atoms on which the radius of gyrations will be printed")
    
    parser.add_argument('-b', '--start-time', type=int,
                        help="Time (ns) of the first frame to calculate.")

    parser.add_argument('-e', '--end-time', type=int,
                        help="Time (ns) of the last frame to calculate.")

    parser.add_argument('-dt', '--delta-time', type=int,
                        help="Intervals (ns) between two calculated frames.")
    
    parser.add_argument('-pbc', '--treat-pbc', action='store_true', default=False,
                        help="Treat broken molecule at periodic boundaries.")

    parser.add_argument('-ov', '--output-verbose', type=str, required=False, 
                        help="XVG file to write verbose Rg for each group as a function of time.")

    parser.add_argument('-oa', '--output-average', type=str, required=False, 
                        help="XVG file to write Rg averaged for group as a function of time.")

    parser.add_argument('-oh', '--output-histogram', type=str, required=False, 
                        help="XVG file to write Rg distributions.")

    parser.add_argument('-bw', '--bin-width', type=float, default=0.1,
                        help="Bin width of output distribution, unit is nanometer.")

    args = parser.parse_args(argv)

    return args

def gyrate(args):

    if not (args.output_average or args.output_histogram or args.output_verbose):
        print("ERROR: No output file specified.")
        quit()
    
    # load trajectory into memory

    try:
        trajectory = trajectory_class(args.run_input, args.index, args.input)
    except:
        print("## An exception occurred when trying to open trajectory file %s." % args.input)
        quit()

    # We treat time for analysis and generate frame for analysis

    start_frame, end_frame, interval_frame = trajectory.time2frame(args.start_time, args.end_time, args.delta_time)

    if args.selection_calculate is not None:
        print(f"## Will use groups {','.join([f"{i}" for i in args.selection_calculate])} for Rg calculations.")
        selections = [trajectory.getSelection(f"group {gid}")[0] for gid in args.selection_calculate]
        selection_names = [f"group{i}" for i in args.selection_calculate]
    else:
        trajectory.index.print_all()
        selections, selection_names = trajectory.getSelection_interactive_multiple()
    
    if args.treat_pbc is True:
        trajectory.Universe.trajectory.add_transformations(unwrap(trajectory.Universe.atoms))
    
    rg_lists = [list() for i in range(len(selections))]
    time_list = list()

    print(f"## Will calculate radius of gyration for {len(selections)} groups.")
    print(f"## Start of radius of gyration calculations.")

    for ts in tqdm(trajectory.Universe.trajectory[start_frame:end_frame:interval_frame]):

        time_list.append(trajectory.Universe.trajectory.time / 1000)
        for selection_id, selection in enumerate(selections):
            pos = selection.positions
            masses = selection.masses 
            rg = radius_of_gyration(pos, masses)
            rg_lists[selection_id].append(rg)
    
    rg_matrix = np.array(rg_lists) / 10.0
    
    # Write output

    if args.output_verbose is not None:
        verbose_filename = validate_extension(args.output_verbose, "xvg")
        xlabel = "Time (ns)"
        ylabel = "Radius of gyration (nm)"
        title = "Radius of gyration"
        legends = selection_names

        write_xvg(verbose_filename, time_list, rg_matrix, title=title, xlabel=xlabel,ylabel=ylabel,legends=legends)
    
    if args.output_average is not None:
        average_filename = validate_extension(args.output_average, "xvg")
        xlabel = "Time (ns)"
        ylabel = "Averaged radius of gyration (nm)"
        title = "Averaged radius of gyration"
        legends = ["Averaged"]

        write_xvg(average_filename, time_list, rg_matrix.mean(axis=0), title=title, xlabel=xlabel,ylabel=ylabel,legends=legends)
    
    if args.output_histogram is not None:
        histogram_filename = validate_extension(args.output_histogram, "xvg")
        rg_flatten = rg_matrix.flatten()
        bin_width = args.bin_width

        data_min = np.floor(rg_flatten.min() / bin_width) * bin_width
        data_max = np.ceil(rg_flatten.max() / bin_width) * bin_width

        bins = np.arange(data_min, data_max + bin_width, bin_width)

        hist, bin_edges = np.histogram(rg_flatten, bins)
        hist_pdf = hist / rg_flatten.shape[0] / bin_width

        xlabel = "Bin lower edge (nm)"
        ylabel = "PDF"
        title = "Distribution of rg profiles"

        write_xvg(histogram_filename, bin_edges[:-1], hist_pdf, title=title, xlabel=xlabel, ylabel=ylabel)


        


from dropps.share.command_class import single_command
gyrate_commands = single_command(prog, getargs_gyrate, gyrate, desc)