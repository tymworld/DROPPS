# inter-chain distance calculation tool in DROPPS package by Yiming Tang @ Fudan
# Development started on July 21 2025

from argparse import ArgumentParser
from dropps.share.trajectory import trajectory_class
from dropps.fileio.filename_control import validate_extension
from dropps.fileio.xvg_reader import write_xvg

from tqdm import tqdm
import numpy as np

import warnings
from Bio import BiopythonDeprecationWarning

warnings.filterwarnings("ignore", category=BiopythonDeprecationWarning)

from MDAnalysis.transformations import unwrap

prog = "inter_distance"
desc = '''This program calculate the inter-chain distance profiles of atom pair.'''

def getargs_inter_distance(argv):

    parser = ArgumentParser(prog=prog, description=desc)

    parser.add_argument('-s', '--run-input', type=str, required=True, 
                        help="TPR file containing all information for a simulation run.")
    
    parser.add_argument('-f', '--input', type=str, required=True, 
                    help="XTC file which is taken as input trajectory.")
    
    parser.add_argument('-n', '--index', type=str, required=False,
                        help="Index file containing non-default groups.")
    
    parser.add_argument('-ref', '--reference', type=int,
                        help="Reference group contaning atoms for each chain, one atom for each chain")
    
    parser.add_argument('-sel', '--selection', type=int,
                        help="Selection group contaning atoms for each chain, one atom for each chain")
    
    parser.add_argument('-b', '--start-time', type=int,
                        help="Time (ns) of the first frame to calculate.")

    parser.add_argument('-e', '--end-time', type=int,
                        help="Time (ns) of the last frame to calculate.")

    parser.add_argument('-dt', '--delta-time', type=int,
                        help="Intervals (ns) between two calculated frames.")
    
    parser.add_argument('-pbc', '--treat-pbc', action='store_true', default=False,
                        help="Treat broken molecule at periodic boundaries.")

    parser.add_argument('-oa', '--output-average', type=str,
                        help="Output of chain-averaged distance as a function of time.")
    
    parser.add_argument('-ov', '--output-verbose', type=str,
                        help="Verbose output of all distances.")

        
    args = parser.parse_args(argv)
    return args

def inter_distance(args):

    # load trajectory into memory

    try:
        trajectory = trajectory_class(args.run_input, args.index, args.input)
    except:
        print("## An exception occurred when trying to open trajectory file %s." % args.input)
        quit()
    
    if args.treat_pbc is True:
        print(f"## Distance will be calculated using periodic boundary conditions.")
        trajectory.Universe.trajectory.add_transformations(unwrap(trajectory.Universe.atoms))
    else:
        print(f"## WARNING: Distance will be calculated without periodic boundary conditions.")

    # We treat time for analysis and generate frame for analysis

    start_frame, end_frame, interval_frame = trajectory.time2frame(args.start_time, args.end_time, args.delta_time)

    frame_list = list(range(start_frame, end_frame, interval_frame))

    # We now generate atom groups for calculations

    if args.reference is not None:
        print(f"## Will use group {args.reference} for distance calculations.")
        reference = trajectory.getSelection(f"group {args.reference}")
        reference_name = f"group{args.reference}"
    else:
        trajectory.index.print_all()
        reference, reference_name = trajectory.getSelection_interactive("distance calculation reference")
    
    if args.selection is not None:
        print(f"## Will use group {args.selection} for distance calculations.")
        selection = trajectory.getselection(f"group {args.selection}")
        selection_name = f"group{args.selection}"
    else:
        trajectory.index.print_all()
        selection, selection_name = trajectory.getSelection_interactive("distance calculation selection")
    
    # We now split the indices
    reference_splitchains = trajectory.index.splitch_indices(reference.indices)
    selection_splitchains = trajectory.index.splitch_indices(selection.indices)

    if len(set([len(group) for group in reference_splitchains])) > 1:
        print(f"ERROR: multiple atoms in single chain for reference group.")
        quit()
    if len(set([len(group) for group in selection_splitchains])) > 1:
        print(f"ERROR: multiple atoms in single chain for selection group.")
        quit()
    if len(reference) != len(selection):
        print(f"ERROR: Number of atoms in reference and selection groups not match.")
        quit()
    
    
    pairs = np.array([[a1, a2] for a1 in reference.indices for a2 in selection.indices])

    distances = np.zeros((len(frame_list),len(reference) * len(selection)))
    time_list = list()

    for ts_index, ts in tqdm(enumerate(trajectory.Universe.trajectory[frame_list])):

        time_list.append(trajectory.Universe.trajectory.time / 1000)
        pos1 = trajectory.Universe.atoms[pairs[:, 0]].positions
        pos2 = trajectory.Universe.atoms[pairs[:, 1]].positions

        # Compute vectors
        v1 = pos1 - pos2

        # Normalize
        v1_norm = np.linalg.norm(v1, axis=1)

        distances[ts_index, :] = v1_norm
    
    distances_matrix = distances.reshape(len(time_list), len(reference), len(selection))

    distances_no_diag = distances_matrix.copy()
    n = distances_matrix.shape[1]
    idx = np.arange(n)
    distances_no_diag[:, idx, idx] = 0
    
    print(f"## Calculation ended.")
    print(f"## We will perform statistics and write to files.")

    chain_index_list = [trajectory.get_chainID(atom) for atom in reference.indices]

    if args.output_verbose is not None:

        verbose_filename = validate_extension(args.output_verbose, "xvg")

        legends = [f"Chain {chain_index_list[j]} - Chain {chain_index_list[i]}"
                   for i in range(len(chain_index_list)) for j in range(i)]
        
        distances_verbose = np.array([[distances_matrix[time_index, j, i]
                                       for i in range(len(chain_index_list)) for j in range(i)]
                                       for time_index in range(len(time_list))])
        
        write_xvg(verbose_filename, time_list, distances_verbose,
                  title="Distance", xlabel="Time (ns)", ylabel="Distance (nm)",
                  legends=legends)
        
        print(f"## Calculated and written verbose distance profiles to {verbose_filename}.")
    
    if args.output_average is not None:
        average_filename = validate_extension(args.output_average, "xvg")

        avg_off_diag = distances_no_diag.sum(axis=(1, 2)) / (n * n - n)
        print(avg_off_diag.shape)
        
        write_xvg(average_filename, time_list, avg_off_diag,
                  title="Averaged distances", xlabel="Time (ns)", ylabel="Distance (nm)")
        print(f"## Calculated and written averaged distance for each time point and each pair to {average_filename}.")
    


from dropps.share.command_class import single_command
inter_distance_commands = single_command("odist", getargs_inter_distance, inter_distance, desc)
