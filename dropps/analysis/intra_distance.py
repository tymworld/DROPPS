# intra-chain distance calculation tool in DROPPS package by Yiming Tang @ Fudan
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

prog = "intra_distance"
desc = '''This program calculate the intra-chain distance profiles of a number of atom pairs.'''

def getargs_intra_distance(argv):

    parser = ArgumentParser(prog=prog, description=desc)

    parser.add_argument('-s', '--run-input', type=str, required=True, 
                        help="TPR file containing all information for a simulation run.")
    
    parser.add_argument('-f', '--input', type=str, required=True, 
                    help="XTC file which is taken as input trajectory.")
    
    parser.add_argument('-n', '--index', type=str, required=False,
                        help="Index file containing non-default groups.")
    
    parser.add_argument('-sel', '--selection', type=int, nargs='+',
                        help="Selection groups contaning pair of atoms for each chain, chain will be splitted.")
    
    parser.add_argument('-b', '--start-time', type=int,
                        help="Time (ns) of the first frame to calculate.")

    parser.add_argument('-e', '--end-time', type=int,
                        help="Time (ns) of the last frame to calculate.")

    parser.add_argument('-dt', '--delta-time', type=int,
                        help="Intervals (ns) between two calculated frames.")
    
    parser.add_argument('-pbc', '--treat-pbc', action='store_true', default=False,
                        help="Treat broken molecule at periodic boundaries.")

    parser.add_argument('-ot', '--output-time', type=str,
                        help="Output of all distances (chain averaged) as a function of time.")
    
    parser.add_argument('-op', '--output-pair', type=str,
                        help="Output of all pairs (time averaged) as a function of residue index.")
    
    parser.add_argument('-ops', '--output-pair-statistic', type=str,
                        help="Statistic of all distances as a function of residue index (mean and std).")
    
    parser.add_argument('-ov', '--output-verbose', type=str,
                        help="Verbose output of all distances.")
    
    args = parser.parse_args(argv)
    return args


def intra_distance(args):

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

    if args.selection is not None:
        print(f"## Will use groups {','.join([f"{i}" for i in args.selection])} for distance calculations.")
        groups = [trajectory.getSelection(f"group {gid}")[0] for gid in args.selection_calculate]
        groups_names = [f"group{i}" for i in args.selection_calculate]
    else:
        trajectory.index.print_all()
        groups, groups_names = trajectory.getSelection_interactive_multiple("distance calculation pairs")
    
    # We now split the indices
    groups_splitchains = [trajectory.index.splitch_indices(group.indices) for group in groups] # n_pair * n_chain * (2)

    # We check chains

    if len(set(list(map(len, groups_splitchains)))) != 1:
        print(f"ERROR: At least two of your input groups contain different number of chains.")
        quit()
    
    issues = [(i, j) for i, group in enumerate(groups_splitchains)
                for j, item in enumerate(group)
                if not isinstance(item, (list, tuple)) or len(item) != 2]
    if issues:
        for i, j in issues:
            print(f"ERROR: Element at [{i}][{j}] is not of length 2.")
            quit()
    
    groups_splitchains = np.array(groups_splitchains) # n_pair * n_chain * (2)

    n_pairs = groups_splitchains.shape[0]
    n_chains = groups_splitchains.shape[1]
    print(f"## Will calculate {n_pairs} distance pairs for each of {n_chains} chains.")

    groups_splitchains_flatten = groups_splitchains.reshape(-1, 2)
    
    distances = np.zeros((len(frame_list),groups_splitchains_flatten.shape[0]))
    time_list = list()

    for ts_index, ts in tqdm(enumerate(trajectory.Universe.trajectory[frame_list])):

        time_list.append(trajectory.Universe.trajectory.time / 1000)
        pos1 = trajectory.Universe.atoms[groups_splitchains_flatten[:, 0]].positions
        pos2 = trajectory.Universe.atoms[groups_splitchains_flatten[:, 1]].positions

        # Compute vectors
        v1 = pos1 - pos2

        # Normalize
        v1_norm = np.linalg.norm(v1, axis=1)

        distances[ts_index, :] = v1_norm
    
    distances = distances.reshape(len(frame_list), groups_splitchains.shape[0], groups_splitchains.shape[1]) / 10.0
    # Shape of distance: n_frame * n_pair * n_chain
    
    print(f"## Calculation ended.")
    print(f"## We will perform statistics and write to files.")

    if args.output_verbose is not None:
        verbose_filename = validate_extension(args.output_verbose, "xvg")

        legends = [f"Chain {trajectory.get_chainID(chain[0])} Pair {group_name}" for chain in groups_splitchains_flatten for group_name in groups_names]
        write_xvg(verbose_filename, time_list, distances.transpose(0, 2, 1).reshape(distances.shape[0],-1).transpose(), 
                  title="Distance", xlabel="Time (ns)", ylabel="Distance (nm)",
                  legends=legends)
        print(f"## Calculated and written verbose distance profiles to {verbose_filename}.")
    
    if args.output_time is not None:
        time_filename = validate_extension(args.output_time, "xvg")
        legends = [f"Pair {group_name}" for group_name in groups_names]
        write_xvg(time_filename, time_list, np.mean(distances, axis=2).transpose(),
                  title="Averaged distances", xlabel="Time (ns)", ylabel="Distance (nm)",
                  legends=legends)
        print(f"## Calculated and written averaged distance for each time point and each pair to {time_filename}.")
    
    if args.output_pair is not None:
        residue_filename = validate_extension(args.output_pair, "xvg")
        legends = [f"Chain {trajectory.get_chainID(chain[0])}" for chain in groups_splitchains_flatten]
        write_xvg(residue_filename, [i+1 for i in range(len(groups_names))], np.mean(distances, axis=0).transpose(),
                  title="Averaged distances", xlabel="Pair index", ylabel="Distance (nm)",
                  legends=legends)
        print(f"## Calculated and written averaged distance for each frame and each pair to {residue_filename}.")
    
    if args.output_pair_statistic is not None:
        statistic_filename = validate_extension(args.output_pair_statistic, "xvg")

        distance_per_residue = distances.transpose(1,0,2).reshape(n_pairs,-1)
        mean = distance_per_residue.mean(axis=1)
        std = distance_per_residue.std(axis=1)

        mean_std = np.array([mean, std])
        legends = ["Mean", "Standard error"]
        write_xvg(statistic_filename, [i+1 for i in range(len(groups_names))], mean_std,
            title="Mean and std for distances", xlabel="Pair index", ylabel="Distance (nm)",
            legends=legends)
        print(f"## Calculated and written statistic for each distance to {statistic_filename}.")

    

from dropps.share.command_class import single_command
intra_distance_commands = single_command("idist", getargs_intra_distance, intra_distance, desc)
