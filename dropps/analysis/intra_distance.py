# intra-chain distance calculation tool in DROPPS package by Yiming Tang @ Fudan
# Development started on July 21 2025

from dropps.share.argument_parser import ArgumentParser
from dropps.share.trajectory import trajectory_class
from dropps.fileio.filename_control import validate_extension
from dropps.fileio.xvg_reader import write_xvg
from dropps.analysis.distance_core import (
    EndpointPlan,
    Entity,
    build_distance_job,
    dimensions_angstrom_to_nm,
    iter_distance_blocks,
    project_endpoint,
)

from tqdm import tqdm
import numpy as np

import warnings
from Bio import BiopythonDeprecationWarning

warnings.filterwarnings("ignore", category=BiopythonDeprecationWarning)

prog = "idist"
desc = "Deprecated alias for legacy intra-chain distance profiles."

def getargs_intra_distance(argv):

    parser = ArgumentParser(prog=prog, description=desc)

    parser.add_argument('-s', '--run-input', type=str, required=True, 
                        help="Input DROPPS run file (.tpr) containing the system and simulation settings.")
    
    parser.add_argument('-f', '--input', type=str, required=True, 
                    help="Input trajectory file (.xtc).")
    
    parser.add_argument('-n', '--index', type=str, required=False,
                        help="Optional index file (.ndx) defining additional atom groups.")
    
    parser.add_argument('-sel', '--selection', type=int, nargs='+',
                        help="Index groups containing one bead pair per chain; if omitted, prompt interactively.")
    
    parser.add_argument('-b', '--start-time', type=float,
                        help="First trajectory time to analyze, in ns.")

    parser.add_argument('-e', '--end-time', type=float,
                        help="Last trajectory time to analyze, in ns.")

    parser.add_argument('-dt', '--delta-time', type=float,
                        help="Approximate interval between analyzed frames, in ns.")
    
    parser.add_argument('-pbc', '--treat-pbc', action='store_true', default=False,
                        help="Unwrap molecules across periodic boundaries before analysis.")

    parser.add_argument('-ot', '--output-time', type=str,
                        help="Output chain-averaged distance time series (.xvg); the extension is added if omitted.")
    
    parser.add_argument('-op', '--output-pair', type=str,
                        help="Output time-averaged distance profile by pair (.xvg); the extension is added if omitted.")
    
    parser.add_argument('-ops', '--output-pair-statistic', type=str,
                        help="Output pair-wise distance mean and standard deviation (.xvg); the extension is added if omitted.")
    
    parser.add_argument('-ov', '--output-verbose', type=str,
                        help="Output per-chain distance time series (.xvg); the extension is added if omitted.")
    
    args = parser.parse_args(argv)
    return args


def intra_distance(args):

    print(
        "## WARNING: 'idist' is deprecated; use 'dps distance --scope "
        "intra-chain' for new analyses."
    )

    if all(output is None for output in (
        args.output_time,
        args.output_pair,
        args.output_pair_statistic,
        args.output_verbose,
    )):
        raise ValueError("at least one output must be requested")

    # load trajectory into memory

    try:
        trajectory = trajectory_class(args.run_input, args.index, args.input)
    except Exception as exc:
        print("## An exception occurred when trying to open trajectory file %s." % args.input)
        print(f"## Root cause: {exc}")
        quit()
    
    if args.treat_pbc is True:
        print(f"## Distance will be calculated using minimum-image periodic boundary conditions.")
    else:
        print(f"## WARNING: Distance will be calculated without periodic boundary conditions.")

    # We treat time for analysis and generate frame for analysis

    start_frame, end_frame, interval_frame = trajectory.time2frame(args.start_time, args.end_time, args.delta_time)

    frame_list = list(range(start_frame, end_frame + 1, interval_frame))

    # We now generate atom groups for calculations

    if args.selection is not None:
        group_ids = ",".join(str(i) for i in args.selection)
        print(f"## Will use groups {group_ids} for distance calculations.")
        groups = [trajectory.getSelection(f"group {gid}")[0] for gid in args.selection]
        groups_names = [f"group{i}" for i in args.selection]
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

    reference_plan = EndpointPlan(
        "legacy-idist-reference",
        tuple(
            Entity(
                np.asarray([atom_index], dtype=np.int64),
                np.asarray([trajectory.id2chainID[atom_index]], dtype=np.int64),
                f"legacy-idist-reference[{pair_index}]",
            )
            for pair_index, atom_index in enumerate(
                groups_splitchains_flatten[:, 0]
            )
        ),
        "atom",
    )
    selection_plan = EndpointPlan(
        "legacy-idist-selection",
        tuple(
            Entity(
                np.asarray([atom_index], dtype=np.int64),
                np.asarray([trajectory.id2chainID[atom_index]], dtype=np.int64),
                f"legacy-idist-selection[{pair_index}]",
            )
            for pair_index, atom_index in enumerate(
                groups_splitchains_flatten[:, 1]
            )
        ),
        "atom",
    )
    distance_plan = build_distance_job(
        "legacy-idist",
        reference_plan,
        selection_plan,
        "corresponding",
        "pair",
    )
    masses = np.asarray(trajectory.id2masses, dtype=np.float64)
    
    distances = np.zeros((len(frame_list),groups_splitchains_flatten.shape[0]))
    time_list = list()

    for ts_index, ts in tqdm(enumerate(trajectory.Universe.trajectory[frame_list])):

        time_list.append(ts.time / 1000)
        positions = np.asarray(ts.positions, dtype=np.float64) / 10.0
        dimensions = (
            dimensions_angstrom_to_nm(ts.dimensions)
            if args.treat_pbc
            else None
        )
        reference_positions = project_endpoint(
            reference_plan,
            positions,
            masses,
            dimensions,
            args.treat_pbc,
        )
        selection_positions = project_endpoint(
            selection_plan,
            positions,
            masses,
            dimensions,
            args.treat_pbc,
        )
        frame_distances = np.concatenate(
            [
                block.values
                for block in iter_distance_blocks(
                    distance_plan,
                    reference_positions,
                    selection_positions,
                    dimensions,
                    args.treat_pbc,
                )
            ]
        )
        distances[ts_index, :] = frame_distances
    
    distances = distances.reshape(len(frame_list), groups_splitchains.shape[0], groups_splitchains.shape[1])
    # Shape of distance: n_frame * n_pair * n_chain
    
    print(f"## Calculation ended.")
    print(f"## We will perform statistics and write to files.")

    chain_ids = [
        trajectory.get_chainID(groups_splitchains[0, chain_index, 0])
        for chain_index in range(n_chains)
    ]

    if args.output_verbose is not None:
        verbose_filename = validate_extension(args.output_verbose, "xvg")

        legends = [
            f"Chain {chain_id} Pair {group_name}"
            for chain_id in chain_ids
            for group_name in groups_names
        ]
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
        legends = [f"Chain {chain_id}" for chain_id in chain_ids]
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
        legends = ["Mean", "Standard deviation"]
        write_xvg(statistic_filename, [i+1 for i in range(len(groups_names))], mean_std,
            title="Mean and std for distances", xlabel="Pair index", ylabel="Distance (nm)",
            legends=legends)
        print(f"## Calculated and written statistic for each distance to {statistic_filename}.")

    

from dropps.share.command_class import single_command
intra_distance_commands = single_command("idist", getargs_intra_distance, intra_distance, desc)
