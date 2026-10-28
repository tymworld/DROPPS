# inter-chain distance calculation tool in DROPPS package by Yiming Tang @ Fudan
# Development started on July 21 2025

from dropps.share.argument_parser import ArgumentParser
from dropps.share.trajectory import trajectory_class
from dropps.fileio.filename_control import validate_extension
from dropps.fileio.xvg_reader import write_xvg
from dropps.analysis.distance_core import (
    build_distance_job,
    build_endpoint_plan,
    dimensions_angstrom_to_nm,
    iter_active_entity_pair_blocks,
    iter_distance_blocks,
    project_endpoint,
)

from tqdm import tqdm
import numpy as np

import warnings
from Bio import BiopythonDeprecationWarning

warnings.filterwarnings("ignore", category=BiopythonDeprecationWarning)

from MDAnalysis.lib.distances import minimize_vectors

prog = "odist"
desc = "Deprecated alias for legacy inter-chain distance profiles."

def getargs_inter_distance(argv):

    parser = ArgumentParser(prog=prog, description=desc)

    parser.add_argument('-s', '--run-input', type=str, required=True, 
                        help="Input DROPPS run file (.tpr) containing the system and simulation settings.")
    
    parser.add_argument('-f', '--input', type=str, required=True, 
                    help="Input trajectory file (.xtc).")
    
    parser.add_argument('-n', '--index', type=str, required=False,
                        help="Optional index file (.ndx) defining additional atom groups.")
    
    parser.add_argument('-ref', '--reference', type=int,
                        help="Reference index group containing one bead per chain; if omitted, prompt interactively.")
    
    parser.add_argument('-sel', '--selection', type=int,
                        help="Selection index group containing one bead per chain; if omitted, prompt interactively.")
    
    parser.add_argument('-b', '--start-time', type=float,
                        help="First trajectory time to analyze, in ns.")

    parser.add_argument('-e', '--end-time', type=float,
                        help="Last trajectory time to analyze, in ns.")

    parser.add_argument('-dt', '--delta-time', type=float,
                        help="Approximate interval between analyzed frames, in ns.")
    
    parser.add_argument('-pbc', '--treat-pbc', action='store_true', default=False,
                        help="Unwrap molecules across periodic boundaries before analysis.")

    parser.add_argument('-oa', '--output-average', type=str,
                        help="Output chain-averaged distance time series (.xvg); the extension is added if omitted.")
    
    parser.add_argument('-ov', '--output-verbose', type=str,
                        help="Output all pair distances as a function of time (.xvg); the extension is added if omitted.")

        
    args = parser.parse_args(argv)
    return args


def _interchain_pairs(reference_indices, selection_indices,
                      reference_chain_ids, selection_chain_ids):
    """Return atom pairs whose two atoms belong to different chains."""
    pairs = [
        (reference_index, selection_index)
        for reference_index, reference_chain in zip(reference_indices, reference_chain_ids)
        for selection_index, selection_chain in zip(selection_indices, selection_chain_ids)
        if reference_chain != selection_chain
    ]
    if not pairs:
        raise ValueError("reference and selection groups contain no inter-chain pairs")
    return np.asarray(pairs, dtype=int)


def _distance_vectors_nm(pos1, pos2, dimensions=None):
    """Calculate displacement-vector lengths in nm, optionally using minimum images."""
    vectors = np.asarray(pos1, dtype=float) - np.asarray(pos2, dtype=float)
    if dimensions is not None:
        box = np.asarray(dimensions, dtype=float)
        if box.size < 6 or not np.all(np.isfinite(box[:6])) or np.any(box[:3] <= 0):
            raise ValueError("valid periodic box dimensions are required for --treat-pbc")
        vectors = minimize_vectors(vectors, box)
    return np.linalg.norm(vectors, axis=1) / 10.0

def inter_distance(args):

    print(
        "## WARNING: 'odist' is deprecated; use 'dps distance --scope "
        "inter-chain' for new analyses."
    )

    if args.output_average is None and args.output_verbose is None:
        raise ValueError("at least one output must be requested with -oa or -ov")

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

    if args.reference is not None:
        print(f"## Will use group {args.reference} for distance calculations.")
        reference, reference_name = trajectory.getSelection(f"group {args.reference}")
    else:
        trajectory.index.print_all()
        reference, reference_name = trajectory.getSelection_interactive("distance calculation reference")
    
    if args.selection is not None:
        print(f"## Will use group {args.selection} for distance calculations.")
        selection, selection_name = trajectory.getSelection(f"group {args.selection}")
    else:
        trajectory.index.print_all()
        selection, selection_name = trajectory.getSelection_interactive("distance calculation selection")
    
    # We now split the indices
    reference_splitchains = trajectory.index.splitch_indices(reference.indices)
    selection_splitchains = trajectory.index.splitch_indices(selection.indices)

    if not reference_splitchains or any(len(group) != 1 for group in reference_splitchains):
        raise ValueError("reference group must contain exactly one atom per chain")
    if not selection_splitchains or any(len(group) != 1 for group in selection_splitchains):
        raise ValueError("selection group must contain exactly one atom per chain")
    if len(reference) != len(selection):
        raise ValueError("reference and selection groups must contain the same number of atoms")

    reference_chain_ids = [trajectory.get_chainID(index) for index in reference.indices]
    selection_chain_ids = [trajectory.get_chainID(index) for index in selection.indices]
    if set(reference_chain_ids) != set(selection_chain_ids):
        raise ValueError("reference and selection groups must contain the same chains")

    reference_plan = build_endpoint_plan(
        reference.indices,
        reference_name,
        "atom",
        "atom",
        trajectory.id2chainID,
        trajectory.id2resID,
    )
    selection_plan = build_endpoint_plan(
        selection.indices,
        selection_name,
        "atom",
        "atom",
        trajectory.id2chainID,
        trajectory.id2resID,
    )
    distance_plan = build_distance_job(
        "legacy-odist",
        reference_plan,
        selection_plan,
        "inter-chain",
        "pair",
    )
    pair_chain_ids = []
    for entity_pairs in iter_active_entity_pair_blocks(distance_plan):
        pair_chain_ids.extend(
            (
                reference_plan.entities[int(ref_entity)].chain_ids[0],
                selection_plan.entities[int(sel_entity)].chain_ids[0],
            )
            for ref_entity, sel_entity in entity_pairs
        )
    masses = np.asarray(trajectory.id2masses, dtype=np.float64)

    distances = np.zeros((len(frame_list), distance_plan.channel_count))
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
        distances[ts_index, :] = np.concatenate(
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
    
    print(f"## Calculation ended.")
    print(f"## We will perform statistics and write to files.")

    if args.output_verbose is not None:

        verbose_filename = validate_extension(args.output_verbose, "xvg")

        legends = [
            f"Reference chain {reference_chain} - selection chain {selection_chain}"
            for reference_chain, selection_chain in pair_chain_ids
        ]
        
        write_xvg(verbose_filename, time_list, distances.transpose(),
                  title="Distance", xlabel="Time (ns)", ylabel="Distance (nm)",
                  legends=legends)
        
        print(f"## Calculated and written verbose distance profiles to {verbose_filename}.")
    
    if args.output_average is not None:
        average_filename = validate_extension(args.output_average, "xvg")

        avg_off_diag = distances.mean(axis=1)
        
        write_xvg(average_filename, time_list, avg_off_diag,
                  title="Averaged distances", xlabel="Time (ns)", ylabel="Distance (nm)")
        print(f"## Calculated and written averaged distance for each time point and each pair to {average_filename}.")
    


from dropps.share.command_class import single_command
inter_distance_commands = single_command("odist", getargs_inter_distance, inter_distance, desc)
