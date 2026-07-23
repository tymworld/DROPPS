"""Low-memory residue contact maps for the ``dps contact.ng`` command."""

from dropps.share.argument_parser import ArgumentParser
import warnings

import numpy as np
from tqdm import tqdm

from Bio import BiopythonDeprecationWarning
from MDAnalysis.lib.nsgrid import FastNS
from MDAnalysis.transformations import wrap

from dropps.analysis.contact_ng_core import (
    INTER_RR,
    INTER_RS,
    INTER_SS,
    INTRA_R,
    INTRA_S,
    StreamingContactAccumulator,
    build_contact_layout,
)
from dropps.analysis.contact_cli import add_contact_arguments
from dropps.fileio.filename_control import validate_extension
from dropps.fileio.xpm_reader import write_xpm
from dropps.fileio.xvg_reader import write_xvg
from dropps.share.command_class import single_command
from dropps.share.trajectory import trajectory_class


warnings.filterwarnings("ignore", category=BiopythonDeprecationWarning)

prog = "contact.ng"
desc = "Calculate residue-level contact maps with memory-efficient aggregation."


MAP_OUTPUT_ATTRIBUTES = {
    INTER_RS: "output_inter_reference_selection",
    INTER_RR: "output_inter_reference_reference",
    INTER_SS: "output_inter_selection_selection",
    INTRA_R: "output_intra_reference",
    INTRA_S: "output_intra_selection",
}

TIME_OUTPUT_ATTRIBUTES = {
    INTER_RS: "output_time_inter_reference_selection",
    INTER_RR: "output_time_inter_reference_reference",
    INTER_SS: "output_time_inter_selection_selection",
    INTRA_R: "output_time_intra_reference",
    INTRA_S: "output_time_intra_selection",
}


def getargs_contact_ng(argv):
    """Parse the same interface as ``contact``, plus two memory controls."""

    parser = ArgumentParser(
        prog=prog,
        description=desc,
        epilog="At least one contact-map or contact-number output option is required.",
    )
    add_contact_arguments(parser)
    parser.add_argument(
        "--search-mode",
        choices=["fast", "stream"],
        default="fast",
        help=(
            "Neighbor-search strategy: fast stores all pairs from one frame; "
            "stream bounds pair memory by searching in "
            "query batches."
        ),
    )
    parser.add_argument(
        "--search-batch-size",
        type=int,
        default=4096,
        metavar="N",
        help=(
            "Number of query atoms per neighborhood-search batch in stream mode. "
            "Smaller values reduce peak memory at some performance cost."
        ),
    )
    parser.add_argument(
        "--pair-chunk-size",
        type=int,
        default=250000,
        metavar="N",
        help="Maximum number of neighbor pairs aggregated at once.",
    )

    args = parser.parse_args(argv)
    if args.search_batch_size <= 0:
        parser.error("--search-batch-size must be greater than zero")
    if args.pair_chunk_size <= 0:
        parser.error("--pair-chunk-size must be greater than zero")
    return args


def _requested_outputs(args, attribute_map):
    return {
        category: getattr(args, attribute)
        for category, attribute in attribute_map.items()
        if getattr(args, attribute) is not None
    }


def _process_neighbor_search_frame(
    accumulator,
    positions,
    box,
    cutoff,
    pbc,
    search_batch_size,
    pair_chunk_size,
    atom_cutoffs=None,
):
    """Search and aggregate one frame without materializing all frame pairs."""

    positions = np.asarray(positions, dtype=np.float32)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (N, 3)")
    if positions.shape[0] != accumulator.layout.n_atoms:
        raise ValueError("positions and contact layout contain different atom counts")

    if atom_cutoffs is not None:
        atom_cutoffs = np.asarray(atom_cutoffs, dtype=np.float64)
        if atom_cutoffs.shape != (positions.shape[0],):
            raise ValueError("atom_cutoffs must contain one value per search atom")

    neighborhood = FastNS(cutoff, positions, box=box, pbc=pbc)
    accumulator.begin_frame()

    for query_start in range(0, positions.shape[0], search_batch_size):
        query_stop = min(query_start + search_batch_size, positions.shape[0])
        result = neighborhood.search(positions[query_start:query_stop])
        pairs = result.get_pairs()
        if pairs.size == 0:
            continue

        distances = result.get_pair_distances() if atom_cutoffs is not None else None

        for pair_start in range(0, pairs.shape[0], pair_chunk_size):
            pair_stop = min(pair_start + pair_chunk_size, pairs.shape[0])
            pair_chunk = pairs[pair_start:pair_stop]

            query_indices = pair_chunk[:, 0].astype(np.int64, copy=False) + query_start
            neighbor_indices = pair_chunk[:, 1].astype(np.int64, copy=False)

            # search() reports both orientations when the query is a subset of
            # the initialized coordinates.  Keeping neighbor > query exactly
            # reproduces self_search()'s unique unordered pairs and drops self.
            keep = neighbor_indices > query_indices

            if atom_cutoffs is not None:
                pair_cutoffs = 0.5 * (
                    atom_cutoffs[query_indices] + atom_cutoffs[neighbor_indices]
                )
                keep &= distances[pair_start:pair_stop] <= pair_cutoffs

            if np.any(keep):
                accumulator.accumulate(
                    query_indices[keep], neighbor_indices[keep]
                )

    return accumulator.finish_frame()


def _process_fast_neighbor_search_frame(
    accumulator,
    positions,
    box,
    cutoff,
    pbc,
    pair_chunk_size,
    atom_cutoffs=None,
):
    """Search a frame once, then aggregate its pairs in bounded-size chunks.

    Unlike the streaming path, this keeps the full frame's neighbor-pair array
    in memory.  It is substantially faster because ``self_search()`` evaluates
    each unordered pair only once.
    """

    positions = np.asarray(positions, dtype=np.float32)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (N, 3)")
    if positions.shape[0] != accumulator.layout.n_atoms:
        raise ValueError("positions and contact layout contain different atom counts")

    if atom_cutoffs is not None:
        atom_cutoffs = np.asarray(atom_cutoffs, dtype=np.float64)
        if atom_cutoffs.shape != (positions.shape[0],):
            raise ValueError("atom_cutoffs must contain one value per search atom")

    result = FastNS(cutoff, positions, box=box, pbc=pbc).self_search()
    pairs = result.get_pairs()
    distances = result.get_pair_distances() if atom_cutoffs is not None else None

    accumulator.begin_frame()
    for pair_start in range(0, pairs.shape[0], pair_chunk_size):
        pair_stop = min(pair_start + pair_chunk_size, pairs.shape[0])
        pair_chunk = pairs[pair_start:pair_stop]

        if atom_cutoffs is None:
            accumulator.accumulate(pair_chunk[:, 0], pair_chunk[:, 1])
            continue

        pair_cutoffs = atom_cutoffs[pair_chunk].mean(axis=1)
        keep = distances[pair_start:pair_stop] <= pair_cutoffs
        if np.any(keep):
            accumulator.accumulate(
                pair_chunk[keep, 0], pair_chunk[keep, 1]
            )

    return accumulator.finish_frame()


def _select_groups_and_chains(trajectory, args):
    if args.reference_group is None or args.selection_group is None:
        trajectory.index.print_all()

    if args.reference_group is not None:
        print(f"## Will use group {args.reference_group} as reference group.")
        reference_group, reference_group_name = trajectory.getSelection(
            f"group {args.reference_group}"
        )
    else:
        reference_group, reference_group_name = trajectory.getSelection_interactive(
            "reference group"
        )

    if args.selection_group is not None:
        print(f"## Will use group {args.selection_group} as selection group.")
        selection_group, selection_group_name = trajectory.getSelection(
            f"group {args.selection_group}"
        )
    else:
        selection_group, selection_group_name = trajectory.getSelection_interactive(
            "selection group (can be empty)"
        )

    ref_length = len(reference_group.indices)
    sel_length = len(selection_group.indices)
    print(f"## Reference group '{reference_group_name}' has {ref_length} atoms.")
    print(f"## Selection group '{selection_group_name}' has {sel_length} atoms.")

    if ref_length == 0:
        raise RuntimeError("Reference group has zero atoms.")

    has_selection = sel_length > 0
    if not has_selection:
        print("## WARNING: Selection group has zero atoms. Please make sure this is intended.")

    reference_chains = trajectory.index.splitch_indices(reference_group.indices)
    ref_chain_lengths = [len(chain) for chain in reference_chains]
    if not ref_chain_lengths or len(set(ref_chain_lengths)) != 1:
        raise RuntimeError("Chains in reference groups are not same in length.")
    print(
        f"## Processed {len(reference_chains)} chains of length "
        f"{ref_chain_lengths[0]} in reference group."
    )

    selection_chains = None
    if has_selection:
        selection_chains = trajectory.index.splitch_indices(selection_group.indices)
        sel_chain_lengths = [len(chain) for chain in selection_chains]
        if not sel_chain_lengths or len(set(sel_chain_lengths)) != 1:
            raise RuntimeError("Chains in selection groups are not same in length.")
        print(
            f"## Processed {len(selection_chains)} chains of length "
            f"{sel_chain_lengths[0]} in selection group."
        )

        if np.array_equal(np.asarray(reference_chains), np.asarray(selection_chains)):
            print("## WARNING: Reference and selection groups are identical.")

    return (
        reference_group,
        reference_group_name,
        reference_chains,
        selection_group,
        selection_group_name,
        selection_chains,
        has_selection,
    )


def _validate_selection_outputs(has_selection, map_outputs, time_outputs):
    if has_selection:
        return

    invalid = (set(map_outputs) | set(time_outputs)).intersection(
        (INTER_RS, INTER_SS, INTRA_S)
    )
    if invalid:
        raise RuntimeError(
            "Selection group is empty, but output files for the selection group are specified."
        )


def _write_maps(args, map_outputs, maps):
    for category, filename_raw in map_outputs.items():
        filename = validate_extension(filename_raw, args.output_type)
        matrix = maps[category]
        if args.output_type == "dat":
            np.savetxt(filename, matrix, fmt="%.6f", delimiter="\t")
        elif args.output_type == "xlsx":
            from pandas import DataFrame

            DataFrame(matrix).to_excel(filename, index=False, header=False)
        elif args.output_type == "xpm":
            write_xpm(matrix, filename)


def _category_legend(category, reference_name, selection_name):
    return {
        INTER_RS: f"inter-{reference_name}-{selection_name}",
        INTER_RR: f"inter-{reference_name}-{reference_name}",
        INTER_SS: f"inter-{selection_name}-{selection_name}",
        INTRA_R: f"intra-{reference_name}",
        INTRA_S: f"intra-{selection_name}",
    }[category]


def _write_time_series(
    time_outputs,
    time_list,
    time_series,
    reference_name,
    selection_name,
):
    for category, filename_raw in time_outputs.items():
        filename = validate_extension(filename_raw, "xvg")
        legend = _category_legend(category, reference_name, selection_name)
        ylabel = f"Contact number ({legend})"
        title = f"Time evolution of contact number ({legend})"
        write_xvg(
            filename,
            time_list,
            time_series[category],
            title=title,
            xlabel="Time (ns)",
            ylabel=ylabel,
            legends=[legend],
        )


def contact_ng(args):
    try:
        trajectory = trajectory_class(args.run_input, args.index, args.input)
    except Exception as exc:
        raise RuntimeError(
            f"An exception occurred when trying to open trajectory file {args.input}: {exc}"
        ) from exc

    # Preserve contact's coordinate transformation and PBC interaction.
    trajectory.Universe.trajectory.add_transformations(wrap(trajectory.Universe.atoms))
    start_frame, end_frame, interval_frame = trajectory.time2frame(
        args.start_time, args.end_time, args.delta_time
    )

    (
        reference_group,
        reference_group_name,
        reference_chains,
        selection_group,
        selection_group_name,
        selection_chains,
        has_selection,
    ) = _select_groups_and_chains(trajectory, args)

    map_outputs = _requested_outputs(args, MAP_OUTPUT_ATTRIBUTES)
    time_outputs = _requested_outputs(args, TIME_OUTPUT_ATTRIBUTES)
    _validate_selection_outputs(has_selection, map_outputs, time_outputs)
    if not map_outputs and not time_outputs:
        raise RuntimeError("No output files are specified, exiting.")

    if has_selection:
        search_atom_indices = np.union1d(
            np.asarray(reference_group.indices, dtype=np.int64),
            np.asarray(selection_group.indices, dtype=np.int64),
        )
    else:
        search_atom_indices = np.unique(
            np.asarray(reference_group.indices, dtype=np.int64)
        )
    search_atoms = trajectory.Universe.atoms[search_atom_indices]

    layout = build_contact_layout(
        search_atom_indices,
        reference_chains,
        selection_chains if has_selection else None,
    )
    accumulator = StreamingContactAccumulator(
        layout,
        map_categories=map_outputs,
        time_categories=time_outputs,
        intra_average=args.intra_average,
        remove_diagonal=args.remove_diagonal,
    )

    if args.cutoff_scheme == "global":
        print(f"## Will use a global cutoff of {args.cutoff} nm.")
        cutoff = args.cutoff * 10.0
        search_atom_cutoffs = None
    elif args.cutoff_scheme == "residue":
        print(
            "## Will use a residue responsive cutoff with a sigma multiplier "
            f"of {args.cutoff_multiplier}."
        )
        cutoff_vector = np.asarray(
            [args.cutoff_multiplier * sigma * 10.0 for sigma in trajectory.sigmas],
            dtype=np.float64,
        )
        if cutoff_vector.size <= int(search_atom_indices[-1]):
            raise RuntimeError("Sigma data does not cover every selected atom.")
        search_atom_cutoffs = cutoff_vector[search_atom_indices]
        cutoff = float(search_atom_cutoffs.max())
    else:
        raise RuntimeError(f"Unknown cutoff scheme {args.cutoff_scheme}.")

    print("## We will now start calculating.")
    if map_outputs:
        print("## Contact maps will be calculated and outputted.")
    if time_outputs:
        print("## Time evolution of contact numbers will be calculated and outputted.")
    print(f"## Neighborhood search will be performed with cutoff {cutoff / 10.0} nm.")
    if args.search_mode == "fast":
        print(
            "## Fast search will store all neighbor pairs from one frame and use "
            f"{args.pair_chunk_size} pairs per aggregation chunk."
        )
    else:
        print(
            f"## Streaming search uses {args.search_batch_size} query atoms per batch "
            f"and {args.pair_chunk_size} pairs per aggregation chunk."
        )
    print(
        f"## Processing frames from {start_frame} to {end_frame} "
        f"with interval {interval_frame}."
    )

    time_list = []
    time_series = {category: [] for category in time_outputs}
    frames = trajectory.Universe.trajectory[start_frame:end_frame:interval_frame]

    for ts in tqdm(frames):
        time_list.append(ts.time / 1000.0)
        frame_kwargs = {
            "accumulator": accumulator,
            "positions": search_atoms.positions,
            "box": ts.dimensions,
            "cutoff": cutoff,
            "pbc": args.treat_pbc,
            "pair_chunk_size": args.pair_chunk_size,
            "atom_cutoffs": search_atom_cutoffs,
        }
        if args.search_mode == "fast":
            frame_counts = _process_fast_neighbor_search_frame(**frame_kwargs)
        else:
            frame_counts = _process_neighbor_search_frame(
                search_batch_size=args.search_batch_size,
                **frame_kwargs,
            )
        for category in time_outputs:
            time_series[category].append(frame_counts[category])

    if not time_list:
        raise RuntimeError("No trajectory frames matched the requested time range.")
    print("## Calculation completed.")

    if map_outputs:
        _write_maps(args, map_outputs, accumulator.averaged_maps())
    if time_outputs:
        _write_time_series(
            time_outputs,
            time_list,
            time_series,
            reference_group_name,
            selection_group_name,
        )


contact_ng_commands = single_command(
    "contact.ng", getargs_contact_ng, contact_ng, desc
)
