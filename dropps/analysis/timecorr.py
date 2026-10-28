# residue-resolved time correlation analysis in DROPPS package by Yiming Tang @ Fudan

from argparse import SUPPRESS

from dropps.share.argument_parser import ArgumentParser
import numpy as np
from tqdm import tqdm

from dropps.fileio.filename_control import validate_extension
from dropps.fileio.xpm_reader import write_xpm

from MDAnalysis.transformations import wrap
from MDAnalysis.lib.nsgrid import FastNS

from dropps.share.trajectory import trajectory_class
from dropps.analysis.contact import compute_residue_contact_maps_fixed, remove_near_diagonal
from dropps.analysis.time_core import uniform_frame_interval_ns

import warnings
from Bio import BiopythonDeprecationWarning
warnings.filterwarnings("ignore", category=BiopythonDeprecationWarning)

prog = "timecorr"
desc = "Calculate residue-contact correlation functions and lifetime matrices."


def _stack_series(series_list):
    if len(series_list) == 0:
        return None
    return np.stack(series_list, axis=0).astype(np.float64, copy=False)


def _autocorrelation_tensor(series, max_lag_frames=None):
    series = np.asarray(series, dtype=np.float64)
    if series.ndim != 3:
        raise ValueError("series must have shape (n_frames, n_row, n_col)")

    n_frames = series.shape[0]
    if n_frames == 0:
        raise ValueError("series must contain at least one frame")

    if max_lag_frames is None:
        n_lag = n_frames
    else:
        n_lag = min(max(1, int(max_lag_frames)), n_frames)

    nfft = 1
    while nfft < (2 * n_frames - 1):
        nfft *= 2

    spectrum = np.fft.rfft(series, n=nfft, axis=0)
    corr = np.fft.irfft(spectrum * np.conjugate(spectrum), n=nfft, axis=0)[:n_lag].real

    normalization = np.arange(n_frames, n_frames - n_lag, -1, dtype=np.float64).reshape(-1, 1, 1)
    corr /= normalization

    zero_lag = corr[0]
    corr = np.divide(
        corr,
        zero_lag[None, :, :],
        out=np.zeros_like(corr),
        where=zero_lag[None, :, :] > 0,
    )
    return corr


def _lifetime_matrix(corr, lag_times, threshold):
    if corr.ndim != 3:
        raise ValueError("corr must have shape (n_lag, n_row, n_col)")
    if len(lag_times) != corr.shape[0]:
        raise ValueError("lag_times length does not match corr")

    zero_lag = corr[0]
    lifetime = np.zeros_like(zero_lag, dtype=np.float64)

    if corr.shape[0] == 1:
        lifetime[zero_lag > 0] = lag_times[0]
        return lifetime, int(np.count_nonzero(zero_lag > 0))

    observed = zero_lag > 0
    below = corr[1:] < threshold
    crossed = below.any(axis=0) & observed

    if np.any(crossed):
        first_cross = np.argmax(below, axis=0) + 1
        row_idx, col_idx = np.where(crossed)
        cross_idx = first_cross[row_idx, col_idx]

        prev_idx = cross_idx - 1
        prev_corr = corr[prev_idx, row_idx, col_idx]
        next_corr = corr[cross_idx, row_idx, col_idx]
        prev_time = lag_times[prev_idx]
        next_time = lag_times[cross_idx]

        denominator = next_corr - prev_corr
        interpolated = np.where(
            denominator == 0,
            next_time,
            prev_time + (threshold - prev_corr) * (next_time - prev_time) / denominator,
        )
        lifetime[row_idx, col_idx] = interpolated

    uncrossed = observed & (~crossed)
    if np.any(uncrossed):
        lifetime[uncrossed] = lag_times[-1]

    return lifetime, int(np.count_nonzero(uncrossed))


def _write_matrix(filename_raw, matrix, output_type):
    filename = validate_extension(filename_raw, output_type)
    if output_type == "dat":
        np.savetxt(filename, matrix, fmt="%.6f", delimiter="\t")
    elif output_type == "xlsx":
        from pandas import DataFrame
        DataFrame(matrix).to_excel(filename, index=False, header=False)
    elif output_type == "xpm":
        write_xpm(matrix, filename)
    else:
        raise ValueError(f"Unknown output type {output_type}")
    print(f"## Matrix written to {filename}.")


def _write_corr_archive(filename_raw, lag_times, corr):
    filename = validate_extension(filename_raw, "npz")
    np.savez_compressed(filename, lag_times=lag_times, correlation=corr)
    print(f"## Time correlation archive written to {filename}.")


def _average_corr_matrix(corr):
    if corr.ndim != 3:
        raise ValueError("corr must have shape (n_lag, n_row, n_col)")
    return corr.mean(axis=0)


def getargs_timecorr(argv):

    parser = ArgumentParser(
        prog=prog,
        description=desc,
        epilog=(
            "At least one correlation output (-ocrs, -ocrr, -ocss, -ocr, or "
            "-ocs) is required in non-verbose mode."
        ),
    )

    parser.add_argument('-s', '--run-input', type=str, required=True,
                        help="Input DROPPS run file (.tpr) containing the system and simulation settings.")

    parser.add_argument('-f', '--input', type=str, required=True,
                        help="Input trajectory file (.xtc).")

    parser.add_argument('-n', '--index', type=str, required=False,
                        help="Optional index file (.ndx) defining additional atom groups.")

    parser.add_argument('-ref', '--reference-group', type=str,
                        help="Reference index group forming the x axis of each residue-pair matrix.")

    parser.add_argument('-sel', '--selection-group', type=str,
                        help="Optional selection index group forming the y axis of cross-group matrices.")

    parser.add_argument('-cs', '-k', '--cutoff-scheme', type=str, choices=["global", "residue"], required=True,
                        help="Contact-cutoff scheme: one global distance or residue-specific distances.")

    parser.add_argument('-c', '--cutoff', type=float, default=0.7,
                        help="Global contact cutoff, in nm.")

    parser.add_argument('-cm', '--cutoff-multiplier', type=float, default=1.2,
                        help="Multiplier applied to residue sigma values in the residue cutoff scheme.")

    parser.add_argument('-rd', '--remove-diagonal', type=int, default=2,
                        help="Number of main and neighboring diagonals removed from intra-chain matrices.")

    parser.add_argument('-thr', '--threshold', type=float, default=0.5,
                        help="Normalized-correlation threshold used to estimate contact lifetimes.")

    parser.add_argument('-ml', '--max-lag', type=float,
                        help="Maximum correlation lag, in ns; if omitted, use all analyzed frames.")

    parser.add_argument('-b', '--start-time', type=float,
                        help="First trajectory time to analyze, in ns.")

    parser.add_argument('-e', '--end-time', type=float,
                        help="Last trajectory time to analyze, in ns.")

    parser.add_argument('-dt', '--delta-time', type=float,
                        help="Approximate interval between analyzed frames, in ns.")

    parser.add_argument('-pbc', '--treat-pbc', action='store_true', default=False,
                        help="Apply periodic-boundary distances during contact calculation.")

    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help="Write lag-resolved correlation tensors and lifetime matrices instead of only lag-averaged matrices.")

    parser.add_argument('-ot', '--average-output-type', dest='average_output_type',
                        choices=["dat", "xlsx"], default="dat",
                        help="File format for non-verbose averaged correlation matrices.")
    parser.add_argument('--otpye', dest='average_output_type', choices=["dat", "xlsx"],
                        default=SUPPRESS, help=SUPPRESS)

    parser.add_argument('-ocrs', '--output-corr-inter-reference-selection', type=str,
                        help="Output inter-chain reference-selection correlations; verbose mode writes .npz, otherwise --average-output-type is used.")

    parser.add_argument('-ocrr', '--output-corr-inter-reference-reference', type=str,
                        help="Output inter-chain reference-reference correlations; verbose mode writes .npz, otherwise --average-output-type is used.")

    parser.add_argument('-ocss', '--output-corr-inter-selection-selection', type=str,
                        help="Output inter-chain selection-selection correlations; verbose mode writes .npz, otherwise --average-output-type is used.")

    parser.add_argument('-ocr', '--output-corr-intra-reference', type=str,
                        help="Output intra-chain reference correlations; verbose mode writes .npz, otherwise --average-output-type is used.")

    parser.add_argument('-ocs', '--output-corr-intra-selection', type=str,
                        help="Output intra-chain selection correlations; verbose mode writes .npz, otherwise --average-output-type is used.")

    parser.add_argument('-olrs', '--output-lifetime-inter-reference-selection', type=str,
                        help="Output inter-chain reference-selection lifetime matrix in verbose mode.")

    parser.add_argument('-olrr', '--output-lifetime-inter-reference-reference', type=str,
                        help="Output inter-chain reference-reference lifetime matrix in verbose mode.")

    parser.add_argument('-olss', '--output-lifetime-inter-selection-selection', type=str,
                        help="Output inter-chain selection-selection lifetime matrix in verbose mode.")

    parser.add_argument('-olr', '--output-lifetime-intra-reference', type=str,
                        help="Output intra-chain reference lifetime matrix in verbose mode.")

    parser.add_argument('-ols', '--output-lifetime-intra-selection', type=str,
                        help="Output intra-chain selection lifetime matrix in verbose mode.")

    parser.add_argument('-otype', '--output-type', type=str, choices=["dat", "xpm", "xlsx"], default="dat",
                        help="File format for verbose lifetime-matrix outputs.")

    args = parser.parse_args(argv)
    return args


def timecorr(args):

    if not 0.0 < args.threshold <= 1.0:
        raise ValueError("correlation threshold must be in (0, 1]")
    if args.remove_diagonal < 0:
        raise ValueError("remove-diagonal must be non-negative")
    if args.cutoff_scheme == "global" and args.cutoff <= 0.0:
        raise ValueError("global cutoff must be positive")
    if args.cutoff_scheme == "residue" and args.cutoff_multiplier <= 0.0:
        raise ValueError("residue cutoff multiplier must be positive")

    try:
        trajectory = trajectory_class(args.run_input, args.index, args.input)
    except Exception as exc:
        print("## An exception occurred when trying to open trajectory file %s." % args.input)
        print(f"## Root cause: {exc}")
        quit()

    trajectory.Universe.trajectory.add_transformations(wrap(trajectory.Universe.atoms))

    start_frame, end_frame, interval_frame = trajectory.time2frame(args.start_time, args.end_time, args.delta_time)

    if args.reference_group is None or args.selection_group is None:
        trajectory.index.print_all()

    if args.reference_group is not None:
        print(f"## Will use group {args.reference_group} as reference group.")
        reference_group, reference_group_name = trajectory.getSelection(f"group {args.reference_group}")
    else:
        reference_group, reference_group_name = trajectory.getSelection_interactive("reference group")

    if args.selection_group is not None:
        print(f"## Will use group {args.selection_group} as selection group.")
        selection_group, selection_group_name = trajectory.getSelection(f"group {args.selection_group}")
    else:
        selection_group, selection_group_name = trajectory.getSelection_interactive("selection group (can be empty)")

    ref_length = len(reference_group.indices)
    sel_length = len(selection_group.indices)

    print(f"## Reference group '{reference_group_name}' has {ref_length} atoms.")
    print(f"## Selection group '{selection_group_name}' has {sel_length} atoms.")

    if ref_length == 0:
        print("ERROR: Reference group has zero atoms.")
        quit()
    if sel_length == 0:
        print("## WARNING: Selection group has zero atoms. Please make sure this is intended.")
        has_selection = False
    else:
        has_selection = True

    reference_chains = trajectory.index.splitch_indices(reference_group.indices)
    chain_length_list_reference = [len(chain) for chain in reference_chains]
    if len(set(chain_length_list_reference)) != 1:
        print("ERROR: Chains in reference groups are not same in length.")
        quit()
    chain_length_reference = chain_length_list_reference[0]
    print(f"## Processed {len(reference_chains)} chains of length {chain_length_reference} in reference group.")

    if has_selection:
        selection_chains = trajectory.index.splitch_indices(selection_group.indices)
        chain_length_list_selection = [len(chain) for chain in selection_chains]
        if len(set(chain_length_list_selection)) != 1:
            print("ERROR: Chains in selection groups are not same in length.")
            quit()
        chain_length_selection = chain_length_list_selection[0]
        print(f"## Processed {len(selection_chains)} chains of length {chain_length_selection} in selection group.")

        c1 = np.array(reference_chains)
        c2 = np.array(selection_chains)
        if np.array_equal(c1, c2):
            print("## WARNING: Reference and selection groups are identical.")
    else:
        selection_chains = None

    output_map = {
        "inter_rs": (
            args.output_corr_inter_reference_selection,
            args.output_lifetime_inter_reference_selection,
            f"inter-{reference_group_name}-{selection_group_name}",
        ),
        "inter_rr": (
            args.output_corr_inter_reference_reference,
            args.output_lifetime_inter_reference_reference,
            f"inter-{reference_group_name}-{reference_group_name}",
        ),
        "inter_ss": (
            args.output_corr_inter_selection_selection,
            args.output_lifetime_inter_selection_selection,
            f"inter-{selection_group_name}-{selection_group_name}",
        ) if has_selection else None,
        "intra_r": (
            args.output_corr_intra_reference,
            args.output_lifetime_intra_reference,
            f"intra-{reference_group_name}",
        ),
        "intra_s": (
            args.output_corr_intra_selection,
            args.output_lifetime_intra_selection,
            f"intra-{selection_group_name}",
        ) if has_selection else None,
    }

    requested_outputs = {}
    for category, outputs in output_map.items():
        if outputs is None:
            requested_outputs[category] = False
            continue
        output_corr, output_lifetime, _ = outputs
        if args.verbose:
            requested_outputs[category] = any((output_corr is not None, output_lifetime is not None))
        else:
            requested_outputs[category] = output_corr is not None

    if not has_selection:
        if any((
            requested_outputs["inter_rs"],
            requested_outputs["inter_ss"],
            requested_outputs["intra_s"],
        )):
            print("ERROR: Selection group is empty, but output files for selection group are specified.")
            quit()

    if not any(requested_outputs.values()):
        if args.verbose:
            print("## ERROR: No output files are specified, exiting.")
        else:
            print("## ERROR: Non-verbose mode requires at least one correlation output file (-ocrs/-ocrr/-ocss/-ocr/-ocs).")
        quit()

    if args.cutoff_scheme == "global":
        print(f"## Will use a glocal cutoff of {args.cutoff} nm.")
        cutoff = args.cutoff * 10.0
        cutoff_vector = None
    elif args.cutoff_scheme == "residue":
        print(f"## Will use a residue responsive cutoff will a sigma multiplier of {args.cutoff_multiplier}")
        cutoff_vector = np.array([args.cutoff_multiplier * sigma * 10.0 for sigma in trajectory.sigmas])
        cutoff = cutoff_vector.max()
    else:
        print(f"ERROR: Unknown cutoff scheme {args.cutoff_scheme}.")
        quit()

    series_dict = {key: [] for key, value in requested_outputs.items() if value}

    if has_selection:
        search_atom_indices = np.sort(np.asarray(list(set(reference_group.indices).union(set(selection_group.indices))), dtype=np.int64))
    else:
        search_atom_indices = np.sort(np.asarray(reference_group.indices, dtype=np.int64))
    search_atoms = trajectory.Universe.atoms[search_atom_indices]

    time_list = []

    print(f"## Neighborhood search will be performed with cutoff {cutoff / 10.0} nm.")
    print(f"## Processing frames from {start_frame} to {end_frame} with interval {interval_frame}.")

    for ts in tqdm(trajectory.Universe.trajectory[start_frame:end_frame + 1:interval_frame]):
        time_list.append(ts.time / 1000.0)

        pos = search_atoms.positions
        box = ts.dimensions
        ns = FastNS(cutoff, pos, box=box, pbc=args.treat_pbc)
        search_result = ns.self_search()

        pairs = search_result.get_pairs()
        pairs_atomid = search_atom_indices[pairs]

        if args.cutoff_scheme == "residue":
            cutoffs = cutoff_vector[pairs_atomid].mean(axis=1)
            pair_distances = search_result.get_pair_distances()
            mask = pair_distances <= cutoffs
            pairs_atomid = pairs_atomid[mask]

        maps = compute_residue_contact_maps_fixed(
            pairs_atomid,
            reference_chains,
            selection_chains if has_selection else None,
            undirected=True,
            unique_pairs=False,
            drop_self_pairs=True,
            dtype=np.int64,
        )

        ref_intra = maps["ref_intra"]
        ref_ref = maps["ref_ref_inter"]
        n_ref = ref_intra.shape[0]

        if requested_outputs.get("intra_r"):
            intra_r = remove_near_diagonal(ref_intra.sum(axis=0), args.remove_diagonal)
            series_dict["intra_r"].append(intra_r)

        if requested_outputs.get("inter_rr"):
            mask_ref = ~np.eye(n_ref, dtype=bool)
            blocks_ref_ref = ref_ref[mask_ref]
            inter_rr = (
                blocks_ref_ref.sum(axis=0)
                if blocks_ref_ref.size
                else np.zeros((chain_length_reference, chain_length_reference), dtype=np.int64)
            )
            series_dict["inter_rr"].append(inter_rr)

        if has_selection:
            sel_intra = maps["sel_intra"]
            ref_sel = maps["ref_sel_inter"]
            sel_sel = maps["sel_sel_inter"]
            n_sel = sel_intra.shape[0]

            if requested_outputs.get("intra_s"):
                intra_s = remove_near_diagonal(sel_intra.sum(axis=0), args.remove_diagonal)
                series_dict["intra_s"].append(intra_s)

            if requested_outputs.get("inter_rs"):
                inter_rs = ref_sel.sum(axis=(0, 1))
                series_dict["inter_rs"].append(inter_rs)

            if requested_outputs.get("inter_ss"):
                mask_sel = ~np.eye(n_sel, dtype=bool)
                blocks_sel_sel = sel_sel[mask_sel]
                inter_ss = (
                    blocks_sel_sel.sum(axis=0)
                    if blocks_sel_sel.size
                    else np.zeros((chain_length_selection, chain_length_selection), dtype=np.int64)
                )
                series_dict["inter_ss"].append(inter_ss)

    print("## Time-series collection completed.")

    frame_dt_ns = uniform_frame_interval_ns(time_list)
    n_frames = len(time_list)
    if args.max_lag is None:
        max_lag_frames = n_frames
    else:
        if args.max_lag <= 0:
            print("ERROR: Maximum lag time must be positive.")
            quit()
        max_lag_frames = max(1, int(np.floor(args.max_lag / frame_dt_ns)) + 1)
    max_lag_frames = min(max_lag_frames, n_frames)
    lag_times = np.arange(max_lag_frames, dtype=np.float64) * frame_dt_ns

    print(f"## Correlation functions will be evaluated for {max_lag_frames} lag points.")

    for category, series_list in series_dict.items():
        series = _stack_series(series_list)
        corr = _autocorrelation_tensor(series, max_lag_frames=max_lag_frames)

        output_corr, output_lifetime, legend = output_map[category]
        if args.verbose:
            lifetime, censored_count = _lifetime_matrix(corr, lag_times, args.threshold)
            if output_corr is not None:
                _write_corr_archive(output_corr, lag_times, corr)
            if output_lifetime is not None:
                _write_matrix(output_lifetime, lifetime, args.output_type)

            active_pairs = int(np.count_nonzero(corr[0] > 0))
            print(f"## Completed {legend}: {active_pairs} residue pairs observed.")
            if censored_count > 0:
                print(f"## WARNING: {censored_count} residue pairs in {legend} did not decay below threshold {args.threshold} within the sampled lag window.")
        else:
            average_corr = _average_corr_matrix(corr)
            _write_matrix(output_corr, average_corr, args.average_output_type)

            active_pairs = int(np.count_nonzero(corr[0] > 0))
            print(f"## Completed {legend}: {active_pairs} residue pairs observed in the lag-averaged matrix.")


from dropps.share.command_class import single_command
timecorr_commands = single_command("timecorr", getargs_timecorr, timecorr, desc)
