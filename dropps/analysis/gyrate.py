# gyrate tool in DROPPS package by Yiming Tang @ Fudan
# Development started on July 18 2025

from dropps.share.argument_parser import ArgumentParser
from dropps.share.command_class import single_command
from dropps.share.trajectory import trajectory_class
from tqdm import tqdm
import numpy as np

from dropps.fileio.xvg_reader import write_xvg
from dropps.fileio.filename_control import validate_extension

import warnings
from Bio import BiopythonDeprecationWarning

warnings.filterwarnings("ignore", category=BiopythonDeprecationWarning)

from MDAnalysis.transformations import unwrap  # noqa: E402 - suppress import warning above

prog = "gyrate"
desc = "Calculate chain radii of gyration and optional distributions."


def radius_of_gyration(positions, masses):
    com = np.average(positions, axis=0, weights=masses)
    squared_distances = np.sum((positions - com) ** 2, axis=1)
    return np.sqrt(np.average(squared_distances, weights=masses))


def getargs_gyrate(argv):
    parser = ArgumentParser(
        prog=prog,
        description=desc,
        epilog="At least one output option (-ov, -oa, or -oh) is required.",
    )

    parser.add_argument(
        "-s",
        "--run-input",
        type=str,
        required=True,
        help="Input DROPPS run file (.tpr) containing the system and simulation settings.",
    )

    parser.add_argument(
        "-f", "--input", type=str, required=True, help="Input trajectory file (.xtc)."
    )

    parser.add_argument(
        "-n",
        "--index",
        type=str,
        required=False,
        help="Optional index file (.ndx) defining additional atom groups.",
    )

    parser.add_argument(
        "-sel",
        "--selection-calculate",
        type=int,
        nargs="+",
        help="Index groups whose chain radii of gyration are calculated; if omitted, prompt interactively.",
    )

    parser.add_argument(
        "-b", "--start-time", type=int, help="First trajectory time to analyze, in ns."
    )

    parser.add_argument(
        "-e", "--end-time", type=int, help="Last trajectory time to analyze, in ns."
    )

    parser.add_argument(
        "-dt",
        "--delta-time",
        type=int,
        help="Approximate interval between analyzed frames, in ns.",
    )

    parser.add_argument(
        "-pbc",
        "--treat-pbc",
        action="store_true",
        default=False,
        help="Unwrap molecules across periodic boundaries before analysis.",
    )

    parser.add_argument(
        "-ov",
        "--output-verbose",
        type=str,
        required=False,
        help="Output per-chain radius-of-gyration time series (.xvg); the extension is added if omitted.",
    )

    parser.add_argument(
        "-oa",
        "--output-average",
        type=str,
        required=False,
        help="Output group-averaged radius-of-gyration time series (.xvg); the extension is added if omitted.",
    )

    parser.add_argument(
        "-oh",
        "--output-histogram",
        type=str,
        required=False,
        help="Output radius-of-gyration distribution (.xvg); the extension is added if omitted.",
    )

    parser.add_argument(
        "-bw",
        "--bin-width",
        type=float,
        default=0.1,
        help="Histogram bin width, in nm.",
    )

    args = parser.parse_args(argv)

    return args


def gyrate(args):
    if not (args.output_average or args.output_histogram or args.output_verbose):
        print("ERROR: No output file specified.")
        quit()

    # load trajectory into memory

    try:
        trajectory = trajectory_class(args.run_input, args.index, args.input)
    except Exception as exc:
        print(
            "## An exception occurred when trying to open trajectory file %s."
            % args.input
        )
        print(f"## Root cause: {exc}")
        quit()

    # We treat time for analysis and generate frame for analysis

    start_frame, end_frame, interval_frame = trajectory.time2frame(
        args.start_time, args.end_time, args.delta_time
    )

    if args.selection_calculate is not None:
        group_ids = ",".join(str(i) for i in args.selection_calculate)
        print(f"## Will use groups {group_ids} for Rg calculations.")
        selections = [
            trajectory.getSelection(f"group {gid}")[0]
            for gid in args.selection_calculate
        ]
        selection_names = [f"group{i}" for i in args.selection_calculate]
    else:
        trajectory.index.print_all()
        selections, selection_names = trajectory.getSelection_interactive_multiple()

    if args.treat_pbc is True:
        trajectory.Universe.trajectory.add_transformations(
            unwrap(trajectory.Universe.atoms)
        )

    rg_lists = [list() for i in range(len(selections))]
    time_list = list()

    print(f"## Will calculate radius of gyration for {len(selections)} groups.")
    print("## Start of radius of gyration calculations.")

    for ts in tqdm(
        trajectory.Universe.trajectory[start_frame : end_frame + 1 : interval_frame]
    ):
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

        write_xvg(
            verbose_filename,
            time_list,
            rg_matrix,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            legends=legends,
        )

    if args.output_average is not None:
        average_filename = validate_extension(args.output_average, "xvg")
        xlabel = "Time (ns)"
        ylabel = "Averaged radius of gyration (nm)"
        title = "Averaged radius of gyration"
        legends = ["Averaged"]

        write_xvg(
            average_filename,
            time_list,
            rg_matrix.mean(axis=0),
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            legends=legends,
        )

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

        write_xvg(
            histogram_filename,
            bin_edges[:-1],
            hist_pdf,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
        )


gyrate_commands = single_command(prog, getargs_gyrate, gyrate, desc)
