"""One-dimensional density profiles with periodic slab centering."""

from __future__ import annotations

import numpy as np
from tqdm import tqdm

from dropps.analysis.coexistence_core import (
    center_periodic_fractions,
    centering_schedule,
    histogram_density,
)
from dropps.analysis.density_core import (
    ANGSTROM_TO_NM,
    AXIS_TO_INDEX,
    CHARGE_PER_NM3_TO_E_MOL_PER_ML,
    DALTON_PER_NM3_TO_MG_PER_ML,
    DEFAULT_DENSITY_BIN_WIDTH_NM,
    box_geometry_nm,
    fractional_positions,
    profile_center_shift,
    select_frame_indices,
    threshold_profile_center,
)
from dropps.fileio.xvg_reader import write_xvg
from dropps.share.argument_parser import ArgumentParser
from dropps.share.command_class import single_command
from dropps.share.trajectory import trajectory_class


prog = "density"
desc = "Calculate a one-dimensional mass- or charge-density profile."


def _frame_windows(start_frame, end_frame, interval_frame):
    """Return legacy inclusive windows with exclusive stop indices.

    Kept for callers that imported the helper.  The density command now samples
    frames directly by physical time and no longer averages unequal windows.
    """

    if end_frame < start_frame:
        raise ValueError("end_frame must not be earlier than start_frame.")
    if interval_frame < 1:
        raise ValueError("interval_frame must be at least 1.")

    windows = []
    window_start = start_frame
    while window_start <= end_frame:
        window_stop = min(window_start + interval_frame, end_frame + 1)
        windows.append((window_start, window_stop))
        window_start = window_stop
    return windows


def shift_density_center(
    density_profiles,
    dense_phase_threshold,
    density_profile_for_center,
):
    """Roll periodic profiles using the largest thresholded dense region."""

    profiles = np.asarray(density_profiles, dtype=float)
    if profiles.ndim == 1:
        profiles = profiles[np.newaxis, :]
    if profiles.ndim != 2:
        raise ValueError("Density profiles must be a 1D or 2D array.")
    center, _ = threshold_profile_center(
        density_profile_for_center,
        dense_phase_threshold,
    )
    shift = profile_center_shift(profiles.shape[1], center)
    return [np.roll(profile, shift) for profile in profiles]


def getargs_density(argv):
    parser = ArgumentParser(prog=prog, description=desc)
    parser.add_argument(
        "-s",
        "--run-input",
        required=True,
        help="Input DROPPS run file (.tpr) containing the system and simulation settings.",
    )
    parser.add_argument(
        "-f",
        "--input",
        required=True,
        help="Input trajectory file (.xtc).",
    )
    parser.add_argument(
        "-n",
        "--index",
        help="Optional index file (.ndx) defining additional atom groups.",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output density-profile file (.xvg); the extension is added if omitted.",
    )
    parser.add_argument(
        "-x",
        "--axis",
        choices=("x", "y", "z"),
        default="z",
        help="Axis along which to calculate the density profile.",
    )
    parser.add_argument(
        "-tp",
        "--type",
        choices=("mass", "charge"),
        default="mass",
        help="Density quantity to calculate.",
    )
    parser.add_argument(
        "-b",
        "--start-time",
        type=float,
        help="First trajectory time to analyze, in ns.",
    )
    parser.add_argument(
        "-e",
        "--end-time",
        type=float,
        help="Last trajectory time to analyze, in ns.",
    )
    parser.add_argument(
        "-dt",
        "--delta-time",
        type=float,
        help="Approximate interval between analyzed frames, in ns.",
    )
    parser.add_argument(
        "--bin-width",
        type=float,
        default=DEFAULT_DENSITY_BIN_WIDTH_NM,
        help="Requested density-bin width, in nm.",
    )
    parser.add_argument(
        "-selfit",
        "--selection-fit",
        type=int,
        help="Index group used to locate the dense phase; if omitted, prompt interactively.",
    )
    parser.add_argument(
        "-sel",
        "--selection-calculate",
        type=int,
        nargs="+",
        help="Index groups whose density profiles are written; if omitted, prompt interactively.",
    )
    parser.add_argument(
        "--center-mode",
        choices=("frame", "block", "none"),
        default="frame",
        help=(
            "Slab centering strategy: frame aligns each analyzed frame; block "
            "uses one center per contiguous --blocks "
            "segment; none applies no shift."
        ),
    )
    parser.add_argument(
        "--blocks",
        type=int,
        default=5,
        help="Number of contiguous groups used by --center-mode block.",
    )
    parser.add_argument(
        "-nc",
        "--no-center",
        action="store_true",
        help="Compatibility alias for --center-mode none.",
    )
    parser.add_argument(
        "-t",
        "--dense-phase-threshold",
        default=0.5,
        type=float,
        help="Dense-phase threshold as a fraction of the maximum reference density.",
    )
    return parser.parse_args(argv)


def _select_density_groups(trajectory, args):
    if args.selection_fit is None or args.selection_calculate is None:
        trajectory.index.print_all()

    if args.selection_fit is None:
        fit_group, fit_group_name = trajectory.getSelection_interactive(
            "group for dense phase determination"
        )
    else:
        fit_group, fit_group_name = trajectory.getSelection(
            f"group {args.selection_fit}",
            "dense phase determination",
        )
    if len(fit_group) == 0:
        raise ValueError("Dense-phase reference selection is empty.")

    if args.selection_calculate is None:
        density_groups, density_group_names = (
            trajectory.getSelection_interactive_multiple(
                "groups for density calculation"
            )
        )
    else:
        selected = [
            trajectory.getSelection(
                f"group {group_id}",
                "density calculation",
            )
            for group_id in args.selection_calculate
        ]
        density_groups = [group for group, _ in selected]
        density_group_names = [name for _, name in selected]
    if not density_groups:
        raise ValueError("At least one density calculation group is required.")
    if any(len(group) == 0 for group in density_groups):
        raise ValueError("Density calculation selections must not be empty.")
    return fit_group, fit_group_name, density_groups, density_group_names


def density(args):
    if not np.isfinite(args.bin_width) or args.bin_width <= 0.0:
        raise ValueError("Bin width must be positive and finite.")
    if args.blocks <= 0:
        raise ValueError("Number of centering blocks must be positive.")
    if (
        not np.isfinite(args.dense_phase_threshold)
        or not 0.0 < args.dense_phase_threshold < 1.0
    ):
        raise ValueError("Dense-phase threshold must be finite and between 0 and 1.")

    center_mode = "none" if args.no_center else args.center_mode
    output_file_name = (
        args.output if args.output.endswith(".xvg") else args.output + ".xvg"
    )
    trajectory = trajectory_class(args.run_input, args.index, args.input)
    fit_group, _, density_groups, density_group_names = _select_density_groups(
        trajectory,
        args,
    )

    reference_indices = np.asarray(fit_group.indices, dtype=int)
    reference_weights = np.asarray(fit_group.masses, dtype=float)
    if (
        not np.all(np.isfinite(reference_weights))
        or np.any(reference_weights < 0.0)
        or reference_weights.sum() <= 0.0
    ):
        raise ValueError(
            "Dense-phase reference group must have positive finite masses."
        )

    group_indices = [np.asarray(group.indices, dtype=int) for group in density_groups]
    if args.type == "mass":
        group_weights = [
            np.asarray(group.masses, dtype=float) for group in density_groups
        ]
        conversion = DALTON_PER_NM3_TO_MG_PER_ML
        title = "Mass density"
        ylabel = "Mass density (mg/mL)"
    else:
        group_weights = [
            np.asarray(group.charges, dtype=float) for group in density_groups
        ]
        conversion = CHARGE_PER_NM3_TO_E_MOL_PER_ML
        title = "Charge density"
        ylabel = "Charge density (e mol/mL)"
    if any(not np.all(np.isfinite(weights)) for weights in group_weights):
        raise ValueError("Density weights must be finite.")

    analyzed_trajectory = trajectory.Universe.trajectory
    frame_indices, actual_interval_ns = select_frame_indices(
        analyzed_trajectory,
        args.start_time,
        args.end_time,
        args.delta_time,
    )
    first_timestep = analyzed_trajectory[frame_indices[0]]
    _, first_lengths_nm, _ = box_geometry_nm(first_timestep)
    axis_index = AXIS_TO_INDEX[args.axis]
    bins = max(1, int(np.ceil(first_lengths_nm[axis_index] / args.bin_width)))

    if center_mode == "none":
        frame_centers = np.zeros(len(frame_indices), dtype=float)
        schedule = centering_schedule(
            frame_centers,
            np.zeros(len(frame_indices), dtype=float),
            mode="none",
            blocks=args.blocks,
        )
    else:
        frame_centers = []
        frame_center_orders = []
        for frame_index in tqdm(frame_indices, desc="## Locating dense phase"):
            timestep = analyzed_trajectory[frame_index]
            box_nm, _, volume_nm3 = box_geometry_nm(timestep)
            positions_nm = np.asarray(timestep.positions, dtype=float) * ANGSTROM_TO_NM
            fractions = fractional_positions(positions_nm, box_nm)
            reference_profile = histogram_density(
                fractions[reference_indices, axis_index],
                reference_weights,
                volume_nm3,
                bins,
            )
            center, order = threshold_profile_center(
                reference_profile,
                args.dense_phase_threshold,
            )
            frame_centers.append(center)
            frame_center_orders.append(order)
        schedule = centering_schedule(
            np.asarray(frame_centers, dtype=float),
            np.asarray(frame_center_orders, dtype=float),
            mode=center_mode,
            blocks=args.blocks,
        )

    frame_profiles = []
    normal_lengths_nm = []
    for local_index, frame_index in enumerate(
        tqdm(frame_indices, desc="## Density profiles")
    ):
        timestep = analyzed_trajectory[frame_index]
        box_nm, lengths_nm, volume_nm3 = box_geometry_nm(timestep)
        positions_nm = np.asarray(timestep.positions, dtype=float) * ANGSTROM_TO_NM
        fractions = fractional_positions(positions_nm, box_nm)
        axis_fractions = fractions[:, axis_index]
        if center_mode != "none":
            axis_fractions = center_periodic_fractions(
                axis_fractions,
                schedule["centers"][local_index],
            )
        frame_profiles.append(
            [
                histogram_density(
                    axis_fractions[indices],
                    weights,
                    volume_nm3,
                    bins,
                    conversion,
                )
                for indices, weights in zip(group_indices, group_weights)
            ]
        )
        normal_lengths_nm.append(float(lengths_nm[axis_index]))

    frame_profiles = np.asarray(frame_profiles, dtype=float)
    normal_lengths_nm = np.asarray(normal_lengths_nm, dtype=float)
    mean_length_nm = float(normal_lengths_nm.mean())
    if float(np.ptp(normal_lengths_nm) / mean_length_nm) > 0.01:
        print(
            "## WARNING: The selected-axis cell length varies by more than 1%; "
            "profiles are averaged in fractional box coordinates."
        )
    coordinates_nm = (np.arange(bins, dtype=float) + 0.5) / bins * mean_length_nm
    density_profiles = frame_profiles.mean(axis=0)

    write_xvg(
        output_file_name,
        coordinates_nm,
        density_profiles,
        title=title,
        xlabel=f"{args.axis} Axis (nm)",
        ylabel=ylabel,
        legends=density_group_names,
    )
    if len(frame_indices) == 1:
        interval_text = "a single frame"
    else:
        interval_text = f"median interval {actual_interval_ns:g} ns"
    print(
        f"## Density profile from {len(frame_indices)} frame(s) with "
        f"{interval_text} written to {output_file_name}."
    )


density_commands = single_command("density", getargs_density, density, desc)
