"""Planar dense/dilute coexistence analysis for DROPPS trajectories."""

from __future__ import annotations

from dropps.share.argument_parser import ArgumentParser
import csv
from pathlib import Path

import numpy as np
from tqdm import tqdm

from dropps.analysis.coexistence_core import (
    center_periodic_fractions,
    centering_schedule,
    contiguous_block_statistics,
    fit_slab_profile_blocks,
    histogram_density,
    plateau_masks,
)
from dropps.analysis.density_core import (
    ANGSTROM_TO_NM,
    AXIS_TO_INDEX,
    DALTON_PER_NM3_TO_MG_PER_ML,
    DEFAULT_DENSITY_BIN_WIDTH_NM,
    PS_TO_NS as PS_TO_NS,
    box_geometry_nm as _box_geometry_nm,
    fractional_positions as _fractional_positions,
    select_frame_indices,
    threshold_profile_center,
)
from dropps.fileio.filename_control import validate_extension
from dropps.share.command_class import single_command
from dropps.share.trajectory import trajectory_class


prog = "coexistence"
desc = "Analyze dense/dilute coexistence and interface width in a planar slab trajectory."

def _select_frame_indices(
    trajectory,
    start_ns,
    end_ns,
    delta_ns,
    *,
    require_uniform=False,
):
    return select_frame_indices(
        trajectory,
        start_ns,
        end_ns,
        delta_ns,
        minimum_frames=2,
        require_uniform=require_uniform,
    )


def _select_groups(trajectory, args):
    if args.reference_group is None:
        trajectory.index.print_all()
        reference, reference_name = trajectory.getSelection_interactive(
            "reference group used to locate the dense slab"
        )
    else:
        reference, reference_name = trajectory.getSelection(
            f"group {args.reference_group}",
            "reference group used to locate the dense slab",
        )
    if len(reference) == 0:
        raise ValueError("Reference selection is empty.")

    groups = [(reference_name, reference)]
    known_indices = {tuple(np.asarray(reference.indices, dtype=int).tolist())}
    for group_id in args.selection_groups or []:
        selection, selection_name = trajectory.getSelection(
            f"group {group_id}", "coexistence density group"
        )
        if len(selection) == 0:
            raise ValueError(f"Selection group {group_id} is empty.")
        index_key = tuple(np.asarray(selection.indices, dtype=int).tolist())
        if index_key in known_indices:
            continue
        known_indices.add(index_key)
        groups.append((selection_name, selection))
    return reference, groups


def _output_paths(output, summary_output):
    profile_path = Path(validate_extension(output, "xvg"))
    if summary_output is None:
        summary_path = profile_path.with_suffix(".csv")
    else:
        summary_path = Path(validate_extension(summary_output, "csv"))
    if profile_path.resolve() == summary_path.resolve():
        raise ValueError("Profile and summary outputs must use different paths.")
    return profile_path, summary_path


def _format_scalar(value):
    if np.isnan(value):
        return "nan"
    if np.isposinf(value):
        return "inf"
    if np.isneginf(value):
        return "-inf"
    return f"{float(value):.12g}"


def _write_profile_xvg(
    path,
    coordinates_nm,
    mean_profiles,
    group_names,
    fit_statistics,
    density_label,
    metadata,
):
    fit = fit_statistics["fit"]
    fit_sem = fit_statistics["sem"]
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# DROPPS planar dense/dilute coexistence analysis\n")
        handle.write(
            f"# Method: center_mode={metadata['center_mode']} + "
            "symmetric double-tanh fit\n"
        )
        handle.write(
            f"# reference_group={metadata['reference_group']} axis={metadata['axis']} "
            f"frames={metadata['frames']} blocks={fit_statistics['blocks']} "
            f"centering_groups={metadata['centering_groups']} "
            f"dense_phase_threshold={metadata['dense_phase_threshold']:.10g} "
            f"requested_bin_width_nm={metadata['requested_bin_width_nm']:.10g} "
            f"actual_bin_width_nm={metadata['actual_bin_width_nm']:.10g}\n"
        )
        handle.write(
            f"# fit_dense={fit['dense_density']:.12g} "
            f"fit_dense_block_sem={fit_sem['dense_density']:.12g} "
            f"fit_dilute={fit['dilute_density']:.12g} "
            f"fit_dilute_block_sem={fit_sem['dilute_density']:.12g}\n"
        )
        handle.write(
            f"# slab_width_nm={fit['slab_width_nm']:.12g} "
            f"slab_width_block_sem_nm={fit_sem['slab_width_nm']:.12g} "
            f"interface_parameter_nm={fit['interface_parameter_nm']:.12g} "
            f"interface_width_10_90_nm={fit['interface_width_10_90_nm']:.12g} "
            f"interface_width_block_sem_nm={fit_sem['interface_width_10_90_nm']:.12g}\n"
        )
        handle.write(
            f"# fit_r_squared={fit['r_squared']:.12g} "
            f"fit_rmse={fit['rmse']:.12g} "
            f"mean_center_order={metadata['mean_center_order']:.12g} "
            f"min_center_order={metadata['min_center_order']:.12g} "
            f"mean_frame_center_order={metadata['mean_frame_center_order']:.12g}\n"
        )
        handle.write('@    title "Planar coexistence density profile"\n')
        handle.write('@    xaxis  label "Centered slab coordinate (nm)"\n')
        handle.write(f'@    yaxis  label "{density_label}"\n')
        handle.write("@TYPE xy\n")
        for series_id, name in enumerate(group_names):
            handle.write(f'@ s{series_id} legend "{name}"\n')
        handle.write(f'@ s{len(group_names)} legend "{group_names[0]} double-tanh fit"\n')

        output_profiles = list(mean_profiles) + [fit["fitted_profile"]]
        for bin_index, coordinate in enumerate(coordinates_nm):
            fields = [f"{coordinate:.10g}"]
            fields.extend(
                f"{profile[bin_index]:.12g}" for profile in output_profiles
            )
            handle.write("\t".join(fields) + "\n")


def _write_summary_csv(
    path,
    group_names,
    density_unit,
    dense_statistics,
    dilute_statistics,
    ratio,
    ratio_sem,
    fit_statistics,
    metadata,
):
    fit = fit_statistics["fit"]
    fit_sem = fit_statistics["sem"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("# DROPPS planar dense/dilute coexistence summary\n")
        handle.write(
            f"# reference_group={metadata['reference_group']},axis={metadata['axis']},"
            f"frames={metadata['frames']},blocks={fit_statistics['blocks']},"
            f"center_mode={metadata['center_mode']},"
            f"centering_groups={metadata['centering_groups']}\n"
        )
        handle.write(
            f"# slab_width_nm={fit['slab_width_nm']:.12g},"
            f"slab_width_block_sem_nm={fit_sem['slab_width_nm']:.12g},"
            f"interface_width_10_90_nm={fit['interface_width_10_90_nm']:.12g},"
            f"interface_width_block_sem_nm={fit_sem['interface_width_10_90_nm']:.12g},"
            f"fit_r_squared={fit['r_squared']:.12g}\n"
        )
        writer = csv.writer(handle)
        writer.writerow(
            (
                "group",
                "density_unit",
                "dense_mean",
                "dense_block_sem",
                "dilute_mean",
                "dilute_block_sem",
                "dense_to_dilute_ratio",
                "ratio_block_sem",
            )
        )
        for group_index, group_name in enumerate(group_names):
            writer.writerow(
                (
                    group_name,
                    density_unit,
                    _format_scalar(dense_statistics["mean"][group_index]),
                    _format_scalar(dense_statistics["sem"][group_index]),
                    _format_scalar(dilute_statistics["mean"][group_index]),
                    _format_scalar(dilute_statistics["sem"][group_index]),
                    _format_scalar(ratio[group_index]),
                    _format_scalar(ratio_sem[group_index]),
                )
            )


def getargs_coexistence(argv):
    parser = ArgumentParser(prog=prog, description=desc)
    parser.add_argument(
        "-s",
        "--run-input",
        required=True,
        help="Input DROPPS run file (.tpr) containing the system and simulation settings.",
    )
    parser.add_argument(
        "-f", "--input", required=True, help="Input planar-slab trajectory file (.xtc)."
    )
    parser.add_argument(
        "-n", "--index", help="Optional index file (.ndx) defining additional atom groups."
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output density-profile file (.xvg); the extension is added if omitted.",
    )
    parser.add_argument(
        "--summary-output",
        help="Optional output summary file (.csv); if omitted, use the profile path with a .csv suffix.",
    )
    parser.add_argument(
        "-ref",
        "--reference-group",
        type=int,
        help="Index group used to locate and fit the dense slab.",
    )
    parser.add_argument(
        "-sel",
        "--selection-groups",
        type=int,
        nargs="+",
        help="Additional index groups for dense/dilute plateau densities.",
    )
    parser.add_argument(
        "--axis",
        choices=("x", "y", "z"),
        default="z",
        help="Interface-normal axis.",
    )
    parser.add_argument(
        "--center-mode",
        choices=("frame", "block", "none"),
        default="frame",
        help=(
            "Slab centering strategy: frame aligns every analyzed frame; block "
            "uses one center per contiguous --blocks segment; none applies no "
            "shift and expects an already box-centered slab."
        ),
    )
    parser.add_argument(
        "-tp",
        "--type",
        dest="density_type",
        choices=("mass", "number"),
        default="mass",
        help="Mass density (mg/mL) or selected-particle number density (nm^-3).",
    )
    parser.add_argument(
        "--bin-width",
        type=float,
        default=DEFAULT_DENSITY_BIN_WIDTH_NM,
        help=(
            "Requested density-bin width, in nm. The default matches the "
            "density command."
        ),
    )
    parser.add_argument(
        "-t",
        "--dense-phase-threshold",
        type=float,
        default=0.5,
        help=(
            "Dense-phase threshold as a fraction of the maximum reference "
            "density; shared with the density command."
        ),
    )
    parser.add_argument(
        "-b", "--start-time", type=float, help="First trajectory time to analyze, in ns."
    )
    parser.add_argument(
        "-e", "--end-time", type=float, help="Last trajectory time to analyze, in ns."
    )
    parser.add_argument(
        "-dt",
        "--delta-time",
        type=float,
        help="Approximate interval between analyzed frames, in ns.",
    )
    parser.add_argument(
        "--blocks",
        type=int,
        default=5,
        help="Number of contiguous blocks used for standard errors.",
    )
    parser.add_argument(
        "--plateau-buffer",
        type=float,
        default=2.0,
        help=(
            "Distance excluded on each side of a fitted interface, in units of "
            "the tanh interface parameter."
        ),
    )
    return parser.parse_args(argv)


def coexistence(args):
    if not np.isfinite(args.bin_width) or args.bin_width <= 0.0:
        raise ValueError("Bin width must be positive and finite.")
    if args.blocks <= 0:
        raise ValueError("Number of blocks must be positive.")
    if not np.isfinite(args.plateau_buffer) or args.plateau_buffer < 0.0:
        raise ValueError("Plateau buffer must be finite and non-negative.")
    if (
        not np.isfinite(args.dense_phase_threshold)
        or not 0.0 < args.dense_phase_threshold < 1.0
    ):
        raise ValueError("Dense-phase threshold must be finite and between 0 and 1.")

    trajectory = trajectory_class(args.run_input, args.index, args.input)
    reference, groups = _select_groups(trajectory, args)
    reference_indices = np.asarray(reference.indices, dtype=int)
    group_names = [name for name, _ in groups]
    group_indices = [np.asarray(group.indices, dtype=int) for _, group in groups]
    if args.density_type == "mass":
        group_weights = [np.asarray(group.masses, dtype=float) for _, group in groups]
        density_conversion = DALTON_PER_NM3_TO_MG_PER_ML
        density_label = "Mass density (mg/mL)"
        density_unit = "mg/mL"
    else:
        group_weights = [np.ones(len(group), dtype=float) for _, group in groups]
        density_conversion = 1.0
        density_label = "Selected-particle number density (nm^-3)"
        density_unit = "nm^-3"
    if any(
        not np.all(np.isfinite(weights)) or np.any(weights < 0.0)
        for weights in group_weights
    ):
        raise ValueError("Density weights must be finite and non-negative.")
    reference_weights = group_weights[0]
    if reference_weights.sum() <= 0.0:
        raise ValueError("Reference density group must have positive total weight.")

    frame_indices, actual_interval_ns = _select_frame_indices(
        trajectory.Universe.trajectory,
        args.start_time,
        args.end_time,
        args.delta_time,
    )
    first_timestep = trajectory.Universe.trajectory[frame_indices[0]]
    _, first_lengths, _ = _box_geometry_nm(first_timestep)
    axis_index = AXIS_TO_INDEX[args.axis]
    bins = max(12, int(np.ceil(first_lengths[axis_index] / args.bin_width)))

    if args.center_mode == "none":
        frame_centers = np.zeros(len(frame_indices), dtype=float)
        frame_center_orders = np.full(len(frame_indices), np.nan, dtype=float)
        schedule = centering_schedule(
            frame_centers,
            np.zeros(len(frame_indices), dtype=float),
            mode="none",
            blocks=args.blocks,
        )
    else:
        frame_centers = []
        frame_center_orders = []
        for frame_index in tqdm(frame_indices, desc="## Locating dense slab"):
            timestep = trajectory.Universe.trajectory[frame_index]
            box_nm, _, volume_nm3 = _box_geometry_nm(timestep)
            positions_nm = (
                np.asarray(timestep.positions, dtype=float) * ANGSTROM_TO_NM
            )
            fractional = _fractional_positions(positions_nm, box_nm)
            reference_profile = histogram_density(
                fractional[reference_indices, axis_index],
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
        frame_centers = np.asarray(frame_centers, dtype=float)
        frame_center_orders = np.asarray(frame_center_orders, dtype=float)
        schedule = centering_schedule(
            frame_centers,
            frame_center_orders,
            mode=args.center_mode,
            blocks=args.blocks,
        )

    frame_profiles = []
    normal_lengths_nm = []
    for local_index, frame_index in enumerate(
        tqdm(frame_indices, desc="## Coexistence profiles")
    ):
        timestep = trajectory.Universe.trajectory[frame_index]
        box_nm, lengths_nm, volume_nm3 = _box_geometry_nm(timestep)
        positions_nm = np.asarray(timestep.positions, dtype=float) * ANGSTROM_TO_NM
        fractional = _fractional_positions(positions_nm, box_nm)
        if args.center_mode == "none":
            centered_axis = fractional[:, axis_index]
            try:
                reference_profile = histogram_density(
                    fractional[reference_indices, axis_index],
                    reference_weights,
                    volume_nm3,
                    bins,
                )
                _, frame_center_orders[local_index] = threshold_profile_center(
                    reference_profile,
                    args.dense_phase_threshold,
                )
            except ValueError:
                frame_center_orders[local_index] = 0.0
        else:
            centered_axis = center_periodic_fractions(
                fractional[:, axis_index], schedule["centers"][local_index]
            )

        frame_profiles.append(
            [
                histogram_density(
                    centered_axis[indices],
                    weights,
                    volume_nm3,
                    bins,
                    density_conversion,
                )
                for indices, weights in zip(group_indices, group_weights)
            ]
        )
        normal_lengths_nm.append(float(lengths_nm[axis_index]))

    frame_profiles = np.asarray(frame_profiles, dtype=float)
    normal_lengths_nm = np.asarray(normal_lengths_nm, dtype=float)
    mean_length_nm = float(normal_lengths_nm.mean())
    relative_length_span = float(np.ptp(normal_lengths_nm) / mean_length_nm)
    if relative_length_span > 0.01:
        print(
            "## WARNING: The interface-normal cell length varies by more than 1%. "
            "Profiles are averaged in fractional box coordinates."
        )
    coordinates_nm = (
        (np.arange(bins, dtype=float) + 0.5) / bins - 0.5
    ) * mean_length_nm

    fit_statistics = fit_slab_profile_blocks(
        coordinates_nm, frame_profiles[:, 0, :], args.blocks
    )
    fit = fit_statistics["fit"]
    dense_mask, dilute_mask = plateau_masks(
        coordinates_nm,
        fit["half_width_nm"],
        fit["interface_parameter_nm"],
        args.plateau_buffer,
    )
    dense_by_frame = frame_profiles[:, :, dense_mask].mean(axis=2)
    dilute_by_frame = frame_profiles[:, :, dilute_mask].mean(axis=2)
    dense_statistics = contiguous_block_statistics(dense_by_frame, args.blocks)
    dilute_statistics = contiguous_block_statistics(dilute_by_frame, args.blocks)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = dense_statistics["mean"] / dilute_statistics["mean"]
        block_ratios = (
            dense_statistics["block_means"] / dilute_statistics["block_means"]
        )
    ratio_sem = np.full(len(groups), np.nan, dtype=float)
    if fit_statistics["blocks"] > 1:
        for group_index in range(len(groups)):
            finite_ratios = block_ratios[
                np.isfinite(block_ratios[:, group_index]), group_index
            ]
            if finite_ratios.size > 1:
                ratio_sem[group_index] = float(
                    finite_ratios.std(ddof=1) / np.sqrt(finite_ratios.size)
                )

    profile_path, summary_path = _output_paths(args.output, args.summary_output)
    centering_orders = schedule["order_values"]
    if centering_orders.size > 0:
        mean_center_order = float(centering_orders.mean())
        min_center_order = float(centering_orders.min())
        mean_frame_center_order = float(frame_center_orders.mean())
    else:
        mean_center_order = float("nan")
        min_center_order = float("nan")
        mean_frame_center_order = float(frame_center_orders.mean())
    metadata = {
        "reference_group": group_names[0],
        "axis": args.axis,
        "frames": len(frame_indices),
        "center_mode": args.center_mode,
        "centering_groups": schedule["groups"],
        "dense_phase_threshold": args.dense_phase_threshold,
        "requested_bin_width_nm": args.bin_width,
        "actual_bin_width_nm": mean_length_nm / bins,
        "mean_center_order": mean_center_order,
        "min_center_order": min_center_order,
        "mean_frame_center_order": mean_frame_center_order,
    }
    _write_profile_xvg(
        profile_path,
        coordinates_nm,
        frame_profiles.mean(axis=0),
        group_names,
        fit_statistics,
        density_label,
        metadata,
    )
    _write_summary_csv(
        summary_path,
        group_names,
        density_unit,
        dense_statistics,
        dilute_statistics,
        ratio,
        ratio_sem,
        fit_statistics,
        metadata,
    )

    print(
        f"## Analyzed {len(frame_indices)} frames with median interval "
        f"{actual_interval_ns:g} ns; "
        f"fit R^2 = {fit['r_squared']:.6g}."
    )
    print(
        f"## Center mode: {args.center_mode}; "
        f"centering groups: {schedule['groups']}."
    )
    print(
        f"## Reference fit: dense={fit['dense_density']:.8g}, "
        f"dilute={fit['dilute_density']:.8g} {density_unit}; "
        f"10-90% interface width={fit['interface_width_10_90_nm']:.6g} nm."
    )
    if fit["r_squared"] < 0.8:
        print(
            "## WARNING: Slab fit R^2 is below 0.8. Inspect the XVG profile and "
            "verify equilibration, centering, reference selection, and box geometry."
        )
    if centering_orders.size > 0 and mean_center_order < 0.05:
        print(
            "## WARNING: The reference group's mode-specific mean centering order "
            "is below 0.05; the system may be nearly homogeneous or the selected "
            "centering interval may be too long."
        )
    if args.center_mode == "none" and mean_frame_center_order < 0.05:
        print(
            "## WARNING: The unshifted reference group's mean frame centering "
            "order is below 0.05; the system may be nearly homogeneous."
        )
    print(f"## Mean profiles and fitted interface written to {profile_path.resolve()}.")
    print(f"## Coexisting phase summary written to {summary_path.resolve()}.")


coexistence_commands = single_command(
    "coexistence", getargs_coexistence, coexistence, desc
)
