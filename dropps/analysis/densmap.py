"""Planar two-dimensional mass, number, and charge density maps."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from tqdm import tqdm

from dropps.analysis.coexistence_core import (
    center_periodic_fractions,
    centering_schedule,
)
from dropps.analysis.density_core import (
    ANGSTROM_TO_NM,
    AXIS_TO_INDEX,
    DALTON_PER_NM3_TO_MG_PER_ML,
    box_geometry_nm,
    fractional_positions,
    select_frame_indices,
)
from dropps.analysis.densmap_core import (
    DEFAULT_DENSMAP_BIN_WIDTH_NM,
    density_grid_shape,
    histogram_density_map,
    periodic_plane_center,
    plane_axes,
    slice_mask_and_thickness,
)
from dropps.fileio.xpm_reader import write_xpm
from dropps.share.argument_parser import ArgumentParser
from dropps.share.command_class import single_command
from dropps.share.trajectory import trajectory_class


prog = "densmap"
desc = "Calculate planar two-dimensional mass, number, or charge density maps."

_AXIS_LABELS = ("x", "y", "z")
_NORMALIZATION_ALIASES = {
    "volume": "volume",
    "area": "area",
    "count": "count",
    "nm-3": "volume",
    "nm-2": "area",
}


def getargs_densmap(argv):
    parser = ArgumentParser(prog=prog, description=desc)
    parser.add_argument(
        "-s",
        "--run-input",
        required=True,
        help="Input DROPPS run file (.tpr) containing topology masses and charges.",
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
        "-od",
        "--output",
        required=True,
        help=(
            "Output GROMACS-style text matrix (.dat). With multiple -sel groups, "
            "the group number and name are appended to this path."
        ),
    )
    parser.add_argument(
        "-oxpm",
        "--xpm-output",
        help=(
            "Optional grayscale XPM map. With multiple -sel groups, the group "
            "number and name are appended to this path."
        ),
    )
    parser.add_argument(
        "-aver",
        "--average-axis",
        choices=("x", "y", "z"),
        default="z",
        help="Direction to average over; z produces an x-y map.",
    )
    parser.add_argument(
        "-tp",
        "--type",
        choices=("mass", "number", "charge"),
        default="number",
        help="Particle property accumulated in each grid cell.",
    )
    parser.add_argument(
        "-unit",
        "--normalization",
        choices=("volume", "area", "count", "nm-3", "nm-2"),
        default="volume",
        help=(
            "volume/nm-3 gives a true 3D density, area/nm-2 gives a projected "
            "surface density, and count gives mean weight per cell."
        ),
    )
    parser.add_argument(
        "-bin",
        "--bin-width",
        type=float,
        default=DEFAULT_DENSMAP_BIN_WIDTH_NM,
        help="Requested grid spacing in both retained directions, in nm.",
    )
    parser.add_argument(
        "-n1",
        "--bins-first",
        type=int,
        default=0,
        help="Grid cells along the first retained axis; zero derives it from -bin.",
    )
    parser.add_argument(
        "-n2",
        "--bins-second",
        type=int,
        default=0,
        help="Grid cells along the second retained axis; zero derives it from -bin.",
    )
    parser.add_argument(
        "-xmin",
        "--average-min",
        type=float,
        help="Optional lower coordinate included along the averaging axis, in nm.",
    )
    parser.add_argument(
        "-xmax",
        "--average-max",
        type=float,
        help="Optional upper coordinate excluded along the averaging axis, in nm.",
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
        "-sel",
        "--selection",
        type=int,
        nargs="+",
        help=(
            "Index groups to map; if omitted, prompt for one or more groups "
            "interactively."
        ),
    )
    parser.add_argument(
        "-selfit",
        "--selection-fit",
        type=int,
        help=(
            "Index group used for optional planar centering. If omitted, the "
            "first mapped group is used."
        ),
    )
    parser.add_argument(
        "--center-mode",
        choices=("frame", "block", "none"),
        default="none",
        help=(
            "Periodic planar alignment: frame follows a compact droplet each "
            "frame, block uses one center per contiguous block, and none keeps "
            "the trajectory coordinates."
        ),
    )
    parser.add_argument(
        "--blocks",
        type=int,
        default=5,
        help="Number of contiguous groups used by --center-mode block.",
    )
    return parser.parse_args(argv)


def _select_groups(trajectory, args):
    if args.selection is None:
        trajectory.index.print_all()
        groups, names = trajectory.getSelection_interactive_multiple(
            "two-dimensional density mapping"
        )
    else:
        selected = [
            trajectory.getSelection(f"group {group_id}", "density mapping")
            for group_id in args.selection
        ]
        groups = [group for group, _ in selected]
        names = [name for _, name in selected]

    if not groups:
        raise ValueError("At least one density-map selection is required.")
    if any(len(group) == 0 for group in groups):
        raise ValueError("Density-map selections must not be empty.")

    if args.center_mode == "none" or args.selection_fit is None:
        center_group = groups[0]
    else:
        center_group, _ = trajectory.getSelection(
            f"group {args.selection_fit}",
            "density-map centering",
        )
    if len(center_group) == 0:
        raise ValueError("Density-map centering selection must not be empty.")
    return groups, names, center_group


def _quantity_configuration(group, density_type, normalization):
    if density_type == "mass":
        weights = np.asarray(group.masses, dtype=float)
        if np.any(weights < 0.0):
            raise ValueError("Mass density requires non-negative particle masses.")
        if normalization == "volume":
            return weights, DALTON_PER_NM3_TO_MG_PER_ML, "mg/mL"
        if normalization == "area":
            return weights, 1.0, "Da/nm^2"
        return weights, 1.0, "Da/cell"

    if density_type == "number":
        weights = np.ones(len(group), dtype=float)
        unit = {
            "volume": "nm^-3",
            "area": "nm^-2",
            "count": "count/cell",
        }[normalization]
        return weights, 1.0, unit

    weights = np.asarray(group.charges, dtype=float)
    unit = {
        "volume": "e/nm^3",
        "area": "e/nm^2",
        "count": "e/cell",
    }[normalization]
    return weights, 1.0, unit


def _safe_label(label):
    cleaned = re.sub(r"[^\w.-]+", "-", str(label), flags=re.UNICODE).strip("-.")
    return cleaned or "selection"


def _group_output_paths(base_path, group_names, suffix):
    base = Path(base_path)
    suffix = suffix if suffix.startswith(".") else "." + suffix
    if len(group_names) == 1:
        if base.suffix.lower() != suffix.lower():
            base = Path(str(base) + suffix)
        return [base]

    if base.suffix.lower() == suffix.lower():
        prefix = base.with_suffix("")
    else:
        prefix = base
    return [
        Path(f"{prefix}.{index}-{_safe_label(name)}{suffix}")
        for index, name in enumerate(group_names, start=1)
    ]


def write_density_map_dat(
    path,
    first_coordinates_nm,
    second_coordinates_nm,
    density_map,
    metadata,
):
    """Write a numeric matrix with coordinates in the first row and column."""

    first = np.asarray(first_coordinates_nm, dtype=float)
    second = np.asarray(second_coordinates_nm, dtype=float)
    values = np.asarray(density_map, dtype=float)
    if first.ndim != 1 or second.ndim != 1:
        raise ValueError("Density-map coordinates must be one-dimensional.")
    if values.shape != (first.size, second.size):
        raise ValueError("Density-map dimensions do not match its coordinates.")
    if not (
        np.all(np.isfinite(first))
        and np.all(np.isfinite(second))
        and np.all(np.isfinite(values))
    ):
        raise ValueError("Density-map output must be finite.")

    path = Path(path)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# DROPPS planar two-dimensional density map\n")
        for key, value in metadata.items():
            handle.write(f"# {key}={value}\n")
        handle.write(
            "# The top row contains second-axis cell centers; the first column "
            "contains first-axis cell centers (nm).\n"
        )
        handle.write("0\t" + "\t".join(f"{value:.12g}" for value in second) + "\n")
        for coordinate, row in zip(first, values):
            handle.write(
                f"{coordinate:.12g}\t"
                + "\t".join(f"{value:.12g}" for value in row)
                + "\n"
            )


def _validate_args(args):
    if not np.isfinite(args.bin_width) or args.bin_width <= 0.0:
        raise ValueError("Bin width must be positive and finite.")
    if args.bins_first < 0 or args.bins_second < 0:
        raise ValueError("Explicit grid sizes must be non-negative.")
    if args.blocks <= 0:
        raise ValueError("Number of centering blocks must be positive.")
    if args.average_min is not None and not np.isfinite(args.average_min):
        raise ValueError("Minimum averaging coordinate must be finite.")
    if args.average_max is not None and not np.isfinite(args.average_max):
        raise ValueError("Maximum averaging coordinate must be finite.")
    if (
        args.average_min is not None
        and args.average_max is not None
        and args.average_max <= args.average_min
    ):
        raise ValueError("Maximum averaging coordinate must exceed the minimum.")


def densmap(args):
    _validate_args(args)
    normalization = _NORMALIZATION_ALIASES[args.normalization]
    trajectory = trajectory_class(args.run_input, args.index, args.input)
    density_groups, density_group_names, center_group = _select_groups(
        trajectory,
        args,
    )

    group_indices = [np.asarray(group.indices, dtype=int) for group in density_groups]
    configurations = [
        _quantity_configuration(group, args.type, normalization)
        for group in density_groups
    ]
    group_weights = [configuration[0] for configuration in configurations]
    conversion = configurations[0][1]
    unit = configurations[0][2]
    if any(not np.all(np.isfinite(weights)) for weights in group_weights):
        raise ValueError("Density-map weights must be finite.")

    center_indices = None
    center_weights = None
    if args.center_mode != "none":
        center_indices = np.asarray(center_group.indices, dtype=int)
        center_weights = np.asarray(center_group.masses, dtype=float)
        if (
            not np.all(np.isfinite(center_weights))
            or np.any(center_weights < 0.0)
            or center_weights.sum() <= 0.0
        ):
            raise ValueError("Centering selection must have positive finite total mass.")

    analyzed_trajectory = trajectory.Universe.trajectory
    frame_indices, actual_interval_ns = select_frame_indices(
        analyzed_trajectory,
        args.start_time,
        args.end_time,
        args.delta_time,
    )
    average_index = AXIS_TO_INDEX[args.average_axis]
    retained_axes = plane_axes(args.average_axis)
    first_timestep = analyzed_trajectory[frame_indices[0]]
    _, first_lengths_nm, _ = box_geometry_nm(first_timestep)
    bins = density_grid_shape(
        first_lengths_nm[list(retained_axes)],
        args.bin_width,
        args.bins_first,
        args.bins_second,
    )

    schedules = None
    if args.center_mode != "none":
        frame_centers = []
        frame_strengths = []
        for frame_index in tqdm(frame_indices, desc="## Locating 2D density center"):
            timestep = analyzed_trajectory[frame_index]
            box_nm, _, _ = box_geometry_nm(timestep)
            positions_nm = np.asarray(timestep.positions, dtype=float) * ANGSTROM_TO_NM
            fractions = fractional_positions(positions_nm, box_nm)
            centers, strengths = periodic_plane_center(
                fractions[np.ix_(center_indices, retained_axes)],
                center_weights,
            )
            frame_centers.append(centers)
            frame_strengths.append(strengths)
        frame_centers = np.asarray(frame_centers, dtype=float)
        frame_strengths = np.asarray(frame_strengths, dtype=float)
        schedules = [
            centering_schedule(
                frame_centers[:, dimension],
                frame_strengths[:, dimension],
                mode=args.center_mode,
                blocks=args.blocks,
            )
            for dimension in range(2)
        ]

    map_sums = np.zeros((len(density_groups), bins[0], bins[1]), dtype=float)
    retained_lengths_nm = []
    for local_index, frame_index in enumerate(
        tqdm(frame_indices, desc="## Two-dimensional density maps")
    ):
        timestep = analyzed_trajectory[frame_index]
        box_nm, lengths_nm, _ = box_geometry_nm(timestep)
        positions_nm = np.asarray(timestep.positions, dtype=float) * ANGSTROM_TO_NM
        fractions = fractional_positions(positions_nm, box_nm)
        plane_fractions = fractions[:, retained_axes].copy()
        if schedules is not None:
            for dimension in range(2):
                plane_fractions[:, dimension] = center_periodic_fractions(
                    plane_fractions[:, dimension],
                    schedules[dimension]["centers"][local_index],
                )

        slice_mask, averaging_thickness_nm = slice_mask_and_thickness(
            fractions[:, average_index],
            lengths_nm[average_index],
            args.average_min,
            args.average_max,
        )
        plane_lengths = lengths_nm[list(retained_axes)]
        for group_index, (indices, weights) in enumerate(
            zip(group_indices, group_weights)
        ):
            included = slice_mask[indices]
            map_sums[group_index] += histogram_density_map(
                plane_fractions[indices][included],
                weights[included],
                plane_lengths,
                averaging_thickness_nm,
                bins,
                normalization,
                conversion,
            )
        retained_lengths_nm.append(plane_lengths)

    density_maps = map_sums / len(frame_indices)
    retained_lengths_nm = np.asarray(retained_lengths_nm, dtype=float)
    mean_lengths_nm = retained_lengths_nm.mean(axis=0)
    for dimension, axis_index in enumerate(retained_axes):
        relative_range = float(
            np.ptp(retained_lengths_nm[:, dimension]) / mean_lengths_nm[dimension]
        )
        if relative_range > 0.01:
            print(
                f"## WARNING: The {_AXIS_LABELS[axis_index]} box length varies by "
                "more than 1%; maps are averaged in fractional box coordinates."
            )

    first_coordinates_nm = (
        (np.arange(bins[0], dtype=float) + 0.5) / bins[0] * mean_lengths_nm[0]
    )
    second_coordinates_nm = (
        (np.arange(bins[1], dtype=float) + 0.5) / bins[1] * mean_lengths_nm[1]
    )
    data_paths = _group_output_paths(args.output, density_group_names, ".dat")
    xpm_paths = (
        _group_output_paths(args.xpm_output, density_group_names, ".xpm")
        if args.xpm_output is not None
        else [None] * len(density_groups)
    )

    if len(frame_indices) == 1:
        interval_text = "single frame"
    else:
        interval_text = f"{actual_interval_ns:.12g} ns median interval"
    first_axis = _AXIS_LABELS[retained_axes[0]]
    second_axis = _AXIS_LABELS[retained_axes[1]]
    for name, density_map, data_path, xpm_path in zip(
        density_group_names,
        density_maps,
        data_paths,
        xpm_paths,
    ):
        metadata = {
            "selection": name,
            "density_type": args.type,
            "normalization": normalization,
            "unit": unit,
            "average_axis": args.average_axis,
            "average_min_nm": (
                "box-origin" if args.average_min is None else args.average_min
            ),
            "average_max_nm": (
                "box-length" if args.average_max is None else args.average_max
            ),
            "first_axis": first_axis,
            "second_axis": second_axis,
            "grid": f"{bins[0]}x{bins[1]}",
            "requested_bin_width_nm": args.bin_width,
            "mean_first_length_nm": f"{mean_lengths_nm[0]:.12g}",
            "mean_second_length_nm": f"{mean_lengths_nm[1]:.12g}",
            "frames": len(frame_indices),
            "sampling": interval_text,
            "center_mode": args.center_mode,
        }
        write_density_map_dat(
            data_path,
            first_coordinates_nm,
            second_coordinates_nm,
            density_map,
            metadata,
        )
        print(f"## {name} {args.type} density map written to {data_path}.")
        if xpm_path is not None:
            # XPM rows run top-to-bottom; transpose and flip so the first axis is
            # horizontal and the second-axis origin appears at the lower edge.
            write_xpm(
                np.flipud(density_map.T),
                xpm_path,
                f"{name} {args.type} density ({unit})",
            )
            print(f"## {name} density-map preview written to {xpm_path}.")


densmap_commands = single_command("densmap", getargs_densmap, densmap, desc)
