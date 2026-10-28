"""Legacy group and phase-resolved molecular radius-of-gyration analysis."""

from __future__ import annotations

import csv
import warnings
from pathlib import Path

import numpy as np
from Bio import BiopythonDeprecationWarning
from MDAnalysis.transformations import unwrap
from tqdm import tqdm

from dropps.analysis.coexistence import (
    ANGSTROM_TO_NM,
    AXIS_TO_INDEX,
    DALTON_PER_NM3_TO_MG_PER_ML,
    PS_TO_NS,
    _box_geometry_nm,
    _fractional_positions,
    _select_frame_indices,
)
from dropps.analysis.coexistence_core import (
    center_periodic_fractions,
    centering_schedule,
    fit_slab_profile_blocks,
    histogram_density,
)
from dropps.analysis.density_core import (
    DEFAULT_DENSITY_BIN_WIDTH_NM,
    threshold_profile_center,
)
from dropps.analysis.exchange import _build_molecules
from dropps.analysis.exchange_core import (
    DENSE,
    DILUTE,
    INTERFACE,
    STATE_NAMES,
    classify_phase_positions,
    periodic_weighted_mean_fraction,
    phase_cutoffs,
)
from dropps.analysis.gyrate_core import (
    periodic_radius_of_gyration,
    phase_rg_histogram,
    phase_rg_summary,
    phase_rg_time_series,
)
from dropps.fileio.filename_control import validate_extension
from dropps.fileio.xvg_reader import write_xvg
from dropps.share.argument_parser import ArgumentParser
from dropps.share.command_class import single_command
from dropps.share.trajectory import trajectory_class

warnings.filterwarnings("ignore", category=BiopythonDeprecationWarning)

prog = "gyrate"
desc = "Calculate chain radii of gyration, including optional dense/dilute resolution."

PHASE_CODES = {"dense": DENSE, "interface": INTERFACE, "dilute": DILUTE}


def radius_of_gyration(positions, masses):
    com = np.average(positions, axis=0, weights=masses)
    squared_distances = np.sum((positions - com) ** 2, axis=1)
    return np.sqrt(np.average(squared_distances, weights=masses))


def getargs_gyrate(argv):
    parser = ArgumentParser(
        prog=prog,
        description=desc,
        epilog="At least one output option (-ov, -oa, -oh, or -op) is required.",
    )

    parser.add_argument(
        "-s",
        "--run-input",
        required=True,
        help="Input DROPPS run file (.tpr) containing the system and simulation settings.",
    )
    parser.add_argument(
        "-f", "--input", required=True, help="Input trajectory file (.xtc)."
    )
    parser.add_argument(
        "-n",
        "--index",
        help="Optional index file (.ndx) defining additional atom groups.",
    )
    parser.add_argument(
        "-sel",
        "--selection-calculate",
        type=int,
        nargs="+",
        help=(
            "Index groups whose radii of gyration are calculated; if omitted, "
            "prompt interactively. Phase-resolved mode splits each group into molecules."
        ),
    )
    parser.add_argument(
        "-b",
        "--start-time",
        type=float,
        help="First trajectory time to analyze, in ns.",
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
        "-pbc",
        "--treat-pbc",
        action="store_true",
        help="Unwrap molecules across periodic boundaries before legacy group analysis.",
    )
    parser.add_argument(
        "-ov",
        "--output-verbose",
        help=(
            "Output per-selected-group radius-of-gyration time series (.xvg); "
            "the extension is added if omitted."
        ),
    )
    parser.add_argument(
        "-oa",
        "--output-average",
        help=(
            "Output the mean radius of gyration across selected groups (.xvg); "
            "the extension is added if omitted."
        ),
    )
    parser.add_argument(
        "-oh",
        "--output-histogram",
        help=(
            "Output the pooled selected-group/frame radius-of-gyration distribution "
            "(.xvg); the extension is added if omitted."
        ),
    )

    parser.add_argument(
        "-op",
        "--output-phase",
        help=(
            "Output phase-resolved mean Rg file (.xvg); companion observation, "
            "summary, histogram, and reference-profile files are also written."
        ),
    )

    parser.add_argument(
        "-bw",
        "--bin-width",
        type=float,
        default=0.1,
        help="Histogram bin width, in nm.",
    )

    parser.add_argument(
        "-ref",
        "--reference-group",
        type=int,
        help="Index group used to locate and fit the dense slab in phase-resolved mode.",
    )
    parser.add_argument(
        "--axis",
        choices=("x", "y", "z"),
        default="z",
        help="Interface-normal axis in phase-resolved mode.",
    )
    parser.add_argument(
        "--center-mode",
        choices=("frame", "block", "none"),
        default="frame",
        help=(
            "Slab centering used for phase classification: frame, contiguous "
            "block, or none."
        ),
    )
    parser.add_argument(
        "--phases",
        choices=tuple(PHASE_CODES),
        nargs="+",
        default=("dense", "dilute"),
        help="Phase regions included in phase-resolved means, summaries, and PDFs.",
    )
    parser.add_argument(
        "--density-bin-width",
        type=float,
        default=DEFAULT_DENSITY_BIN_WIDTH_NM,
        help="Requested reference density-bin width in phase-resolved mode, in nm.",
    )
    parser.add_argument(
        "--dense-phase-threshold",
        type=float,
        default=0.5,
        help="Fraction of maximum reference density used to locate the dense slab.",
    )
    parser.add_argument(
        "--interface-threshold",
        type=float,
        default=0.1,
        help=(
            "Normalized dilute-side threshold defining the interface state; "
            "0.1 uses the fitted 10--90%% region."
        ),
    )
    parser.add_argument(
        "--blocks",
        type=int,
        default=5,
        help="Number of fit/statistics blocks and block-centering groups.",
    )
    parser.add_argument(
        "--no-phase-observations",
        action="store_true",
        help="Skip the potentially large per-frame, per-molecule phase CSV.",
    )

    args = parser.parse_args(argv)

    return args


def _format_scalar(value):
    value = float(value)
    if np.isnan(value):
        return "nan"
    if np.isposinf(value):
        return "inf"
    if np.isneginf(value):
        return "-inf"
    return f"{value:.12g}"


def _phase_output_paths(output, write_observations):
    mean_path = Path(validate_extension(output, "xvg"))
    stem = mean_path.stem
    paths = {
        "mean": mean_path,
        "summary": mean_path.with_name(f"{stem}.summary.csv"),
        "histogram": mean_path.with_name(f"{stem}.histogram.xvg"),
        "profile": mean_path.with_name(f"{stem}.profile.xvg"),
    }
    if write_observations:
        paths["observations"] = mean_path.with_name(f"{stem}.observations.csv")
    resolved = [path.resolve() for path in paths.values()]
    if len(set(resolved)) != len(resolved):
        raise ValueError("Phase-resolved gyrate output paths must be distinct.")
    return paths


def _filename_token(value):
    token = "".join(
        character if character.isalnum() or character in {".", "_", "-"} else "-"
        for character in str(value).strip()
    ).strip(".-_")
    return token or "group"


def _group_phase_value_paths(output, group_names):
    mean_path = Path(validate_extension(output, "xvg"))
    return {
        group_index: mean_path.with_name(
            f"{mean_path.stem}.group{group_index + 1:02d}-"
            f"{_filename_token(group_name)}.values.csv"
        )
        for group_index, group_name in enumerate(group_names)
    }


def _validate_output_paths(args):
    paths = []
    for value in (
        args.output_verbose,
        args.output_average,
        args.output_histogram,
    ):
        if value is not None:
            paths.append(Path(validate_extension(value, "xvg")))
    if args.output_phase is not None:
        paths.extend(
            _phase_output_paths(
                args.output_phase, not args.no_phase_observations
            ).values()
        )
    resolved = [path.resolve() for path in paths]
    if len(resolved) != len(set(resolved)):
        raise ValueError("Gyrate output paths must be distinct.")


def _select_phase_groups(trajectory, args):
    if args.reference_group is None:
        trajectory.index.print_all()
        reference, reference_name = trajectory.getSelection_interactive(
            "reference group used to locate and fit the dense slab"
        )
    else:
        reference, reference_name = trajectory.getSelection(
            f"group {args.reference_group}",
            "reference group used to locate and fit the dense slab",
        )
    if len(reference) == 0:
        raise ValueError("Reference selection is empty.")

    if args.selection_calculate:
        groups = []
        for group_id in args.selection_calculate:
            selection, name = trajectory.getSelection(
                f"group {group_id}", "molecules used for phase-resolved Rg"
            )
            if len(selection) == 0:
                raise ValueError(f"Selection group {group_id} is empty.")
            groups.append((name, selection))
    else:
        groups = [(reference_name, reference)]
    return reference, reference_name, groups


def _write_phase_mean_xvg(path, times_ns, phase_series, group_names, metadata):
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# DROPPS phase-resolved molecular radius of gyration\n")
        handle.write(
            f"# reference_group={metadata['reference_group']} axis={metadata['axis']} "
            f"frames={metadata['frames']} center_mode={metadata['center_mode']} "
            f"centering_groups={metadata['centering_groups']} "
            f"dense_phase_threshold={metadata['dense_phase_threshold']:.12g} "
            f"interface_threshold={metadata['interface_threshold']:.12g}\n"
        )
        handle.write(
            "# Values are frame-local arithmetic means over molecules currently "
            "classified in each phase; nan means that no molecule occupied that phase.\n"
        )
        handle.write('@    title "Phase-resolved molecular radius of gyration"\n')
        handle.write('@    xaxis  label "Time (ns)"\n')
        handle.write('@    yaxis  label "Mean radius of gyration (nm)"\n')
        handle.write("@TYPE xy\n")
        for series_id, (group_index, phase) in enumerate(phase_series["series"]):
            handle.write(
                f'@ s{series_id} legend "{group_names[group_index]} '
                f'{STATE_NAMES[phase]}"\n'
            )
        for frame_index, time_ns in enumerate(times_ns):
            fields = [_format_scalar(time_ns)]
            fields.extend(
                _format_scalar(value)
                for value in phase_series["mean_rg_nm"][frame_index]
            )
            handle.write("\t".join(fields) + "\n")


def _write_phase_observations_csv(
    path,
    times_ns,
    coordinates_nm,
    states,
    rg_nm,
    molecules,
):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "frame",
                "time_ns",
                "group",
                "molecule_id",
                "chain_id",
                "coordinate_nm",
                "phase",
                "rg_nm",
            )
        )
        for frame_index, time_ns in enumerate(times_ns):
            for molecule_index, molecule in enumerate(molecules):
                writer.writerow(
                    (
                        frame_index,
                        _format_scalar(time_ns),
                        molecule["group"],
                        molecule["molecule_id"],
                        molecule["chain_id"],
                        _format_scalar(coordinates_nm[frame_index, molecule_index]),
                        STATE_NAMES[int(states[frame_index, molecule_index])],
                        _format_scalar(rg_nm[frame_index, molecule_index]),
                    )
                )


def _write_group_phase_values(paths, states, rg_nm, molecule_groups):
    molecule_groups = np.asarray(molecule_groups, dtype=int)
    for group_index, path in paths.items():
        group_mask = molecule_groups == group_index
        dense_values = rg_nm[(states == DENSE) & group_mask[np.newaxis, :]]
        dilute_values = rg_nm[(states == DILUTE) & group_mask[np.newaxis, :]]
        row_count = max(dense_values.size, dilute_values.size)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("dense_rg_nm", "dilute_rg_nm"))
            for row_index in range(row_count):
                writer.writerow(
                    (
                        (
                            _format_scalar(dense_values[row_index])
                            if row_index < dense_values.size
                            else ""
                        ),
                        (
                            _format_scalar(dilute_values[row_index])
                            if row_index < dilute_values.size
                            else ""
                        ),
                    )
                )


def _write_phase_summary_csv(path, rows, group_names):
    fields = (
        "group",
        "phase",
        "molecule_count",
        "observations",
        "occupancy_fraction",
        "occupied_frames",
        "observed_molecules",
        "mean_rg_nm",
        "std_rg_nm",
        "median_rg_nm",
        "min_rg_nm",
        "max_rg_nm",
        "block_sem_rg_nm",
        "blocks_used",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "group": group_names[row["group_index"]],
                    "phase": STATE_NAMES[row["phase"]],
                    **{
                        key: (
                            _format_scalar(value)
                            if isinstance(value, (float, np.floating))
                            else value
                        )
                        for key, value in row.items()
                        if key not in {"group_index", "phase"}
                    },
                }
            )


def _write_phase_histogram_xvg(path, histogram, group_names):
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# DROPPS phase-resolved molecular Rg distributions\n")
        handle.write('@    title "Phase-resolved molecular Rg distributions"\n')
        handle.write('@    xaxis  label "Radius of gyration (nm)"\n')
        handle.write('@    yaxis  label "Probability density (nm\\S-1\\N)"\n')
        handle.write("@TYPE xy\n")
        for series_id, (group_index, phase) in enumerate(histogram["series"]):
            handle.write(
                f'@ s{series_id} legend "{group_names[group_index]} '
                f'{STATE_NAMES[phase]}"\n'
            )
        for bin_index, center_nm in enumerate(histogram["bin_centers_nm"]):
            fields = [_format_scalar(center_nm)]
            fields.extend(
                _format_scalar(value)
                for value in histogram["density_per_nm"][bin_index]
            )
            handle.write("\t".join(fields) + "\n")


def _write_phase_profile_xvg(
    path,
    coordinates_nm,
    mean_profile,
    fit_statistics,
    metadata,
    dense_cutoff_nm,
    dilute_cutoff_nm,
):
    fit = fit_statistics["fit"]
    fit_sem = fit_statistics["sem"]
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# DROPPS phase-resolved Rg reference density profile\n")
        handle.write(
            f"# reference_group={metadata['reference_group']} axis={metadata['axis']} "
            f"frames={metadata['frames']} center_mode={metadata['center_mode']} "
            f"centering_groups={metadata['centering_groups']} "
            f"blocks={fit_statistics['blocks']}\n"
        )
        handle.write(
            f"# dense_phase_threshold={metadata['dense_phase_threshold']:.12g} "
            f"interface_threshold={metadata['interface_threshold']:.12g} "
            f"dense_cutoff_nm={dense_cutoff_nm:.12g} "
            f"dilute_cutoff_nm={dilute_cutoff_nm:.12g}\n"
        )
        handle.write(
            f"# fit_dense_mg_per_ml={fit['dense_density']:.12g} "
            f"fit_dilute_mg_per_ml={fit['dilute_density']:.12g} "
            f"slab_width_nm={fit['slab_width_nm']:.12g} "
            f"slab_width_block_sem_nm={fit_sem['slab_width_nm']:.12g} "
            f"interface_width_10_90_nm={fit['interface_width_10_90_nm']:.12g} "
            f"fit_r_squared={fit['r_squared']:.12g}\n"
        )
        handle.write('@    title "Phase-resolved Rg reference density profile"\n')
        handle.write('@    xaxis  label "Centered slab coordinate (nm)"\n')
        handle.write('@    yaxis  label "Mass density (mg/mL)"\n')
        handle.write("@TYPE xy\n")
        handle.write(f'@ s0 legend "{metadata["reference_group"]}"\n')
        handle.write('@ s1 legend "double-tanh fit"\n')
        for coordinate, density, fitted in zip(
            coordinates_nm, mean_profile, fit["fitted_profile"]
        ):
            handle.write(f"{coordinate:.10g}\t{density:.12g}\t{fitted:.12g}\n")


def _phase_gyrate(args, trajectory):
    if not np.isfinite(args.bin_width) or args.bin_width <= 0.0:
        raise ValueError("Histogram bin width must be positive and finite.")
    if not np.isfinite(args.density_bin_width) or args.density_bin_width <= 0.0:
        raise ValueError("Density bin width must be positive and finite.")
    if int(args.blocks) != args.blocks or args.blocks <= 0:
        raise ValueError("Number of blocks must be positive.")
    if (
        not np.isfinite(args.dense_phase_threshold)
        or not 0.0 < args.dense_phase_threshold < 1.0
    ):
        raise ValueError("Dense-phase threshold must be finite and between 0 and 1.")
    if (
        not np.isfinite(args.interface_threshold)
        or not 0.0 < args.interface_threshold < 0.5
    ):
        raise ValueError("Interface threshold must lie strictly between 0 and 0.5.")
    if len(set(args.phases)) != len(args.phases):
        raise ValueError("Phase names must not be repeated.")

    reference, reference_name, groups = _select_phase_groups(trajectory, args)
    molecules, _ = _build_molecules(trajectory, groups)
    group_names = [name for name, _ in groups]
    reference_indices = np.asarray(reference.indices, dtype=int)
    reference_weights = np.asarray(reference.masses, dtype=float)
    if (
        not np.all(np.isfinite(reference_weights))
        or np.any(reference_weights < 0.0)
        or float(reference_weights.sum()) <= 0.0
    ):
        raise ValueError("Reference group must have positive finite masses.")

    analyzed_trajectory = trajectory.Universe.trajectory
    frame_indices, actual_interval_ns = _select_frame_indices(
        analyzed_trajectory,
        args.start_time,
        args.end_time,
        args.delta_time,
    )
    first_timestep = analyzed_trajectory[frame_indices[0]]
    _, first_lengths_nm, _ = _box_geometry_nm(first_timestep)
    axis_index = AXIS_TO_INDEX[args.axis]
    bins = max(
        12,
        int(np.ceil(first_lengths_nm[axis_index] / args.density_bin_width)),
    )

    if args.center_mode == "none":
        schedule = centering_schedule(
            np.zeros(len(frame_indices), dtype=float),
            np.zeros(len(frame_indices), dtype=float),
            mode="none",
            blocks=args.blocks,
        )
    else:
        frame_centers = []
        frame_center_orders = []
        for frame_index in tqdm(frame_indices, desc="## Locating dense slab for Rg"):
            timestep = analyzed_trajectory[frame_index]
            box_nm, _, volume_nm3 = _box_geometry_nm(timestep)
            positions_nm = np.asarray(timestep.positions, dtype=float) * ANGSTROM_TO_NM
            fractional = _fractional_positions(positions_nm, box_nm)
            reference_profile = histogram_density(
                fractional[reference_indices, axis_index],
                reference_weights,
                volume_nm3,
                bins,
            )
            center, order = threshold_profile_center(
                reference_profile, args.dense_phase_threshold
            )
            frame_centers.append(center)
            frame_center_orders.append(order)
        schedule = centering_schedule(
            np.asarray(frame_centers, dtype=float),
            np.asarray(frame_center_orders, dtype=float),
            mode=args.center_mode,
            blocks=args.blocks,
        )

    frame_count = len(frame_indices)
    molecule_count = len(molecules)
    times_ns = np.empty(frame_count, dtype=float)
    normal_lengths_nm = np.empty(frame_count, dtype=float)
    reference_profiles = np.empty((frame_count, bins), dtype=float)
    molecular_coordinates_nm = np.empty((frame_count, molecule_count), dtype=float)
    molecular_rg_nm = np.empty((frame_count, molecule_count), dtype=float)
    for local_index, frame_index in enumerate(
        tqdm(frame_indices, desc="## Calculating phase-resolved molecular Rg")
    ):
        timestep = analyzed_trajectory[frame_index]
        box_nm, lengths_nm, volume_nm3 = _box_geometry_nm(timestep)
        positions_nm = np.asarray(timestep.positions, dtype=float) * ANGSTROM_TO_NM
        fractional = _fractional_positions(positions_nm, box_nm)
        centered_axis = fractional[:, axis_index]
        if args.center_mode != "none":
            centered_axis = center_periodic_fractions(
                centered_axis, schedule["centers"][local_index]
            )
        reference_profiles[local_index] = histogram_density(
            centered_axis[reference_indices],
            reference_weights,
            volume_nm3,
            bins,
            DALTON_PER_NM3_TO_MG_PER_ML,
        )
        for molecule_index, molecule in enumerate(molecules):
            molecule_fraction = periodic_weighted_mean_fraction(
                fractional[molecule["indices"], axis_index], molecule["weights"]
            )
            if args.center_mode != "none":
                molecule_fraction = float(
                    center_periodic_fractions(
                        molecule_fraction, schedule["centers"][local_index]
                    )
                )
            molecular_coordinates_nm[local_index, molecule_index] = (
                molecule_fraction - 0.5
            ) * lengths_nm[axis_index]
            molecular_rg_nm[local_index, molecule_index] = periodic_radius_of_gyration(
                positions_nm[molecule["indices"]], molecule["weights"], box_nm
            )
        times_ns[local_index] = float(timestep.time) * PS_TO_NS
        normal_lengths_nm[local_index] = float(lengths_nm[axis_index])

    mean_length_nm = float(normal_lengths_nm.mean())
    relative_length_span = float(np.ptp(normal_lengths_nm) / mean_length_nm)
    if relative_length_span > 0.01:
        print(
            "## WARNING: The interface-normal cell length varies by more than 1%. "
            "Phase distances use each frame's cell length."
        )
    coordinates_nm = (
        (np.arange(bins, dtype=float) + 0.5) / bins - 0.5
    ) * mean_length_nm
    fit_statistics = fit_slab_profile_blocks(
        coordinates_nm, reference_profiles, args.blocks
    )
    fit = fit_statistics["fit"]
    dense_cutoff_nm, dilute_cutoff_nm, _ = phase_cutoffs(
        fit["half_width_nm"],
        fit["interface_parameter_nm"],
        threshold=args.interface_threshold,
        cell_length_nm=mean_length_nm,
    )
    states = classify_phase_positions(
        molecular_coordinates_nm, dense_cutoff_nm, dilute_cutoff_nm
    )
    molecule_groups = np.asarray(
        [molecule["group_index"] for molecule in molecules], dtype=int
    )
    requested_phases = tuple(PHASE_CODES[name] for name in args.phases)
    phase_series = phase_rg_time_series(
        molecular_rg_nm,
        states,
        molecule_groups,
        len(groups),
        requested_phases,
    )
    summary_rows = phase_rg_summary(
        molecular_rg_nm,
        states,
        molecule_groups,
        len(groups),
        requested_phases,
        blocks=args.blocks,
    )
    histogram = phase_rg_histogram(
        molecular_rg_nm,
        states,
        molecule_groups,
        len(groups),
        requested_phases,
        args.bin_width,
    )
    empty_series = [row for row in summary_rows if row["observations"] == 0]
    if empty_series:
        labels = ", ".join(
            f"{group_names[row['group_index']]} {STATE_NAMES[row['phase']]}"
            for row in empty_series
        )
        print(
            f"## WARNING: No molecules occupied {labels}; corresponding means, "
            "summary values, and histogram densities are nan."
        )

    paths = _phase_output_paths(args.output_phase, not args.no_phase_observations)
    value_paths = _group_phase_value_paths(args.output_phase, group_names)
    metadata = {
        "reference_group": reference_name,
        "axis": args.axis,
        "frames": frame_count,
        "center_mode": args.center_mode,
        "centering_groups": schedule["groups"],
        "dense_phase_threshold": args.dense_phase_threshold,
        "interface_threshold": args.interface_threshold,
    }
    _write_phase_mean_xvg(paths["mean"], times_ns, phase_series, group_names, metadata)
    _write_phase_summary_csv(paths["summary"], summary_rows, group_names)
    _write_phase_histogram_xvg(paths["histogram"], histogram, group_names)
    _write_phase_profile_xvg(
        paths["profile"],
        coordinates_nm,
        reference_profiles.mean(axis=0),
        fit_statistics,
        metadata,
        dense_cutoff_nm,
        dilute_cutoff_nm,
    )
    if "observations" in paths:
        _write_phase_observations_csv(
            paths["observations"],
            times_ns,
            molecular_coordinates_nm,
            states,
            molecular_rg_nm,
            molecules,
        )
    _write_group_phase_values(value_paths, states, molecular_rg_nm, molecule_groups)

    print(
        f"## Phase-resolved Rg analyzed {frame_count} frames with median interval "
        f"{actual_interval_ns:g} ns and {molecule_count} molecules in "
        f"{len(groups)} group(s)."
    )
    print(
        f"## Fixed phase cutoffs: dense <= {dense_cutoff_nm:.6g} nm, "
        f"dilute >= {dilute_cutoff_nm:.6g} nm; fit R^2 = {fit['r_squared']:.6g}."
    )
    if fit["r_squared"] < 0.8:
        print(
            "## WARNING: Slab fit R^2 is below 0.8. Inspect the profile and verify "
            "equilibration, centering, reference selection, and box geometry."
        )
    for key in ("mean", "observations", "summary", "histogram", "profile"):
        if key in paths:
            print(f"## Phase {key} output written to {paths[key].resolve()}.")
    for group_index, path in value_paths.items():
        print(
            f"## Raw dense/dilute Rg values for {group_names[group_index]} written "
            f"to {path.resolve()}."
        )


def gyrate(args):
    if not (
        args.output_average
        or args.output_histogram
        or args.output_verbose
        or args.output_phase
    ):
        print("ERROR: No output file specified.")
        quit()
    _validate_output_paths(args)

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

    if args.output_phase is not None:
        _phase_gyrate(args, trajectory)
        if not (args.output_average or args.output_histogram or args.output_verbose):
            return

    # We treat time for legacy group-level analysis and generate frames.

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

    rg_lists = [[] for _ in selections]
    time_list = list()

    print(f"## Will calculate radius of gyration for {len(selections)} groups.")
    print("## Start of radius of gyration calculations.")

    frames = trajectory.Universe.trajectory[
        start_frame : end_frame + 1 : interval_frame
    ]
    for _ in tqdm(frames):
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
