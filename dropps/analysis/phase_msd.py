"""Phase-conditioned molecular MSD analysis for planar slabs."""

from __future__ import annotations

from dropps.share.argument_parser import ArgumentParser
import csv
from pathlib import Path

import numpy as np
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
    periodic_center_fraction,
)
from dropps.analysis.exchange import _build_molecules, _select_groups
from dropps.analysis.exchange_core import (
    DENSE,
    DILUTE,
    INTERFACE,
    STATE_NAMES,
    classify_phase_positions,
    periodic_weighted_mean_fraction,
    phase_cutoffs,
)
from dropps.analysis.phase_msd_core import (
    phase_conditioned_msd,
    unwrap_fractional_trajectories,
)
from dropps.fileio.filename_control import validate_extension
from dropps.share.command_class import single_command
from dropps.share.trajectory import trajectory_class


prog = "phase-msd"
desc = "Calculate phase-conditioned molecular MSD with phase-translation removal in a planar slab."

MSD_DIMENSIONS = {
    "xyz": (0, 1, 2),
    "xy": (0, 1),
    "yz": (1, 2),
    "xz": (0, 2),
    "x": (0,),
    "y": (1,),
    "z": (2,),
}
PHASE_CODES = {"dense": DENSE, "interface": INTERFACE, "dilute": DILUTE}


def _format_scalar(value):
    value = float(value)
    if np.isnan(value):
        return "nan"
    if np.isposinf(value):
        return "inf"
    if np.isneginf(value):
        return "-inf"
    return f"{value:.12g}"


def _output_paths(output):
    msd_path = Path(validate_extension(output, "xvg"))
    csv_path = msd_path.with_suffix(".csv")
    profile_path = msd_path.with_name(f"{msd_path.stem}.profile.xvg")
    paths = {"msd": msd_path, "csv": csv_path, "profile": profile_path}
    resolved = [path.resolve() for path in paths.values()]
    if len(set(resolved)) != len(resolved):
        raise ValueError("Phase-MSD output paths must be distinct.")
    return paths


def _write_profile_xvg(
    path,
    coordinates_nm,
    mean_profile,
    fit_statistics,
    metadata,
    dense_cutoff_nm,
    dilute_cutoff_nm,
):
    fit = fit_statistics["fit"]
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# DROPPS phase-MSD reference density profile\n")
        handle.write(
            f"# reference_group={metadata['reference_group']} "
            f"axis={metadata['axis']} frames={metadata['frames']} "
            f"center_mode={metadata['center_mode']} "
            f"centering_groups={metadata['centering_groups']} "
            f"blocks={fit_statistics['blocks']}\n"
        )
        handle.write(
            f"# interface_threshold={metadata['interface_threshold']:.12g} "
            f"dense_cutoff_nm={dense_cutoff_nm:.12g} "
            f"dilute_cutoff_nm={dilute_cutoff_nm:.12g} "
            f"fit_r_squared={fit['r_squared']:.12g}\n"
        )
        handle.write(
            f"# fit_dense_mg_per_ml={fit['dense_density']:.12g} "
            f"fit_dilute_mg_per_ml={fit['dilute_density']:.12g} "
            f"slab_width_nm={fit['slab_width_nm']:.12g} "
            f"interface_width_10_90_nm={fit['interface_width_10_90_nm']:.12g}\n"
        )
        handle.write('@    title "Phase-MSD reference density profile"\n')
        handle.write('@    xaxis  label "Centered slab coordinate (nm)"\n')
        handle.write('@    yaxis  label "Mass density (mg/mL)"\n')
        handle.write("@TYPE xy\n")
        handle.write(f'@ s0 legend "{metadata["reference_group"]}"\n')
        handle.write('@ s1 legend "double-tanh fit"\n')
        for coordinate, density, fitted in zip(
            coordinates_nm, mean_profile, fit["fitted_profile"]
        ):
            handle.write(f"{coordinate:.10g}\t{density:.12g}\t{fitted:.12g}\n")


def _write_msd_xvg(path, results, metadata):
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# DROPPS phase-conditioned molecular MSD\n")
        handle.write(
            "# intrinsic_msd = raw_msd - mass-weighted cohort translation_msd\n"
        )
        handle.write(
            f"# dimensions={metadata['msd_type']} axis={metadata['axis']} "
            f"center_mode={metadata['center_mode']} "
            f"interface_threshold={metadata['interface_threshold']:.12g} "
            f"min_particles={metadata['min_particles']} "
            f"blocks={metadata['blocks']}\n"
        )
        handle.write(
            "# Centering is used only for phase classification; dynamics use "
            "periodically unwrapped laboratory-frame molecular centers.\n"
        )
        handle.write('@    title "Phase-conditioned molecular MSD"\n')
        handle.write('@    xaxis  label "Lag time (ns)"\n')
        handle.write('@    yaxis  label "MSD (nm\\S2\\N)"\n')
        handle.write("@TYPE xy\n")
        series = []
        for result in results:
            label = (
                f"{result['group']} {result['phase']} "
                f"{result['conditioning']}"
            )
            series.append((result, "intrinsic_msd_nm2"))
            series.append((result, "raw_msd_nm2"))
            handle.write(f'@ s{len(series) - 2} legend "{label} intrinsic"\n')
            handle.write(f'@ s{len(series) - 1} legend "{label} raw"\n')

        row_count = len(results[0]["rows"])
        for row_index in range(row_count):
            fields = [
                _format_scalar(results[0]["rows"][row_index]["lag_time_ns"])
            ]
            for result, value_key in series:
                fields.append(_format_scalar(result["rows"][row_index][value_key]))
            handle.write("\t".join(fields) + "\n")


def _write_msd_csv(path, results):
    fields = (
        "group",
        "phase",
        "conditioning",
        "lag_frames",
        "lag_time_ns",
        "raw_msd_nm2",
        "intrinsic_msd_nm2",
        "translation_msd_nm2",
        "raw_block_sem_nm2",
        "intrinsic_block_sem_nm2",
        "translation_block_sem_nm2",
        "contributing_pairs",
        "time_origins",
        "mean_particles_per_origin",
        "blocks_used",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            for row in result["rows"]:
                writer.writerow(
                    {
                        "group": result["group"],
                        "phase": result["phase"],
                        "conditioning": result["conditioning"],
                        **{
                            key: (
                                _format_scalar(value)
                                if isinstance(value, (float, np.floating))
                                else value
                            )
                            for key, value in row.items()
                        },
                    }
                )


def getargs_phase_msd(argv):
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
        help="Output MSD file (.xvg); matching .csv and .profile.xvg files are also written.",
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
        help="Index groups whose molecules are tracked; if omitted, use the reference group.",
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
            "Centering used for phase classification: frame, contiguous block, "
            "or none. Dynamics are always calculated from unwrapped raw molecular "
            "positions."
        ),
    )
    parser.add_argument(
        "-t",
        "--msd-type",
        choices=tuple(MSD_DIMENSIONS),
        default="xyz",
        help="Cartesian dimensions included in the MSD.",
    )
    parser.add_argument(
        "--phases",
        choices=tuple(PHASE_CODES),
        nargs="+",
        default=("dense", "dilute"),
        help="Phase regions to analyze.",
    )
    parser.add_argument(
        "--conditioning",
        choices=("continuous", "intermittent", "both"),
        default="both",
        help=(
            "continuous requires phase membership throughout each interval; "
            "intermittent requires it only at both endpoints."
        ),
    )
    parser.add_argument(
        "--max-lag",
        type=float,
        help="Maximum lag time, in ns; if omitted, use half the analyzed trajectory.",
    )
    parser.add_argument(
        "--min-particles",
        type=int,
        default=2,
        help=(
            "Minimum eligible molecules at one time origin for phase-translation "
            "removal."
        ),
    )
    parser.add_argument(
        "--bin-width",
        type=float,
        default=0.2,
        help="Requested reference density-bin width, in nm.",
    )
    parser.add_argument(
        "--interface-threshold",
        type=float,
        default=0.1,
        help=(
            "Normalized dilute-side threshold defining the interface state; 0.1 "
            "uses the fitted 10--90%% region. Must be between 0 and 0.5."
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
        help="Number of contiguous time-origin blocks for standard errors.",
    )
    return parser.parse_args(argv)


def phase_msd(args):
    if not np.isfinite(args.bin_width) or args.bin_width <= 0.0:
        raise ValueError("Bin width must be positive and finite.")
    if args.blocks <= 0:
        raise ValueError("Number of blocks must be positive.")
    if args.min_particles <= 0:
        raise ValueError("Minimum particles must be positive.")
    if (
        not np.isfinite(args.interface_threshold)
        or not 0.0 < args.interface_threshold < 0.5
    ):
        raise ValueError("Interface threshold must lie strictly between 0 and 0.5.")
    if args.max_lag is not None and (
        not np.isfinite(args.max_lag) or args.max_lag <= 0.0
    ):
        raise ValueError("Maximum lag time must be positive and finite.")

    trajectory = trajectory_class(args.run_input, args.index, args.input)
    reference, reference_name, groups = _select_groups(
        trajectory, args, "molecules used for phase-MSD analysis"
    )
    molecules, _ = _build_molecules(trajectory, groups)
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
    bins = max(12, int(np.ceil(first_lengths_nm[axis_index] / args.bin_width)))

    if args.center_mode == "none":
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
        for frame_index in tqdm(frame_indices, desc="## Locating dense slab"):
            timestep = analyzed_trajectory[frame_index]
            box_nm, _, _ = _box_geometry_nm(timestep)
            positions_nm = np.asarray(timestep.positions, dtype=float) * ANGSTROM_TO_NM
            fractional = _fractional_positions(positions_nm, box_nm)
            center, order = periodic_center_fraction(
                fractional[reference_indices, axis_index], reference_weights
            )
            frame_centers.append(center)
            frame_center_orders.append(order)
        schedule = centering_schedule(
            np.asarray(frame_centers, dtype=float),
            np.asarray(frame_center_orders, dtype=float),
            mode=args.center_mode,
            blocks=args.blocks,
        )

    times_ns = []
    box_lengths_nm = []
    reference_profiles = []
    molecule_fractions = np.empty(
        (len(frame_indices), len(molecules), 3), dtype=float
    )
    molecular_normal_coordinates_nm = np.empty(
        (len(frame_indices), len(molecules)), dtype=float
    )
    for local_index, frame_index in enumerate(
        tqdm(frame_indices, desc="## Tracking phase-resolved dynamics")
    ):
        timestep = analyzed_trajectory[frame_index]
        box_nm, lengths_nm, volume_nm3 = _box_geometry_nm(timestep)
        positions_nm = np.asarray(timestep.positions, dtype=float) * ANGSTROM_TO_NM
        fractional = _fractional_positions(positions_nm, box_nm)
        centered_axis = (
            fractional[:, axis_index]
            if args.center_mode == "none"
            else center_periodic_fractions(
                fractional[:, axis_index], schedule["centers"][local_index]
            )
        )
        reference_profiles.append(
            histogram_density(
                centered_axis[reference_indices],
                reference_weights,
                volume_nm3,
                bins,
                DALTON_PER_NM3_TO_MG_PER_ML,
            )
        )

        for molecule_index, molecule in enumerate(molecules):
            for coordinate_index in range(3):
                molecule_fractions[local_index, molecule_index, coordinate_index] = (
                    periodic_weighted_mean_fraction(
                        fractional[molecule["indices"], coordinate_index],
                        molecule["weights"],
                    )
                )
            centered_molecule_fraction = molecule_fractions[
                local_index, molecule_index, axis_index
            ]
            if args.center_mode != "none":
                centered_molecule_fraction = float(
                    center_periodic_fractions(
                        centered_molecule_fraction,
                        schedule["centers"][local_index],
                    )
                )
            molecular_normal_coordinates_nm[local_index, molecule_index] = (
                centered_molecule_fraction - 0.5
            ) * lengths_nm[axis_index]

        times_ns.append(float(timestep.time) * PS_TO_NS)
        box_lengths_nm.append(lengths_nm)

    times_ns = np.asarray(times_ns, dtype=float)
    box_lengths_nm = np.asarray(box_lengths_nm, dtype=float)
    reference_profiles = np.asarray(reference_profiles, dtype=float)
    mean_lengths_nm = box_lengths_nm.mean(axis=0)
    relative_length_span = np.max(
        np.ptp(box_lengths_nm, axis=0) / mean_lengths_nm
    )
    if relative_length_span > 0.01:
        print(
            "## WARNING: A cell length varies by more than 1%. Phase MSD excludes "
            "pure affine box rescaling and is best interpreted for fixed-volume "
            "trajectories."
        )

    coordinates_nm = (
        (np.arange(bins, dtype=float) + 0.5) / bins - 0.5
    ) * mean_lengths_nm[axis_index]
    fit_statistics = fit_slab_profile_blocks(
        coordinates_nm, reference_profiles, args.blocks
    )
    fit = fit_statistics["fit"]
    dense_cutoff_nm, dilute_cutoff_nm, _ = phase_cutoffs(
        fit["half_width_nm"],
        fit["interface_parameter_nm"],
        threshold=args.interface_threshold,
        cell_length_nm=mean_lengths_nm[axis_index],
    )
    states = classify_phase_positions(
        molecular_normal_coordinates_nm, dense_cutoff_nm, dilute_cutoff_nm
    )
    unwrapped_positions_nm = unwrap_fractional_trajectories(
        molecule_fractions, box_lengths_nm
    )

    if args.max_lag is None:
        max_lag_frames = len(frame_indices) // 2
    else:
        max_lag_frames = int(np.floor(args.max_lag / actual_interval_ns + 1.0e-10))
        max_lag_frames = min(max_lag_frames, len(frame_indices) - 1)
        if max_lag_frames < 1:
            raise ValueError(
                "Maximum lag is shorter than the analyzed frame interval."
            )
    conditioning_modes = (
        ("continuous", "intermittent")
        if args.conditioning == "both"
        else (args.conditioning,)
    )
    molecule_groups = np.asarray(
        [molecule["group_index"] for molecule in molecules], dtype=int
    )
    molecule_masses = np.asarray(
        [float(molecule["weights"].sum()) for molecule in molecules], dtype=float
    )
    results = []
    for group_index, (group_name, _) in enumerate(groups):
        group_mask = molecule_groups == group_index
        for phase_name in dict.fromkeys(args.phases):
            phase_code = PHASE_CODES[phase_name]
            for conditioning in conditioning_modes:
                rows = phase_conditioned_msd(
                    unwrapped_positions_nm[:, group_mask],
                    states[:, group_mask],
                    molecule_masses[group_mask],
                    phase_code,
                    dimensions=MSD_DIMENSIONS[args.msd_type],
                    conditioning=conditioning,
                    max_lag_frames=max_lag_frames,
                    min_particles=args.min_particles,
                    blocks=args.blocks,
                )
                for row in rows:
                    lag = row["lag_frames"]
                    row["lag_time_ns"] = (
                        0.0
                        if lag == 0
                        else float(np.mean(times_ns[lag:] - times_ns[:-lag]))
                    )
                results.append(
                    {
                        "group": group_name,
                        "phase": STATE_NAMES[phase_code],
                        "conditioning": conditioning,
                        "rows": rows,
                    }
                )

    paths = _output_paths(args.output)
    metadata = {
        "reference_group": reference_name,
        "axis": args.axis,
        "frames": len(frame_indices),
        "center_mode": args.center_mode,
        "centering_groups": schedule["groups"],
        "interface_threshold": args.interface_threshold,
        "msd_type": args.msd_type,
        "min_particles": args.min_particles,
        "blocks": args.blocks,
    }
    _write_msd_xvg(paths["msd"], results, metadata)
    _write_msd_csv(paths["csv"], results)
    _write_profile_xvg(
        paths["profile"],
        coordinates_nm,
        reference_profiles.mean(axis=0),
        fit_statistics,
        metadata,
        dense_cutoff_nm,
        dilute_cutoff_nm,
    )

    missing_curves = sum(
        not any(np.isfinite(row["intrinsic_msd_nm2"]) for row in result["rows"][1:])
        for result in results
    )
    print(
        f"## Analyzed {len(frame_indices)} frames every {actual_interval_ns:g} ns; "
        f"tracked {len(molecules)} molecules to {max_lag_frames} lag frames."
    )
    print(
        "## Phase translation was removed separately for every group, phase, "
        "conditioning mode, time origin, and lag using its eligible molecular cohort."
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
    if missing_curves:
        print(
            f"## WARNING: {missing_curves} phase-conditioned curves have no valid "
            "nonzero lag because fewer than --min-particles molecules qualified."
        )
    print(f"## Phase-MSD curves written to {paths['msd'].resolve()}.")
    print(f"## Detailed phase-MSD table written to {paths['csv'].resolve()}.")
    print(f"## Reference phase profile written to {paths['profile'].resolve()}.")


phase_msd_commands = single_command("phase-msd", getargs_phase_msd, phase_msd, desc)
