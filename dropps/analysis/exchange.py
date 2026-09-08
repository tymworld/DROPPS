"""Molecular exchange and phase-residence analysis for planar slabs."""

from __future__ import annotations

from dropps.share.argument_parser import ArgumentParser
import csv
from pathlib import Path

import numpy as np
from tqdm import tqdm

from dropps.analysis.density_core import (
    ANGSTROM_TO_NM,
    AXIS_TO_INDEX,
    DALTON_PER_NM3_TO_MG_PER_ML,
    PS_TO_NS,
    box_geometry_nm as _box_geometry_nm,
    fractional_positions as _fractional_positions,
    select_frame_indices,
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
from dropps.analysis.exchange_core import (
    DENSE,
    DILUTE,
    INTERFACE,
    STATE_NAMES,
    classify_phase_positions,
    extract_exchange_events,
    extract_phase_episodes,
    kaplan_meier,
    periodic_weighted_mean_fraction,
    phase_cutoffs,
)
from dropps.share.command_class import single_command
from dropps.share.trajectory import trajectory_class


prog = "exchange"
desc = "Analyze molecular exchange and phase residence times across a planar slab interface."


def _select_frame_indices(trajectory, start_ns, end_ns, delta_ns):
    return select_frame_indices(
        trajectory,
        start_ns,
        end_ns,
        delta_ns,
        minimum_frames=2,
    )


def _format_scalar(value):
    value = float(value)
    if np.isnan(value):
        return "nan"
    if np.isposinf(value):
        return "inf"
    if np.isneginf(value):
        return "-inf"
    return f"{value:.12g}"


def _select_groups(
    trajectory,
    args,
    selection_help="molecules used for exchange analysis",
):
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

    if args.selection_groups:
        groups = []
        for group_id in args.selection_groups:
            selection, name = trajectory.getSelection(
                f"group {group_id}", selection_help
            )
            if len(selection) == 0:
                raise ValueError(f"Selection group {group_id} is empty.")
            groups.append((name, selection))
    else:
        groups = [(reference_name, reference)]
    return reference, reference_name, groups


def _build_molecules(trajectory, groups):
    molecules = []
    group_sizes = []
    for group_index, (group_name, selection) in enumerate(groups):
        molecule_indices = trajectory.index.splitch_indices(
            np.asarray(selection.indices, dtype=int).tolist()
        )
        if not molecule_indices:
            raise ValueError(f"Selection '{group_name}' contains no molecules.")
        group_sizes.append(len(molecule_indices))
        for molecule_id, atom_indices in enumerate(molecule_indices):
            indices = np.asarray(atom_indices, dtype=int)
            weights = np.asarray(trajectory.Universe.atoms[indices].masses, dtype=float)
            if (
                not np.all(np.isfinite(weights))
                or np.any(weights < 0.0)
                or float(weights.sum()) <= 0.0
            ):
                raise ValueError(
                    f"Molecule {molecule_id} in '{group_name}' must have positive "
                    "finite selected-atom masses."
                )
            molecules.append(
                {
                    "group_index": group_index,
                    "group": group_name,
                    "molecule_id": molecule_id,
                    "chain_id": int(trajectory.get_chainID(int(indices[0]))),
                    "indices": indices,
                    "weights": weights,
                }
            )
    return molecules, group_sizes


def _output_paths(prefix, write_states):
    prefix = Path(prefix)
    suffixes = {
        "profile": ".profile.xvg",
        "events": ".events.csv",
        "residence": ".residence.csv",
        "survival": ".survival.csv",
        "summary": ".summary.csv",
    }
    if write_states:
        suffixes["states"] = ".states.csv"
    paths = {key: Path(f"{prefix}{suffix}") for key, suffix in suffixes.items()}
    resolved = [path.resolve() for path in paths.values()]
    if len(set(resolved)) != len(resolved):
        raise ValueError("Exchange output paths must be distinct.")
    return paths


def _write_profile_xvg(
    path,
    coordinates_nm,
    profile,
    fit_statistics,
    metadata,
    dense_cutoff_nm,
    dilute_cutoff_nm,
):
    fit = fit_statistics["fit"]
    fit_sem = fit_statistics["sem"]
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# DROPPS phase-exchange reference density profile\n")
        handle.write(
            f"# reference_group={metadata['reference_group']} "
            f"axis={metadata['axis']} frames={metadata['frames']} "
            f"center_mode={metadata['center_mode']} "
            f"centering_groups={metadata['centering_groups']} "
            f"blocks={fit_statistics['blocks']}\n"
        )
        handle.write(
            f"# interface_threshold={metadata['interface_threshold']:.12g} "
            f"dense_phase_threshold={metadata['dense_phase_threshold']:.12g} "
            f"dense_cutoff_nm={dense_cutoff_nm:.12g} "
            f"dilute_cutoff_nm={dilute_cutoff_nm:.12g}\n"
        )
        handle.write(
            f"# fit_dense_mg_per_ml={fit['dense_density']:.12g} "
            f"fit_dilute_mg_per_ml={fit['dilute_density']:.12g} "
            f"slab_width_nm={fit['slab_width_nm']:.12g} "
            f"slab_width_block_sem_nm={fit_sem['slab_width_nm']:.12g}\n"
        )
        handle.write(
            f"# interface_parameter_nm={fit['interface_parameter_nm']:.12g} "
            f"interface_width_10_90_nm={fit['interface_width_10_90_nm']:.12g} "
            f"interface_width_block_sem_nm="
            f"{fit_sem['interface_width_10_90_nm']:.12g} "
            f"fit_r_squared={fit['r_squared']:.12g}\n"
        )
        handle.write('@    title "Exchange reference density profile"\n')
        handle.write('@    xaxis  label "Centered slab coordinate (nm)"\n')
        handle.write('@    yaxis  label "Mass density (mg/mL)"\n')
        handle.write("@TYPE xy\n")
        handle.write(f'@ s0 legend "{metadata["reference_group"]}"\n')
        handle.write('@ s1 legend "double-tanh fit"\n')
        for coordinate, density, fitted in zip(
            coordinates_nm, profile, fit["fitted_profile"]
        ):
            handle.write(f"{coordinate:.10g}\t{density:.12g}\t{fitted:.12g}\n")


def _write_states_csv(path, times_ns, coordinates_nm, states, molecules):
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
                    )
                )


def _write_residence_csv(path, episodes, molecules):
    fields = (
        "group",
        "molecule_id",
        "chain_id",
        "phase",
        "start_frame",
        "end_frame",
        "start_time_ns",
        "end_time_ns",
        "duration_ns",
        "left_censored",
        "right_censored",
        "completed",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for episode in episodes:
            molecule = molecules[episode["molecule_index"]]
            writer.writerow(
                {
                    "group": molecule["group"],
                    "molecule_id": molecule["molecule_id"],
                    "chain_id": molecule["chain_id"],
                    "phase": STATE_NAMES[episode["phase"]],
                    "start_frame": episode["start_frame"],
                    "end_frame": episode["end_frame"],
                    "start_time_ns": _format_scalar(episode["start_time_ns"]),
                    "end_time_ns": _format_scalar(episode["end_time_ns"]),
                    "duration_ns": _format_scalar(episode["duration_ns"]),
                    "left_censored": int(episode["left_censored"]),
                    "right_censored": int(episode["right_censored"]),
                    "completed": int(
                        not episode["left_censored"] and not episode["right_censored"]
                    ),
                }
            )


def _write_events_csv(path, events, molecules):
    fields = (
        "group",
        "molecule_id",
        "chain_id",
        "source_phase",
        "destination_phase",
        "departure_time_ns",
        "arrival_time_ns",
        "transition_time_ns",
        "waiting_time_since_last_exchange_ns",
        "waiting_left_censored",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for event in events:
            molecule = molecules[event["molecule_index"]]
            writer.writerow(
                {
                    "group": molecule["group"],
                    "molecule_id": molecule["molecule_id"],
                    "chain_id": molecule["chain_id"],
                    "source_phase": STATE_NAMES[event["source_phase"]],
                    "destination_phase": STATE_NAMES[event["destination_phase"]],
                    "departure_time_ns": _format_scalar(event["departure_time_ns"]),
                    "arrival_time_ns": _format_scalar(event["arrival_time_ns"]),
                    "transition_time_ns": _format_scalar(event["transition_time_ns"]),
                    "waiting_time_since_last_exchange_ns": _format_scalar(
                        event["waiting_time_since_last_exchange_ns"]
                    ),
                    "waiting_left_censored": int(event["waiting_left_censored"]),
                }
            )


def _group_phase_statistics(
    group_names,
    group_sizes,
    molecules,
    times_ns,
    episodes,
    events,
):
    observation_time = float(times_ns[-1] - times_ns[0])
    molecule_groups = np.asarray(
        [molecule["group_index"] for molecule in molecules], dtype=int
    )
    survival_rows = []
    summary_rows = []

    for group_index, (group_name, molecule_count) in enumerate(
        zip(group_names, group_sizes)
    ):
        group_molecule_indices = set(np.flatnonzero(molecule_groups == group_index))
        for phase in (DENSE, INTERFACE, DILUTE):
            selected_episodes = [
                row
                for row in episodes
                if row["molecule_index"] in group_molecule_indices
                and row["phase"] == phase
            ]
            exposure_ns = float(sum(row["duration_ns"] for row in selected_episodes))
            completed = [
                row["duration_ns"]
                for row in selected_episodes
                if not row["left_censored"] and not row["right_censored"]
            ]
            km_episodes = [row for row in selected_episodes if not row["left_censored"]]
            km_rows = kaplan_meier(
                [row["duration_ns"] for row in km_episodes],
                [not row["right_censored"] for row in km_episodes],
            )
            for row in km_rows:
                survival_rows.append(
                    {
                        "group": group_name,
                        "phase": STATE_NAMES[phase],
                        **row,
                    }
                )
            km_median = next(
                (
                    row["time_ns"]
                    for row in km_rows
                    if row["time_ns"] > 0.0 and row["survival"] <= 0.5
                ),
                float("nan"),
            )
            exchange_count = sum(
                event["molecule_index"] in group_molecule_indices
                and event["source_phase"] == phase
                for event in events
            )
            event_rate = (
                exchange_count / exposure_ns * 1000.0
                if phase in (DENSE, DILUTE) and exposure_ns > 0.0
                else float("nan")
            )
            summary_rows.append(
                {
                    "group": group_name,
                    "phase": STATE_NAMES[phase],
                    "molecule_count": molecule_count,
                    "occupancy_fraction": (
                        exposure_ns / (molecule_count * observation_time)
                    ),
                    "exposure_ns": exposure_ns,
                    "completed_residences": len(completed),
                    "mean_completed_residence_ns": (
                        float(np.mean(completed)) if completed else float("nan")
                    ),
                    "median_completed_residence_ns": (
                        float(np.median(completed)) if completed else float("nan")
                    ),
                    "km_median_ns": km_median,
                    "exits_to_opposite_bulk": exchange_count,
                    "event_rate_per_exposure_us": event_rate,
                }
            )
    return survival_rows, summary_rows


def _write_survival_csv(path, rows):
    fields = ("group", "phase", "time_ns", "survival", "at_risk", "events", "censored")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "group": row["group"],
                    "phase": row["phase"],
                    "time_ns": _format_scalar(row["time_ns"]),
                    "survival": _format_scalar(row["survival"]),
                    "at_risk": row["at_risk"],
                    "events": row["events"],
                    "censored": row["censored"],
                }
            )


def _write_summary_csv(path, rows):
    fields = (
        "group",
        "phase",
        "molecule_count",
        "occupancy_fraction",
        "exposure_ns",
        "completed_residences",
        "mean_completed_residence_ns",
        "median_completed_residence_ns",
        "km_median_ns",
        "exits_to_opposite_bulk",
        "event_rate_per_exposure_us",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        _format_scalar(value)
                        if isinstance(value, (float, np.floating))
                        else value
                    )
                    for key, value in row.items()
                }
            )


def getargs_exchange(argv):
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
        "-n",
        "--index",
        help="Optional index file (.ndx) defining additional atom groups.",
    )
    parser.add_argument(
        "-o",
        "--output-prefix",
        required=True,
        help="Output prefix for .profile.xvg, .events.csv, .residence.csv, .survival.csv, .summary.csv, and optional .states.csv files.",
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
            "Slab centering strategy: frame aligns every analyzed frame; block "
            "uses one center per contiguous --blocks "
            "segment; none expects an already centered slab."
        ),
    )
    parser.add_argument(
        "--bin-width",
        type=float,
        default=DEFAULT_DENSITY_BIN_WIDTH_NM,
        help="Requested reference density-bin width, in nm; shared with density.",
    )
    parser.add_argument(
        "--dense-phase-threshold",
        type=float,
        default=0.5,
        help=(
            "Dense-phase threshold as a fraction of the maximum reference "
            "density; shared with the density command."
        ),
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
        "--blocks",
        type=int,
        default=5,
        help="Number of fit/statistics blocks and block-centering groups.",
    )
    parser.add_argument(
        "--no-states",
        action="store_true",
        help="Skip the potentially large per-frame .states.csv output file.",
    )
    return parser.parse_args(argv)


def exchange(args):
    if not np.isfinite(args.bin_width) or args.bin_width <= 0.0:
        raise ValueError("Bin width must be positive and finite.")
    if args.blocks <= 0:
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

    trajectory = trajectory_class(args.run_input, args.index, args.input)
    reference, reference_name, groups = _select_groups(trajectory, args)
    molecules, group_sizes = _build_molecules(trajectory, groups)
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
    bins = max(12, int(np.ceil(first_lengths_nm[axis_index] / args.bin_width)))

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

    times_ns = []
    normal_lengths_nm = []
    reference_profiles = []
    molecular_coordinates_nm = np.empty(
        (len(frame_indices), len(molecules)), dtype=float
    )
    for local_index, frame_index in enumerate(
        tqdm(frame_indices, desc="## Tracking molecular phases")
    ):
        timestep = analyzed_trajectory[frame_index]
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

        times_ns.append(float(timestep.time) * PS_TO_NS)
        normal_lengths_nm.append(float(lengths_nm[axis_index]))

    times_ns = np.asarray(times_ns, dtype=float)
    normal_lengths_nm = np.asarray(normal_lengths_nm, dtype=float)
    reference_profiles = np.asarray(reference_profiles, dtype=float)
    mean_length_nm = float(normal_lengths_nm.mean())
    relative_length_span = float(np.ptp(normal_lengths_nm) / mean_length_nm)
    if relative_length_span > 0.01:
        print(
            "## WARNING: The interface-normal cell length varies by more than 1%. "
            "The profile is fitted in fractional bins while molecular distances "
            "use each frame's cell length."
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
    episodes = extract_phase_episodes(times_ns, states)
    events = extract_exchange_events(times_ns, states)
    survival_rows, summary_rows = _group_phase_statistics(
        group_names,
        group_sizes,
        molecules,
        times_ns,
        episodes,
        events,
    )

    paths = _output_paths(args.output_prefix, not args.no_states)
    metadata = {
        "reference_group": reference_name,
        "axis": args.axis,
        "frames": len(frame_indices),
        "center_mode": args.center_mode,
        "centering_groups": schedule["groups"],
        "interface_threshold": args.interface_threshold,
        "dense_phase_threshold": args.dense_phase_threshold,
    }
    _write_profile_xvg(
        paths["profile"],
        coordinates_nm,
        reference_profiles.mean(axis=0),
        fit_statistics,
        metadata,
        dense_cutoff_nm,
        dilute_cutoff_nm,
    )
    _write_events_csv(paths["events"], events, molecules)
    _write_residence_csv(paths["residence"], episodes, molecules)
    _write_survival_csv(paths["survival"], survival_rows)
    _write_summary_csv(paths["summary"], summary_rows)
    if "states" in paths:
        _write_states_csv(
            paths["states"], times_ns, molecular_coordinates_nm, states, molecules
        )

    print(
        f"## Analyzed {len(frame_indices)} frames with median interval "
        f"{actual_interval_ns:g} ns; "
        f"tracked {len(molecules)} molecules and confirmed {len(events)} exchanges."
    )
    print(
        f"## Fixed phase cutoffs from the mean reference profile: "
        f"dense <= {dense_cutoff_nm:.6g} nm, "
        f"dilute >= {dilute_cutoff_nm:.6g} nm; fit R^2 = "
        f"{fit['r_squared']:.6g}."
    )
    if fit["r_squared"] < 0.8:
        print(
            "## WARNING: Slab fit R^2 is below 0.8. Inspect the profile and verify "
            "equilibration, centering, reference selection, and box geometry."
        )
    zero_time_events = sum(event["transition_time_ns"] == 0.0 for event in events)
    if zero_time_events:
        print(
            f"## WARNING: {zero_time_events} exchange events crossed directly between "
            "bulk states at the sampled resolution; their transition time is reported "
            "as zero. Analyze more frequent frames to resolve the interface passage."
        )
    for key in ("profile", "events", "residence", "survival", "summary", "states"):
        if key in paths:
            print(f"## {key.capitalize()} output written to {paths[key].resolve()}.")


exchange_commands = single_command("exchange", getargs_exchange, exchange, desc)
