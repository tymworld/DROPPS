"""PBC-aware positional RMSF analysis for systems with molecule copies."""

from __future__ import annotations

import csv
import warnings
from collections import defaultdict
from pathlib import Path

import MDAnalysis as mda
import numpy as np
from MDAnalysis.coordinates.PDB import PDBWriter
from MDAnalysis.exceptions import NoDataError
from MDAnalysis.lib.distances import minimize_vectors
from MDAnalysis.lib.mdamath import triclinic_vectors
from tqdm import tqdm

from dropps.analysis.image_interaction_core import build_unwrap_plan
from dropps.analysis.rmsd_core import batched_kabsch_residuals
from dropps.analysis.rmsf_core import (
    CoordinateMoments,
    aggregate_site_values,
    b_factors_from_covariance,
    rmsf_from_covariance,
)
from dropps.fileio.filename_control import validate_extension
from dropps.share.argument_parser import ArgumentParser
from dropps.share.command_class import single_command
from dropps.share.trajectory import trajectory_class


prog = "rmsf"
desc = "Calculate PBC-aware atomic or residue RMS fluctuations."

ANGSTROM_PER_NM = 10.0
PS_PER_NS = 1000.0


def getargs_rmsf(argv):
    parser = ArgumentParser(
        prog=prog,
        description=desc,
        epilog=(
            "RMSF is sqrt(trace(<(x-<x>)(x-<x>)^T>)). Complete molecules "
            "are reconstructed across PBC before every molecule is fitted to "
            "its own reference coordinates by default."
        ),
    )
    parser.add_argument(
        "-s",
        "--run-input",
        required=True,
        help="Input DROPPS run file (.tpr) containing topology and coordinates.",
    )
    parser.add_argument(
        "-f",
        "--input",
        required=True,
        help="Input trajectory file (.xtc or .pdb).",
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
        help=(
            "Output RMSF profile (.xvg); a matching per-molecule .csv is "
            "also written unless --summary-only is used."
        ),
    )
    parser.add_argument(
        "-od",
        "--output-deviation",
        help="Optional positional RMS deviation from the reference (.xvg).",
    )
    parser.add_argument(
        "-ox",
        "--output-average",
        help="Optional average aligned coordinates with B factors (.pdb).",
    )
    parser.add_argument(
        "-oq",
        "--output-reference",
        help="Optional reference coordinates with B factors (.pdb).",
    )

    selection_source = parser.add_mutually_exclusive_group(section="input")
    selection_source.add_argument(
        "-sel",
        "--selection-groups",
        type=int,
        nargs="+",
        metavar="GROUP",
        help=(
            "Index groups containing atoms across one or more molecule copies; "
            "molecules are split automatically."
        ),
    )
    selection_source.add_argument(
        "--select",
        "--selection-expression",
        dest="selection_expressions",
        action="append",
        metavar="EXPR",
        help=(
            "DROPPS selection expression; repeat the option to calculate more "
            "than one RMSF profile."
        ),
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
        "-ref",
        "--reference-time",
        type=float,
        help=(
            "Reference time in ns; the nearest saved frame is used. By default "
            "use the first analyzed frame."
        ),
    )
    parser.add_argument(
        "--fit-mode",
        choices=("none", "molecule", "selection"),
        default="molecule",
        help=(
            "Least-squares fit scope: none, every molecule independently, or "
            "all atoms in each selection together."
        ),
    )
    parser.add_argument(
        "--fit-weighting",
        choices=("mass", "uniform"),
        default="mass",
        help="Weights used for fit centers and rotations.",
    )
    grouping = parser.add_mutually_exclusive_group()
    grouping.add_argument(
        "--grouping",
        choices=("atom", "residue"),
        default=None,
        help="Report fluctuations for individual atoms or residues; defaults to atom.",
    )
    grouping.add_argument(
        "-res",
        "--residue",
        dest="grouping",
        action="store_const",
        const="residue",
        help="Calculate mass-weighted residue RMSF (gmx rmsf compatibility).",
    )
    parser.add_argument(
        "--group-weighting",
        choices=("mass", "uniform"),
        default="mass",
        help="Weights used to average atomic covariance within each residue.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Skip the detailed CSV and write only requested summary outputs.",
    )
    args = parser.parse_args(argv)
    if args.grouping is None:
        args.grouping = "atom"
    return args


def _unique_labels(names):
    counts = defaultdict(int)
    labels = []
    for name in names:
        base = str(name)
        counts[base] += 1
        labels.append(base if counts[base] == 1 else f"{base}#{counts[base]}")
    return labels


def _resolve_selections(trajectory, args):
    selections = []
    names = []
    if args.selection_groups:
        for group_id in args.selection_groups:
            selection, name = trajectory.getSelection(
                str(group_id),
                "RMSF region",
            )
            selections.append(selection)
            names.append(name)
    elif args.selection_expressions:
        for expression in args.selection_expressions:
            selection, _ = trajectory.getSelection(expression, "RMSF region")
            selections.append(selection)
            names.append(expression)
    else:
        trajectory.index.print_all()
        selections, names = trajectory.getSelection_interactive_multiple(
            "RMSF regions (one selection may span many molecules)"
        )

    if not selections:
        raise ValueError("At least one RMSF selection is required.")
    labels = _unique_labels(names)
    for label, selection in zip(labels, selections):
        if len(selection) == 0:
            raise ValueError(f"RMSF selection '{label}' is empty.")
    return list(zip(labels, selections))


def _topology_bonds(trajectory):
    try:
        return np.asarray(trajectory.Universe.bonds.indices, dtype=np.int64)
    except NoDataError:
        return np.empty((0, 2), dtype=np.int64)


def _ordered_indices_by_chain(indices, atom_chain_ids):
    groups = {}
    for index in np.asarray(indices, dtype=np.int64):
        groups.setdefault(int(atom_chain_ids[int(index)]), []).append(int(index))
    return [
        (chain_id, np.asarray(group, dtype=np.int64))
        for chain_id, group in groups.items()
    ]


def _valid_weights(values, label):
    values = np.asarray(values, dtype=np.float64)
    if (
        values.ndim != 1
        or not np.all(np.isfinite(values))
        or np.any(values < 0.0)
        or float(values.sum()) <= 0.0
    ):
        raise ValueError(f"{label} weights must be finite with positive total.")
    return values


def _residue_sites(global_indices, trajectory):
    groups = {}
    atom_resids = trajectory.index.atom_resids
    atom_resnames = trajectory.index.atom_resnames
    atom_resindices = np.asarray(
        trajectory.Universe.atoms.resindices,
        dtype=np.int64,
    )
    for local_index, global_index in enumerate(global_indices):
        global_index = int(global_index)
        residue_index = int(atom_resindices[global_index])
        if residue_index not in groups:
            groups[residue_index] = {
                "residue_id": int(atom_resids[global_index]),
                "residue_name": str(atom_resnames[global_index]),
                "local_indices": [],
            }
        groups[residue_index]["local_indices"].append(local_index)

    sites = []
    residue_groups = []
    for residue_ordinal, residue in enumerate(
        groups.values(),
        start=1,
    ):
        residue_id = residue["residue_id"]
        residue_name = residue["residue_name"]
        local_indices = residue["local_indices"]
        residue_groups.append(np.asarray(local_indices, dtype=np.int64))
        sites.append(
            {
                "site_ordinal": residue_ordinal,
                "x": residue_id,
                "residue_id": residue_id,
                "residue_name": residue_name,
                "atom_name": "",
                "atom_global_index": "",
            }
        )
    return residue_groups, sites


def _atom_sites(global_indices, complete_molecule_indices, trajectory):
    molecule_position = {
        int(global_index): local_index
        for local_index, global_index in enumerate(complete_molecule_indices, start=1)
    }
    sites = []
    for site_ordinal, global_index in enumerate(global_indices, start=1):
        global_index = int(global_index)
        sites.append(
            {
                "site_ordinal": site_ordinal,
                "x": molecule_position[global_index],
                "residue_id": int(trajectory.index.atom_resids[global_index]),
                "residue_name": str(trajectory.index.atom_resnames[global_index]),
                "atom_name": str(trajectory.index.atom_names[global_index]),
                "atom_global_index": global_index,
            }
        )
    return sites


def _site_signature(sites):
    return tuple(
        (
            int(site["x"]),
            int(site["residue_id"]),
            str(site["residue_name"]),
            str(site["atom_name"]),
        )
        for site in sites
    )


def _build_geometry_plan(trajectory, selections, args):
    atom_chain_ids = np.asarray(trajectory.id2chainID, dtype=np.int64)
    all_atom_indices = np.arange(atom_chain_ids.size, dtype=np.int64)
    complete_molecules = dict(
        _ordered_indices_by_chain(all_atom_indices, atom_chain_ids)
    )
    selection_molecules = []
    selected_chain_ids = []
    union_selected = []

    for label, selection in selections:
        indices = np.sort(np.asarray(selection.indices, dtype=np.int64))
        if np.unique(indices).size != indices.size:
            raise ValueError(f"RMSF selection '{label}' contains duplicate atoms.")
        molecules = _ordered_indices_by_chain(indices, atom_chain_ids)
        selection_molecules.append(molecules)
        union_selected.extend(int(index) for index in indices)
        selected_chain_ids.extend(chain_id for chain_id, _ in molecules)

    selected_chain_ids = list(dict.fromkeys(selected_chain_ids))
    union_selected = np.asarray(
        list(dict.fromkeys(union_selected)),
        dtype=np.int64,
    )
    unwrap_plan = build_unwrap_plan(
        [complete_molecules[chain_id] for chain_id in selected_chain_ids],
        union_selected,
        _topology_bonds(trajectory),
    )
    union_position = {
        int(global_index): position
        for position, global_index in enumerate(union_selected)
    }

    masses = np.asarray(trajectory.Universe.atoms.masses, dtype=np.float64)
    plans = []
    for (label, _), molecule_groups in zip(selections, selection_molecules):
        molecules = []
        flat_position_rows = []
        flat_global_rows = []
        batches_by_size = defaultdict(list)
        profile_signatures = {}
        profile_rows = defaultdict(list)
        flat_offset = 0

        for molecule_row, (chain_id, global_indices) in enumerate(molecule_groups):
            molecule_name = str(trajectory.index.atom_molnames[int(global_indices[0])])
            complete_indices = complete_molecules[chain_id]
            position_indices = np.asarray(
                [union_position[int(index)] for index in global_indices],
                dtype=np.int64,
            )
            fit_weights = (
                masses[global_indices]
                if args.fit_weighting == "mass"
                else np.ones(global_indices.size, dtype=np.float64)
            )
            group_weights = (
                masses[global_indices]
                if args.group_weighting == "mass"
                else np.ones(global_indices.size, dtype=np.float64)
            )
            fit_weights = _valid_weights(
                fit_weights,
                f"Selection '{label}', chain {chain_id} fit",
            )
            group_weights = _valid_weights(
                group_weights,
                f"Selection '{label}', chain {chain_id} grouping",
            )

            atom_sites = _atom_sites(global_indices, complete_indices, trajectory)
            residue_groups, residue_sites = _residue_sites(global_indices, trajectory)
            sites = atom_sites if args.grouping == "atom" else residue_sites
            signature = _site_signature(sites)
            expected = profile_signatures.setdefault(molecule_name, signature)
            if signature != expected:
                raise ValueError(
                    f"Selection '{label}' does not select equivalent "
                    f"{args.grouping} sites in every {molecule_name} molecule."
                )

            flat_indices = np.arange(
                flat_offset,
                flat_offset + global_indices.size,
                dtype=np.int64,
            )
            flat_offset += int(global_indices.size)
            molecule = {
                "molecule_row": molecule_row,
                "molecule_id": molecule_row,
                "chain_id": chain_id,
                "molecule_name": molecule_name,
                "global_indices": global_indices,
                "flat_indices": flat_indices,
                "fit_weights": fit_weights,
                "group_weights": group_weights,
                "residue_groups": residue_groups,
                "sites": sites,
            }
            molecules.append(molecule)
            profile_rows[molecule_name].append(molecule_row)
            flat_position_rows.append(position_indices)
            flat_global_rows.append(global_indices)
            batches_by_size[int(global_indices.size)].append(
                (
                    molecule_row,
                    position_indices,
                    flat_indices,
                    fit_weights,
                )
            )

        batches = []
        for atom_count, rows in batches_by_size.items():
            batches.append(
                {
                    "atom_count": atom_count,
                    "molecule_rows": np.asarray(
                        [row[0] for row in rows], dtype=np.int64
                    ),
                    "position_indices": np.stack([row[1] for row in rows]),
                    "flat_indices": np.stack([row[2] for row in rows]),
                    "weights": np.stack([row[3] for row in rows]),
                }
            )

        plans.append(
            {
                "label": label,
                "molecules": molecules,
                "batches": batches,
                "position_indices": np.concatenate(flat_position_rows),
                "global_indices": np.concatenate(flat_global_rows),
                "fit_weights": np.concatenate(
                    [molecule["fit_weights"] for molecule in molecules]
                ),
                "profile_rows": dict(profile_rows),
            }
        )
    return unwrap_plan, plans


def _validate_dimensions(dimensions):
    if dimensions is None:
        raise ValueError("Trajectory frame does not contain a periodic box.")
    dimensions = np.asarray(dimensions, dtype=np.float64)
    if dimensions.shape != (6,) or not np.all(np.isfinite(dimensions)):
        raise ValueError("Trajectory frame has invalid periodic box dimensions.")
    cell = np.asarray(triclinic_vectors(dimensions, dtype=np.float64))
    if cell.shape != (3, 3) or float(np.linalg.det(cell)) <= 0.0:
        raise ValueError("Trajectory frame has a non-positive periodic box volume.")
    return dimensions


def _unwrap_selected_positions(all_positions, plan, dimensions):
    dimensions = _validate_dimensions(dimensions)
    raw = np.asarray(all_positions[plan.atom_indices], dtype=np.float64)
    unwrapped = np.empty_like(raw)
    unwrapped[plan.roots] = raw[plan.roots]
    for parents, children in plan.levels:
        vectors = raw[children] - raw[parents]
        unwrapped[children] = unwrapped[parents] + minimize_vectors(
            vectors,
            dimensions,
        )
    return unwrapped[plan.selected_local_indices] / ANGSTROM_PER_NM


def _canonical_positions(positions, plan, dimensions):
    dimensions_nm = _validate_dimensions(dimensions).copy()
    dimensions_nm[:3] /= ANGSTROM_PER_NM
    canonical = np.empty_like(plan["reference"])
    for batch in plan["batches"]:
        mobile = positions[batch["position_indices"]]
        reference = plan["reference"][batch["flat_indices"]]
        weights = batch["weights"]
        totals = np.sum(weights, axis=1)
        mobile_centers = np.einsum("mai,ma->mi", mobile, weights) / totals[:, None]
        reference_centers = (
            np.einsum("mai,ma->mi", reference, weights) / totals[:, None]
        )
        displacements = mobile_centers - reference_centers
        image_shifts = minimize_vectors(displacements, dimensions_nm) - displacements
        canonical[batch["flat_indices"]] = mobile + image_shifts[:, None, :]
    return canonical


def _align_positions(positions, plan, dimensions, fit_mode):
    mobile = _canonical_positions(positions, plan, dimensions)
    reference = plan["reference"]
    if fit_mode == "none":
        return mobile
    if fit_mode == "selection":
        residuals = batched_kabsch_residuals(
            mobile[None, :, :],
            reference[None, :, :],
            plan["fit_weights"],
        )[0]
        return residuals + reference
    if fit_mode == "molecule":
        aligned = np.empty_like(mobile)
        for batch in plan["batches"]:
            residuals = batched_kabsch_residuals(
                mobile[batch["flat_indices"]],
                reference[batch["flat_indices"]],
                batch["weights"],
            )
            aligned[batch["flat_indices"]] = (
                residuals + reference[batch["flat_indices"]]
            )
        return aligned
    raise ValueError(f"Unknown RMSF fit mode: {fit_mode}")


def _reference_frame(trajectory, first_analyzed_frame, reference_time_ns):
    if reference_time_ns is None:
        return int(first_analyzed_frame)
    if not np.isfinite(reference_time_ns):
        raise ValueError("Reference time must be finite.")

    frames = trajectory.Universe.trajectory
    times_ns = np.asarray(
        [float(frames[index].time) / PS_PER_NS for index in range(frames.n_frames)],
        dtype=np.float64,
    )
    if times_ns.size == 0 or not np.all(np.isfinite(times_ns)):
        raise ValueError("Trajectory does not contain valid frame timestamps.")
    if np.any(np.diff(times_ns) <= 0.0):
        raise ValueError("Trajectory frame timestamps must be strictly increasing.")
    tolerance = max(abs(times_ns[0]), abs(times_ns[-1]), 1.0) * 1.0e-10
    if reference_time_ns < times_ns[0] - tolerance:
        raise ValueError("Reference time precedes the first trajectory frame.")
    if reference_time_ns > times_ns[-1] + tolerance:
        raise ValueError("Reference time follows the last trajectory frame.")
    return int(np.argmin(np.abs(times_ns - float(reference_time_ns))))


def _output_paths(args):
    paths = {"rmsf": Path(validate_extension(args.output, "xvg"))}
    if not args.summary_only:
        paths["details"] = paths["rmsf"].with_suffix(".csv")
    if args.output_deviation:
        paths["deviation"] = Path(validate_extension(args.output_deviation, "xvg"))
    if args.output_average:
        paths["average"] = Path(validate_extension(args.output_average, "pdb"))
    if args.output_reference:
        paths["reference"] = Path(validate_extension(args.output_reference, "pdb"))
    resolved = [path.expanduser().resolve() for path in paths.values()]
    if len(set(resolved)) != len(resolved):
        raise ValueError("RMSF output paths must be distinct.")
    return paths


def _finalize_selection(plan, grouping):
    atom_covariance = plan["moments"].covariance
    atom_reference_msd = plan["moments"].reference_mean_square_displacement
    molecules = []
    for molecule in plan["molecules"]:
        flat = molecule["flat_indices"]
        covariance = atom_covariance[flat]
        reference_msd = atom_reference_msd[flat]
        if grouping == "residue":
            covariance = aggregate_site_values(
                covariance,
                molecule["residue_groups"],
                molecule["group_weights"],
            )
            reference_msd = aggregate_site_values(
                reference_msd,
                molecule["residue_groups"],
                molecule["group_weights"],
            )
        molecules.append(
            {
                **molecule,
                "covariance": covariance,
                "rmsf": rmsf_from_covariance(covariance),
                "reference_rmsd": np.sqrt(np.maximum(reference_msd, 0.0)),
                "b_factor": b_factors_from_covariance(covariance),
            }
        )

    profiles = []
    for molecule_name, molecule_rows in plan["profile_rows"].items():
        profile_molecules = [molecules[row] for row in molecule_rows]
        rmsf_values = np.stack([molecule["rmsf"] for molecule in profile_molecules])
        deviation_values = np.stack(
            [molecule["reference_rmsd"] for molecule in profile_molecules]
        )
        profiles.append(
            {
                "molecule_name": molecule_name,
                "molecule_count": len(profile_molecules),
                "sites": profile_molecules[0]["sites"],
                "rmsf_mean": rmsf_values.mean(axis=0),
                "rmsf_std": rmsf_values.std(axis=0),
                "deviation_mean": deviation_values.mean(axis=0),
                "deviation_std": deviation_values.std(axis=0),
            }
        )
    return {"label": plan["label"], "molecules": molecules, "profiles": profiles}


def _quoted(text):
    return str(text).replace("\\", "\\\\").replace('"', '\\"')


def _write_profile_xvg(path, results, grouping, value, metadata):
    is_rmsf = value == "rmsf"
    y_label = "RMSF (nm)" if is_rmsf else "RMS deviation (nm)"
    title = "Root mean square fluctuation" if is_rmsf else "Reference deviation"
    profiles = [
        (result["label"], profile)
        for result in results
        for profile in result["profiles"]
    ]
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"# DROPPS PBC-aware {value}\n")
        handle.write(f"# {metadata}\n")
        handle.write(
            "# y is the molecule mean; dy is the population standard deviation "
            "across equivalent molecule copies.\n"
        )
        handle.write(f'@    title "{title}"\n')
        x_label = "Residue" if grouping == "residue" else "Atom in molecule"
        handle.write(f'@    xaxis  label "{x_label}"\n')
        handle.write(f'@    yaxis  label "{y_label}"\n')
        handle.write("@TYPE xydy\n")
        for series_id, (selection_label, profile) in enumerate(profiles):
            legend = _quoted(
                f"{selection_label}/{profile['molecule_name']} "
                f"(n={profile['molecule_count']})"
            )
            handle.write(f'@ s{series_id} legend "{legend}"\n')

        mean_key = f"{value}_mean" if is_rmsf else "deviation_mean"
        std_key = f"{value}_std" if is_rmsf else "deviation_std"
        for profile_id, (_, profile) in enumerate(profiles):
            handle.write(
                f"# profile={profile_id} molecule_name="
                f"{profile['molecule_name']} molecules={profile['molecule_count']}\n"
            )
            for site, mean, std in zip(
                profile["sites"],
                profile[mean_key],
                profile[std_key],
            ):
                handle.write(f"{site['x']:8d} {mean:.12g} {std:.12g}\n")
            if profile_id + 1 < len(profiles):
                handle.write("&\n")


def _write_details(path, results, metadata_fields):
    fields = (
        "selection",
        "molecule_id",
        "chain_id",
        "molecule_name",
        "site_ordinal",
        "atom_global_index",
        "atom_name",
        "residue_id",
        "residue_name",
        "rmsf_nm",
        "reference_rmsd_nm",
        "b_factor_a2",
        "u_xx_nm2",
        "u_xy_nm2",
        "u_xz_nm2",
        "u_yy_nm2",
        "u_yz_nm2",
        "u_zz_nm2",
        *metadata_fields.keys(),
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            for molecule in result["molecules"]:
                for site, rmsf_value, deviation, b_factor, covariance in zip(
                    molecule["sites"],
                    molecule["rmsf"],
                    molecule["reference_rmsd"],
                    molecule["b_factor"],
                    molecule["covariance"],
                ):
                    writer.writerow(
                        {
                            "selection": result["label"],
                            "molecule_id": molecule["molecule_id"],
                            "chain_id": molecule["chain_id"],
                            "molecule_name": molecule["molecule_name"],
                            **{
                                key: site[key]
                                for key in (
                                    "site_ordinal",
                                    "atom_global_index",
                                    "atom_name",
                                    "residue_id",
                                    "residue_name",
                                )
                            },
                            "rmsf_nm": f"{rmsf_value:.12g}",
                            "reference_rmsd_nm": f"{deviation:.12g}",
                            "b_factor_a2": f"{b_factor:.12g}",
                            "u_xx_nm2": f"{covariance[0, 0]:.12g}",
                            "u_xy_nm2": f"{covariance[0, 1]:.12g}",
                            "u_xz_nm2": f"{covariance[0, 2]:.12g}",
                            "u_yy_nm2": f"{covariance[1, 1]:.12g}",
                            "u_yz_nm2": f"{covariance[1, 2]:.12g}",
                            "u_zz_nm2": f"{covariance[2, 2]:.12g}",
                            **metadata_fields,
                        }
                    )


def _atom_b_factors(plan, grouping):
    atom_covariance = plan["moments"].covariance
    if grouping == "atom":
        return b_factors_from_covariance(atom_covariance)

    b_factors = np.empty(atom_covariance.shape[0], dtype=np.float64)
    for molecule in plan["molecules"]:
        flat = molecule["flat_indices"]
        residue_covariance = aggregate_site_values(
            atom_covariance[flat],
            molecule["residue_groups"],
            molecule["group_weights"],
        )
        residue_b_factors = b_factors_from_covariance(residue_covariance)
        for group, value in zip(molecule["residue_groups"], residue_b_factors):
            b_factors[flat[group]] = value
    return b_factors


def _write_pdb(path, universe, global_indices, coordinates_nm, b_factors, dimensions):
    output_universe = mda.Merge(universe.atoms[global_indices])
    try:
        output_universe.atoms.tempfactors
    except NoDataError:
        output_universe.add_TopologyAttr("tempfactors", b_factors)
    else:
        output_universe.atoms.tempfactors = b_factors
    output_universe.atoms.positions = np.asarray(coordinates_nm) * ANGSTROM_PER_NM
    output_universe.dimensions = _validate_dimensions(dimensions)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Found no information for attr:",
            category=UserWarning,
        )
        with PDBWriter(
            str(path), n_atoms=len(global_indices), multiframe=False
        ) as writer:
            writer.write(output_universe.atoms)


def rmsf(args):
    paths = _output_paths(args)
    trajectory = trajectory_class(args.run_input, args.index, args.input)
    start_frame, end_frame, interval_frame = trajectory.time2frame(
        args.start_time,
        args.end_time,
        args.delta_time,
    )
    frame_indices = list(range(start_frame, end_frame + 1, interval_frame))
    selections = _resolve_selections(trajectory, args)
    if ("average" in paths or "reference" in paths) and len(selections) != 1:
        raise ValueError("PDB outputs require exactly one RMSF selection.")

    unwrap_plan, plans = _build_geometry_plan(trajectory, selections, args)
    frames = trajectory.Universe.trajectory
    reference_frame = _reference_frame(
        trajectory,
        frame_indices[0],
        args.reference_time,
    )
    reference_timestep = frames[reference_frame]
    reference_time_ns = float(reference_timestep.time) / PS_PER_NS
    reference_dimensions = np.asarray(reference_timestep.dimensions).copy()
    reference_union = _unwrap_selected_positions(
        reference_timestep.positions,
        unwrap_plan,
        reference_timestep.dimensions,
    )
    for plan in plans:
        plan["reference"] = reference_union[plan["position_indices"]]
        plan["moments"] = CoordinateMoments(
            plan["reference"].shape[0],
            reference=plan["reference"],
        )

    print(
        f"## RMSF: {len(frame_indices)} frame(s), {len(plans)} selection(s), "
        f"{sum(len(plan['molecules']) for plan in plans)} molecule-region(s)."
    )
    print(
        f"## Reference frame {reference_frame} at {reference_time_ns:.12g} ns; "
        f"fit={args.fit_mode}, fit weighting={args.fit_weighting}."
    )
    print(
        f"## Output grouping={args.grouping}, group weighting={args.group_weighting}."
    )
    print("## Complete selected molecules will be unwrapped from topology bonds.")

    for frame_index in tqdm(frame_indices, desc="## Calculating molecular RMSF"):
        timestep = frames[frame_index]
        union_positions = _unwrap_selected_positions(
            timestep.positions,
            unwrap_plan,
            timestep.dimensions,
        )
        for plan in plans:
            aligned = _align_positions(
                union_positions,
                plan,
                timestep.dimensions,
                args.fit_mode,
            )
            plan["moments"].update(aligned)

    results = [_finalize_selection(plan, args.grouping) for plan in plans]
    metadata_fields = {
        "frames": len(frame_indices),
        "reference_frame": reference_frame,
        "reference_time_ns": f"{reference_time_ns:.12g}",
        "fit_mode": args.fit_mode,
        "fit_weighting": args.fit_weighting,
        "grouping": args.grouping,
        "group_weighting": args.group_weighting,
        "pbc": "complete-molecule-bonded-unwrap",
    }
    metadata = " ".join(f"{key}={value}" for key, value in metadata_fields.items())
    _write_profile_xvg(paths["rmsf"], results, args.grouping, "rmsf", metadata)
    if "deviation" in paths:
        _write_profile_xvg(
            paths["deviation"],
            results,
            args.grouping,
            "deviation",
            metadata,
        )
    if "details" in paths:
        _write_details(paths["details"], results, metadata_fields)

    if "average" in paths or "reference" in paths:
        plan = plans[0]
        b_factors = _atom_b_factors(plan, args.grouping)
        if "average" in paths:
            _write_pdb(
                paths["average"],
                trajectory.Universe,
                plan["global_indices"],
                plan["moments"].mean,
                b_factors,
                reference_dimensions,
            )
        if "reference" in paths:
            _write_pdb(
                paths["reference"],
                trajectory.Universe,
                plan["global_indices"],
                plan["reference"],
                b_factors,
                reference_dimensions,
            )

    written = ", ".join(str(path) for path in paths.values())
    print(f"## Wrote RMSF outputs: {written}.")


rmsf_commands = single_command(prog, getargs_rmsf, rmsf, desc)
