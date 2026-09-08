"""Per-molecule, PBC-aware conformational RMSD analysis."""

from __future__ import annotations

import csv
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path

import numpy as np
from MDAnalysis.exceptions import NoDataError
from MDAnalysis.lib.distances import minimize_vectors
from MDAnalysis.lib.mdamath import triclinic_vectors
from tqdm import tqdm

from dropps.analysis.image_interaction_core import build_unwrap_plan
from dropps.analysis.rmsd_core import (
    batched_kabsch_residuals,
    batched_residual_rmsd,
)
from dropps.fileio.filename_control import validate_extension
from dropps.share.argument_parser import ArgumentParser
from dropps.share.command_class import single_command
from dropps.share.trajectory import trajectory_class


prog = "rmsd"
desc = "Calculate PBC-aware RMSD with configurable fitting and output granularity."

ANGSTROM_PER_NM = 10.0
PS_PER_NS = 1000.0


def getargs_rmsd(argv):
    parser = ArgumentParser(
        prog=prog,
        description=desc,
        epilog=(
            "Select a region once across all molecule copies. DROPPS splits it "
            "by topology molecule and unwraps each complete molecule. Fitting "
            "and RMSD reporting can then operate per molecule or per selection."
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
        help=(
            "Output RMSD time series (.xvg); a matching detailed .csv is also "
            "written unless --summary-only is used."
        ),
    )

    selection_source = parser.add_mutually_exclusive_group(section="input")
    selection_source.add_argument(
        "-sel",
        "--selection-groups",
        type=int,
        nargs="+",
        metavar="GROUP",
        help=(
            "Index groups containing selected atoms across any number of "
            "molecules; molecules are split automatically."
        ),
    )
    selection_source.add_argument(
        "--select",
        "--selection-expression",
        dest="selection_expressions",
        action="append",
        metavar="EXPR",
        help=(
            "DROPPS selection expression applied across all molecules; repeat "
            "the option to analyze more than one region."
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
            "Coordinate fitting before RMSD: none preserves physical "
            "translation/rotation, molecule fits every molecule independently, "
            "and selection applies one fit to all atoms in each selection."
        ),
    )
    parser.add_argument(
        "--output-mode",
        choices=("molecule", "selection"),
        default="molecule",
        help=(
            "RMSD calculation/output granularity: one value per molecule or "
            "one pooled value for all atoms in each selection."
        ),
    )
    parser.add_argument(
        "--mass-weighted",
        action="store_true",
        help="Use atom masses for centering, fitting, and RMSD.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Skip the detailed CSV and write only the XVG time series.",
    )
    return parser.parse_args(argv)


def _unique_labels(names):
    totals = defaultdict(int)
    labels = []
    for name in names:
        base = str(name)
        totals[base] += 1
        labels.append(base if totals[base] == 1 else f"{base}#{totals[base]}")
    return labels


def _resolve_selections(trajectory, args):
    selections = []
    names = []
    if args.selection_groups:
        for group_id in args.selection_groups:
            selection, name = trajectory.getSelection(
                f"group {group_id}",
                "molecular RMSD region",
            )
            selections.append(selection)
            names.append(name)
    elif args.selection_expressions:
        for expression in args.selection_expressions:
            selection, _ = trajectory.getSelection(
                expression,
                "molecular RMSD region",
            )
            selections.append(selection)
            names.append(expression)
    else:
        trajectory.index.print_all()
        selections, names = trajectory.getSelection_interactive_multiple(
            "molecular RMSD regions (one selection may span many molecules)"
        )

    if not selections:
        raise ValueError("At least one RMSD selection is required.")
    for name, selection in zip(names, selections):
        if len(selection) == 0:
            raise ValueError(f"RMSD selection '{name}' is empty.")
    return list(zip(_unique_labels(names), selections))


def _topology_bonds(trajectory):
    try:
        return np.asarray(trajectory.Universe.bonds.indices, dtype=np.int64)
    except NoDataError:
        return np.empty((0, 2), dtype=np.int64)


def _split_indices_by_chain(indices, atom_chain_ids):
    """Split sorted atom indices in one pass using topology molecule IDs."""

    indices = np.asarray(indices, dtype=np.int64)
    if indices.ndim != 1 or indices.size == 0:
        raise ValueError("Cannot split an empty or non-vector atom selection.")
    chain_ids = atom_chain_ids[indices]
    boundaries = np.flatnonzero(chain_ids[1:] != chain_ids[:-1]) + 1
    return [
        (int(atom_chain_ids[int(group[0])]), group)
        for group in np.split(indices, boundaries)
    ]


def _build_geometry_plan(trajectory, selections, mass_weighted):
    atom_chain_ids = np.asarray(trajectory.id2chainID, dtype=np.int64)
    selection_molecule_indices = []
    selected_chain_ids = []
    union_selected_indices = []

    for _, selection in selections:
        indices = np.sort(np.asarray(selection.indices, dtype=np.int64))
        if np.unique(indices).size != indices.size:
            raise ValueError("An RMSD selection contains duplicate atom indices.")
        molecule_indices = _split_indices_by_chain(indices, atom_chain_ids)
        selection_molecule_indices.append(molecule_indices)
        union_selected_indices.extend(int(index) for index in indices)
        selected_chain_ids.extend(chain_id for chain_id, _ in molecule_indices)

    union_selected_indices = np.asarray(
        list(dict.fromkeys(union_selected_indices)),
        dtype=np.int64,
    )
    selected_chain_ids = list(dict.fromkeys(selected_chain_ids))
    system_molecules = dict(
        _split_indices_by_chain(
            np.arange(atom_chain_ids.size, dtype=np.int64),
            atom_chain_ids,
        )
    )
    molecule_atom_indices = [
        system_molecules[chain_id] for chain_id in selected_chain_ids
    ]
    unwrap_plan = build_unwrap_plan(
        molecule_atom_indices,
        union_selected_indices,
        _topology_bonds(trajectory),
    )

    union_position = {
        int(atom_index): position
        for position, atom_index in enumerate(union_selected_indices)
    }
    masses = np.asarray(trajectory.Universe.atoms.masses, dtype=float)
    selection_plans = []
    single_atom_molecules = 0

    for (label, _), molecule_indices_by_chain in zip(
        selections,
        selection_molecule_indices,
    ):
        molecules = []
        batches_by_size = defaultdict(list)
        flat_position_rows = []
        flat_weight_rows = []
        flat_offset = 0
        for molecule_id, (chain_id, molecule_indices) in enumerate(
            molecule_indices_by_chain
        ):
            position_indices = np.asarray(
                [union_position[int(index)] for index in molecule_indices],
                dtype=np.int64,
            )
            molecule_weights = (
                masses[molecule_indices].copy()
                if mass_weighted
                else np.ones(molecule_indices.size, dtype=float)
            )
            if (
                not np.all(np.isfinite(molecule_weights))
                or np.any(molecule_weights < 0.0)
                or float(molecule_weights.sum()) <= 0.0
            ):
                raise ValueError(
                    f"Molecule {molecule_id} in selection '{label}' has invalid "
                    "selected-atom masses."
                )
            molecule_name = str(
                trajectory.index.atom_molnames[int(molecule_indices[0])]
            )
            molecules.append(
                {
                    "molecule_id": molecule_id,
                    "chain_id": chain_id,
                    "molecule_name": molecule_name,
                    "selected_atoms": int(molecule_indices.size),
                }
            )
            flat_indices = np.arange(
                flat_offset,
                flat_offset + molecule_indices.size,
                dtype=np.int64,
            )
            flat_offset += int(molecule_indices.size)
            flat_position_rows.append(position_indices)
            flat_weight_rows.append(molecule_weights)
            batches_by_size[int(molecule_indices.size)].append(
                (molecule_id, position_indices, molecule_weights, flat_indices)
            )
            single_atom_molecules += int(molecule_indices.size == 1)

        batches = []
        for atom_count, rows in batches_by_size.items():
            batches.append(
                {
                    "atom_count": atom_count,
                    "molecule_rows": np.asarray(
                        [row[0] for row in rows], dtype=np.int64
                    ),
                    "position_indices": np.stack([row[1] for row in rows]),
                    "weights": np.stack([row[2] for row in rows]),
                    "flat_indices": np.stack([row[3] for row in rows]),
                }
            )
        selection_plans.append(
            {
                "label": label,
                "molecules": molecules,
                "batches": batches,
                "position_indices": np.concatenate(flat_position_rows),
                "weights": np.concatenate(flat_weight_rows),
            }
        )

    if single_atom_molecules:
        print(
            f"## WARNING: {single_atom_molecules} selected molecule-region(s) "
            "contain one atom; --fit-mode molecule makes their RMSD zero."
        )
    return unwrap_plan, selection_plans


def _validate_box_dimensions(dimensions):
    if dimensions is None:
        raise ValueError("Trajectory frame does not contain a periodic box.")
    dimensions = np.asarray(dimensions, dtype=float)
    if dimensions.shape != (6,) or not np.all(np.isfinite(dimensions)):
        raise ValueError("Trajectory frame has invalid periodic box dimensions.")
    cell = np.asarray(triclinic_vectors(dimensions, dtype=np.float64))
    if cell.shape != (3, 3) or float(np.linalg.det(cell)) <= 0.0:
        raise ValueError("Trajectory frame has a non-positive periodic box volume.")
    return dimensions


def _unwrap_selected_positions(all_positions, plan, dimensions):
    dimensions = _validate_box_dimensions(dimensions)
    raw = np.asarray(all_positions[plan.atom_indices], dtype=np.float64)
    unwrapped = np.empty_like(raw)
    unwrapped[plan.roots] = raw[plan.roots]
    for parents, children in plan.levels:
        bond_vectors = raw[children] - raw[parents]
        minimum_vectors = minimize_vectors(bond_vectors, dimensions)
        unwrapped[children] = unwrapped[parents] + minimum_vectors
    return unwrapped[plan.selected_local_indices] / ANGSTROM_PER_NM


def _box_dimensions_nm(dimensions):
    dimensions_nm = _validate_box_dimensions(dimensions).copy()
    dimensions_nm[:3] /= ANGSTROM_PER_NM
    return dimensions_nm


def _canonicalize_selection_positions(positions, selection_plan, dimensions):
    """Choose the current periodic image nearest each reference molecule.

    This lattice-image normalization is part of PBC handling and does not
    remove physical sub-box translation or rotation in ``fit-mode=none``.
    """

    dimensions_nm = _box_dimensions_nm(dimensions)
    canonical = np.empty_like(selection_plan["reference"])
    for batch in selection_plan["batches"]:
        mobile = positions[batch["position_indices"]]
        reference = selection_plan["reference"][batch["flat_indices"]]
        weights = batch["weights"]
        weight_sums = np.sum(weights, axis=1)
        mobile_centers = (
            np.einsum("mai,ma->mi", mobile, weights) / weight_sums[:, np.newaxis]
        )
        reference_centers = (
            np.einsum("mai,ma->mi", reference, weights) / weight_sums[:, np.newaxis]
        )
        center_displacements = mobile_centers - reference_centers
        minimum_displacements = minimize_vectors(
            center_displacements,
            dimensions_nm,
        )
        image_shifts = minimum_displacements - center_displacements
        canonical[batch["flat_indices"]] = mobile + image_shifts[:, np.newaxis, :]
    return canonical


def _calculate_selection_rmsd(
    positions,
    selection_plan,
    dimensions,
    fit_mode,
    output_mode,
):
    """Apply the requested fit and return molecule- or selection-level RMSD."""

    mobile = _canonicalize_selection_positions(
        positions,
        selection_plan,
        dimensions,
    )
    reference = selection_plan["reference"]
    weights = selection_plan["weights"]

    if fit_mode == "none":
        residuals = mobile - reference
    elif fit_mode == "selection":
        residuals = batched_kabsch_residuals(
            mobile[np.newaxis, :, :],
            reference[np.newaxis, :, :],
            weights,
        )[0]
    elif fit_mode == "molecule":
        residuals = np.empty_like(mobile)
        for batch in selection_plan["batches"]:
            residuals[batch["flat_indices"]] = batched_kabsch_residuals(
                mobile[batch["flat_indices"]],
                reference[batch["flat_indices"]],
                batch["weights"],
            )
    else:
        raise ValueError(f"Unknown RMSD fit mode: {fit_mode}")

    if output_mode == "selection":
        return float(
            batched_residual_rmsd(
                residuals[np.newaxis, :, :],
                weights,
            )[0]
        )
    if output_mode == "molecule":
        molecule_rmsd = np.empty(len(selection_plan["molecules"]), dtype=float)
        for batch in selection_plan["batches"]:
            molecule_rmsd[batch["molecule_rows"]] = batched_residual_rmsd(
                residuals[batch["flat_indices"]],
                batch["weights"],
            )
        return molecule_rmsd
    raise ValueError(f"Unknown RMSD output mode: {output_mode}")


def _reference_frame(trajectory, first_analyzed_frame, reference_time_ns):
    if reference_time_ns is None:
        return int(first_analyzed_frame)
    if not np.isfinite(reference_time_ns):
        raise ValueError("Reference time must be finite.")

    analyzed_trajectory = trajectory.Universe.trajectory
    frame_times_ns = np.asarray(
        [
            float(analyzed_trajectory[index].time) / PS_PER_NS
            for index in range(analyzed_trajectory.n_frames)
        ],
        dtype=float,
    )
    if frame_times_ns.size == 0 or not np.all(np.isfinite(frame_times_ns)):
        raise ValueError("Trajectory does not contain valid frame timestamps.")
    if np.any(np.diff(frame_times_ns) <= 0.0):
        raise ValueError("Trajectory frame timestamps must be strictly increasing.")
    first_time_ns = float(frame_times_ns[0])
    last_time_ns = float(frame_times_ns[-1])
    tolerance = max(abs(first_time_ns), abs(last_time_ns), 1.0) * 1.0e-10
    if reference_time_ns < first_time_ns - tolerance:
        raise ValueError("Reference time precedes the first trajectory frame.")
    if reference_time_ns > last_time_ns + tolerance:
        raise ValueError("Reference time follows the last trajectory frame.")
    return int(np.argmin(np.abs(frame_times_ns - float(reference_time_ns))))


def _output_paths(output, summary_only):
    summary_path = Path(validate_extension(output, "xvg"))
    paths = {"summary": summary_path}
    if not summary_only:
        paths["details"] = summary_path.with_suffix(".csv")
    resolved = [path.resolve() for path in paths.values()]
    if len(set(resolved)) != len(resolved):
        raise ValueError("RMSD output paths must be distinct.")
    return paths


def _quoted_label(label):
    return str(label).replace("\\", "\\\\").replace('"', '\\"')


def _write_xvg_header(
    handle,
    selection_plans,
    reference_frame,
    reference_time_ns,
    weighting,
    fit_mode,
    output_mode,
):
    handle.write("# DROPPS PBC-aware RMSD\n")
    handle.write(
        f"# reference_frame={reference_frame} "
        f"reference_time_ns={reference_time_ns:.12g} "
        f"weighting={weighting} pbc=complete-molecule-bonded-unwrap "
        f"fit_mode={fit_mode} output_mode={output_mode}\n"
    )
    title = "Per-molecule RMSD" if output_mode == "molecule" else "Selection RMSD"
    handle.write(f'@    title "{title}"\n')
    handle.write('@    xaxis  label "Time (ns)"\n')
    handle.write('@    yaxis  label "RMSD (nm)"\n')
    handle.write("@TYPE xy\n")
    series_id = 0
    for plan in selection_plans:
        label = _quoted_label(plan["label"])
        if output_mode == "molecule":
            handle.write(f'@ s{series_id} legend "{label} mean"\n')
            handle.write(f'@ s{series_id + 1} legend "{label} std"\n')
            series_id += 2
        else:
            handle.write(f'@ s{series_id} legend "{label}"\n')
            series_id += 1


def rmsd(args):
    trajectory = trajectory_class(args.run_input, args.index, args.input)
    start_frame, end_frame, interval_frame = trajectory.time2frame(
        args.start_time,
        args.end_time,
        args.delta_time,
    )
    frame_indices = list(range(start_frame, end_frame + 1, interval_frame))
    selections = _resolve_selections(trajectory, args)
    unwrap_plan, selection_plans = _build_geometry_plan(
        trajectory,
        selections,
        args.mass_weighted,
    )

    reference_frame = _reference_frame(
        trajectory,
        frame_indices[0],
        args.reference_time,
    )
    analyzed_trajectory = trajectory.Universe.trajectory
    reference_timestep = analyzed_trajectory[reference_frame]
    reference_time_ns = float(reference_timestep.time) / PS_PER_NS
    reference_positions = _unwrap_selected_positions(
        reference_timestep.positions,
        unwrap_plan,
        reference_timestep.dimensions,
    )
    for selection_plan in selection_plans:
        selection_plan["reference"] = reference_positions[
            selection_plan["position_indices"]
        ]

    weighting = "mass" if args.mass_weighted else "uniform"
    paths = _output_paths(args.output, args.summary_only)
    molecule_count = sum(
        len(selection_plan["molecules"]) for selection_plan in selection_plans
    )
    print(
        f"## RMSD selections: {len(selection_plans)}; represented "
        f"molecule-regions: {molecule_count}."
    )
    print(
        f"## Reference: frame {reference_frame}, {reference_time_ns:.12g} ns; "
        f"weighting: {weighting}."
    )
    print(f"## Fit mode: {args.fit_mode}; output mode: {args.output_mode}.")
    print("## Complete molecules will be unwrapped from topology bonds in every frame.")

    common_fields = (
        "frame",
        "time_ns",
        "selection",
    )
    molecule_fields = common_fields + (
        "molecule_id",
        "chain_id",
        "molecule_name",
        "selected_atoms",
        "reference_frame",
        "reference_time_ns",
        "weighting",
        "fit_mode",
        "output_mode",
        "rmsd_nm",
    )
    selection_fields = common_fields + (
        "molecules",
        "selected_atoms",
        "reference_frame",
        "reference_time_ns",
        "weighting",
        "fit_mode",
        "output_mode",
        "rmsd_nm",
    )
    with ExitStack() as stack:
        summary_handle = stack.enter_context(
            paths["summary"].open("w", encoding="utf-8")
        )
        if "details" in paths:
            detail_handle = stack.enter_context(
                paths["details"].open(
                    "w",
                    encoding="utf-8",
                    newline="",
                )
            )
            detail_writer = csv.DictWriter(
                detail_handle,
                fieldnames=(
                    molecule_fields
                    if args.output_mode == "molecule"
                    else selection_fields
                ),
            )
            detail_writer.writeheader()
        else:
            detail_writer = None

        _write_xvg_header(
            summary_handle,
            selection_plans,
            reference_frame,
            reference_time_ns,
            weighting,
            args.fit_mode,
            args.output_mode,
        )
        for frame_index in tqdm(frame_indices, desc="## Calculating molecular RMSD"):
            timestep = analyzed_trajectory[frame_index]
            time_ns = float(timestep.time) / PS_PER_NS
            positions = _unwrap_selected_positions(
                timestep.positions,
                unwrap_plan,
                timestep.dimensions,
            )
            summary_values = []
            for selection_plan in selection_plans:
                values = _calculate_selection_rmsd(
                    positions,
                    selection_plan,
                    timestep.dimensions,
                    args.fit_mode,
                    args.output_mode,
                )
                if args.output_mode == "molecule":
                    molecule_rmsd = values
                    summary_values.extend(
                        (
                            float(molecule_rmsd.mean()),
                            float(molecule_rmsd.std()),
                        )
                    )
                    if detail_writer is not None:
                        for molecule, value in zip(
                            selection_plan["molecules"],
                            molecule_rmsd,
                        ):
                            detail_writer.writerow(
                                {
                                    "frame": frame_index,
                                    "time_ns": f"{time_ns:.12g}",
                                    "selection": selection_plan["label"],
                                    **molecule,
                                    "reference_frame": reference_frame,
                                    "reference_time_ns": (f"{reference_time_ns:.12g}"),
                                    "weighting": weighting,
                                    "fit_mode": args.fit_mode,
                                    "output_mode": args.output_mode,
                                    "rmsd_nm": f"{float(value):.12g}",
                                }
                            )
                else:
                    selection_rmsd = float(values)
                    summary_values.append(selection_rmsd)
                    if detail_writer is not None:
                        detail_writer.writerow(
                            {
                                "frame": frame_index,
                                "time_ns": f"{time_ns:.12g}",
                                "selection": selection_plan["label"],
                                "molecules": len(selection_plan["molecules"]),
                                "selected_atoms": int(selection_plan["weights"].size),
                                "reference_frame": reference_frame,
                                "reference_time_ns": f"{reference_time_ns:.12g}",
                                "weighting": weighting,
                                "fit_mode": args.fit_mode,
                                "output_mode": args.output_mode,
                                "rmsd_nm": f"{selection_rmsd:.12g}",
                            }
                        )
            fields = [f"{time_ns:.12g}"]
            fields.extend(f"{value:.12g}" for value in summary_values)
            summary_handle.write("\t".join(fields) + "\n")

    print(f"## Wrote RMSD summary to {paths['summary']}.")
    if "details" in paths:
        scope = "per-molecule" if args.output_mode == "molecule" else "per-selection"
        print(f"## Wrote {scope} RMSD details to {paths['details']}.")


rmsd_commands = single_command(prog, getargs_rmsd, rmsd, desc)
