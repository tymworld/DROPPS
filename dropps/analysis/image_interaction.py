"""Detect molecular contacts with periodic images in a trajectory."""

from __future__ import annotations

import csv
from itertools import product
from pathlib import Path

import numpy as np
from MDAnalysis.exceptions import NoDataError
from MDAnalysis.lib.distances import minimize_vectors
from MDAnalysis.lib.mdamath import triclinic_vectors
from MDAnalysis.lib.nsgrid import FastNS
from tqdm import tqdm

from dropps.analysis.image_interaction_core import (
    analyze_image_edges,
    build_unwrap_plan,
    canonicalize_image_contacts,
)
from dropps.share.argument_parser import ArgumentParser
from dropps.share.command_class import single_command
from dropps.share.trajectory import trajectory_class


prog = "pbcontact"
desc = (
    "Measure direct and network-mediated interactions with periodic molecular images."
)

ANGSTROM_PER_NM = 10.0
PS_PER_NS = 1000.0


def getargs_image_interaction(argv):
    parser = ArgumentParser(
        prog=prog,
        description=desc,
        epilog=(
            "A non-zero-winding cycle in the molecule contact graph is counted "
            "as an image interaction, including paths through any number of "
            "intermediate molecules."
        ),
    )
    parser.add_argument(
        "-s",
        "--run-input",
        type=str,
        required=True,
        help="Input DROPPS run file (.tpr) containing topology and coordinates.",
    )
    parser.add_argument(
        "-f",
        "--input",
        type=str,
        required=True,
        help="Input trajectory file (.xtc).",
    )
    parser.add_argument(
        "-n",
        "--index",
        type=str,
        help="Optional index file (.ndx) defining additional atom groups.",
    )
    parser.add_argument(
        "-o",
        "--output-prefix",
        type=str,
        required=True,
        help="Output prefix for .frames.csv and .summary.csv files.",
    )
    parser.add_argument(
        "--molecule-output",
        type=str,
        help=(
            "Optional detailed CSV with the image-interaction state of every "
            "selected molecule in every analyzed frame."
        ),
    )
    parser.add_argument(
        "-sel",
        "--selection-group",
        type=int,
        help=(
            "Index group whose atoms define contacts; molecules represented by "
            "the group are analyzed. If omitted, prompt interactively."
        ),
    )
    parser.add_argument(
        "-c",
        "--cutoff",
        type=float,
        default=0.7,
        help="Atom-contact cutoff in nm.",
    )
    parser.add_argument(
        "-mc",
        "--minimum-contacts",
        type=int,
        default=1,
        help=("Minimum atom-pair contacts required for one molecule-to-image edge."),
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
    args = parser.parse_args(argv)
    if not np.isfinite(args.cutoff) or args.cutoff <= 0.0:
        parser.error("--cutoff must be finite and greater than zero")
    if args.minimum_contacts <= 0:
        parser.error("--minimum-contacts must be greater than zero")
    return args


def _select_molecules(trajectory, selection):
    selected_indices = np.asarray(selection.indices, dtype=np.int64)
    if selected_indices.size == 0:
        raise ValueError("The selected contact group is empty.")

    atom_chain_ids = np.asarray(trajectory.id2chainID, dtype=np.int64)
    selected_chain_ids = list(
        dict.fromkeys(int(chain_id) for chain_id in atom_chain_ids[selected_indices])
    )
    molecule_atoms = [
        np.flatnonzero(atom_chain_ids == chain_id).astype(np.int64)
        for chain_id in selected_chain_ids
    ]

    try:
        bonds = np.asarray(trajectory.Universe.bonds.indices, dtype=np.int64)
    except NoDataError:
        bonds = np.empty((0, 2), dtype=np.int64)

    plan = build_unwrap_plan(molecule_atoms, selected_indices, bonds)
    return plan, np.asarray(selected_chain_ids, dtype=np.int64)


def _unwrap_selected_positions(all_positions, plan, dimensions):
    raw = np.asarray(all_positions[plan.atom_indices], dtype=np.float64)
    unwrapped = np.empty_like(raw)
    unwrapped[plan.roots] = raw[plan.roots]
    for parents, children in plan.levels:
        bond_vectors = raw[children] - raw[parents]
        minimum_vectors = minimize_vectors(bond_vectors, dimensions)
        unwrapped[children] = unwrapped[parents] + minimum_vectors
    return unwrapped[plan.selected_local_indices]


def _box_matrix(dimensions):
    dimensions = np.asarray(dimensions, dtype=np.float64)
    if dimensions.shape != (6,) or not np.all(np.isfinite(dimensions)):
        raise ValueError("Trajectory frame has invalid periodic box dimensions.")
    cell = np.asarray(triclinic_vectors(dimensions, dtype=np.float64), dtype=np.float64)
    determinant = float(np.linalg.det(cell))
    if not np.isfinite(determinant) or determinant <= 0.0:
        raise ValueError("Trajectory frame has a non-positive periodic box volume.")
    return cell


def _shortest_lattice_vector(cell):
    # DROPPS/OpenMM uses reduced triclinic boxes. Searching this local stencil
    # covers their shortest lattice vector and is more conservative than using
    # only the three stored box-vector lengths.
    translations = np.asarray(
        [value for value in product(range(-2, 3), repeat=3) if value != (0, 0, 0)],
        dtype=np.float64,
    )
    vectors = translations @ cell
    return float(np.sqrt(np.min(np.einsum("ij,ij->i", vectors, vectors))))


def _wrap_positions(positions, cell):
    inverse_cell = np.linalg.inv(cell)
    fractions = np.asarray(positions, dtype=np.float64) @ inverse_cell
    image_indices = np.floor(fractions).astype(np.int64)
    wrapped = (fractions - image_indices) @ cell
    return wrapped, image_indices, inverse_cell


def _pair_image_shifts(wrapped, image_indices, pairs, dimensions, inverse_cell):
    raw_vectors = wrapped[pairs[:, 1]] - wrapped[pairs[:, 0]]
    minimum_vectors = minimize_vectors(raw_vectors, dimensions)
    shift_values = (minimum_vectors - raw_vectors) @ inverse_cell
    minimum_shifts = np.rint(shift_values).astype(np.int64)
    if not np.allclose(shift_values, minimum_shifts, atol=2e-5, rtol=0.0):
        raise RuntimeError(
            "Could not resolve a neighbor contact to an integer box image."
        )
    return minimum_shifts + image_indices[pairs[:, 0]] - image_indices[pairs[:, 1]]


def _analyze_frame(all_positions, dimensions, plan, cutoff_angstrom, minimum_contacts):
    cell = _box_matrix(dimensions)
    shortest_vector = _shortest_lattice_vector(cell)
    if cutoff_angstrom >= 0.5 * shortest_vector:
        raise ValueError(
            f"The cutoff ({cutoff_angstrom / ANGSTROM_PER_NM:.6g} nm) must be "
            "less than half the shortest periodic lattice vector "
            f"({0.5 * shortest_vector / ANGSTROM_PER_NM:.6g} nm for this frame)."
        )

    selected_positions = _unwrap_selected_positions(all_positions, plan, dimensions)
    wrapped, image_indices, inverse_cell = _wrap_positions(selected_positions, cell)
    neighbor_result = FastNS(
        cutoff_angstrom,
        np.asarray(wrapped, dtype=np.float32),
        box=np.asarray(dimensions, dtype=np.float32),
        pbc=True,
    ).self_search()
    pairs = neighbor_result.get_pairs()

    if pairs.size == 0:
        edges = np.empty((0, 5), dtype=np.int64)
        edge_contact_counts = np.empty(0, dtype=np.int64)
    else:
        image_shifts = _pair_image_shifts(
            wrapped, image_indices, pairs, dimensions, inverse_cell
        )
        molecule_ids = plan.selected_molecule_ids
        edges, edge_contact_counts = canonicalize_image_contacts(
            molecule_ids[pairs[:, 0]],
            molecule_ids[pairs[:, 1]],
            image_shifts,
            minimum_contacts=minimum_contacts,
        )

    result = analyze_image_edges(int(plan.roots.size), edges)
    return result, edges, edge_contact_counts


def _output_paths(prefix, molecule_output):
    paths = {
        "frames": Path(f"{prefix}.frames.csv"),
        "summary": Path(f"{prefix}.summary.csv"),
    }
    if molecule_output is not None:
        paths["molecules"] = Path(molecule_output)
    resolved = [path.resolve() for path in paths.values()]
    if len(set(resolved)) != len(resolved):
        raise ValueError("Image-interaction output paths must be distinct.")
    return paths


def _write_summary(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("metric", "value"))
        writer.writerows(rows)


def image_interaction(args):
    trajectory = trajectory_class(args.run_input, args.index, args.input)
    start_frame, end_frame, interval_frame = trajectory.time2frame(
        args.start_time, args.end_time, args.delta_time
    )

    if args.selection_group is None:
        trajectory.index.print_all()
        selection, selection_name = trajectory.getSelection_interactive(
            "contact atoms for periodic-image interaction analysis"
        )
    else:
        selection, selection_name = trajectory.getSelection(
            f"group {args.selection_group}",
            "contact atoms for periodic-image interaction analysis",
        )

    plan, chain_ids = _select_molecules(trajectory, selection)
    molecule_count = int(plan.roots.size)
    cutoff_angstrom = args.cutoff * ANGSTROM_PER_NM
    paths = _output_paths(args.output_prefix, args.molecule_output)

    print(
        f"## Analyzing {molecule_count} molecules represented by "
        f"{plan.selected_local_indices.size} atoms in group '{selection_name}'."
    )
    print(
        f"## Contact cutoff: {args.cutoff} nm; minimum contacts per "
        f"molecule-image edge: {args.minimum_contacts}."
    )
    print(
        "## Direct contacts and indirect paths through any number of molecules "
        "will be included."
    )

    frame_fields = (
        "frame",
        "time_ns",
        "molecules",
        "molecule_image_edges",
        "atom_contacts_in_edges",
        "components",
        "wrapped_components",
        "winding_conflicts",
        "mirror_molecules",
        "direct_mirror_molecules",
        "indirect_only_mirror_molecules",
        "mirror_percentage",
        "has_mirror_interaction",
    )
    molecule_fields = (
        "frame",
        "time_ns",
        "molecule_id",
        "chain_id",
        "component_id",
        "mirror_interaction",
        "direct_mirror_interaction",
    )

    frame_count = 0
    frames_with_mirror = 0
    frames_with_direct = 0
    affected_observations = 0
    direct_observations = 0
    indirect_only_observations = 0
    ever_affected = np.zeros(molecule_count, dtype=bool)
    ever_direct = np.zeros(molecule_count, dtype=bool)

    frames_handle = paths["frames"].open("w", encoding="utf-8", newline="")
    molecule_handle = None
    try:
        frame_writer = csv.DictWriter(frames_handle, fieldnames=frame_fields)
        frame_writer.writeheader()
        molecule_writer = None
        if "molecules" in paths:
            molecule_handle = paths["molecules"].open("w", encoding="utf-8", newline="")
            molecule_writer = csv.DictWriter(
                molecule_handle, fieldnames=molecule_fields
            )
            molecule_writer.writeheader()

        frames = trajectory.Universe.trajectory[
            start_frame : end_frame + 1 : interval_frame
        ]
        for ts in tqdm(frames):
            result, edges, edge_contact_counts = _analyze_frame(
                trajectory.Universe.atoms.positions,
                ts.dimensions,
                plan,
                cutoff_angstrom,
                args.minimum_contacts,
            )
            time_ns = float(ts.time / PS_PER_NS)
            affected_count = int(np.count_nonzero(result.affected))
            direct_count = int(np.count_nonzero(result.direct))
            indirect_only_count = int(
                np.count_nonzero(result.affected & ~result.direct)
            )
            component_count = int(np.unique(result.component_labels).size)
            has_mirror = affected_count > 0

            frame_writer.writerow(
                {
                    "frame": int(ts.frame),
                    "time_ns": f"{time_ns:.12g}",
                    "molecules": molecule_count,
                    "molecule_image_edges": int(edges.shape[0]),
                    "atom_contacts_in_edges": int(edge_contact_counts.sum()),
                    "components": component_count,
                    "wrapped_components": result.wrapped_components,
                    "winding_conflicts": result.winding_conflicts,
                    "mirror_molecules": affected_count,
                    "direct_mirror_molecules": direct_count,
                    "indirect_only_mirror_molecules": indirect_only_count,
                    "mirror_percentage": f"{100.0 * affected_count / molecule_count:.12g}",
                    "has_mirror_interaction": int(has_mirror),
                }
            )

            if molecule_writer is not None:
                for molecule_id in range(molecule_count):
                    molecule_writer.writerow(
                        {
                            "frame": int(ts.frame),
                            "time_ns": f"{time_ns:.12g}",
                            "molecule_id": molecule_id,
                            "chain_id": int(chain_ids[molecule_id]),
                            "component_id": int(result.component_labels[molecule_id]),
                            "mirror_interaction": int(result.affected[molecule_id]),
                            "direct_mirror_interaction": int(
                                result.direct[molecule_id]
                            ),
                        }
                    )

            frame_count += 1
            frames_with_mirror += int(has_mirror)
            frames_with_direct += int(direct_count > 0)
            affected_observations += affected_count
            direct_observations += direct_count
            indirect_only_observations += indirect_only_count
            ever_affected |= result.affected
            ever_direct |= result.direct
    finally:
        frames_handle.close()
        if molecule_handle is not None:
            molecule_handle.close()

    if frame_count == 0:
        raise RuntimeError("No trajectory frames matched the requested time range.")

    molecule_observations = frame_count * molecule_count
    frame_percentage = 100.0 * frames_with_mirror / frame_count
    molecule_percentage = 100.0 * affected_observations / molecule_observations
    direct_percentage = 100.0 * direct_observations / molecule_observations
    indirect_percentage = 100.0 * indirect_only_observations / molecule_observations
    summary_rows = (
        ("selection_group", selection_name),
        ("selected_atoms", int(plan.selected_local_indices.size)),
        ("selected_molecules", molecule_count),
        ("cutoff_nm", f"{args.cutoff:.12g}"),
        ("minimum_contacts", args.minimum_contacts),
        ("analyzed_frames", frame_count),
        ("frames_with_mirror_interaction", frames_with_mirror),
        ("frame_percentage", f"{frame_percentage:.12g}"),
        ("molecule_observations", molecule_observations),
        ("mirror_molecule_observations", affected_observations),
        ("molecule_percentage", f"{molecule_percentage:.12g}"),
        ("direct_mirror_molecule_observations", direct_observations),
        ("direct_molecule_percentage", f"{direct_percentage:.12g}"),
        ("indirect_only_mirror_molecule_observations", indirect_only_observations),
        ("indirect_only_molecule_percentage", f"{indirect_percentage:.12g}"),
        ("molecules_ever_affected", int(np.count_nonzero(ever_affected))),
        (
            "molecules_ever_affected_percentage",
            f"{100.0 * np.count_nonzero(ever_affected) / molecule_count:.12g}",
        ),
        ("molecules_ever_directly_affected", int(np.count_nonzero(ever_direct))),
        (
            "frames_with_direct_mirror_interaction",
            frames_with_direct,
        ),
    )
    _write_summary(paths["summary"], summary_rows)

    print(f"## Frames with mirror interaction: {frame_percentage:.6g}%")
    print(f"## Mirror molecule-frame observations: {molecule_percentage:.6g}%")
    print(f"## Per-frame results written to {paths['frames']}.")
    print(f"## Summary written to {paths['summary']}.")
    if "molecules" in paths:
        print(f"## Per-molecule details written to {paths['molecules']}.")


image_interaction_commands = single_command(
    "pbcontact",
    getargs_image_interaction,
    image_interaction,
    desc,
)
