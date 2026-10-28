"""Infer secondary-structure-like states from residue-level CG trajectories."""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter, defaultdict
from contextlib import ExitStack
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from MDAnalysis.exceptions import NoDataError
from MDAnalysis.lib.distances import apply_PBC, minimize_vectors
from MDAnalysis.lib.mdamath import triclinic_vectors
from MDAnalysis.lib.nsgrid import FastNS
from scipy.spatial import cKDTree
from tqdm import tqdm

from dropps import __version__
from dropps.analysis.image_interaction_core import build_unwrap_plan
from dropps.analysis.ss_core import (
    DEFAULT_PSEA_PROFILE,
    STATE_CODES,
    STATE_COIL,
    STATE_HELIX,
    STATE_STRAND,
    apply_strand_pairing,
    classify_secondary_structure,
    consensus_states,
    detect_interchain_strand_pairing,
    state_fractions,
)
from dropps.fileio.filename_control import validate_extension
from dropps.fileio.xpm_reader import write_discrete_xpm
from dropps.fileio.xvg_reader import write_xvg
from dropps.share.argument_parser import ArgumentParser
from dropps.share.command_class import single_command
from dropps.share.trajectory import trajectory_class


prog = "ss"
desc = "Infer H/E/C states from residue-level coarse-grained backbone geometry."

ANGSTROM_PER_NM = 10.0

STATE_COLORS = {
    "C": "#FFFFFF",
    "H": "#3366CC",
    "E": "#CC3333",
    "M": "#A6A6A6",
    "X": "#000000",
}
STATE_LABELS = {
    "C": "Coil",
    "H": "Helix",
    "E": "Beta strand",
    "M": "Mixed consensus",
    "X": "Invalid",
}


@dataclass
class SSCohort:
    """Equivalent selected molecule copies analyzed as one ensemble."""

    selection_label: str
    label: str
    molecule_name: str
    chain_ids: np.ndarray
    selected_indices: np.ndarray
    position_indices: np.ndarray
    residue_ids: np.ndarray
    total_type_copies: int


def getargs_ss(argv):
    parser = ArgumentParser(
        prog=prog,
        description=desc,
        epilog=(
            "This is a geometry-based C-alpha/bead-trace inference, not a "
            "hydrogen-bond DSSP assignment. Select exactly one ordered backbone "
            "bead per residue across one or more molecule copies."
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
        help="Optional index file (.ndx) defining backbone selections.",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Long-form per-frame, per-residue state fractions (.csv).",
    )
    parser.add_argument(
        "-map",
        "--map-output",
        help="GROMACS-compatible discrete state map (.xpm).",
    )
    parser.add_argument(
        "-num",
        "--number-output",
        help="Mean C/H/E residue counts per molecule copy versus time (.xvg).",
    )
    parser.add_argument(
        "-occ",
        "--occupancy-output",
        help="Per-residue C/H/E occupancies over frames and copies (.xvg).",
    )
    parser.add_argument(
        "--strand-stat-output",
        help="Per-residue local, paired, and final E occupancies (.xvg).",
    )
    parser.add_argument(
        "--strand-pair-output",
        help="Accepted inter-chain strand pairings for every frame (.csv).",
    )
    parser.add_argument(
        "--plot-prefix",
        section="output",
        help="Also render map, count, and occupancy plots with this prefix.",
    )

    selection_source = parser.add_mutually_exclusive_group(section="input")
    selection_source.add_argument(
        "-sel",
        "--selection-groups",
        type=int,
        nargs="+",
        metavar="GROUP",
        help="Index groups containing one backbone bead per selected residue.",
    )
    selection_source.add_argument(
        "--select",
        "--selection-expression",
        dest="selection_expressions",
        action="append",
        metavar="EXPR",
        help="DROPPS selection expression; repeat for independent selections.",
    )
    parser.add_argument(
        "--method",
        choices=("angle", "psea"),
        default="psea",
        help="Local geometry classifier; psea also uses torsions and span distances.",
    )
    parser.add_argument(
        "--normalize-bonds",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Scale span distances to a 0.38 nm reference backbone bond.",
    )
    parser.add_argument(
        "--min-helix-length",
        type=int,
        default=4,
        help="Minimum number of consecutive residues in an H segment.",
    )
    parser.add_argument(
        "--min-strand-length",
        type=int,
        default=3,
        help="Minimum number of consecutive residues in an E segment.",
    )
    parser.add_argument(
        "--helix-angle",
        type=float,
        nargs=2,
        metavar=("CENTER", "HALF_WIDTH"),
        default=DEFAULT_PSEA_PROFILE.helix_angle,
        help="Virtual-angle center and half-width for H, in degrees.",
    )
    parser.add_argument(
        "--strand-angle",
        type=float,
        nargs=2,
        metavar=("CENTER", "HALF_WIDTH"),
        default=DEFAULT_PSEA_PROFILE.strand_angle,
        help="Virtual-angle center and half-width for E, in degrees.",
    )
    parser.add_argument(
        "--helix-dihedral",
        type=float,
        nargs=2,
        metavar=("CENTER", "HALF_WIDTH"),
        default=DEFAULT_PSEA_PROFILE.helix_dihedral,
        help="Circular pseudo-dihedral center and half-width for H, in degrees.",
    )
    parser.add_argument(
        "--strand-dihedral",
        type=float,
        nargs=2,
        metavar=("CENTER", "HALF_WIDTH"),
        default=DEFAULT_PSEA_PROFILE.strand_dihedral,
        help="Circular pseudo-dihedral center and half-width for E, in degrees.",
    )
    parser.add_argument(
        "--strand-pairing",
        choices=("off", "interchain"),
        default="off",
        help="Require registered inter-chain contacts before retaining local E.",
    )
    parser.add_argument(
        "--strand-contact-scheme",
        choices=("global", "residue"),
        default="global",
        help="Contact cutoff used only for optional strand pairing.",
    )
    parser.add_argument(
        "--strand-contact-cutoff",
        type=float,
        default=0.7,
        help="Global strand-pair contact cutoff in nm.",
    )
    parser.add_argument(
        "--strand-contact-multiplier",
        type=float,
        default=1.2,
        help="Multiplier applied to bead sigma for residue-responsive contacts.",
    )
    parser.add_argument(
        "--strand-axis-angle",
        type=float,
        default=30.0,
        help="Maximum parallel or antiparallel strand-axis mismatch in degrees.",
    )
    parser.add_argument(
        "--strand-lateral-angle",
        type=float,
        default=35.0,
        help="Maximum contact-vector deviation from the lateral plane in degrees.",
    )
    parser.add_argument(
        "--strand-min-pairs",
        type=int,
        default=2,
        help="Minimum registered residue contacts needed for a strand pair.",
    )
    parser.add_argument(
        "--strand-registry-slack",
        type=int,
        default=1,
        help="Allowed residue-index deviation within one strand registry.",
    )
    parser.add_argument(
        "--consensus-threshold",
        type=float,
        default=0.5,
        help="Minimum copy fraction for a categorical state in an ensemble map.",
    )
    parser.add_argument(
        "--pbc",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Unwrap bonded molecules and use minimum-image strand contacts.",
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
        "--plot-formats",
        choices=("png", "pdf", "svg"),
        nargs="+",
        default=("png",),
        help="File formats written when --plot-prefix is set.",
    )
    parser.add_argument(
        "--plot-dpi",
        type=int,
        default=300,
        help="Raster resolution for direct plots.",
    )
    args = parser.parse_args(argv)

    if args.min_helix_length < 1 or args.min_strand_length < 1:
        parser.error("minimum secondary-structure lengths must be positive")
    for option, values in (
        ("--helix-angle", args.helix_angle),
        ("--strand-angle", args.strand_angle),
    ):
        center, half_width = values
        if not 0.0 <= center <= 180.0 or not 0.0 < half_width <= 180.0:
            parser.error(
                f"{option} requires CENTER in [0, 180] and HALF_WIDTH in (0, 180]"
            )
    for option, values in (
        ("--helix-dihedral", args.helix_dihedral),
        ("--strand-dihedral", args.strand_dihedral),
    ):
        center, half_width = values
        if not -180.0 <= center <= 180.0 or not 0.0 < half_width <= 180.0:
            parser.error(
                f"{option} requires CENTER in [-180, 180] and HALF_WIDTH in (0, 180]"
            )
    if args.strand_contact_cutoff <= 0.0:
        parser.error("--strand-contact-cutoff must be greater than zero")
    if args.strand_contact_multiplier <= 0.0:
        parser.error("--strand-contact-multiplier must be greater than zero")
    if not 0.0 <= args.strand_axis_angle < 90.0:
        parser.error("--strand-axis-angle must be in [0, 90)")
    if not 0.0 <= args.strand_lateral_angle < 90.0:
        parser.error("--strand-lateral-angle must be in [0, 90)")
    if args.strand_min_pairs < 1:
        parser.error("--strand-min-pairs must be greater than zero")
    if args.strand_registry_slack < 0:
        parser.error("--strand-registry-slack cannot be negative")
    if not 0.0 <= args.consensus_threshold <= 1.0:
        parser.error("--consensus-threshold must be between zero and one")
    if args.plot_dpi < 1:
        parser.error("--plot-dpi must be greater than zero")
    if args.strand_pair_output and args.strand_pairing != "interchain":
        parser.error("--strand-pair-output requires --strand-pairing interchain")
    args.plot_formats = tuple(dict.fromkeys(args.plot_formats))
    args.helix_angle = tuple(args.helix_angle)
    args.strand_angle = tuple(args.strand_angle)
    args.helix_dihedral = tuple(args.helix_dihedral)
    args.strand_dihedral = tuple(args.strand_dihedral)
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
                f"group {group_id}",
                "secondary-structure backbone",
            )
            selections.append(selection)
            names.append(name)
    elif args.selection_expressions:
        for expression in args.selection_expressions:
            selection, _ = trajectory.getSelection(
                expression,
                "secondary-structure backbone",
            )
            selections.append(selection)
            names.append(expression)
    else:
        trajectory.index.print_all()
        selections, names = trajectory.getSelection_interactive_multiple(
            "secondary-structure backbones"
        )

    if not selections:
        raise ValueError("At least one secondary-structure selection is required.")
    resolved = []
    for label, selection in zip(_unique_labels(names), selections):
        indices = np.asarray(selection.indices, dtype=np.int64)
        if indices.size == 0:
            raise ValueError(f"Secondary-structure selection '{label}' is empty.")
        if np.unique(indices).size != indices.size:
            raise ValueError(
                f"Secondary-structure selection '{label}' contains duplicates."
            )
        resolved.append((label, indices))
    return resolved


def _topology_bonds(trajectory):
    try:
        bonds = np.asarray(trajectory.Universe.bonds.indices, dtype=np.int64)
    except (AttributeError, NoDataError):
        return np.empty((0, 2), dtype=np.int64)
    if bonds.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    return bonds


def _system_molecules(atom_chain_ids):
    groups = {}
    for atom_index, chain_id in enumerate(atom_chain_ids):
        groups.setdefault(int(chain_id), []).append(atom_index)
    return {
        chain_id: np.asarray(indices, dtype=np.int64)
        for chain_id, indices in groups.items()
    }


def _molecule_name(trajectory, chain_id, first_index):
    itp_list = getattr(trajectory.tpr, "itp_list", ())
    if 0 <= chain_id < len(itp_list):
        return str(
            getattr(itp_list[chain_id], "molecule_name", f"molecule-{chain_id}")
        )
    names = getattr(trajectory.index, "atom_molnames", None)
    if names is not None:
        return str(names[first_index])
    return f"molecule-{chain_id}"


def _atom_identity(trajectory, global_index):
    fields = []
    for attribute in ("atom_names", "atom_abbrs", "atom_resnames"):
        values = getattr(trajectory.index, attribute, None)
        fields.append(str(values[global_index]) if values is not None else "")
    return tuple(fields)


def _build_cohorts(trajectory, selections, args):
    atom_chain_ids = np.asarray(trajectory.id2chainID, dtype=np.int64)
    atom_residue_ids = np.asarray(trajectory.id2resID, dtype=np.int64)
    bonds = _topology_bonds(trajectory)
    bond_pairs = {
        tuple(sorted((int(atom_a), int(atom_b)))) for atom_a, atom_b in bonds
    }
    molecules = _system_molecules(atom_chain_ids)
    type_totals = Counter(
        _molecule_name(trajectory, chain_id, int(indices[0]))
        for chain_id, indices in molecules.items()
    )

    cohorts = []
    represented_chain_ids = []
    union_selected = []
    for selection_label, selection_indices in selections:
        selected_set = set(int(value) for value in selection_indices)
        fragments = []
        for chain_id, full_indices in molecules.items():
            selected = np.asarray(
                [int(index) for index in full_indices if int(index) in selected_set],
                dtype=np.int64,
            )
            if selected.size == 0:
                continue
            residue_ids = atom_residue_ids[selected]
            if np.unique(residue_ids).size != selected.size:
                raise ValueError(
                    f"Selection '{selection_label}' has more than one bead for a "
                    f"residue in chain {chain_id}; select one backbone bead per residue."
                )
            if selected.size < 3:
                raise ValueError(
                    f"Selection '{selection_label}' chain {chain_id} needs at least "
                    "three consecutive backbone beads."
                )
            if any(
                tuple(sorted((int(atom_a), int(atom_b)))) not in bond_pairs
                for atom_a, atom_b in zip(selected[:-1], selected[1:])
            ):
                raise ValueError(
                    f"Selection '{selection_label}' chain {chain_id} is not a "
                    "consecutive bonded path in topology order."
                )
            local_lookup = {
                int(global_index): local_index
                for local_index, global_index in enumerate(full_indices)
            }
            local_selection = tuple(local_lookup[int(index)] for index in selected)
            identity = tuple(
                _atom_identity(trajectory, int(index)) for index in selected
            )
            molecule_name = _molecule_name(
                trajectory,
                chain_id,
                int(full_indices[0]),
            )
            signature = (
                local_selection,
                tuple(int(value) for value in residue_ids),
                identity,
            )
            fragments.append(
                {
                    "chain_id": int(chain_id),
                    "molecule_name": molecule_name,
                    "signature": signature,
                    "selected": selected,
                    "residue_ids": residue_ids.astype(np.int64, copy=True),
                }
            )

        grouped = defaultdict(list)
        for fragment in fragments:
            grouped[(fragment["molecule_name"], fragment["signature"])].append(
                fragment
            )
        name_group_counts = Counter(name for name, _ in grouped)
        name_seen = Counter()
        for (molecule_name, _), rows in grouped.items():
            name_seen[molecule_name] += 1
            label = molecule_name
            if name_group_counts[molecule_name] > 1:
                label = f"{molecule_name}#{name_seen[molecule_name]}"
            selected_matrix = np.stack([row["selected"] for row in rows])
            residue_matrix = np.stack([row["residue_ids"] for row in rows])
            if not np.all(residue_matrix == residue_matrix[0]):
                raise ValueError(
                    f"Cohort '{selection_label}:{label}' has inconsistent residue IDs."
                )
            cohort = SSCohort(
                selection_label=selection_label,
                label=label,
                molecule_name=molecule_name,
                chain_ids=np.asarray(
                    [row["chain_id"] for row in rows],
                    dtype=np.int64,
                ),
                selected_indices=selected_matrix,
                position_indices=np.empty_like(selected_matrix),
                residue_ids=residue_matrix[0],
                total_type_copies=int(type_totals[molecule_name]),
            )
            cohorts.append(cohort)
            represented_chain_ids.extend(int(value) for value in cohort.chain_ids)
            union_selected.extend(int(value) for value in selected_matrix.reshape(-1))

    if not cohorts:
        raise ValueError("No molecule cohort was produced from the selections.")
    union_selected = np.asarray(
        list(dict.fromkeys(union_selected)),
        dtype=np.int64,
    )
    union_position = {
        int(atom_index): position
        for position, atom_index in enumerate(union_selected)
    }
    for cohort in cohorts:
        cohort.position_indices = np.asarray(
            [
                [union_position[int(atom_index)] for atom_index in row]
                for row in cohort.selected_indices
            ],
            dtype=np.int64,
        )
    unique_chain_ids = list(dict.fromkeys(represented_chain_ids))
    molecule_atom_indices = [molecules[chain_id] for chain_id in unique_chain_ids]
    return cohorts, molecule_atom_indices, union_selected, bonds


def _validate_box_dimensions(dimensions):
    if dimensions is None:
        raise ValueError("Trajectory frame does not contain a periodic box.")
    values = np.asarray(dimensions, dtype=np.float64)
    if values.shape != (6,) or not np.all(np.isfinite(values)):
        raise ValueError("Trajectory frame has invalid periodic box dimensions.")
    cell = np.asarray(triclinic_vectors(values, dtype=np.float64))
    if cell.shape != (3, 3) or float(np.linalg.det(cell)) <= 0.0:
        raise ValueError("Trajectory frame has a non-positive periodic box volume.")
    return values


def _frame_selected_positions_nm(timestep, unwrap_plan, selected_indices, pbc):
    if not pbc:
        return (
            np.asarray(timestep.positions[selected_indices], dtype=np.float64)
            / ANGSTROM_PER_NM
        )
    dimensions = _validate_box_dimensions(timestep.dimensions)
    raw = np.asarray(timestep.positions[unwrap_plan.atom_indices], dtype=np.float64)
    unwrapped = np.empty_like(raw)
    unwrapped[unwrap_plan.roots] = raw[unwrap_plan.roots]
    for parents, children in unwrap_plan.levels:
        bond_vectors = raw[children] - raw[parents]
        unwrapped[children] = unwrapped[parents] + minimize_vectors(
            bond_vectors,
            dimensions,
        )
    return unwrapped[unwrap_plan.selected_local_indices] / ANGSTROM_PER_NM


def _strand_contacts(trajectory, timestep, cohort, local_strand, args):
    candidate_flat = np.flatnonzero(np.asarray(local_strand, dtype=bool).reshape(-1))
    if candidate_flat.size < 2:
        return np.empty((0, 4), dtype=np.int64), np.empty((0, 3), dtype=np.float64)

    all_global = cohort.selected_indices.reshape(-1)
    candidate_global = all_global[candidate_flat]
    positions = np.asarray(timestep.positions[candidate_global], dtype=np.float32)
    dimensions = None
    if args.pbc:
        dimensions = _validate_box_dimensions(timestep.dimensions)
        positions = np.asarray(apply_PBC(positions, dimensions), dtype=np.float32)

    atom_cutoffs = None
    if args.strand_contact_scheme == "global":
        maximum_cutoff = float(args.strand_contact_cutoff) * ANGSTROM_PER_NM
    else:
        sigmas = getattr(trajectory, "sigmas", None)
        if sigmas is None:
            raise ValueError(
                "Residue-responsive strand contacts require sigma values in the run input."
            )
        sigma_values = np.asarray(sigmas, dtype=np.float64)
        if sigma_values.size <= int(np.max(candidate_global)):
            raise ValueError("Sigma data does not cover every strand candidate bead.")
        atom_cutoffs = (
            float(args.strand_contact_multiplier)
            * sigma_values[candidate_global]
            * ANGSTROM_PER_NM
        )
        maximum_cutoff = float(np.max(atom_cutoffs))

    if args.pbc:
        result = FastNS(
            maximum_cutoff,
            positions,
            box=np.asarray(dimensions, dtype=np.float32),
            pbc=True,
        ).self_search()
        pairs = np.asarray(result.get_pairs(), dtype=np.int64)
        distances = np.asarray(result.get_pair_distances(), dtype=np.float64)
    else:
        pairs = cKDTree(np.asarray(positions, dtype=np.float64)).query_pairs(
            maximum_cutoff,
            output_type="ndarray",
        )
        pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
        if pairs.size:
            distances = np.linalg.norm(
                positions[pairs[:, 1]] - positions[pairs[:, 0]],
                axis=1,
            )
        else:
            distances = np.empty(0, dtype=np.float64)
    if pairs.size == 0:
        return np.empty((0, 4), dtype=np.int64), np.empty((0, 3), dtype=np.float64)

    keep = np.ones(pairs.shape[0], dtype=bool)
    if atom_cutoffs is not None:
        keep &= distances <= np.mean(atom_cutoffs[pairs], axis=1)
    length = cohort.selected_indices.shape[1]
    local_a = candidate_flat[pairs[:, 0]]
    local_b = candidate_flat[pairs[:, 1]]
    chain_a, residue_a = np.divmod(local_a, length)
    chain_b, residue_b = np.divmod(local_b, length)
    keep &= chain_a != chain_b
    if not np.any(keep):
        return np.empty((0, 4), dtype=np.int64), np.empty((0, 3), dtype=np.float64)

    pairs = pairs[keep]
    contact_pairs = np.column_stack(
        (chain_a[keep], residue_a[keep], chain_b[keep], residue_b[keep])
    ).astype(np.int64, copy=False)
    vectors = np.asarray(
        positions[pairs[:, 1]] - positions[pairs[:, 0]],
        dtype=np.float64,
    )
    if args.pbc:
        vectors = minimize_vectors(vectors, dimensions)
    return contact_pairs, vectors / ANGSTROM_PER_NM


def _output_paths(args):
    specifications = (
        ("state", args.output, "csv"),
        ("map", args.map_output, "xpm"),
        ("number", args.number_output, "xvg"),
        ("occupancy", args.occupancy_output, "xvg"),
        ("strand_stat", args.strand_stat_output, "xvg"),
        ("strand_pair", args.strand_pair_output, "csv"),
    )
    paths = {
        key: Path(validate_extension(value, extension))
        for key, value, extension in specifications
        if value is not None
    }
    resolved = [path.resolve() for path in paths.values()]
    if len(resolved) != len(set(resolved)):
        raise ValueError("Secondary-structure output paths must be distinct.")
    return paths


def _cohort_name(cohort):
    return f"{cohort.selection_label}:{cohort.label}"


def _slug(value):
    result = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value)).strip("-._")
    return result or "cohort"


def _cohort_path(path, cohort, cohort_count):
    if cohort_count == 1:
        return path
    return path.with_name(
        f"{path.stem}-{_slug(_cohort_name(cohort))}{path.suffix}"
    )


def _new_runtime(cohort, keep_map):
    length = cohort.residue_ids.size
    return {
        "times": [],
        "map_frames": [] if keep_map else None,
        "numbers": [],
        "occupancy_counts": np.zeros((length, 3), dtype=np.int64),
        "local_e_counts": np.zeros(length, dtype=np.int64),
        "paired_e_counts": np.zeros(length, dtype=np.int64),
        "observations": 0,
    }


def _state_fields():
    return (
        "frame",
        "time_ns",
        "selection",
        "cohort",
        "molecule_name",
        "resid",
        "copies",
        "c_fraction",
        "h_fraction",
        "e_local_fraction",
        "e_paired_fraction",
        "e_final_fraction",
        "dominant",
        "consensus_fraction",
        "strand_pairing",
    )


def _pair_fields():
    return (
        "frame",
        "time_ns",
        "selection",
        "cohort",
        "molecule_name",
        "chain_a",
        "start_resid_a",
        "stop_resid_a",
        "chain_b",
        "start_resid_b",
        "stop_resid_b",
        "orientation",
        "registry",
        "paired_contacts",
        "mean_distance_nm",
        "minimum_distance_nm",
    )


def _write_state_rows(writer, frame, time_ns, cohort, fractions, local_e, paired_e, args):
    consensus = consensus_states(fractions, threshold=args.consensus_threshold)
    confidence = np.max(fractions, axis=1)
    for residue_index, residue_id in enumerate(cohort.residue_ids):
        writer.writerow(
            {
                "frame": int(frame),
                "time_ns": f"{time_ns:.9g}",
                "selection": cohort.selection_label,
                "cohort": cohort.label,
                "molecule_name": cohort.molecule_name,
                "resid": int(residue_id),
                "copies": int(cohort.chain_ids.size),
                "c_fraction": f"{fractions[residue_index, 0]:.9g}",
                "h_fraction": f"{fractions[residue_index, 1]:.9g}",
                "e_local_fraction": f"{local_e[residue_index]:.9g}",
                "e_paired_fraction": (
                    f"{paired_e[residue_index]:.9g}"
                    if args.strand_pairing == "interchain"
                    else ""
                ),
                "e_final_fraction": f"{fractions[residue_index, 2]:.9g}",
                "dominant": STATE_CODES[int(consensus[residue_index])],
                "consensus_fraction": f"{confidence[residue_index]:.9g}",
                "strand_pairing": args.strand_pairing,
            }
        )
    return consensus


def _write_pair_rows(writer, frame, time_ns, cohort, records):
    for record in records:
        writer.writerow(
            {
                "frame": int(frame),
                "time_ns": f"{time_ns:.9g}",
                "selection": cohort.selection_label,
                "cohort": cohort.label,
                "molecule_name": cohort.molecule_name,
                "chain_a": int(cohort.chain_ids[record.chain_a]),
                "start_resid_a": int(cohort.residue_ids[record.start_a]),
                "stop_resid_a": int(cohort.residue_ids[record.stop_a - 1]),
                "chain_b": int(cohort.chain_ids[record.chain_b]),
                "start_resid_b": int(cohort.residue_ids[record.start_b]),
                "stop_resid_b": int(cohort.residue_ids[record.stop_b - 1]),
                "orientation": record.orientation,
                "registry": int(record.registry),
                "paired_contacts": int(record.paired_residues),
                "mean_distance_nm": f"{record.mean_distance_nm:.9g}",
                "minimum_distance_nm": f"{record.minimum_distance_nm:.9g}",
            }
        )


def _write_number_output(path, cohorts, runtimes):
    times = np.asarray(runtimes[0]["times"], dtype=np.float64)
    series = []
    legends = []
    for cohort, runtime in zip(cohorts, runtimes):
        if not np.allclose(times, runtime["times"], rtol=0.0, atol=1.0e-12):
            raise RuntimeError("Cohort time axes unexpectedly differ.")
        numbers = np.asarray(runtime["numbers"], dtype=np.float64)
        for state_index, state_name in enumerate(("C", "H", "E")):
            series.append(numbers[:, state_index])
            legends.append(f"{_cohort_name(cohort)} {state_name}")
    write_xvg(
        path,
        times,
        series,
        title="Secondary-structure-like residue counts",
        xlabel="Time (ns)",
        ylabel="Mean residues per molecule copy",
        legends=legends,
    )


def _write_cohort_outputs(paths, cohorts, runtimes, args):
    cohort_count = len(cohorts)
    written = []
    for cohort, runtime in zip(cohorts, runtimes):
        denominator = float(runtime["observations"])
        occupancy = runtime["occupancy_counts"] / denominator
        local_e = runtime["local_e_counts"] / denominator
        paired_e = runtime["paired_e_counts"] / denominator
        if args.strand_pairing != "interchain":
            paired_e = np.full_like(paired_e, np.nan)
        if "occupancy" in paths:
            output = _cohort_path(paths["occupancy"], cohort, cohort_count)
            write_xvg(
                output,
                cohort.residue_ids,
                occupancy.T,
                title=f"Secondary-structure occupancy: {_cohort_name(cohort)}",
                xlabel="Residue ID",
                ylabel="Occupancy",
                legends=("C", "H", "E"),
            )
            written.append(output)
        if "strand_stat" in paths:
            output = _cohort_path(paths["strand_stat"], cohort, cohort_count)
            write_xvg(
                output,
                cohort.residue_ids,
                (local_e, occupancy[:, 2], paired_e),
                title=f"Strand occupancy diagnostics: {_cohort_name(cohort)}",
                xlabel="Residue ID",
                ylabel="E occupancy",
                legends=("E local", "E final", "E paired"),
            )
            written.append(output)
        if "map" in paths:
            output = _cohort_path(paths["map"], cohort, cohort_count)
            state_values = np.asarray(runtime["map_frames"], dtype=np.uint8).T
            state_chars = np.vectorize(STATE_CODES.__getitem__)(state_values)
            write_discrete_xpm(
                state_chars,
                output,
                colors=STATE_COLORS,
                labels=STATE_LABELS,
                title=f"Secondary structure: {_cohort_name(cohort)}",
                x_axis=runtime["times"],
                y_axis=cohort.residue_ids,
            )
            written.append(output)
    return written


def _plot_outputs(prefix, formats, dpi, cohorts, runtimes):
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        from matplotlib.patches import Patch
    except ImportError as exc:
        raise RuntimeError(
            "Direct plotting requires matplotlib; install the 'dropps[plot]' extra."
        ) from exc

    prefix = Path(prefix)
    cohort_count = len(cohorts)
    outputs = []
    cmap = ListedColormap(
        [STATE_COLORS[key] for key in ("C", "H", "E", "M", "X")]
    )
    legend_handles = [
        Patch(facecolor=STATE_COLORS[key], edgecolor="black", label=STATE_LABELS[key])
        for key in ("C", "H", "E", "M")
    ]
    for cohort, runtime in zip(cohorts, runtimes):
        suffix = "" if cohort_count == 1 else f"-{_slug(_cohort_name(cohort))}"
        times = np.asarray(runtime["times"], dtype=np.float64)
        states = np.asarray(runtime["map_frames"], dtype=np.uint8).T
        residue_ids = cohort.residue_ids
        occupancy = runtime["occupancy_counts"] / float(runtime["observations"])
        numbers = np.asarray(runtime["numbers"], dtype=np.float64)

        figures = []
        fig, axis = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
        if times.size == 1:
            x_extent = (times[0] - 0.5, times[0] + 0.5)
        else:
            half = 0.5 * float(np.median(np.diff(times)))
            x_extent = (times[0] - half, times[-1] + half)
        axis.imshow(
            states,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            vmin=0,
            vmax=4,
            extent=(x_extent[0], x_extent[1], -0.5, residue_ids.size - 0.5),
        )
        tick_step = max(1, int(np.ceil(residue_ids.size / 10.0)))
        tick_positions = np.arange(0, residue_ids.size, tick_step)
        axis.set_yticks(tick_positions, residue_ids[tick_positions])
        axis.set(xlabel="Time (ns)", ylabel="Residue ID", title=_cohort_name(cohort))
        axis.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.01, 1.0))
        figures.append(("map", fig))

        fig, axis = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
        for state_index, (state, color) in enumerate(
            (("C", "#666666"), ("H", STATE_COLORS["H"]), ("E", STATE_COLORS["E"]))
        ):
            axis.plot(times, numbers[:, state_index], label=state, color=color)
        axis.set(
            xlabel="Time (ns)",
            ylabel="Mean residues per molecule copy",
            title=_cohort_name(cohort),
        )
        axis.legend(frameon=False)
        figures.append(("count", fig))

        fig, axis = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
        for state_index, (state, color) in enumerate(
            (("C", "#666666"), ("H", STATE_COLORS["H"]), ("E", STATE_COLORS["E"]))
        ):
            axis.plot(residue_ids, occupancy[:, state_index], label=state, color=color)
        axis.set(
            xlabel="Residue ID",
            ylabel="Occupancy",
            ylim=(0.0, 1.0),
            title=_cohort_name(cohort),
        )
        axis.legend(frameon=False)
        figures.append(("occupancy", fig))

        for kind, figure in figures:
            for file_format in formats:
                output = prefix.with_name(
                    f"{prefix.name}{suffix}-{kind}.{file_format}"
                )
                figure.savefig(output, dpi=dpi)
                outputs.append(output)
            plt.close(figure)
    return outputs


def ss(args):
    trajectory = trajectory_class(args.run_input, args.index, args.input)
    start_frame, end_frame, interval_frame = trajectory.time2frame(
        args.start_time,
        args.end_time,
        args.delta_time,
    )
    frame_indices = range(start_frame, end_frame + 1, interval_frame)
    selections = _resolve_selections(trajectory, args)
    cohorts, molecule_atom_indices, union_selected, bonds = _build_cohorts(
        trajectory,
        selections,
        args,
    )
    unwrap_plan = None
    if args.pbc:
        unwrap_plan = build_unwrap_plan(
            molecule_atom_indices,
            union_selected,
            bonds,
        )
    paths = _output_paths(args)
    profile = replace(
        DEFAULT_PSEA_PROFILE,
        helix_angle=args.helix_angle,
        strand_angle=args.strand_angle,
        helix_dihedral=args.helix_dihedral,
        strand_dihedral=args.strand_dihedral,
    )
    keep_map = bool("map" in paths or args.plot_prefix)
    runtimes = [_new_runtime(cohort, keep_map) for cohort in cohorts]

    print(
        f"## Secondary-structure selections: {len(selections)}; cohorts: "
        f"{len(cohorts)}; represented copies: "
        f"{sum(cohort.chain_ids.size for cohort in cohorts)}."
    )
    print(
        f"## Geometry method={args.method}; strand_pairing={args.strand_pairing}; "
        f"pbc={str(bool(args.pbc)).lower()}."
    )

    with ExitStack() as stack:
        state_handle = stack.enter_context(paths["state"].open("w", newline=""))
        state_writer = csv.DictWriter(state_handle, fieldnames=_state_fields())
        state_writer.writeheader()
        pair_writer = None
        if "strand_pair" in paths:
            pair_handle = stack.enter_context(
                paths["strand_pair"].open("w", newline="")
            )
            pair_writer = csv.DictWriter(pair_handle, fieldnames=_pair_fields())
            pair_writer.writeheader()

        for frame_index in tqdm(frame_indices, desc="Secondary structure"):
            timestep = trajectory.Universe.trajectory[frame_index]
            time_ns = float(timestep.time) / 1000.0
            selected_positions = _frame_selected_positions_nm(
                timestep,
                unwrap_plan,
                union_selected,
                args.pbc,
            )
            for cohort, runtime in zip(cohorts, runtimes):
                coordinates = selected_positions[cohort.position_indices]
                local = classify_secondary_structure(
                    coordinates,
                    method=args.method,
                    min_helix_length=args.min_helix_length,
                    min_strand_length=args.min_strand_length,
                    normalize_bonds=args.normalize_bonds,
                    profile=profile,
                )
                paired = np.zeros_like(local.strand)
                records = []
                if args.strand_pairing == "interchain":
                    contact_pairs, contact_vectors = _strand_contacts(
                        trajectory,
                        timestep,
                        cohort,
                        local.strand,
                        args,
                    )
                    paired, records = detect_interchain_strand_pairing(
                        coordinates,
                        local.strand,
                        contact_pairs,
                        contact_vectors,
                        max_axis_angle_degrees=args.strand_axis_angle,
                        max_lateral_deviation_degrees=args.strand_lateral_angle,
                        min_pairs=args.strand_min_pairs,
                        registry_slack=args.strand_registry_slack,
                        min_strand_length=args.min_strand_length,
                    )
                    states = apply_strand_pairing(local.states, paired)
                else:
                    states = local.states

                fractions = state_fractions(states)
                local_e_fraction = np.mean(local.strand, axis=0)
                paired_e_fraction = np.mean(paired, axis=0)
                consensus = _write_state_rows(
                    state_writer,
                    frame_index,
                    time_ns,
                    cohort,
                    fractions,
                    local_e_fraction,
                    paired_e_fraction,
                    args,
                )
                if pair_writer is not None:
                    _write_pair_rows(
                        pair_writer,
                        frame_index,
                        time_ns,
                        cohort,
                        records,
                    )

                runtime["times"].append(time_ns)
                if runtime["map_frames"] is not None:
                    runtime["map_frames"].append(consensus)
                runtime["numbers"].append(np.sum(fractions, axis=0))
                for state_index, state in enumerate(
                    (STATE_COIL, STATE_HELIX, STATE_STRAND)
                ):
                    runtime["occupancy_counts"][:, state_index] += np.sum(
                        states == state,
                        axis=0,
                    )
                runtime["local_e_counts"] += np.sum(local.strand, axis=0)
                runtime["paired_e_counts"] += np.sum(paired, axis=0)
                runtime["observations"] += int(cohort.chain_ids.size)

    written = [paths["state"]]
    if "number" in paths:
        _write_number_output(paths["number"], cohorts, runtimes)
        written.append(paths["number"])
    written.extend(_write_cohort_outputs(paths, cohorts, runtimes, args))
    if "strand_pair" in paths:
        written.append(paths["strand_pair"])
    if args.plot_prefix:
        written.extend(
            _plot_outputs(
                args.plot_prefix,
                args.plot_formats,
                args.plot_dpi,
                cohorts,
                runtimes,
            )
        )

    for cohort, runtime in zip(cohorts, runtimes):
        occupancy = runtime["occupancy_counts"] / float(runtime["observations"])
        mean_fraction = np.mean(occupancy, axis=0)
        print(
            f"## {_cohort_name(cohort)}: copies={cohort.chain_ids.size}/"
            f"{cohort.total_type_copies}; mean residue fractions "
            f"C={mean_fraction[0]:.4f}, H={mean_fraction[1]:.4f}, "
            f"E={mean_fraction[2]:.4f}."
        )
    print(f"## DROPPS version {__version__}; wrote {len(written)} output file(s).")
    for path in written:
        print(f"## Wrote {path}.")


ss_commands = single_command(prog, getargs_ss, ss, desc)
