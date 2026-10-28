"""Many-protein, PBC-aware polymer size and backbone statistics."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from MDAnalysis.exceptions import NoDataError
from MDAnalysis.lib.distances import minimize_vectors
from MDAnalysis.lib.mdamath import triclinic_vectors
from tqdm import tqdm

from dropps import __version__
from dropps.analysis.image_interaction_core import build_unwrap_plan
from dropps.analysis.polystat_core import (
    RunningMoments,
    calculate_molecular_shape,
    contour_distance_sums,
    end_to_end_distances,
    internal_distance_sums,
    persistence_1e,
    tangent_correlation_sums,
)
from dropps.fileio.filename_control import validate_extension
from dropps.share.argument_parser import ArgumentParser
from dropps.share.command_class import single_command
from dropps.share.trajectory import trajectory_class


prog = "polystat"
desc = "Calculate size, shape, and backbone statistics for many protein copies."

ANGSTROM_PER_NM = 10.0
PS_PER_NS = 1000.0
TARGET_TEMPORARY_BYTES = 64 * 1024**2
LARGE_DETAIL_ROWS = 100_000_000
LARGE_DETAIL_BYTES = 1024**3


@dataclass
class CohortPlan:
    """Equivalent selected molecule copies that can be processed as a batch."""

    selection_label: str
    label: str
    molecule_name: str
    chain_ids: np.ndarray
    selected_indices: np.ndarray
    position_indices: np.ndarray
    weights: np.ndarray
    selected_particles: int
    total_type_copies: int
    residue_range: str
    path_valid: bool
    internal_lags: np.ndarray
    tangent_lags: np.ndarray


def getargs_polystat(argv):
    parser = ArgumentParser(
        prog=prog,
        description=desc,
        epilog=(
            "Select a molecule type or region once across all copies. DROPPS "
            "splits it into topology molecules and keeps different molecule "
            "types in separate cohorts."
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
        help="Compact per-frame cohort statistics (.xvg).",
    )
    parser.add_argument(
        "-i",
        "--internal-distance-output",
        help="Pooled mean-square internal-distance curve (.csv).",
    )
    parser.add_argument(
        "-p",
        "--persistence-output",
        help="Per-frame 1/e persistence length in steps and nm (.xvg).",
    )
    parser.add_argument(
        "--tangent-output",
        help="Pooled backbone tangent-correlation curve (.csv).",
    )
    parser.add_argument(
        "--molecule-output",
        help="Optional long-form per-molecule, per-frame values (.csv).",
    )

    selection_source = parser.add_mutually_exclusive_group(section="input")
    selection_source.add_argument(
        "-sel",
        "--selection-groups",
        type=int,
        nargs="+",
        metavar="GROUP",
        help=(
            "Index groups spanning any number of molecule copies; each group "
            "is one analysis job."
        ),
    )
    selection_source.add_argument(
        "--select",
        "--selection-expression",
        dest="selection_expressions",
        action="append",
        metavar="EXPR",
        help=(
            "DROPPS selection expression; repeat to analyze more than one "
            "molecule type or region."
        ),
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=("ree", "rg", "shape"),
        default=("ree", "rg", "shape"),
        help=(
            "Primary time-series metrics: end-to-end distance, radius of "
            "gyration, and rotation-invariant principal shape."
        ),
    )
    parser.add_argument(
        "--weight",
        choices=("mass", "uniform"),
        default="mass",
        help="Weights used only for gyration centers and tensors.",
    )
    parser.add_argument(
        "--max-lag",
        type=int,
        help="Largest backbone lag; by default use every available lag.",
    )
    parser.add_argument(
        "--lag-step",
        type=int,
        default=1,
        help="Backbone lag spacing; use 2 for the legacy all-trans convention.",
    )
    parser.add_argument(
        "--allow-heterogeneous-cohorts",
        action="store_true",
        help=(
            "Allow one molecule name to have different topology or local "
            "selection signatures and label them as separate cohorts."
        ),
    )
    parser.add_argument(
        "--allow-large-output",
        action="store_true",
        section="parameters",
        help="Allow an estimated per-molecule output larger than safety limits.",
    )
    parser.add_argument(
        "--pbc",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reconstruct complete molecules from topology bonds before analysis.",
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
    if args.max_lag is not None and args.max_lag < 1:
        parser.error("--max-lag must be greater than zero")
    if args.lag_step < 1:
        parser.error("--lag-step must be greater than zero")
    args.metrics = tuple(dict.fromkeys(args.metrics))
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
                "polymer statistics region",
            )
            selections.append(selection)
            names.append(name)
    elif args.selection_expressions:
        for expression in args.selection_expressions:
            selection, _ = trajectory.getSelection(
                expression,
                "polymer statistics region",
            )
            selections.append(selection)
            names.append(expression)
    else:
        trajectory.index.print_all()
        selections, names = trajectory.getSelection_interactive_multiple(
            "polymer statistics regions (one selection may span many molecules)"
        )

    if not selections:
        raise ValueError("At least one polystat selection is required.")
    labels = _unique_labels(names)
    resolved = []
    for label, selection in zip(labels, selections):
        indices = np.asarray(selection.indices, dtype=np.int64)
        if indices.size == 0:
            raise ValueError(f"Polystat selection '{label}' is empty.")
        if np.unique(indices).size != indices.size:
            raise ValueError(f"Polystat selection '{label}' contains duplicates.")
        resolved.append((label, np.sort(indices)))
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
        return str(getattr(itp_list[chain_id], "molecule_name", f"molecule-{chain_id}"))
    names = getattr(trajectory.index, "atom_molnames", None)
    if names is not None:
        return str(names[first_index])
    return f"molecule-{chain_id}"


def _atom_identity(trajectory, global_index):
    index = trajectory.index
    fields = []
    for attribute in ("atom_names", "atom_abbrs", "atom_resids", "atom_resnames"):
        values = getattr(index, attribute, None)
        fields.append(str(values[global_index]) if values is not None else "")
    return tuple(fields)


def _complete_signature(trajectory, full_indices, molecule_bonds, masses):
    global_to_local = {
        int(global_index): local_index
        for local_index, global_index in enumerate(full_indices)
    }
    local_edges = []
    for atom_a, atom_b in molecule_bonds:
        local_a = global_to_local.get(int(atom_a))
        local_b = global_to_local.get(int(atom_b))
        if local_a is None or local_b is None:
            continue
        local_edges.append(tuple(sorted((local_a, local_b))))
    identities = tuple(_atom_identity(trajectory, int(index)) for index in full_indices)
    mass_signature = tuple(float(value).hex() for value in masses[full_indices])
    return (
        int(full_indices.size),
        identities,
        mass_signature,
        tuple(sorted(set(local_edges))),
    )


def _path_is_bonded(selected_indices, bond_pairs):
    return all(
        tuple(sorted((int(atom_a), int(atom_b)))) in bond_pairs
        for atom_a, atom_b in zip(selected_indices[:-1], selected_indices[1:])
    )


def _requested_lags(length, args):
    internal_lags = np.empty(0, dtype=np.int64)
    tangent_lags = np.empty(0, dtype=np.int64)
    if args.internal_distance_output:
        maximum = min(length - 1, args.max_lag or length - 1)
        internal_lags = np.arange(1, maximum + 1, args.lag_step, dtype=np.int64)
        if internal_lags.size == 0:
            raise ValueError("The lag settings leave no internal-distance lag.")
    if args.persistence_output or args.tangent_output:
        maximum = min(length - 2, args.max_lag or length - 2)
        nonzero = np.arange(
            args.lag_step,
            maximum + 1,
            args.lag_step,
            dtype=np.int64,
        )
        if nonzero.size == 0:
            raise ValueError(
                "The lag settings leave no nonzero tangent-correlation lag."
            )
        tangent_lags = np.concatenate((np.asarray([0], dtype=np.int64), nonzero))
    return internal_lags, tangent_lags


def _build_cohorts(trajectory, selections, args):
    atom_chain_ids = np.asarray(trajectory.id2chainID, dtype=np.int64)
    atom_residue_ids = np.asarray(trajectory.id2resID, dtype=np.int64)
    masses = np.asarray(trajectory.Universe.atoms.masses, dtype=np.float64)
    if masses.shape != atom_chain_ids.shape:
        raise ValueError("Topology masses do not match the atom count.")
    global_bonds = _topology_bonds(trajectory)
    bonds_by_chain = defaultdict(list)
    for atom_a, atom_b in global_bonds:
        chain_a = int(atom_chain_ids[int(atom_a)])
        chain_b = int(atom_chain_ids[int(atom_b)])
        if chain_a == chain_b:
            bonds_by_chain[chain_a].append((int(atom_a), int(atom_b)))
    bond_pairs = {
        tuple(sorted((int(atom_a), int(atom_b))))
        for atom_a, atom_b in global_bonds
    }
    molecules = _system_molecules(atom_chain_ids)
    type_totals = Counter(
        _molecule_name(trajectory, chain_id, int(indices[0]))
        for chain_id, indices in molecules.items()
    )

    cohorts = []
    represented_chain_ids = []
    union_selected_indices = []
    for selection_label, selection_indices in selections:
        fragments_by_chain = {}
        for atom_index in selection_indices:
            chain_id = int(atom_chain_ids[int(atom_index)])
            fragments_by_chain.setdefault(chain_id, []).append(int(atom_index))

        fragments = []
        signatures_by_name = defaultdict(set)
        for chain_id, selected_list in fragments_by_chain.items():
            selected = np.asarray(sorted(selected_list), dtype=np.int64)
            full = molecules[chain_id]
            global_to_local = {
                int(global_index): local_index
                for local_index, global_index in enumerate(full)
            }
            local_selection = tuple(global_to_local[int(index)] for index in selected)
            molecule_name = _molecule_name(trajectory, chain_id, int(full[0]))
            complete_signature = _complete_signature(
                trajectory,
                full,
                bonds_by_chain[chain_id],
                masses,
            )
            signature = (complete_signature, local_selection)
            signatures_by_name[molecule_name].add(signature)
            fragments.append(
                {
                    "chain_id": chain_id,
                    "molecule_name": molecule_name,
                    "signature": signature,
                    "selected": selected,
                    "weights": masses[selected],
                    "path_valid": _path_is_bonded(selected, bond_pairs),
                }
            )

        inconsistent = {
            name: len(signatures)
            for name, signatures in signatures_by_name.items()
            if len(signatures) > 1
        }
        if inconsistent and not args.allow_heterogeneous_cohorts:
            details = ", ".join(
                f"{name} ({count} signatures)"
                for name, count in sorted(inconsistent.items())
            )
            raise ValueError(
                f"Selection '{selection_label}' has heterogeneous copies of "
                f"the same molecule name: {details}. Check for missing residues "
                "or add --allow-heterogeneous-cohorts."
            )

        grouped = {}
        for fragment in fragments:
            key = (fragment["molecule_name"], fragment["signature"])
            grouped.setdefault(key, []).append(fragment)
        name_group_counts = Counter(name for name, _ in grouped)
        name_seen = Counter()

        for (molecule_name, _), rows in grouped.items():
            name_seen[molecule_name] += 1
            label = molecule_name
            if name_group_counts[molecule_name] > 1:
                label = f"{molecule_name}#{name_seen[molecule_name]}"
            selected_matrix = np.stack([row["selected"] for row in rows])
            weight_matrix = np.stack([row["weights"] for row in rows])
            chain_ids = np.asarray([row["chain_id"] for row in rows], dtype=np.int64)
            length = selected_matrix.shape[1]
            path_valid = all(row["path_valid"] for row in rows)

            if "ree" in args.metrics and length < 2:
                raise ValueError(
                    f"Cohort '{selection_label}:{label}' needs at least two "
                    "selected particles for end-to-end distance."
                )
            path_outputs = bool(
                args.internal_distance_output
                or args.persistence_output
                or args.tangent_output
            )
            if path_outputs and length < 3 and (
                args.persistence_output or args.tangent_output
            ):
                raise ValueError(
                    f"Cohort '{selection_label}:{label}' needs at least three "
                    "selected particles for tangent correlation."
                )
            if path_outputs and not path_valid:
                raise ValueError(
                    f"Cohort '{selection_label}:{label}' is not a consecutive "
                    "bonded path in topology order. Use a backbone selection or "
                    "request only --metrics ree rg shape."
                )
            if args.weight == "mass" and ({"rg", "shape"} & set(args.metrics)):
                if (
                    not np.all(np.isfinite(weight_matrix))
                    or np.any(weight_matrix < 0.0)
                    or np.any(np.sum(weight_matrix, axis=1) <= 0.0)
                ):
                    raise ValueError(
                        f"Cohort '{selection_label}:{label}' has invalid masses."
                    )

            internal_lags, tangent_lags = _requested_lags(length, args)
            residue_values = atom_residue_ids[selected_matrix.reshape(-1)]
            residue_range = f"{int(np.min(residue_values))}-{int(np.max(residue_values))}"
            cohort = CohortPlan(
                selection_label=selection_label,
                label=label,
                molecule_name=molecule_name,
                chain_ids=chain_ids,
                selected_indices=selected_matrix,
                position_indices=np.empty_like(selected_matrix),
                weights=weight_matrix,
                selected_particles=length,
                total_type_copies=int(type_totals[molecule_name]),
                residue_range=residue_range,
                path_valid=path_valid,
                internal_lags=internal_lags,
                tangent_lags=tangent_lags,
            )
            cohorts.append(cohort)
            represented_chain_ids.extend(int(value) for value in chain_ids)
            union_selected_indices.extend(int(value) for value in selected_matrix.reshape(-1))

    union_selected_indices = np.asarray(
        list(dict.fromkeys(union_selected_indices)),
        dtype=np.int64,
    )
    union_position = {
        int(atom_index): position
        for position, atom_index in enumerate(union_selected_indices)
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
    return cohorts, molecule_atom_indices, union_selected_indices, global_bonds


def _validate_box_dimensions(dimensions):
    if dimensions is None:
        raise ValueError("Trajectory frame does not contain a periodic box.")
    dimensions = np.asarray(dimensions, dtype=np.float64)
    if dimensions.shape != (6,) or not np.all(np.isfinite(dimensions)):
        raise ValueError("Trajectory frame has invalid periodic box dimensions.")
    cell = np.asarray(triclinic_vectors(dimensions, dtype=np.float64))
    if cell.shape != (3, 3) or float(np.linalg.det(cell)) <= 0.0:
        raise ValueError("Trajectory frame has a non-positive periodic box volume.")
    return dimensions


def _frame_selected_positions_nm(timestep, unwrap_plan, selected_indices, pbc):
    if not pbc:
        return np.asarray(timestep.positions[selected_indices], dtype=np.float64) / ANGSTROM_PER_NM
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


def _output_paths(args):
    specifications = (
        ("summary", args.output, "xvg"),
        ("internal", args.internal_distance_output, "csv"),
        ("persistence", args.persistence_output, "xvg"),
        ("tangent", args.tangent_output, "csv"),
        ("molecule", args.molecule_output, "csv"),
    )
    paths = {
        key: Path(validate_extension(value, extension))
        for key, value, extension in specifications
        if value is not None
    }
    resolved = [path.resolve() for path in paths.values()]
    if len(resolved) != len(set(resolved)):
        raise ValueError("Polystat output paths must be distinct.")
    return paths


def _quoted(value):
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def _cohort_prefix(cohort):
    return f"{cohort.selection_label}:{cohort.label}"


def _primary_series(cohorts, metrics):
    series = []
    for cohort in cohorts:
        prefix = _cohort_prefix(cohort)
        if "ree" in metrics:
            series.extend(
                (f"{prefix} Ree RMS", f"{prefix} Ree mean", f"{prefix} Ree std")
            )
        if "rg" in metrics:
            series.extend(
                (f"{prefix} Rg RMS", f"{prefix} Rg mean", f"{prefix} Rg std")
            )
        if "shape" in metrics:
            series.extend(
                (
                    f"{prefix} g1 RMS",
                    f"{prefix} g2 RMS",
                    f"{prefix} g3 RMS",
                    f"{prefix} kappa2 mean",
                )
            )
    return series


def _write_xvg_header(handle, title, ylabel, legends, cohorts, args):
    handle.write("# DROPPS many-protein polymer statistics\n")
    handle.write(
        f"# dropps_version={__version__} run_input={args.run_input!r} "
        f"trajectory={args.input!r}\n"
    )
    handle.write(
        f"# weighting={args.weight} pbc={str(bool(args.pbc)).lower()} "
        f"metrics={','.join(args.metrics)} lag_step={args.lag_step} "
        f"max_lag={args.max_lag if args.max_lag is not None else 'all'}\n"
    )
    for cohort in cohorts:
        handle.write(
            f"# cohort={_cohort_prefix(cohort)!r} copies={cohort.chain_ids.size} "
            f"topology_copies={cohort.total_type_copies} "
            f"selected_particles={cohort.selected_particles} "
            f"residue_range={cohort.residue_range} "
            f"bonded_path={str(cohort.path_valid).lower()}\n"
        )
    handle.write(f'@    title "{_quoted(title)}"\n')
    handle.write('@    xaxis  label "Time (ns)"\n')
    handle.write(f'@    yaxis  label "{_quoted(ylabel)}"\n')
    handle.write("@TYPE xy\n")
    for series_id, legend in enumerate(legends):
        handle.write(f'@ s{series_id} legend "{_quoted(legend)}"\n')


def _detail_fields(metrics):
    fields = [
        "frame",
        "time_ns",
        "selection",
        "cohort",
        "molecule_id",
        "chain_id",
        "molecule_name",
        "selected_particles",
    ]
    if "ree" in metrics:
        fields.append("ree_nm")
    if "rg" in metrics:
        fields.append("rg_nm")
    if "shape" in metrics:
        fields.extend(
            ("lambda1_nm2", "lambda2_nm2", "lambda3_nm2", "kappa2")
        )
    fields.extend(("weighting", "pbc"))
    return fields


def _validate_detail_size(path, frame_count, cohorts, allow_large):
    if path is None:
        return
    rows = frame_count * sum(int(cohort.chain_ids.size) for cohort in cohorts)
    estimated_bytes = rows * 220
    if (
        (rows > LARGE_DETAIL_ROWS or estimated_bytes > LARGE_DETAIL_BYTES)
        and not allow_large
    ):
        raise ValueError(
            f"Per-molecule output is estimated at {rows} rows and "
            f"{estimated_bytes / 1024**3:.2f} GiB; add --allow-large-output "
            "after reviewing the size, or use the compact XVG output."
        )
    if estimated_bytes > LARGE_DETAIL_BYTES:
        print(
            f"## WARNING: Per-molecule output is estimated at {rows} rows "
            f"({estimated_bytes / 1024**3:.2f} GiB)."
        )


def _block_size(selected_particles, lag_analysis):
    factor = 12 if lag_analysis else 4
    bytes_per_molecule = max(1, selected_particles * 3 * 8 * factor)
    return max(1, TARGET_TEMPORARY_BYTES // bytes_per_molecule)


def _new_runtime(cohort, args):
    pooled_names = []
    if "ree" in args.metrics:
        pooled_names.append("ree")
    if "rg" in args.metrics:
        pooled_names.append("rg")
    if "shape" in args.metrics:
        pooled_names.extend(("lambda1", "lambda2", "lambda3", "kappa2"))
    return {
        "pooled": {name: RunningMoments() for name in pooled_names},
        "internal_sum": np.zeros(cohort.internal_lags.size, dtype=np.float64),
        "internal_count": np.zeros(cohort.internal_lags.size, dtype=np.int64),
        "internal_contour_sum": np.zeros(cohort.internal_lags.size, dtype=np.float64),
        "internal_contour_count": np.zeros(cohort.internal_lags.size, dtype=np.int64),
        "tangent_sum": np.zeros(cohort.tangent_lags.size, dtype=np.float64),
        "tangent_count": np.zeros(cohort.tangent_lags.size, dtype=np.int64),
        "tangent_contour_sum": np.zeros(cohort.tangent_lags.size, dtype=np.float64),
        "tangent_contour_count": np.zeros(cohort.tangent_lags.size, dtype=np.int64),
        "tangent_frames": 0,
        "persistence_status": Counter(),
        "persistence_steps": RunningMoments(),
        "persistence_nm": RunningMoments(),
    }


def _update_finite(moments, values):
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size:
        moments.update(finite)


def _format_number(value):
    return f"{float(value):.12g}"


def _write_internal_output(path, cohorts, runtimes):
    fields = (
        "selection",
        "cohort",
        "molecule_name",
        "copies",
        "selected_particles",
        "lag_steps",
        "mean_contour_nm",
        "observations",
        "mean_r2_nm2",
        "rms_distance_nm",
        "r2_per_step_nm2",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for cohort, runtime in zip(cohorts, runtimes):
            for index, lag in enumerate(cohort.internal_lags):
                count = int(runtime["internal_count"][index])
                contour_count = int(runtime["internal_contour_count"][index])
                mean_r2 = runtime["internal_sum"][index] / count
                mean_contour = runtime["internal_contour_sum"][index] / contour_count
                writer.writerow(
                    {
                        "selection": cohort.selection_label,
                        "cohort": cohort.label,
                        "molecule_name": cohort.molecule_name,
                        "copies": int(cohort.chain_ids.size),
                        "selected_particles": cohort.selected_particles,
                        "lag_steps": int(lag),
                        "mean_contour_nm": _format_number(mean_contour),
                        "observations": count,
                        "mean_r2_nm2": _format_number(mean_r2),
                        "rms_distance_nm": _format_number(np.sqrt(mean_r2)),
                        "r2_per_step_nm2": _format_number(mean_r2 / lag),
                    }
                )


def _write_tangent_output(path, cohorts, runtimes):
    fields = (
        "selection",
        "cohort",
        "molecule_name",
        "copies",
        "selected_particles",
        "lag_steps",
        "mean_contour_nm",
        "observations",
        "frames",
        "mean_cosine",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for cohort, runtime in zip(cohorts, runtimes):
            for index, lag in enumerate(cohort.tangent_lags):
                count = int(runtime["tangent_count"][index])
                contour_count = int(runtime["tangent_contour_count"][index])
                mean_cosine = runtime["tangent_sum"][index] / count
                mean_contour = (
                    runtime["tangent_contour_sum"][index] / contour_count
                )
                writer.writerow(
                    {
                        "selection": cohort.selection_label,
                        "cohort": cohort.label,
                        "molecule_name": cohort.molecule_name,
                        "copies": int(cohort.chain_ids.size),
                        "selected_particles": cohort.selected_particles,
                        "lag_steps": int(lag),
                        "mean_contour_nm": _format_number(mean_contour),
                        "observations": count,
                        "frames": runtime["tangent_frames"],
                        "mean_cosine": _format_number(mean_cosine),
                    }
                )


def _print_terminal_summary(cohorts, runtimes, args):
    for cohort, runtime in zip(cohorts, runtimes):
        prefix = _cohort_prefix(cohort)
        print(
            f"## {prefix}: {cohort.chain_ids.size} selected copies of "
            f"{cohort.total_type_copies} topology copies."
        )
        if "ree" in args.metrics:
            value = runtime["pooled"]["ree"].summary()
            print(
                f"##   Ree molecule-frame mean={value.mean:.6g} nm, "
                f"std={value.std:.6g} nm, RMS={value.rms:.6g} nm, n={value.count}."
            )
            print(
                f"##   Ree observed range=[{value.minimum:.6g}, "
                f"{value.maximum:.6g}] nm."
            )
        if "rg" in args.metrics:
            value = runtime["pooled"]["rg"].summary()
            print(
                f"##   Rg molecule-frame mean={value.mean:.6g} nm, "
                f"std={value.std:.6g} nm, RMS={value.rms:.6g} nm, n={value.count}."
            )
            print(
                f"##   Rg observed range=[{value.minimum:.6g}, "
                f"{value.maximum:.6g}] nm."
            )
        if "shape" in args.metrics:
            kappa = runtime["pooled"]["kappa2"].summary()
            print(f"##   mean kappa^2={kappa.mean:.6g}, n={kappa.count}.")
        if args.persistence_output:
            statuses = runtime["persistence_status"]
            valid = runtime["persistence_steps"].summary()
            valid_nm = runtime["persistence_nm"].summary()
            print(
                f"##   persistence valid={statuses['valid']}, "
                f"censored={statuses['censored']}, "
                f"nonpositive={statuses['nonpositive-bracket']}, "
                f"flat={statuses['flat-bracket']}, mean_valid={valid.mean:.6g} "
                f"steps ({valid_nm.mean:.6g} nm)."
            )


def polystat(args):
    trajectory = trajectory_class(args.run_input, args.index, args.input)
    start_frame, end_frame, interval_frame = trajectory.time2frame(
        args.start_time,
        args.end_time,
        args.delta_time,
    )
    frame_indices = range(start_frame, end_frame + 1, interval_frame)
    selections = _resolve_selections(trajectory, args)
    cohorts, molecule_atom_indices, union_selected_indices, bonds = _build_cohorts(
        trajectory,
        selections,
        args,
    )
    if not cohorts:
        raise ValueError("No molecule cohort was produced from the selections.")

    unwrap_plan = None
    if args.pbc:
        unwrap_plan = build_unwrap_plan(
            molecule_atom_indices,
            union_selected_indices,
            bonds,
        )
    paths = _output_paths(args)
    _validate_detail_size(
        paths.get("molecule"),
        len(frame_indices),
        cohorts,
        args.allow_large_output,
    )
    runtimes = [_new_runtime(cohort, args) for cohort in cohorts]
    lag_analysis = bool(
        args.internal_distance_output or args.persistence_output or args.tangent_output
    )

    print(
        f"## Polystat selections: {len(selections)}; cohorts: {len(cohorts)}; "
        f"represented molecule copies: {sum(c.chain_ids.size for c in cohorts)}."
    )
    for cohort in cohorts:
        print(
            f"## Cohort {_cohort_prefix(cohort)}: {cohort.chain_ids.size}/"
            f"{cohort.total_type_copies} copies, {cohort.selected_particles} "
            f"selected particles, residues {cohort.residue_range}."
        )

    with ExitStack() as stack:
        summary_handle = stack.enter_context(
            paths["summary"].open("w", encoding="utf-8")
        )
        _write_xvg_header(
            summary_handle,
            "Many-protein polymer statistics",
            "Distance (nm); kappa2 is dimensionless",
            _primary_series(cohorts, args.metrics),
            cohorts,
            args,
        )

        if "persistence" in paths:
            persistence_handle = stack.enter_context(
                paths["persistence"].open("w", encoding="utf-8")
            )
            persistence_legends = []
            for cohort in cohorts:
                prefix = _cohort_prefix(cohort)
                persistence_legends.extend((f"{prefix} lp steps", f"{prefix} lp nm"))
            _write_xvg_header(
                persistence_handle,
                "Backbone tangent 1/e persistence length",
                "Steps and nm",
                persistence_legends,
                cohorts,
                args,
            )
        else:
            persistence_handle = None

        if "molecule" in paths:
            detail_handle = stack.enter_context(
                paths["molecule"].open("w", encoding="utf-8", newline="")
            )
            detail_writer = csv.DictWriter(
                detail_handle,
                fieldnames=_detail_fields(args.metrics),
            )
            detail_writer.writeheader()
        else:
            detail_writer = None

        analyzed_trajectory = trajectory.Universe.trajectory
        for frame_index in tqdm(frame_indices, desc="## Calculating polymer statistics"):
            timestep = analyzed_trajectory[frame_index]
            time_ns = float(timestep.time) / PS_PER_NS
            selected_positions = _frame_selected_positions_nm(
                timestep,
                unwrap_plan,
                union_selected_indices,
                args.pbc,
            )
            primary_values = []
            persistence_values = []

            for cohort, runtime in zip(cohorts, runtimes):
                frame_moments = {
                    name: RunningMoments() for name in runtime["pooled"]
                }
                frame_tangent_sum = np.zeros(
                    cohort.tangent_lags.size,
                    dtype=np.float64,
                )
                frame_tangent_count = np.zeros(
                    cohort.tangent_lags.size,
                    dtype=np.int64,
                )
                frame_contour_sum = np.zeros(
                    cohort.tangent_lags.size,
                    dtype=np.float64,
                )
                frame_contour_count = np.zeros(
                    cohort.tangent_lags.size,
                    dtype=np.int64,
                )
                block_size = _block_size(cohort.selected_particles, lag_analysis)

                for block_start in range(0, cohort.chain_ids.size, block_size):
                    block_end = min(block_start + block_size, cohort.chain_ids.size)
                    position_rows = cohort.position_indices[block_start:block_end]
                    coordinates = selected_positions[position_rows]
                    weights = cohort.weights[block_start:block_end]
                    shape_result = None
                    if {"rg", "shape"} & set(args.metrics):
                        shape_result = calculate_molecular_shape(
                            coordinates,
                            weights if args.weight == "mass" else None,
                        )

                    if "ree" in args.metrics:
                        ree = (
                            shape_result.end_to_end
                            if shape_result is not None
                            else end_to_end_distances(coordinates)
                        )
                        frame_moments["ree"].update(ree)
                        runtime["pooled"]["ree"].update(ree)
                    else:
                        ree = None
                    if "rg" in args.metrics:
                        rg = shape_result.radius_gyration
                        frame_moments["rg"].update(rg)
                        runtime["pooled"]["rg"].update(rg)
                    else:
                        rg = None
                    if "shape" in args.metrics:
                        for eigen_index in range(3):
                            name = f"lambda{eigen_index + 1}"
                            values = shape_result.eigenvalues[:, eigen_index]
                            frame_moments[name].update(values)
                            runtime["pooled"][name].update(values)
                        kappa2 = shape_result.relative_shape_anisotropy
                        _update_finite(frame_moments["kappa2"], kappa2)
                        _update_finite(runtime["pooled"]["kappa2"], kappa2)
                    else:
                        kappa2 = None

                    if detail_writer is not None:
                        for local_row, chain_id in enumerate(
                            cohort.chain_ids[block_start:block_end]
                        ):
                            row = {
                                "frame": frame_index,
                                "time_ns": _format_number(time_ns),
                                "selection": cohort.selection_label,
                                "cohort": cohort.label,
                                "molecule_id": block_start + local_row,
                                "chain_id": int(chain_id),
                                "molecule_name": cohort.molecule_name,
                                "selected_particles": cohort.selected_particles,
                                "weighting": args.weight,
                                "pbc": str(bool(args.pbc)).lower(),
                            }
                            if ree is not None:
                                row["ree_nm"] = _format_number(ree[local_row])
                            if rg is not None:
                                row["rg_nm"] = _format_number(rg[local_row])
                            if "shape" in args.metrics:
                                for eigen_index in range(3):
                                    row[f"lambda{eigen_index + 1}_nm2"] = _format_number(
                                        shape_result.eigenvalues[local_row, eigen_index]
                                    )
                                row["kappa2"] = _format_number(kappa2[local_row])
                            detail_writer.writerow(row)

                    if cohort.internal_lags.size:
                        values, counts = internal_distance_sums(
                            coordinates,
                            cohort.internal_lags,
                        )
                        contour_values, contour_counts = contour_distance_sums(
                            coordinates,
                            cohort.internal_lags,
                        )
                        runtime["internal_sum"] += values
                        runtime["internal_count"] += counts
                        runtime["internal_contour_sum"] += contour_values
                        runtime["internal_contour_count"] += contour_counts

                    if cohort.tangent_lags.size:
                        values, counts = tangent_correlation_sums(
                            coordinates,
                            cohort.tangent_lags,
                        )
                        contour_values, contour_counts = contour_distance_sums(
                            coordinates,
                            cohort.tangent_lags,
                            tangent_midpoints=True,
                        )
                        frame_tangent_sum += values
                        frame_tangent_count += counts
                        frame_contour_sum += contour_values
                        frame_contour_count += contour_counts
                        if args.tangent_output:
                            runtime["tangent_sum"] += values
                            runtime["tangent_count"] += counts
                            runtime["tangent_contour_sum"] += contour_values
                            runtime["tangent_contour_count"] += contour_counts

                if "ree" in args.metrics:
                    summary = frame_moments["ree"].summary()
                    primary_values.extend((summary.rms, summary.mean, summary.std))
                if "rg" in args.metrics:
                    summary = frame_moments["rg"].summary()
                    primary_values.extend((summary.rms, summary.mean, summary.std))
                if "shape" in args.metrics:
                    primary_values.extend(
                        np.sqrt(
                            [
                                frame_moments[f"lambda{index}"].mean
                                for index in range(1, 4)
                            ]
                        )
                    )
                    primary_values.append(frame_moments["kappa2"].summary().mean)

                if cohort.tangent_lags.size:
                    correlations = frame_tangent_sum / frame_tangent_count
                    mean_contours = frame_contour_sum / frame_contour_count
                    if args.tangent_output:
                        runtime["tangent_frames"] += 1
                    if args.persistence_output:
                        persistence = persistence_1e(
                            correlations,
                            cohort.tangent_lags,
                            mean_contours,
                        )
                        runtime["persistence_status"][persistence.status] += 1
                        if persistence.status == "valid":
                            runtime["persistence_steps"].update([persistence.steps])
                            runtime["persistence_nm"].update(
                                [persistence.contour_length]
                            )
                        persistence_values.extend(
                            (persistence.steps, persistence.contour_length)
                        )

            summary_handle.write(
                "\t".join(
                    [_format_number(time_ns)]
                    + [_format_number(value) for value in primary_values]
                )
                + "\n"
            )
            if persistence_handle is not None:
                persistence_handle.write(
                    "\t".join(
                        [_format_number(time_ns)]
                        + [_format_number(value) for value in persistence_values]
                    )
                    + "\n"
                )

    if "internal" in paths:
        _write_internal_output(paths["internal"], cohorts, runtimes)
    if "tangent" in paths:
        _write_tangent_output(paths["tangent"], cohorts, runtimes)
    _print_terminal_summary(cohorts, runtimes, args)
    print(f"## Wrote compact polystat time series to {paths['summary']}.")
    for key in ("internal", "persistence", "tangent", "molecule"):
        if key in paths:
            print(f"## Wrote {key} output to {paths[key]}.")


polystat_commands = single_command(prog, getargs_polystat, polystat, desc)
