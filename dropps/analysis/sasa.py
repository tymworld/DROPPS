"""Coarse-grained, PBC-aware solvent-accessible surface area analysis."""

from __future__ import annotations

import sys
from collections import defaultdict
from itertools import product
from pathlib import Path

import numpy as np
from MDAnalysis.lib.distances import minimize_vectors
from MDAnalysis.lib.mdamath import triclinic_vectors
from MDAnalysis.lib.nsgrid import FastNS
from tqdm import tqdm

from dropps.analysis.sasa_core import (
    RADIUS_CONVENTION_FACTORS,
    fibonacci_sphere_points,
    forcefield_radii_nm,
    neighbor_vectors_from_pairs,
    orthorhombic_periodic_neighbor_vectors,
    shrake_rupley_areas,
)
from dropps.fileio.filename_control import validate_extension
from dropps.share.argument_parser import ArgumentParser
from dropps.share.command_class import single_command
from dropps.share.trajectory import trajectory_class


prog = "sasa"
desc = "Calculate coarse-grained solvent-accessible surface area."

ANGSTROM_PER_NM = 10.0
PS_PER_NS = 1000.0


def getargs_sasa(argv):
    parser = ArgumentParser(
        prog=prog,
        description=desc,
        epilog=(
            "The surface selection defines all beads that can occlude one "
            "another. Each output selection must be a subset of that surface; "
            "its area is extracted without recalculating an isolated surface."
        ),
    )
    parser.add_argument(
        "-s",
        "--run-input",
        required=True,
        help="Input DROPPS run file (.tpr) containing topology and bead parameters.",
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
        help="Output surface-area time series (.xvg).",
    )
    parser.add_argument(
        "-surface",
        "--surface",
        type=str,
        metavar="SELECTION",
        section="parameters",
        help=(
            "One DROPPS selection expression or index-group number defining the "
            "calculated surface. Default: choose interactively in a terminal, "
            "with an empty response meaning all atoms; otherwise all atoms."
        ),
    )
    parser.add_argument(
        "-output",
        "--output-selection",
        dest="output_selections",
        type=str,
        nargs="+",
        action="append",
        metavar="SELECTION",
        section="parameters",
        help=(
            "Additional surface subsets reported as separate series; accepts "
            "one or more quoted selections and may be repeated. Default: choose "
            "zero or more groups interactively in a terminal; otherwise none."
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
        "--probe-radius",
        type=float,
        default=0.14,
        help="Solvent-probe radius in nm.",
    )
    parser.add_argument(
        "--sphere-points",
        type=int,
        default=960,
        help="Number of deterministic sampling points on every expanded bead sphere.",
    )
    parser.add_argument(
        "--radius-convention",
        choices=tuple(RADIUS_CONVENTION_FACTORS),
        default="sigma-half",
        help=(
            "Mapping from the force-field sigma to a geometric bead radius: "
            "sigma-half uses sigma/2 and lj-minimum-half uses "
            "2^(1/6)*sigma/2."
        ),
    )
    parser.add_argument(
        "--radius-scale",
        type=float,
        default=1.0,
        help="Positive multiplier applied to all resolved bead radii.",
    )
    parser.add_argument(
        "--fixed-radius",
        type=float,
        help=(
            "Use this radius in nm for every bead instead of force-field sigma "
            "values; --radius-scale is still applied."
        ),
    )
    args = parser.parse_args(argv)
    if not np.isfinite(args.probe_radius) or args.probe_radius < 0.0:
        parser.error("--probe-radius must be finite and non-negative")
    if args.sphere_points < 4 or args.sphere_points % 2:
        parser.error("--sphere-points must be an even integer of at least 4")
    if not np.isfinite(args.radius_scale) or args.radius_scale <= 0.0:
        parser.error("--radius-scale must be positive and finite")
    if args.fixed_radius is not None and (
        not np.isfinite(args.fixed_radius) or args.fixed_radius <= 0.0
    ):
        parser.error("--fixed-radius must be positive and finite")
    return args


def _all_atoms_selection(trajectory):
    return trajectory.Universe.atoms, "System"


def _selection_from_spec(trajectory, specification, purpose):
    specification = str(specification).strip()
    if specification.lower() in {"all", "all atoms"}:
        selection, name = _all_atoms_selection(trajectory)
        print(f"## Selected {len(selection)} atoms for {purpose}.\n")
        return selection, name
    return trajectory.getSelection(specification, purpose)


def _flatten_output_specs(nested_specs):
    if nested_specs is None:
        return []
    return [specification for group in nested_specs for specification in group]


def _interactive_enabled():
    return bool(getattr(sys.stdin, "isatty", lambda: False)())


def _resolve_selections(trajectory, args, *, interactive=None):
    """Resolve the calculated surface and zero or more reported subsets."""

    if interactive is None:
        interactive = _interactive_enabled()
    groups_printed = False

    if args.surface is not None:
        surface, surface_name = _selection_from_spec(
            trajectory,
            args.surface,
            "SASA surface calculation",
        )
    elif interactive:
        trajectory.index.print_all()
        groups_printed = True
        try:
            specification = input(
                "> Surface selection [all atoms]: "
            ).strip()
        except EOFError:
            specification = ""
        if specification:
            surface, surface_name = _selection_from_spec(
                trajectory,
                specification,
                "SASA surface calculation",
            )
        else:
            surface, surface_name = _all_atoms_selection(trajectory)
            print(
                f"## Selected all {len(surface)} atoms for SASA surface calculation.\n"
            )
    else:
        surface, surface_name = _all_atoms_selection(trajectory)
        print(
            f"## Selected all {len(surface)} atoms for SASA surface calculation.\n"
        )

    output_specs = _flatten_output_specs(args.output_selections)
    if output_specs:
        outputs = [
            _selection_from_spec(
                trajectory,
                specification,
                "additional SASA output",
            )
            for specification in output_specs
        ]
    elif interactive:
        if not groups_printed:
            trajectory.index.print_all()
        selections, names = trajectory.getSelection_interactive_multiple(
            "additional SASA outputs (each must be a surface subset)"
        )
        outputs = list(zip(selections, names))
    else:
        outputs = []

    surface_indices = np.asarray(surface.indices, dtype=np.int64)
    if surface_indices.ndim != 1 or surface_indices.size == 0:
        raise ValueError("The SASA surface selection is empty.")
    if np.unique(surface_indices).size != surface_indices.size:
        raise ValueError("The SASA surface selection contains duplicate atoms.")

    surface_lookup = np.full(trajectory.num_atoms(), -1, dtype=np.int64)
    surface_lookup[surface_indices] = np.arange(surface_indices.size, dtype=np.int64)
    resolved_outputs = []
    for output, name in outputs:
        output_indices = np.asarray(output.indices, dtype=np.int64)
        if output_indices.ndim != 1 or output_indices.size == 0:
            raise ValueError(f"SASA output selection {name!r} is empty.")
        if np.unique(output_indices).size != output_indices.size:
            raise ValueError(
                f"SASA output selection {name!r} contains duplicate atoms."
            )
        local_indices = surface_lookup[output_indices]
        outside = output_indices[local_indices < 0]
        if outside.size:
            preview = ", ".join(str(int(value)) for value in outside[:8])
            suffix = "..." if outside.size > 8 else ""
            raise ValueError(
                f"SASA output selection {name!r} is not a subset of surface "
                f"{surface_name!r}; outside atom indices: {preview}{suffix}."
            )
        resolved_outputs.append(
            {
                "name": str(name),
                "global_indices": output_indices,
                "surface_indices": local_indices,
            }
        )
    return surface, str(surface_name), resolved_outputs


def _unique_labels(labels):
    counts = defaultdict(int)
    unique = []
    for label in labels:
        label = str(label)
        counts[label] += 1
        unique.append(label if counts[label] == 1 else f"{label}#{counts[label]}")
    return unique


def _box_geometry_nm(dimensions):
    values = np.asarray(dimensions, dtype=np.float64)
    if values.shape != (6,) or not np.all(np.isfinite(values)):
        raise ValueError("A SASA trajectory frame has invalid periodic box dimensions.")
    values = values.copy()
    values[:3] /= ANGSTROM_PER_NM
    if np.any(values[:3] <= 0.0):
        raise ValueError("A SASA trajectory frame has non-positive box lengths.")
    cell = np.asarray(triclinic_vectors(values, dtype=np.float64), dtype=np.float64)
    if cell.shape != (3, 3) or float(np.linalg.det(cell)) <= 0.0:
        raise ValueError("A SASA trajectory frame has a non-positive box volume.")
    return values, cell


def _shortest_lattice_vector(cell):
    translations = np.asarray(
        [value for value in product(range(-2, 3), repeat=3) if value != (0, 0, 0)],
        dtype=np.float64,
    )
    vectors = translations @ cell
    return float(np.sqrt(np.min(np.einsum("ij,ij->i", vectors, vectors))))


def _wrap_triclinic(positions, cell):
    inverse = np.linalg.inv(cell)
    fractions = np.asarray(positions, dtype=np.float64) @ inverse
    return (fractions - np.floor(fractions)) @ cell


def _triclinic_periodic_neighbor_vectors(
    positions_nm,
    expanded_radii_nm,
    dimensions_nm,
    cell_nm,
):
    maximum_cutoff = 2.0 * float(np.max(expanded_radii_nm))
    half_shortest = 0.5 * _shortest_lattice_vector(cell_nm)
    if maximum_cutoff >= half_shortest:
        raise ValueError(
            f"The maximum SASA overlap cutoff ({maximum_cutoff:.6g} nm) must "
            "be less than half the shortest periodic lattice vector "
            f"({half_shortest:.6g} nm)."
        )

    wrapped = _wrap_triclinic(positions_nm, cell_nm)
    pairs = FastNS(
        maximum_cutoff,
        np.asarray(wrapped, dtype=np.float32),
        box=np.asarray(dimensions_nm, dtype=np.float32),
        pbc=True,
    ).self_search().get_pairs()
    pairs = np.asarray(pairs, dtype=np.int64)
    if pairs.size == 0:
        pairs = np.empty((0, 2), dtype=np.int64)
        vectors = np.empty((0, 3), dtype=np.float64)
    else:
        pairs = np.unique(np.sort(pairs.reshape(-1, 2), axis=1), axis=0)
        raw_vectors = wrapped[pairs[:, 1]] - wrapped[pairs[:, 0]]
        vectors = np.asarray(
            minimize_vectors(raw_vectors, dimensions_nm),
            dtype=np.float64,
        )
    return neighbor_vectors_from_pairs(
        positions_nm.shape[0],
        expanded_radii_nm,
        pairs,
        vectors,
    )


def _periodic_neighbor_vectors(positions_nm, expanded_radii_nm, dimensions):
    dimensions_nm, cell_nm = _box_geometry_nm(dimensions)
    if np.allclose(dimensions_nm[3:], 90.0, rtol=0.0, atol=1.0e-5):
        return orthorhombic_periodic_neighbor_vectors(
            positions_nm,
            expanded_radii_nm,
            dimensions_nm[:3],
        )
    return _triclinic_periodic_neighbor_vectors(
        positions_nm,
        expanded_radii_nm,
        dimensions_nm,
        cell_nm,
    )


def _resolved_radii(trajectory, args):
    atom_count = trajectory.num_atoms()
    if args.fixed_radius is not None:
        radii = np.full(
            atom_count,
            float(args.fixed_radius) * float(args.radius_scale),
            dtype=np.float64,
        )
        description = "fixed"
    else:
        radii = forcefield_radii_nm(
            trajectory.tpr.itp_list,
            convention=args.radius_convention,
            scale=args.radius_scale,
        )
        description = args.radius_convention
    if radii.shape != (atom_count,):
        raise ValueError(
            f"Resolved {radii.size} SASA radii for a topology with "
            f"{atom_count} atoms."
        )
    return radii, description


def _quoted_label(label):
    return str(label).replace("\\", "\\\\").replace('"', '\\"')


def _write_sasa_xvg(path, times_ns, values_nm2, labels, metadata):
    path = Path(path)
    values = np.asarray(values_nm2, dtype=np.float64)
    times = np.asarray(times_ns, dtype=np.float64)
    if values.ndim != 2 or values.shape != (times.size, len(labels)):
        raise ValueError("SASA output data do not match the time axis and labels.")
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# DROPPS coarse-grained solvent-accessible surface area\n")
        handle.write(
            "# "
            + " ".join(f"{key}={value}" for key, value in metadata.items())
            + "\n"
        )
        for label, series in zip(labels, values.T):
            handle.write(
                f"# statistics label={_quoted_label(label)} "
                f"mean_nm2={float(np.mean(series)):.12g} "
                f"std_nm2={float(np.std(series)):.12g}\n"
            )
        handle.write('@    title "Solvent-accessible surface area"\n')
        handle.write('@    xaxis  label "Time (ns)"\n')
        handle.write('@    yaxis  label "SASA (nm^2)"\n')
        handle.write("@TYPE xy\n")
        for series_id, label in enumerate(labels):
            handle.write(
                f'@ s{series_id} legend "{_quoted_label(label)}"\n'
            )
        for time_ns, row in zip(times, values):
            handle.write(
                f"{time_ns:.12g}\t"
                + "\t".join(f"{float(value):.12g}" for value in row)
                + "\n"
            )


def sasa(args):
    trajectory = trajectory_class(args.run_input, args.index, args.input)
    start_frame, end_frame, interval_frame = trajectory.time2frame(
        args.start_time,
        args.end_time,
        args.delta_time,
    )
    frame_indices = list(range(start_frame, end_frame + 1, interval_frame))
    surface, surface_name, output_groups = _resolve_selections(trajectory, args)
    surface_global_indices = np.asarray(surface.indices, dtype=np.int64)

    all_radii_nm, radius_description = _resolved_radii(trajectory, args)
    surface_radii_nm = all_radii_nm[surface_global_indices]
    expanded_radii_nm = surface_radii_nm + float(args.probe_radius)
    sphere_points = fibonacci_sphere_points(args.sphere_points)

    labels = _unique_labels(
        [f"Surface:{surface_name}", *[group["name"] for group in output_groups]]
    )
    times_ns = []
    rows = []
    print(
        f"## SASA surface {surface_name!r}: {len(surface)} beads; "
        f"additional outputs: {len(output_groups)}."
    )
    print(
        f"## Radius convention: {radius_description}; scale: "
        f"{args.radius_scale:.6g}; probe: {args.probe_radius:.6g} nm; "
        f"sphere points: {args.sphere_points}."
    )
    analyzed_trajectory = trajectory.Universe.trajectory
    for frame_index in tqdm(frame_indices, desc="## Calculating SASA"):
        timestep = analyzed_trajectory[frame_index]
        positions_nm = np.asarray(surface.positions, dtype=np.float64) / ANGSTROM_PER_NM
        neighbor_indices, neighbor_vectors = _periodic_neighbor_vectors(
            positions_nm,
            expanded_radii_nm,
            timestep.dimensions,
        )
        atom_areas_nm2 = shrake_rupley_areas(
            expanded_radii_nm,
            neighbor_indices,
            neighbor_vectors,
            sphere_points,
        )
        row = [float(np.sum(atom_areas_nm2))]
        row.extend(
            float(np.sum(atom_areas_nm2[group["surface_indices"]]))
            for group in output_groups
        )
        times_ns.append(float(timestep.time) / PS_PER_NS)
        rows.append(row)

    output_path = validate_extension(args.output, "xvg")
    metadata = {
        "surface": _quoted_label(surface_name),
        "surface_atoms": len(surface),
        "output_groups": len(output_groups),
        "radius_convention": radius_description,
        "radius_scale": f"{args.radius_scale:.12g}",
        "probe_radius_nm": f"{args.probe_radius:.12g}",
        "sphere_points": args.sphere_points,
        "pbc": "minimum-image-per-frame",
    }
    if args.fixed_radius is not None:
        metadata["fixed_radius_nm"] = f"{args.fixed_radius:.12g}"
    _write_sasa_xvg(output_path, times_ns, rows, labels, metadata)
    print(f"## Wrote SASA time series to {output_path}.")


sasa_commands = single_command(prog, getargs_sasa, sasa, desc)
