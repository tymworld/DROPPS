"""Radial distribution functions for atom and molecular position selections."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from MDAnalysis.exceptions import NoDataError
from MDAnalysis.lib.distances import minimize_vectors
from tqdm import tqdm

from dropps import __version__
from dropps.analysis.density_core import select_frame_indices
from dropps.analysis.distance_core import (
    build_endpoint_plan,
    project_endpoint,
    validate_dimensions,
)
from dropps.analysis.image_interaction_core import build_unwrap_plan
from dropps.analysis.rdf_core import (
    DEFAULT_RDF_BIN_WIDTH_NM,
    RdfAccumulator,
    RdfPositionSet,
    box_measure,
    build_radial_grid,
    cell_vectors,
    frame_pair_histogram,
    frame_surface_histogram,
    safe_periodic_radius,
    topology_exclusion_keys,
)
from dropps.fileio.filename_control import validate_extension
from dropps.share.argument_parser import ArgumentParser
from dropps.share.command_class import single_command
from dropps.share.topology_exclusions import build_exclusion_pairs
from dropps.share.trajectory import trajectory_class


prog = "rdf"
desc = "Calculate radial distribution functions and cumulative coordination numbers."

ANGSTROM_PER_NM = 10.0


def getargs_rdf(argv):
    parser = ArgumentParser(
        prog=prog,
        description=desc,
        epilog=(
            "One reference selection is compared with every repeated target "
            "selection. Atom selections can be retained as positions or "
            "grouped into topology-aware COG/COM positions."
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
        help="Output radial-distribution curves (.xvg).",
    )
    parser.add_argument(
        "-cn",
        "--output-cumulative",
        help="Optional cumulative coordination-number curves (.xvg).",
    )
    parser.add_argument(
        "-ref",
        "--reference",
        metavar="SELECTION",
        section="input",
        help="Reference DROPPS selection expression or index-group number.",
    )
    parser.add_argument(
        "-sel",
        "--selection",
        dest="selections",
        action="append",
        metavar="SELECTION",
        section="input",
        help="Target DROPPS selection; repeat to write multiple RDF curves.",
    )
    parser.add_argument(
        "--position",
        choices=("atom", "cog", "com"),
        default="atom",
        help="Common position construction for reference and targets.",
    )
    parser.add_argument(
        "--ref-position",
        choices=("atom", "cog", "com"),
        help="Reference position construction overriding --position.",
    )
    parser.add_argument(
        "--sel-position",
        choices=("atom", "cog", "com"),
        help="Target position construction overriding --position.",
    )
    parser.add_argument(
        "--grouping",
        choices=("all", "chain", "molecule", "residue", "atom"),
        default="all",
        help="Common grouping applied before COG/COM position construction.",
    )
    parser.add_argument(
        "--ref-grouping",
        choices=("all", "chain", "molecule", "residue", "atom"),
        help="Reference grouping overriding --grouping.",
    )
    parser.add_argument(
        "--sel-grouping",
        choices=("all", "chain", "molecule", "residue", "atom"),
        help="Target grouping overriding --grouping.",
    )
    parser.add_argument(
        "--geometry",
        choices=("3d", "xy"),
        default="3d",
        help="Use spherical 3D distances or distances projected into x-y.",
    )
    parser.add_argument(
        "--pbc",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use periodic minimum-image distances.",
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
        "--bin-width",
        type=float,
        default=DEFAULT_RDF_BIN_WIDTH_NM,
        help="Radial bin width in nm.",
    )
    parser.add_argument(
        "--cut",
        type=float,
        default=0.0,
        help="Strict lower distance cutoff in nm.",
    )
    parser.add_argument(
        "--rmax",
        type=float,
        help="Requested maximum RDF radius in nm; otherwise choose a safe value.",
    )
    parser.add_argument(
        "--exclude",
        choices=("none", "topology", "same-molecule"),
        default="none",
        help="Candidate-pair exclusion policy; normalization remains unchanged.",
    )
    parser.add_argument(
        "--surface",
        choices=("none", "chain", "molecule", "residue"),
        default="none",
        help="Use closest selected reference atom per chain or residue.",
    )
    parser.add_argument(
        "--norm",
        dest="normalization",
        choices=("rdf", "number-density", "number_density", "none"),
        help="Normalization. Default: rdf; surface mode defaults to none.",
    )
    args = parser.parse_args(argv)

    for option in ("bin_width", "cut", "rmax"):
        value = getattr(args, option)
        if value is not None and not np.isfinite(value):
            parser.error(f"--{option.replace('_', '-')} must be finite")
    if args.bin_width <= 0.0:
        parser.error("--bin-width must be greater than zero")
    if args.cut < 0.0:
        parser.error("--cut must be non-negative")
    if args.rmax is not None and args.rmax <= 0.0:
        parser.error("--rmax must be greater than zero")
    if args.rmax is not None and args.cut >= args.rmax:
        parser.error("--cut must be smaller than --rmax")
    if (args.reference is None) != (args.selections is None):
        parser.error("provide both -ref and at least one -sel, or omit both for interactive mode")

    args.surface = "chain" if args.surface == "molecule" else args.surface
    args.grouping = _normalize_grouping(args.grouping)
    args.ref_grouping = _normalize_grouping(args.ref_grouping)
    args.sel_grouping = _normalize_grouping(args.sel_grouping)
    if args.normalization == "number_density":
        args.normalization = "number-density"

    reference_position = args.ref_position or args.position
    target_position = args.sel_position or args.position
    if args.surface != "none":
        if reference_position != "atom":
            parser.error("--surface requires atom reference positions")
        if args.ref_grouping is not None or args.grouping != "all":
            parser.error("--surface defines reference grouping; use --sel-grouping for targets")
        if args.exclude == "topology":
            parser.error("--surface cannot be combined with --exclude topology")
        if args.normalization not in (None, "none"):
            parser.error("--surface only supports --norm none")
        args.normalization = "none"
    elif args.normalization is None:
        args.normalization = "rdf"

    if args.exclude == "topology" and (
        reference_position != "atom" or target_position != "atom"
    ):
        parser.error("--exclude topology requires atom positions on both sides")
    return args


def _normalize_grouping(value):
    return "chain" if value == "molecule" else value


def _unique_labels(labels):
    seen = defaultdict(int)
    result = []
    for label in labels:
        base = str(label)
        seen[base] += 1
        result.append(base if seen[base] == 1 else f"{base}#{seen[base]}")
    return result


def _resolve_selections(trajectory, args):
    if args.reference is None:
        trajectory.index.print_all()
        reference, reference_label = trajectory.getSelection_interactive(
            "RDF reference"
        )
        targets, target_labels = trajectory.getSelection_interactive_multiple(
            "RDF targets"
        )
    else:
        reference, _ = trajectory.getSelection(args.reference, "RDF reference")
        reference_label = args.reference
        targets = []
        target_labels = []
        for specification in args.selections:
            target, _ = trajectory.getSelection(specification, "RDF target")
            targets.append(target)
            target_labels.append(specification)

    if len(reference) == 0:
        raise ValueError("RDF reference selection is empty.")
    if not targets:
        raise ValueError("At least one RDF target selection is required.")
    if any(len(target) == 0 for target in targets):
        raise ValueError("RDF target selections must not be empty.")
    return reference, str(reference_label), targets, _unique_labels(target_labels)


def _endpoint_position_count(endpoint):
    if endpoint.position_type == "atom":
        return sum(entity.atom_indices.size for entity in endpoint.entities)
    return len(endpoint.entities)


def _build_endpoint(trajectory, selection, label, grouping, position_type):
    indices = np.asarray(selection.indices, dtype=np.int64)
    return build_endpoint_plan(
        indices,
        label,
        grouping,
        position_type,
        np.asarray(trajectory.id2chainID, dtype=np.int64),
        np.asarray(trajectory.id2resID, dtype=np.int64),
    )


def _topology_bonds(trajectory):
    try:
        return np.asarray(trajectory.Universe.bonds.indices, dtype=np.int64)
    except NoDataError:
        return np.empty((0, 2), dtype=np.int64)


def _build_center_unwrap_plan(trajectory, endpoints, pbc):
    center_endpoints = [
        endpoint for endpoint in endpoints if endpoint.position_type != "atom"
    ]
    if not pbc or not center_endpoints:
        return None
    selected = np.unique(
        np.concatenate(
            [
                entity.atom_indices
                for endpoint in center_endpoints
                for entity in endpoint.entities
            ]
        )
    )
    chain_ids = np.asarray(trajectory.id2chainID, dtype=np.int64)
    represented_chains = list(
        dict.fromkeys(int(chain_ids[index]) for index in selected)
    )
    molecule_groups = [
        np.flatnonzero(chain_ids == chain_id).astype(np.int64)
        for chain_id in represented_chains
    ]
    return build_unwrap_plan(
        molecule_groups,
        selected,
        _topology_bonds(trajectory),
    )


def _frame_positions_nm(timestep, unwrap_plan, pbc):
    raw_nm = np.asarray(timestep.positions, dtype=np.float64) / ANGSTROM_PER_NM
    if not pbc or unwrap_plan is None:
        return raw_nm
    dimensions = validate_dimensions(timestep.dimensions)
    raw = np.asarray(
        timestep.positions[unwrap_plan.atom_indices],
        dtype=np.float64,
    )
    unwrapped = np.empty_like(raw)
    unwrapped[unwrap_plan.roots] = raw[unwrap_plan.roots]
    for parents, children in unwrap_plan.levels:
        bond_vectors = raw[children] - raw[parents]
        unwrapped[children] = (
            unwrapped[parents] + minimize_vectors(bond_vectors, dimensions)
        )
    selected_global = unwrap_plan.atom_indices[unwrap_plan.selected_local_indices]
    raw_nm[selected_global] = (
        unwrapped[unwrap_plan.selected_local_indices] / ANGSTROM_PER_NM
    )
    return raw_nm


def _dimensions_nm(timestep):
    dimensions = np.asarray(timestep.dimensions, dtype=np.float64)
    if dimensions.shape != (6,) or not np.all(np.isfinite(dimensions)):
        raise ValueError("Trajectory frame does not contain a finite periodic box.")
    dimensions = dimensions.copy()
    dimensions[:3] /= ANGSTROM_PER_NM
    cell_vectors(dimensions)
    return dimensions


def _project_position_set(endpoint, positions_nm, masses, dimensions_nm, pbc):
    projected = project_endpoint(
        endpoint,
        positions_nm,
        masses,
        dimensions_nm,
        pbc,
    )
    coordinates = []
    atom_indices = []
    memberships = []
    for entity, entity_positions in zip(endpoint.entities, projected):
        coordinates.append(np.asarray(entity_positions, dtype=np.float64))
        if endpoint.position_type == "atom":
            atom_indices.extend(int(index) for index in entity.atom_indices)
            memberships.extend(
                frozenset((int(chain_id),)) for chain_id in entity.atom_chain_ids
            )
        else:
            atom_indices.append(-1)
            memberships.append(frozenset(int(value) for value in entity.chain_ids))
    return RdfPositionSet(
        np.vstack(coordinates),
        np.asarray(atom_indices, dtype=np.int64),
        tuple(memberships),
    )


def _surface_reference_position_set(indices, positions_nm, chain_ids):
    indices = np.asarray(indices, dtype=np.int64)
    return RdfPositionSet(
        positions_nm[indices],
        indices,
        tuple(frozenset((int(chain_ids[index]),)) for index in indices),
    )


def _surface_groups(indices, chain_ids, residue_ids, surface):
    mapping = {}
    group_ids = []
    for atom_index in np.asarray(indices, dtype=np.int64):
        if surface == "chain":
            key = int(chain_ids[atom_index])
        else:
            key = (int(chain_ids[atom_index]), int(residue_ids[atom_index]))
        if key not in mapping:
            mapping[key] = len(mapping)
        group_ids.append(mapping[key])
    if not mapping:
        raise ValueError("Surface RDF reference contains no chain/residue groups.")
    return np.asarray(group_ids, dtype=np.int64), len(mapping)


def _radius_prepass(frames, frame_indices, args):
    minimum_safe = np.inf
    limiting_frame = None
    limiting_time_ns = None
    largest_edge = 0.0
    largest_edge_frame = None
    selected_times_ns = []

    for frame_index in frame_indices:
        timestep = frames[frame_index]
        dimensions_nm = _dimensions_nm(timestep)
        time_ns = float(timestep.time) / 1000.0
        selected_times_ns.append(time_ns)
        if args.pbc:
            safe = safe_periodic_radius(dimensions_nm, args.geometry)
            if safe < minimum_safe:
                minimum_safe = safe
                limiting_frame = int(frame_index)
                limiting_time_ns = time_ns
        else:
            edge = float(np.max(np.linalg.norm(cell_vectors(dimensions_nm), axis=1)))
            if edge > largest_edge:
                largest_edge = edge
                largest_edge_frame = int(frame_index)
                limiting_time_ns = time_ns

    if args.pbc:
        if args.rmax is not None:
            tolerance = max(1.0e-12, minimum_safe * 1.0e-12)
            if args.rmax > minimum_safe + tolerance:
                raise ValueError(
                    f"Requested --rmax {args.rmax:.8g} nm exceeds the safe "
                    f"periodic radius {minimum_safe:.8g} nm at frame "
                    f"{limiting_frame} (time {limiting_time_ns:.8g} ns)."
                )
            requested_limit = float(args.rmax)
            source = "explicit"
        else:
            requested_limit = 0.99 * minimum_safe
            source = "automatic-pbc"
    else:
        requested_limit = (
            float(args.rmax) if args.rmax is not None else 3.0 * largest_edge
        )
        source = "explicit" if args.rmax is not None else "automatic-no-pbc"
        limiting_frame = largest_edge_frame

    grid = build_radial_grid(args.bin_width, requested_limit, args.geometry)
    if args.cut >= grid.actual_rmax:
        raise ValueError(
            f"--cut {args.cut:.8g} nm must be below actual-rmax "
            f"{grid.actual_rmax:.8g} nm after bin alignment."
        )
    return grid, {
        "source": source,
        "safe_radius_nm": minimum_safe if args.pbc else None,
        "limiting_frame": limiting_frame,
        "limiting_time_ns": limiting_time_ns,
        "selected_times_ns": selected_times_ns,
    }


def _output_paths(args):
    paths = {
        "rdf": Path(validate_extension(args.output, "xvg")),
    }
    if args.output_cumulative is not None:
        paths["cumulative"] = Path(
            validate_extension(args.output_cumulative, "xvg")
        )
    resolved = [path.resolve() for path in paths.values()]
    if len(resolved) != len(set(resolved)):
        raise ValueError("RDF output paths must be distinct.")
    return paths


def _quoted(value):
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def _write_xvg(path, x, series, title, xlabel, ylabel, legends, metadata):
    values = np.asarray(series, dtype=np.float64)
    if values.ndim == 1:
        values = values[np.newaxis, :]
    if values.shape != (len(legends), len(x)):
        raise ValueError("RDF output series do not match their grid/legends.")
    with path.open("w", encoding="utf-8") as handle:
        for line in metadata:
            handle.write(f"# {line}\n")
        handle.write(f'@    title "{_quoted(title)}"\n')
        handle.write(f'@    xaxis  label "{_quoted(xlabel)}"\n')
        handle.write(f'@    yaxis  label "{_quoted(ylabel)}"\n')
        handle.write("@TYPE xy\n")
        for index, legend in enumerate(legends):
            handle.write(f'@ s{index} legend "{_quoted(legend)}"\n')
        for column in range(len(x)):
            row = [float(x[column]), *(float(item[column]) for item in values)]
            handle.write("\t".join(f"{value:.8g}" for value in row) + "\n")


def _metadata_lines(
    args,
    reference_label,
    target_labels,
    reference_position,
    reference_grouping,
    target_position,
    target_grouping,
    frame_indices,
    radius_info,
    grid,
    accumulator,
):
    times = radius_info["selected_times_ns"]
    lines = [
        f"command=dps rdf version={__version__}",
        f'reference="{_quoted(reference_label)}" position={reference_position} grouping={reference_grouping}',
        (
            f"geometry={args.geometry} pbc={str(args.pbc).lower()} "
            f"surface={args.surface} exclusion={args.exclude} "
            f"normalization={args.normalization} exclusion_denominator=unchanged"
        ),
        (
            f"frames={len(frame_indices)} first_frame={frame_indices[0]} "
            f"last_frame={frame_indices[-1]} start_time_ns={times[0]:.8g} "
            f"end_time_ns={times[-1]:.8g}"
        ),
        (
            f"bin_width_nm={grid.bin_width:.8g} cut_nm={args.cut:.8g} "
            f"requested_rmax_nm={grid.requested_limit:.8g} "
            f"actual_rmax_nm={grid.actual_rmax:.8g} "
            f"rmax_source={radius_info['source']}"
        ),
        (
            f"limiting_frame={radius_info['limiting_frame']} "
            f"limiting_time_ns={radius_info['limiting_time_ns']:.8g}"
        ),
        (
            f"mean_reference_count={accumulator.mean_reference_count:.8g} "
            f"mean_box_measure={accumulator.mean_measure:.8g}"
        ),
    ]
    if radius_info["safe_radius_nm"] is not None:
        lines.append(
            f"minimum_safe_periodic_radius_nm={radius_info['safe_radius_nm']:.8g}"
        )
    for index, (label, count, density) in enumerate(
        zip(
            target_labels,
            accumulator.mean_target_counts,
            accumulator.mean_target_densities,
        )
    ):
        lines.append(
            f'target_{index}="{_quoted(label)}" position={target_position} '
            f"grouping={target_grouping} mean_count={count:.8g} "
            f"mean_density={density:.8g}"
        )
    return lines


def rdf(args):
    output_paths = _output_paths(args)
    trajectory = trajectory_class(args.run_input, args.index, args.input)
    reference, reference_label, targets, target_labels = _resolve_selections(
        trajectory,
        args,
    )

    reference_position = args.ref_position or args.position
    target_position = args.sel_position or args.position
    reference_grouping = args.ref_grouping or args.grouping
    target_grouping = args.sel_grouping or args.grouping
    target_endpoints = [
        _build_endpoint(
            trajectory,
            target,
            label,
            target_grouping,
            target_position,
        )
        for target, label in zip(targets, target_labels)
    ]
    reference_endpoint = None
    if args.surface == "none":
        reference_endpoint = _build_endpoint(
            trajectory,
            reference,
            reference_label,
            reference_grouping,
            reference_position,
        )
        reference_count = _endpoint_position_count(reference_endpoint)
        endpoints = [reference_endpoint, *target_endpoints]
    else:
        reference_count = None
        endpoints = list(target_endpoints)

    chain_ids = np.asarray(trajectory.id2chainID, dtype=np.int64)
    residue_ids = np.asarray(trajectory.id2resID, dtype=np.int64)
    surface_group_ids = None
    if args.surface != "none":
        surface_group_ids, reference_count = _surface_groups(
            reference.indices,
            chain_ids,
            residue_ids,
            args.surface,
        )

    target_counts = [_endpoint_position_count(endpoint) for endpoint in target_endpoints]
    unwrap_plan = _build_center_unwrap_plan(trajectory, endpoints, args.pbc)
    if unwrap_plan is not None:
        print("## PBC-aware COG/COM positions will use bonded molecule reconstruction.")

    frames = trajectory.Universe.trajectory
    frame_indices, _ = select_frame_indices(
        frames,
        args.start_time,
        args.end_time,
        args.delta_time,
    )
    grid, radius_info = _radius_prepass(frames, frame_indices, args)
    if not args.pbc and args.normalization == "rdf":
        print(
            "WARNING: --no-pbc RDF normalization retains the box density, but "
            "finite-box boundary effects can prevent a large-r plateau at one."
        )

    topology_keys = None
    if args.exclude == "topology":
        exclusion_pairs = build_exclusion_pairs(trajectory.tpr.itp_list)
        topology_keys = topology_exclusion_keys(
            exclusion_pairs,
            trajectory.num_atoms(),
        )

    workload = int(reference_count) * int(sum(target_counts)) * len(frame_indices)
    print(
        f"## RDF plan: {reference_count} reference position/entity count; "
        f"target position counts {target_counts}; {grid.bin_count} bins; "
        f"actual-rmax {grid.actual_rmax:.8g} nm."
    )
    print(f"## Conservative candidate workload before cutoff: {workload} pairs.")

    masses = np.asarray(trajectory.id2masses, dtype=np.float64)
    accumulator = RdfAccumulator(len(target_endpoints), grid)
    reference_indices = np.asarray(reference.indices, dtype=np.int64)
    for frame_index in tqdm(frame_indices, desc="Calculating RDF"):
        timestep = frames[frame_index]
        dimensions_nm = _dimensions_nm(timestep)
        positions_nm = _frame_positions_nm(timestep, unwrap_plan, args.pbc)
        target_position_sets = [
            _project_position_set(
                endpoint,
                positions_nm,
                masses,
                dimensions_nm,
                args.pbc,
            )
            for endpoint in target_endpoints
        ]

        histograms = []
        if args.surface == "none":
            reference_position_set = _project_position_set(
                reference_endpoint,
                positions_nm,
                masses,
                dimensions_nm,
                args.pbc,
            )
            for target_position_set in target_position_sets:
                histograms.append(
                    frame_pair_histogram(
                        reference_position_set,
                        target_position_set,
                        grid,
                        dimensions_nm,
                        pbc=args.pbc,
                        geometry=args.geometry,
                        cut=args.cut,
                        exclusion=args.exclude,
                        topology_keys=topology_keys,
                        atom_count=trajectory.num_atoms(),
                    )
                )
            frame_reference_count = reference_position_set.count
        else:
            reference_atoms = _surface_reference_position_set(
                reference_indices,
                positions_nm,
                chain_ids,
            )
            for target_position_set in target_position_sets:
                histograms.append(
                    frame_surface_histogram(
                        reference_atoms,
                        surface_group_ids,
                        target_position_set,
                        grid,
                        dimensions_nm,
                        pbc=args.pbc,
                        geometry=args.geometry,
                        cut=args.cut,
                        exclusion=args.exclude,
                    )
                )
            frame_reference_count = int(reference_count)

        accumulator.add_frame(
            np.vstack(histograms),
            frame_reference_count,
            [position_set.count for position_set in target_position_sets],
            box_measure(dimensions_nm, args.geometry),
        )

    metadata = _metadata_lines(
        args,
        reference_label,
        target_labels,
        reference_position,
        reference_grouping if args.surface == "none" else args.surface,
        target_position,
        target_grouping,
        frame_indices,
        radius_info,
        grid,
        accumulator,
    )
    if args.normalization == "rdf":
        ylabel = "Radial distribution g(r)"
    elif args.normalization == "number-density":
        power = "-3" if args.geometry == "3d" else "-2"
        ylabel = f"Number density (nm^{power})"
    else:
        ylabel = "Pair count density (nm^-1)"

    _write_xvg(
        output_paths["rdf"],
        grid.radii,
        accumulator.normalized(args.normalization),
        "Radial distribution",
        "r (nm)",
        ylabel,
        target_labels,
        metadata,
    )
    print(f"## Wrote RDF curves to {output_paths['rdf']}.")
    if "cumulative" in output_paths:
        _write_xvg(
            output_paths["cumulative"],
            grid.cumulative_radii,
            accumulator.cumulative(),
            "Cumulative coordination number",
            "r (nm)",
            "Cumulative number",
            target_labels,
            metadata,
        )
        print(f"## Wrote cumulative RDF curves to {output_paths['cumulative']}.")


rdf_commands = single_command(prog, getargs_rdf, rdf, desc)
