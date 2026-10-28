"""Unified fixed-pair and grouped distance trajectory analysis."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from contextlib import ExitStack
from pathlib import Path

import numpy as np
from MDAnalysis.exceptions import NoDataError
from MDAnalysis.lib.distances import minimize_vectors
from tqdm import tqdm

from dropps.analysis.distance_core import (
    ANGSTROM_PER_NM,
    ChannelStatistics,
    HistogramAccumulator,
    RunningStatistics,
    build_distance_job,
    build_endpoint_plan,
    dimensions_angstrom_to_nm,
    iter_channel_descriptors,
    iter_distance_blocks,
    project_endpoint,
    validate_dimensions,
)
from dropps.analysis.image_interaction_core import build_unwrap_plan
from dropps.fileio.filename_control import validate_extension
from dropps.share.argument_parser import ArgumentParser
from dropps.share.command_class import single_command
from dropps.share.trajectory import trajectory_class


prog = "distance"
desc = "Calculate fixed, intra-chain, inter-chain, minimum, and maximum distances."

PS_PER_NS = 1000.0
DEFAULT_BIN_WIDTH_NM = 0.01
DEFAULT_ENTITY_BLOCK_SIZE = 4096
LARGE_CHANNEL_THRESHOLD = 100_000
LARGE_OUTPUT_BYTES = 1024**3


def getargs_distance(argv):
    parser = ArgumentParser(
        prog=prog,
        description=desc,
        epilog=(
            "Select each region once across all chain copies. --scope controls "
            "which chains are related; --mode controls how positions inside "
            "each admitted entity pair are compared."
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
        "-ref",
        "--reference",
        dest="references",
        action="append",
        metavar="SELECTION",
        section="parameters",
        help=(
            "Reference DROPPS selection expression or index-group number; "
            "repeat to define multiple jobs."
        ),
    )
    parser.add_argument(
        "-sel",
        "--selection",
        dest="selections",
        action="append",
        metavar="SELECTION",
        section="parameters",
        help=(
            "Selection-side DROPPS expression or index-group number; repeat "
            "to define multiple jobs."
        ),
    )
    parser.add_argument(
        "--label",
        action="append",
        metavar="LABEL",
        help="Optional output label repeated once per resulting distance job.",
    )
    parser.add_argument(
        "-m",
        "--mode",
        choices=("pair", "min", "max"),
        required=True,
        help=(
            "Distance within each entity pair: zip corresponding positions, "
            "or choose the minimum/maximum Cartesian position-pair norm."
        ),
    )
    parser.add_argument(
        "--scope",
        choices=("corresponding", "intra-chain", "inter-chain", "all"),
        default="corresponding",
        help="Entity relationship used before the distance mode is applied.",
    )
    parser.add_argument(
        "--unique-entity-pairs",
        action="store_true",
        help=(
            "Keep only one direction of a symmetric all/inter-chain entity "
            "product; endpoints must be identical."
        ),
    )
    parser.add_argument(
        "--position",
        choices=("atom", "cog", "com"),
        default="atom",
        help="Common position construction for both endpoint sides.",
    )
    parser.add_argument(
        "--ref-position",
        choices=("atom", "cog", "com"),
        help="Reference-side position construction overriding --position.",
    )
    parser.add_argument(
        "--sel-position",
        choices=("atom", "cog", "com"),
        help="Selection-side position construction overriding --position.",
    )
    parser.add_argument(
        "--grouping",
        choices=("all", "chain", "molecule", "residue", "atom"),
        help=(
            "Common endpoint grouping. Default: chain for intra/inter-chain "
            "scope, otherwise all. 'molecule' is an alias for chain."
        ),
    )
    parser.add_argument(
        "--ref-grouping",
        choices=("all", "chain", "molecule", "residue", "atom"),
        help="Reference-side grouping overriding --grouping.",
    )
    parser.add_argument(
        "--sel-grouping",
        choices=("all", "chain", "molecule", "residue", "atom"),
        help="Selection-side grouping overriding --grouping.",
    )
    parser.add_argument(
        "--exclude",
        choices=("none", "same-atom", "same-molecule"),
        default="none",
        help="Candidate position pairs to remove before output/reduction.",
    )
    parser.add_argument(
        "--pbc",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use minimum-image periodic distances and PBC-aware centers.",
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
        "-oall",
        "-o",
        "--output-all",
        help="Output every scalar distance channel as a time series (.xvg).",
    )
    parser.add_argument(
        "-oav",
        "--output-average",
        help="Output per-frame channel mean and population std per job (.xvg).",
    )
    parser.add_argument(
        "-oxyz",
        "--output-components",
        help="Output dx, dy, and dz for every distance channel (.xvg).",
    )
    parser.add_argument(
        "-ostat",
        "--output-statistics",
        help="Output pooled and/or per-channel time statistics (.csv).",
    )
    parser.add_argument(
        "-oh",
        "--output-histogram",
        help="Output pooled or per-channel distance histograms (.xvg).",
    )
    parser.add_argument(
        "--statistics-scope",
        choices=("pooled", "channel", "both"),
        default="pooled",
        help="Granularity written by --output-statistics.",
    )
    parser.add_argument(
        "--hist-scope",
        choices=("job", "channel"),
        default="job",
        help="Pool histogram samples per job or retain every channel.",
    )
    parser.add_argument(
        "--hist-normalization",
        choices=("probability", "count", "density"),
        default="probability",
        help="Histogram y-axis normalization.",
    )
    parser.add_argument(
        "--bin-width",
        type=float,
        default=DEFAULT_BIN_WIDTH_NM,
        help="Histogram bin width in nm.",
    )
    parser.add_argument(
        "--hist-min",
        type=float,
        help="Optional histogram lower edge in nm.",
    )
    parser.add_argument(
        "--hist-max",
        type=float,
        help="Optional histogram upper edge in nm.",
    )
    parser.add_argument(
        "--allow-large-output",
        action="store_true",
        section="parameters",
        help="Allow detailed outputs containing more than 100000 channels.",
    )
    args = parser.parse_args(argv)
    if not np.isfinite(args.bin_width) or args.bin_width <= 0.0:
        parser.error("--bin-width must be positive and finite")
    for name in ("hist_min", "hist_max"):
        value = getattr(args, name)
        if value is not None and not np.isfinite(value):
            parser.error(f"--{name.replace('_', '-')} must be finite")
    if (
        args.hist_min is not None
        and args.hist_max is not None
        and args.hist_max <= args.hist_min
    ):
        parser.error("--hist-max must be greater than --hist-min")
    return args


def _normalize_grouping(value):
    return "chain" if value == "molecule" else value


def _simple_index_group(trajectory, specification):
    text = str(specification).strip()
    if text.isdigit():
        group_id = int(text)
    else:
        match = re.fullmatch(r"group\s+(.+)", text, flags=re.IGNORECASE)
        if match is None:
            return None
        token = match.group(1).strip()
        if token.isdigit():
            group_id = int(token)
        else:
            normalized = token.replace(" ", "_").lower()
            matches = [
                index
                for index, group in enumerate(trajectory.index.index_groups)
                if group.name.replace(" ", "_").lower() == normalized
            ]
            if len(matches) != 1:
                return None
            group_id = matches[0]
    if not 0 <= group_id < len(trajectory.index.index_groups):
        raise ValueError(f"Unknown index group {group_id}.")
    group = trajectory.index.index_groups[group_id]
    return np.asarray(group.indices, dtype=np.int64), group.name


def _resolve_selection(trajectory, specification, purpose):
    simple = _simple_index_group(trajectory, specification)
    if simple is not None:
        indices, name = simple
    else:
        text = str(specification).strip()
        indices = np.asarray(trajectory.index.phraseSelection(text), dtype=np.int64)
        name = text
    if indices.size == 0:
        raise ValueError(f"Selection '{specification}' for {purpose} is empty.")
    print(f"## Selected {indices.size} atoms for {purpose}: {name}.")
    return indices, str(name)


def _interactive_specs(trajectory):
    trajectory.index.print_all()
    try:
        reference = input("> Reference selection: ").strip()
    except EOFError as exc:
        raise ValueError("Distance selections are required in non-interactive use.") from exc
    if not reference:
        raise ValueError("A reference distance selection is required.")
    selections = []
    print("> Enter one or more selection-side expressions; blank input finishes.")
    while True:
        try:
            selection = input("> Selection: ").strip()
        except EOFError:
            selection = ""
        if not selection:
            break
        selections.append(selection)
    if not selections:
        raise ValueError("At least one selection-side distance selection is required.")
    return [reference], selections


def _broadcast_job_specs(references, selections):
    if len(references) == len(selections):
        return list(zip(references, selections))
    if len(references) == 1:
        return [(references[0], selection) for selection in selections]
    if len(selections) == 1:
        return [(reference, selections[0]) for reference in references]
    raise ValueError(
        "Reference and selection counts must match, or one side must contain "
        "exactly one selection for broadcasting."
    )


def _unique_labels(labels):
    counts = {}
    result = []
    for label in labels:
        base = str(label)
        counts[base] = counts.get(base, 0) + 1
        result.append(base if counts[base] == 1 else f"{base}#{counts[base]}")
    return result


def _build_jobs(trajectory, args):
    references = args.references
    selections = args.selections
    if references is None and selections is None:
        references, selections = _interactive_specs(trajectory)
    elif references is None or selections is None:
        raise ValueError("Both --reference and --selection must be supplied.")
    specs = _broadcast_job_specs(references, selections)
    if args.label is not None and len(args.label) != len(specs):
        raise ValueError("--label must be repeated once per resulting distance job.")

    default_grouping = (
        "chain" if args.scope in {"intra-chain", "inter-chain"} else "all"
    )
    common_grouping = _normalize_grouping(args.grouping) or default_grouping
    ref_grouping = _normalize_grouping(args.ref_grouping) or common_grouping
    sel_grouping = _normalize_grouping(args.sel_grouping) or common_grouping
    ref_position = args.ref_position or args.position
    sel_position = args.sel_position or args.position

    atom_chain_ids = np.asarray(trajectory.id2chainID, dtype=np.int64)
    atom_residue_ids = np.asarray(trajectory.id2resID, dtype=np.int64)
    resolved = []
    automatic_labels = []
    for job_index, (ref_spec, sel_spec) in enumerate(specs):
        ref_indices, ref_name = _resolve_selection(
            trajectory,
            ref_spec,
            f"distance job {job_index + 1} reference",
        )
        sel_indices, sel_name = _resolve_selection(
            trajectory,
            sel_spec,
            f"distance job {job_index + 1} selection",
        )
        automatic_labels.append(f"{ref_name}->{sel_name}")
        resolved.append(
            (
                build_endpoint_plan(
                    ref_indices,
                    ref_name,
                    ref_grouping,
                    ref_position,
                    atom_chain_ids,
                    atom_residue_ids,
                ),
                build_endpoint_plan(
                    sel_indices,
                    sel_name,
                    sel_grouping,
                    sel_position,
                    atom_chain_ids,
                    atom_residue_ids,
                ),
            )
        )
    labels = _unique_labels(args.label or automatic_labels)
    jobs = [
        build_distance_job(
            label,
            reference,
            selection,
            args.scope,
            args.mode,
            args.exclude,
            args.unique_entity_pairs,
        )
        for label, (reference, selection) in zip(labels, resolved)
    ]
    return jobs


def _topology_bonds(trajectory):
    try:
        return np.asarray(trajectory.Universe.bonds.indices, dtype=np.int64)
    except NoDataError:
        return np.empty((0, 2), dtype=np.int64)


def _build_center_unwrap_plan(trajectory, jobs, pbc):
    if not pbc:
        return None
    endpoints = [endpoint for job in jobs for endpoint in (job.reference, job.selection)]
    center_endpoints = [endpoint for endpoint in endpoints if endpoint.position_type != "atom"]
    if not center_endpoints:
        return None
    selected = np.unique(
        np.concatenate(
            [entity.atom_indices for endpoint in center_endpoints for entity in endpoint.entities]
        )
    )
    atom_chain_ids = np.asarray(trajectory.id2chainID, dtype=np.int64)
    represented_chains = list(dict.fromkeys(int(atom_chain_ids[index]) for index in selected))
    molecule_groups = [
        np.flatnonzero(atom_chain_ids == chain_id).astype(np.int64)
        for chain_id in represented_chains
    ]
    return build_unwrap_plan(molecule_groups, selected, _topology_bonds(trajectory))


def _frame_positions_nm(timestep, unwrap_plan, pbc):
    raw_nm = np.asarray(timestep.positions, dtype=np.float64) / ANGSTROM_PER_NM
    if not pbc or unwrap_plan is None:
        return raw_nm
    dimensions = validate_dimensions(timestep.dimensions)
    raw = np.asarray(timestep.positions[unwrap_plan.atom_indices], dtype=np.float64)
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


def _output_paths(args):
    specifications = (
        ("all", args.output_all, "xvg"),
        ("average", args.output_average, "xvg"),
        ("components", args.output_components, "xvg"),
        ("statistics", args.output_statistics, "csv"),
        ("histogram", args.output_histogram, "xvg"),
    )
    paths = {
        key: Path(validate_extension(value, extension))
        for key, value, extension in specifications
        if value is not None
    }
    if not paths:
        raise ValueError(
            "At least one distance output is required: -oall, -oav, -oxyz, "
            "-ostat, or -oh."
        )
    resolved = [path.resolve() for path in paths.values()]
    if len(resolved) != len(set(resolved)):
        raise ValueError("Distance output paths must be distinct.")
    return paths


def _quoted(value):
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def _metadata_lines(jobs, args):
    lines = [
        "# DROPPS unified distance analysis",
        (
            f"# mode={args.mode} scope={args.scope} exclusion={args.exclude} "
            f"pbc={str(bool(args.pbc)).lower()}"
        ),
    ]
    for index, job in enumerate(jobs):
        lines.append(
            f"# job[{index}]={job.label!r} "
            f"reference_position={job.reference.position_type} "
            f"selection_position={job.selection.position_type} "
            f"entity_pairs={job.entity_pair_count} channels={job.channel_count}"
        )
    return lines


def _write_xvg_header(handle, title, ylabel, legends, metadata):
    for line in metadata:
        handle.write(line + "\n")
    handle.write(f'@    title "{_quoted(title)}"\n')
    handle.write('@    xaxis  label "Time (ns)"\n')
    handle.write(f'@    yaxis  label "{_quoted(ylabel)}"\n')
    handle.write("@TYPE xy\n")
    for index, legend in enumerate(legends):
        handle.write(f'@ s{index} legend "{_quoted(legend)}"\n')


def _needs_channel_descriptors(args):
    return bool(
        args.output_all
        or args.output_components
        or (
            args.output_statistics
            and args.statistics_scope in {"channel", "both"}
        )
        or (args.output_histogram and args.hist_scope == "channel")
    )


def _validate_large_outputs(args, jobs, frame_count):
    channel_count = sum(job.channel_count for job in jobs)
    detailed = _needs_channel_descriptors(args)
    if (
        detailed
        and channel_count > LARGE_CHANNEL_THRESHOLD
        and not args.allow_large_output
    ):
        raise ValueError(
            f"Detailed distance output contains {channel_count} channels; add "
            "--allow-large-output after reviewing the output size, or request "
            "pooled statistics/histograms instead."
        )
    scalar_columns = channel_count if args.output_all else 0
    component_columns = channel_count * 3 if args.output_components else 0
    estimated_bytes = frame_count * (scalar_columns + component_columns) * 14
    if estimated_bytes > LARGE_OUTPUT_BYTES:
        print(
            f"## WARNING: Estimated detailed distance time-series size is "
            f"{estimated_bytes / 1024**3:.2f} GiB."
        )
    return channel_count


def _histogram_edges(args, pooled_statistics):
    observed_min = min(stat.minimum for stat in pooled_statistics)
    observed_max = max(stat.maximum for stat in pooled_statistics)
    lower = (
        float(args.hist_min)
        if args.hist_min is not None
        else np.floor(observed_min / args.bin_width) * args.bin_width
    )
    upper = (
        float(args.hist_max)
        if args.hist_max is not None
        else np.ceil(observed_max / args.bin_width) * args.bin_width
    )
    if upper <= lower:
        upper = lower + args.bin_width
    bin_count = max(1, int(np.ceil((upper - lower) / args.bin_width)))
    return lower + np.arange(bin_count + 1, dtype=np.float64) * args.bin_width


def _write_statistics(
    path,
    jobs,
    pooled,
    channel_stats,
    descriptors,
    scope,
    args,
):
    fields = (
        "scope",
        "job",
        "mode",
        "entity_scope",
        "reference_position",
        "selection_position",
        "exclusion",
        "pbc",
        "channel",
        "reference_entity",
        "selection_entity",
        "samples",
        "mean_nm",
        "std_nm",
        "min_nm",
        "max_nm",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        if scope in {"pooled", "both"}:
            for job, statistics in zip(jobs, pooled):
                common = {
                    "job": job.label,
                    "mode": job.mode,
                    "entity_scope": job.scope,
                    "reference_position": job.reference.position_type,
                    "selection_position": job.selection.position_type,
                    "exclusion": job.exclusion,
                    "pbc": str(bool(args.pbc)).lower(),
                }
                writer.writerow(
                    {
                        "scope": "pooled",
                        **common,
                        "channel": "",
                        "reference_entity": "",
                        "selection_entity": "",
                        "samples": statistics.count,
                        "mean_nm": f"{statistics.mean:.12g}",
                        "std_nm": f"{statistics.std:.12g}",
                        "min_nm": f"{statistics.minimum:.12g}",
                        "max_nm": f"{statistics.maximum:.12g}",
                    }
                )
        if scope in {"channel", "both"}:
            for job_index, (job, statistics, job_descriptors) in enumerate(
                zip(jobs, channel_stats, descriptors)
            ):
                std = statistics.std
                for descriptor in job_descriptors:
                    channel = descriptor.index
                    common = {
                        "job": job.label,
                        "mode": job.mode,
                        "entity_scope": job.scope,
                        "reference_position": job.reference.position_type,
                        "selection_position": job.selection.position_type,
                        "exclusion": job.exclusion,
                        "pbc": str(bool(args.pbc)).lower(),
                    }
                    writer.writerow(
                        {
                            "scope": "channel",
                            **common,
                            "channel": descriptor.label,
                            "reference_entity": descriptor.reference_entity,
                            "selection_entity": descriptor.selection_entity,
                            "samples": int(statistics.count[channel]),
                            "mean_nm": f"{statistics.mean[channel]:.12g}",
                            "std_nm": f"{std[channel]:.12g}",
                            "min_nm": f"{statistics.minimum[channel]:.12g}",
                            "max_nm": f"{statistics.maximum[channel]:.12g}",
                        }
                    )


def _evaluate_histogram(
    trajectory,
    frame_indices,
    jobs,
    masses,
    unwrap_plan,
    args,
    edges,
):
    if args.hist_scope == "job":
        histogram = HistogramAccumulator(edges, len(jobs))
        channel_offsets = None
    else:
        offsets = np.cumsum([0, *[job.channel_count for job in jobs]])
        histogram = HistogramAccumulator(edges, int(offsets[-1]))
        channel_offsets = offsets[:-1]

    analyzed = trajectory.Universe.trajectory
    for frame_index in tqdm(frame_indices, desc="## Building distance histogram"):
        timestep = analyzed[frame_index]
        positions = _frame_positions_nm(timestep, unwrap_plan, args.pbc)
        dimensions = (
            dimensions_angstrom_to_nm(timestep.dimensions) if args.pbc else None
        )
        for job_index, job in enumerate(jobs):
            reference = project_endpoint(
                job.reference, positions, masses, dimensions, args.pbc
            )
            selection = project_endpoint(
                job.selection, positions, masses, dimensions, args.pbc
            )
            for block in iter_distance_blocks(
                job, reference, selection, dimensions, args.pbc
            ):
                if args.hist_scope == "job":
                    histogram.update_series(job_index, block.values)
                else:
                    histogram.update_channels(
                        int(channel_offsets[job_index]) + block.start,
                        block.values,
                    )
    return histogram


def _write_histogram_xvg(path, centers, values, legends, ylabel, metadata, args):
    values = np.asarray(values, dtype=np.float64)
    with path.open("w", encoding="utf-8") as handle:
        for line in metadata:
            handle.write(line + "\n")
        handle.write(
            f"# hist_scope={args.hist_scope} "
            f"normalization={args.hist_normalization} "
            f"bin_width_nm={args.bin_width:.12g}\n"
        )
        handle.write('@    title "Distance histogram"\n')
        handle.write('@    xaxis  label "Distance (nm)"\n')
        handle.write(f'@    yaxis  label "{_quoted(ylabel)}"\n')
        handle.write("@TYPE xy\n")
        for index, legend in enumerate(legends):
            handle.write(f'@ s{index} legend "{_quoted(legend)}"\n')
        for bin_index, center in enumerate(centers):
            fields = [f"{float(center):.12g}"]
            fields.extend(f"{float(series[bin_index]):.12g}" for series in values)
            handle.write("\t".join(fields) + "\n")


def distance(args):
    paths = _output_paths(args)
    trajectory = trajectory_class(args.run_input, args.index, args.input)
    jobs = _build_jobs(trajectory, args)
    start_frame, end_frame, interval_frame = trajectory.time2frame(
        args.start_time,
        args.end_time,
        args.delta_time,
    )
    frame_indices = list(range(start_frame, end_frame + 1, interval_frame))
    total_channels = _validate_large_outputs(args, jobs, len(frame_indices))
    for job in jobs:
        print(
            f"## Distance job '{job.label}': {job.entity_pair_count} entity "
            f"pairs, {job.channel_count} channels."
        )
    print(
        f"## Total distance jobs: {len(jobs)}; channels: {total_channels}; "
        f"frames: {len(frame_indices)}."
    )

    unwrap_plan = _build_center_unwrap_plan(trajectory, jobs, args.pbc)
    if unwrap_plan is not None:
        print("## PBC-aware COG/COM positions will reconstruct each chain once per frame.")
    elif args.pbc:
        print("## Atomic displacement vectors will use minimum-image PBC.")
    else:
        print("## WARNING: Distance analysis will not use periodic boundaries.")

    descriptors = None
    if _needs_channel_descriptors(args):
        descriptors = [list(iter_channel_descriptors(job)) for job in jobs]

    metadata = _metadata_lines(jobs, args)
    pooled_statistics = [RunningStatistics() for _ in jobs]
    need_channel_stats = bool(
        args.output_statistics
        and args.statistics_scope in {"channel", "both"}
    )
    channel_statistics = (
        [ChannelStatistics(job.channel_count) for job in jobs]
        if need_channel_stats
        else None
    )
    masses = np.asarray(trajectory.id2masses, dtype=np.float64)

    with ExitStack() as stack:
        all_handle = (
            stack.enter_context(paths["all"].open("w", encoding="utf-8"))
            if "all" in paths
            else None
        )
        average_handle = (
            stack.enter_context(paths["average"].open("w", encoding="utf-8"))
            if "average" in paths
            else None
        )
        component_handle = (
            stack.enter_context(paths["components"].open("w", encoding="utf-8"))
            if "components" in paths
            else None
        )
        if all_handle is not None:
            _write_xvg_header(
                all_handle,
                "Distance",
                "Distance (nm)",
                [descriptor.label for group in descriptors for descriptor in group],
                metadata,
            )
        if average_handle is not None:
            _write_xvg_header(
                average_handle,
                "Distance ensemble summary",
                "Distance (nm)",
                [
                    label
                    for job in jobs
                    for label in (f"{job.label} mean", f"{job.label} std")
                ],
                metadata,
            )
        if component_handle is not None:
            _write_xvg_header(
                component_handle,
                "Distance components",
                "Displacement (nm)",
                [
                    f"{descriptor.label} {axis}"
                    for group in descriptors
                    for descriptor in group
                    for axis in ("dx", "dy", "dz")
                ],
                metadata,
            )

        analyzed = trajectory.Universe.trajectory
        for frame_index in tqdm(frame_indices, desc="## Calculating distances"):
            timestep = analyzed[frame_index]
            time_ns = float(timestep.time) / PS_PER_NS
            positions = _frame_positions_nm(timestep, unwrap_plan, args.pbc)
            dimensions = (
                dimensions_angstrom_to_nm(timestep.dimensions) if args.pbc else None
            )
            if all_handle is not None:
                all_handle.write(f"{time_ns:.12g}")
            if component_handle is not None:
                component_handle.write(f"{time_ns:.12g}")
            average_values = []
            for job_index, job in enumerate(jobs):
                reference = project_endpoint(
                    job.reference, positions, masses, dimensions, args.pbc
                )
                selection = project_endpoint(
                    job.selection, positions, masses, dimensions, args.pbc
                )
                frame_statistics = RunningStatistics()
                for block in iter_distance_blocks(
                    job,
                    reference,
                    selection,
                    dimensions,
                    args.pbc,
                    DEFAULT_ENTITY_BLOCK_SIZE,
                ):
                    pooled_statistics[job_index].update(block.values)
                    if channel_statistics is not None:
                        channel_statistics[job_index].update(block.start, block.values)
                    frame_statistics.update(block.values)
                    if all_handle is not None:
                        for value in block.values:
                            all_handle.write(f"\t{float(value):.12g}")
                    if component_handle is not None:
                        for vector in block.vectors:
                            component_handle.write(
                                "\t" + "\t".join(f"{float(value):.12g}" for value in vector)
                            )
                average_values.extend(
                    (frame_statistics.mean, frame_statistics.std)
                )
            if all_handle is not None:
                all_handle.write("\n")
            if component_handle is not None:
                component_handle.write("\n")
            if average_handle is not None:
                average_handle.write(
                    "\t".join(
                        [f"{time_ns:.12g}", *[f"{value:.12g}" for value in average_values]]
                    )
                    + "\n"
                )

    if "statistics" in paths:
        _write_statistics(
            paths["statistics"],
            jobs,
            pooled_statistics,
            channel_statistics,
            descriptors,
            args.statistics_scope,
            args,
        )

    if "histogram" in paths:
        edges = _histogram_edges(args, pooled_statistics)
        histogram = _evaluate_histogram(
            trajectory,
            frame_indices,
            jobs,
            masses,
            unwrap_plan,
            args,
            edges,
        )
        if args.hist_scope == "job":
            legends = [job.label for job in jobs]
        else:
            legends = [
                descriptor.label for group in descriptors for descriptor in group
            ]
        normalization = histogram.normalized(args.hist_normalization)
        centers = 0.5 * (edges[:-1] + edges[1:])
        ylabel = {
            "count": "Count",
            "probability": "Probability",
            "density": "Probability density (1/nm)",
        }[args.hist_normalization]
        _write_histogram_xvg(
            paths["histogram"],
            centers,
            normalization,
            legends,
            ylabel,
            metadata,
            args,
        )

    for key, path in paths.items():
        print(f"## Wrote distance {key} output to {path}.")


distance_commands = single_command(prog, getargs_distance, distance, desc)
