# assembly calculation tool in DROPPS package by Yiming Tang @ Fudan
# Development started on Jan 26 2026

from argparse import SUPPRESS

from dropps.share.argument_parser import ArgumentParser
from dropps.share.command_class import single_command

from dropps.fileio.filename_control import validate_extension

from dropps.share.trajectory import trajectory_class
from MDAnalysis.transformations import unwrap
from tqdm import tqdm

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from MDAnalysis.lib.nsgrid import FastNS
from MDAnalysis.lib.distances import apply_PBC, minimize_vectors


import numpy as np

prog = "assembly"
desc = "Analyze the formation, size, composition, and shape of molecular assemblies."


def _cluster_members(pairs, labels, n_groups, threshold):
    """Build sorted connected components, including singleton-only frames."""
    counts = np.zeros((n_groups, n_groups), dtype=np.int64)
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    if pairs.size:
        gi = labels[pairs[:, 0]]
        gj = labels[pairs[:, 1]]
        key = gi.astype(np.int64) * n_groups + gj.astype(np.int64)
        one_way = np.bincount(
            key,
            minlength=n_groups * n_groups,
        ).reshape(n_groups, n_groups)
        counts = one_way + one_way.T
        np.fill_diagonal(counts, 0)

    graph = csr_matrix(counts >= threshold)
    n_components, component_labels = connected_components(
        graph,
        directed=False,
        return_labels=True,
    )
    members = [[] for _ in range(n_components)]
    for group_id, component_id in enumerate(component_labels):
        members[component_id].append(group_id)
    members.sort(key=len, reverse=True)
    return members


def _weighted_cluster_fraction(members, selected_group_ids, minimum_size):
    """Return the molecule fraction pooled across all qualifying clusters."""
    selected = set(selected_group_ids)
    qualifying = [component for component in members if len(component) >= minimum_size]
    denominator = sum(len(component) for component in qualifying)
    if denominator == 0:
        return 0.0
    numerator = sum(
        sum(group_id in selected for group_id in component) for component in qualifying
    )
    return numerator / denominator


def _contact_image_shifts(positions, pairs, labels, dimensions):
    """Return whole-group image shifts inferred from minimum-image contacts."""
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    if not pairs.size:
        return {}
    raw_vectors = positions[pairs[:, 1]] - positions[pairs[:, 0]]
    image_shifts = minimize_vectors(raw_vectors, dimensions) - raw_vectors
    shifts = {}
    for pair, shift in zip(pairs, image_shifts):
        source = int(labels[pair[0]])
        target = int(labels[pair[1]])
        if source == target:
            continue
        shifts.setdefault((source, target), np.asarray(shift, dtype=float))
        shifts.setdefault((target, source), -np.asarray(shift, dtype=float))
    return shifts


def _component_translations(component, image_shifts):
    """Propagate periodic image translations over one connected component."""
    component = list(component)
    translations = {component[0]: np.zeros(3, dtype=float)}
    pending = [component[0]]
    component_set = set(component)
    while pending:
        source = pending.pop()
        for (edge_source, target), shift in image_shifts.items():
            if (
                edge_source != source
                or target not in component_set
                or target in translations
            ):
                continue
            translations[target] = translations[source] + shift
            pending.append(target)
    if len(translations) != len(component):
        raise ValueError(
            "could not construct a consistent periodic image for a cluster"
        )
    return translations


def _component_geometry(groups, component, image_shifts=None):
    translations = (
        _component_translations(component, image_shifts)
        if image_shifts is not None
        else {group_id: np.zeros(3, dtype=float) for group_id in component}
    )
    positions = np.concatenate(
        [groups[group_id].positions + translations[group_id] for group_id in component],
        axis=0,
    )
    masses = np.concatenate([groups[group_id].masses for group_id in component])
    return rg_tensor_principal_components(positions, masses)


def _mean_or_nan(values):
    return float(np.mean(values)) if values else float("nan")


def rg_tensor_principal_components(pos, masses=None, box=None):
    """
    Compute:
      - scalar Rg
      - 3 principal components (sqrt of eigenvalues) of the gyration tensor
      - gyration tensor itself (3x3)
    with cluster-local PBC unwrapping (orthorhombic only).

    Parameters
    ----------
    pos : (N, 3) array_like
        Particle coordinates for the cluster (same frame).
    masses : (N,) array_like or None
        Per-particle masses. If None, all masses = 1.
    box : array_like or None
        MDAnalysis dimensions: [lx, ly, lz, alpha, beta, gamma].
        Only orthorhombic supported (angles = 90). If None => no PBC handling.

    Returns
    -------
    rg : float
        Radius of gyration, sqrt(trace(S)).
    rg_principal : (3,) np.ndarray
        Principal components: sqrt(eigenvalues(S)), sorted descending.
    S : (3,3) np.ndarray
        Mass-weighted gyration tensor about COM: S = (1/M) Σ m_i (r_i - r_com)(r_i - r_com)^T
    eigvecs : (3,3) np.ndarray
        Eigenvectors corresponding to rg_principal (columns), sorted descending.
    """
    pos = np.asarray(pos, dtype=np.float64)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"`pos` must be shape (N,3), got {pos.shape}")
    n = pos.shape[0]
    if n == 0:
        raise ValueError("Empty cluster: `pos` has N=0")

    if masses is None:
        m = np.ones(n, dtype=np.float64)
    else:
        m = np.asarray(masses, dtype=np.float64)
        if m.shape != (n,):
            raise ValueError(f"`masses` must be shape (N,), got {m.shape}")
        if np.any(m < 0):
            raise ValueError("`masses` must be non-negative")
        if np.all(m == 0):
            raise ValueError("All masses are zero; COM is undefined")

    # Unwrap cluster locally if box is provided
    if box is not None:
        box = np.asarray(box, dtype=np.float64).ravel()
        if box.size < 3:
            raise ValueError("`box` must provide at least [lx, ly, lz]")
        L = box[:3].copy()

        if box.size >= 6:
            angles = box[3:6]
            if np.any(np.abs(angles - 90.0) > 1e-6):
                raise NotImplementedError(
                    "Only orthorhombic boxes are supported (alpha=beta=gamma=90)."
                )
        if np.any(L <= 0):
            raise ValueError(f"Invalid box lengths: {L}")

        # Anchor on heaviest particle (stable)
        ref_idx = int(np.argmax(m))
        ref = pos[ref_idx]
        d = pos - ref
        d -= L * np.round(d / L)  # minimum image (orthorhombic)
        pos_u = ref + d
    else:
        pos_u = pos

    mtot = m.sum()
    com = (pos_u * m[:, None]).sum(axis=0) / mtot
    dr = pos_u - com

    # Gyration tensor: S = (1/M) Σ m_i dr_i dr_i^T
    # Efficient: (dr.T * m) @ dr  / M
    S = (dr.T * m) @ dr / mtot
    # Numerical symmetry cleanup
    S = 0.5 * (S + S.T)

    # Eigen-decomposition (S is symmetric)
    eigvals, eigvecs = np.linalg.eigh(S)  # ascending
    # guard tiny negative due to numerical roundoff
    eigvals = np.maximum(eigvals, 0.0)

    # Sort descending
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    rg_principal = np.sqrt(eigvals)  # (Rg1, Rg2, Rg3)
    rg = float(np.sqrt(eigvals.sum()))  # sqrt(trace(S))

    return rg, rg_principal, S, eigvecs


def getargs_assembly(argv):
    parser = ArgumentParser(prog=prog, description=desc)

    parser.add_argument(
        "-s",
        "--run-input",
        type=str,
        required=True,
        help="Input DROPPS run file (.tpr) containing the system and simulation settings.",
    )

    parser.add_argument(
        "-f", "--input", type=str, required=True, help="Input trajectory file (.xtc)."
    )

    parser.add_argument(
        "-n",
        "--index",
        type=str,
        required=False,
        help="Optional index file (.ndx) defining additional atom groups.",
    )

    parser.add_argument(
        "-ref",
        "--reference-group",
        type=int,
        help="Reference index group used to determine clusters; if omitted, prompt interactively.",
    )

    parser.add_argument(
        "-sel",
        "--selection-group",
        type=int,
        nargs="+",
        help="Index groups whose molecular fractions in clusters are reported; if omitted, prompt interactively.",
    )

    parser.add_argument(
        "-pbc",
        "--treat-pbc",
        action="store_true",
        default=False,
        help="Apply periodic-boundary distances during cluster detection.",
    )

    parser.add_argument(
        "-c",
        "--cutoff",
        type=float,
        default=0.7,
        help="Bead-contact cutoff used for cluster detection, in nm.",
    )

    parser.add_argument(
        "-t",
        "--threshold",
        type=int,
        default=5,
        help="Minimum number of bead contacts required to connect two chains.",
    )
    parser.add_argument(
        "--threashold", dest="threshold", type=int, default=SUPPRESS, help=SUPPRESS
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
        "-cn",
        "--cluster-number",
        type=str,
        help="Output cluster-count time series (.xvg); the extension is added if omitted.",
    )

    parser.add_argument(
        "-cs",
        "--cluster-size",
        type=str,
        help="Output largest-cluster size time series (.xvg); the extension is added if omitted.",
    )

    parser.add_argument(
        "-csd",
        "--cluster-size-distribution",
        type=str,
        help="Output per-frame cluster-size distributions (.xvg); the extension is added if omitted.",
    )

    parser.add_argument(
        "-mf",
        "--molecule-fraction",
        type=str,
        help="Output selected-group fractions in large clusters (.xvg); the extension is added if omitted.",
    )

    parser.add_argument(
        "-mfc",
        "--molecule-fraction-cutoff",
        type=int,
        default=10,
        help="Minimum cluster size used for molecular-fraction and shape analyses.",
    )

    parser.add_argument(
        "-rgl",
        "--radius-gyration-largest",
        type=str,
        help="Output largest-cluster radius-of-gyration time series (.xvg); the extension is added if omitted.",
    )

    parser.add_argument(
        "-asp",
        "--asphericity",
        type=str,
        help="Output asphericity of large clusters (.xvg); the extension is added if omitted.",
    )

    parser.add_argument(
        "-elp",
        "--ellipticity",
        type=str,
        help="Output ellipticity of large clusters (.xvg); the extension is added if omitted.",
    )

    args = parser.parse_args(argv)
    return args


def assembly(args):
    outputs = (
        args.cluster_number,
        args.cluster_size,
        args.cluster_size_distribution,
        args.molecule_fraction,
        args.radius_gyration_largest,
        args.asphericity,
        args.ellipticity,
    )
    if all(output is None for output in outputs):
        raise ValueError("at least one assembly output must be requested")
    if not np.isfinite(args.cutoff) or args.cutoff <= 0.0:
        raise ValueError("contact cutoff must be a positive finite number")
    if args.threshold <= 0:
        raise ValueError("contact threshold must be positive")
    if args.molecule_fraction_cutoff <= 0:
        raise ValueError("molecule-fraction cutoff must be positive")

    # load trajectory into memory

    try:
        trajectory = trajectory_class(args.run_input, args.index, args.input)
    except Exception as exc:
        print(
            "## An exception occurred when trying to open trajectory file %s."
            % args.input
        )
        print(f"## Root cause: {exc}")
        quit()

    if args.treat_pbc is True:
        print("## Distance will be calculated using periodic boundary conditions.")
        trajectory.Universe.trajectory.add_transformations(
            unwrap(trajectory.Universe.atoms)
        )
    else:
        print(
            "## WARNING: Distance will be calculated without periodic boundary conditions."
        )

    # We treat time for analysis and generate frame for analysis

    start_frame, end_frame, interval_frame = trajectory.time2frame(
        args.start_time, args.end_time, args.delta_time
    )
    # We treat groups

    if args.reference_group is None or args.selection_group is None:
        trajectory.index.print_all()

    if args.reference_group is not None:
        print(
            f"## Will use group {args.reference_group} as reference group to determine clusters."
        )
        reference_group, reference_group_name = trajectory.getSelection(
            f"group {args.reference_group}"
        )
    else:
        reference_group, reference_group_name = trajectory.getSelection_interactive(
            "reference group for cluster determination"
        )

    if args.selection_group is not None:
        selection_group_ids = ",".join(str(i) for i in args.selection_group)
        print(
            f"## Will use groups {selection_group_ids} to print composition of clusters."
        )
        selection_groups = [
            trajectory.getSelection(f"group {gid}")[0] for gid in args.selection_group
        ]
        selection_group_names = [f"group{i}" for i in args.selection_group]
    else:
        selection_groups, selection_group_names = (
            trajectory.getSelection_interactive_multiple(
                "selection groups for cluster composition output"
            )
        )

    # We now split the indices

    reference_chains = trajectory.index.splitch_indices(reference_group.indices)
    selection_chains_list = [
        trajectory.index.splitch_indices(group.indices) for group in selection_groups
    ]

    print("## We now test reference and selection chains.")
    print(f"## There are {len(reference_chains)} reference chains.")
    if not reference_chains:
        raise ValueError("reference group does not contain any chains")
    reference_chain_length = [len(chain) for chain in reference_chains]
    if len(set(reference_chain_length)) != 1:
        raise ValueError(
            f"reference chains have different lengths: {set(reference_chain_length)}"
        )
    else:
        print(f"## Each reference chain has length {reference_chain_length[0]}.")

    reference_chain_lookup = {
        tuple(int(index) for index in chain): chain_id
        for chain_id, chain in enumerate(reference_chains)
    }
    selection_group_chain_indices = []
    for i, selection_chains in enumerate(selection_chains_list):
        selected_ids = []
        for sel_chain in selection_chains:
            key = tuple(int(index) for index in sel_chain)
            if key not in reference_chain_lookup:
                raise ValueError(
                    f"selection group {selection_group_names[i]} contains a chain "
                    "that is absent from the reference group"
                )
            selected_ids.append(reference_chain_lookup[key])
        selection_group_chain_indices.append(selected_ids)

    print("## Test pass, all selection chains are contained in reference chains.")

    # We generate AtomGroups for reference chains

    groups = []
    for idxs in reference_chains:
        idxs = np.asarray(idxs, dtype=np.int64)
        groups.append(trajectory.Universe.atoms[idxs])
    n_groups = len(groups)

    all_atoms = sum(groups[1:], groups[0])
    labels = np.empty(len(all_atoms), dtype=np.int16)
    offset = 0
    for gi, ag in enumerate(groups):
        labels[offset : offset + len(ag)] = gi
        offset += len(ag)

    # We now perform analysis

    time_list = list()

    cluster_number_list = list()
    largest_size_list = list()
    size_distribution_list = list()
    size_distribution_list_verbose = list()
    large_cluster_molecular_fraction_list = [list() for _ in selection_groups]
    rg_largest_list = list()
    asphericity_list = list()
    ellipticity_list = list()

    for ts in tqdm(
        trajectory.Universe.trajectory[start_frame : end_frame + 1 : interval_frame]
    ):
        time_list.append(ts.time / 1000)

        # We now build contact map between reference chains

        pos = np.asarray(all_atoms.positions, dtype=float)

        box = ts.dimensions
        search_pos = apply_PBC(pos, box) if args.treat_pbc else pos
        ns = FastNS(args.cutoff * 10.0, search_pos, box=box, pbc=args.treat_pbc)

        pairs = ns.self_search().get_pairs()
        members = _cluster_members(pairs, labels, n_groups, args.threshold)
        n_components = len(members)
        sizes = [len(m) for m in members]
        image_shifts = (
            _contact_image_shifts(pos, pairs, labels, box) if args.treat_pbc else None
        )

        cluster_number_list.append(n_components)
        largest_size_list.append(sizes[0] if len(sizes) > 0 else 0)
        size_distribution_list.append(sizes)

        # We now determine selection group compositions
        size_distribution_verbose = []
        for comp_id, comp in enumerate(members):
            comp_dict = {}
            for sel_id, selected_ids in enumerate(selection_group_chain_indices):
                count = sum(chain_idx in comp for chain_idx in selected_ids)
                comp_dict[selection_group_names[sel_id]] = count
            size_distribution_verbose.append((len(comp), comp_dict))
        size_distribution_list_verbose.append(size_distribution_verbose)

        # print(sizes)
        # print(size_distribution_verbose)

        # We now calculate molecular fractions in large clusters
        for sel_id, selected_ids in enumerate(selection_group_chain_indices):
            large_cluster_molecular_fraction_list[sel_id].append(
                _weighted_cluster_fraction(
                    members,
                    selected_ids,
                    args.molecule_fraction_cutoff,
                )
            )

        # We now calculate radius of gyration of largest cluster
        if args.radius_gyration_largest is not None:
            largest_cluster = members[0]
            rg, (rg1, rg2, rg3), S, eigvecs = _component_geometry(
                groups,
                largest_cluster,
                image_shifts,
            )

            rg_largest_list.append((rg, rg1, rg2, rg3))

        if args.asphericity is not None or args.ellipticity is not None:
            # We now calculate asphericity of large clusters
            asphericity_list_temp = []
            ellipticity_list_temp = []
            for comp in members:
                if len(comp) >= args.molecule_fraction_cutoff:
                    rg, (rg1, rg2, rg3), S, eigvecs = _component_geometry(
                        groups,
                        comp,
                        image_shifts,
                    )

                    asphericity = (
                        (rg1 - rg2) ** 2 + (rg2 - rg3) ** 2 + (rg3 - rg1) ** 2
                    ) / (2 * (rg1 + rg2 + rg3) ** 2)
                    ellipticity = rg1 / rg3 if rg3 > np.finfo(float).eps else np.nan
                    asphericity_list_temp.append(asphericity)
                    ellipticity_list_temp.append(ellipticity)

            asphericity_list.append(asphericity_list_temp)
            ellipticity_list.append(ellipticity_list_temp)

    # We now output results

    if args.cluster_number is not None:
        cluster_number_filename = validate_extension(args.cluster_number, "xvg")
        with open(cluster_number_filename, "w") as fout:
            fout.write("#Time(ns)    Num_Clusters\n")
            for t, ncl in zip(time_list, cluster_number_list):
                fout.write(f"{t:.3f}    {ncl}\n")

    if args.cluster_size is not None:
        cluster_size_filename = validate_extension(args.cluster_size, "xvg")
        with open(cluster_size_filename, "w") as fout:
            fout.write("#Time(ns)    Largest_Cluster_Size\n")
            for t, sz in zip(time_list, largest_size_list):
                fout.write(f"{t:.3f}    {sz}\n")

    if args.cluster_size_distribution is not None:
        size_distribution_dicts = [
            {size: sizes_of_frame.count(size) for size in set(sizes_of_frame)}
            for sizes_of_frame in size_distribution_list
        ]
        sizes = sorted({k for d in size_distribution_dicts for k in d})

        size_distribution_filename = validate_extension(
            args.cluster_size_distribution, "xvg"
        )
        with open(size_distribution_filename, "w") as fout:
            header = "#Time(ns)    " + "    ".join([f"{sz}" for sz in sizes]) + "\n"
            fout.write(header)
            for t, dist_dict in zip(time_list, size_distribution_dicts):
                line = (
                    f"{t:.3f}    "
                    + "    ".join([f"{dist_dict.get(sz, 0)}" for sz in sizes])
                    + "\n"
                )
                fout.write(line)

    if args.molecule_fraction is not None:
        molecule_fraction_filename = validate_extension(args.molecule_fraction, "xvg")
        with open(molecule_fraction_filename, "w") as fout:
            header = "#Time(ns)    " + "    ".join(selection_group_names) + "\n"
            fout.write(header)
            for i in range(len(time_list)):
                line = (
                    f"{time_list[i]:.3f}    "
                    + "    ".join(
                        [
                            f"{large_cluster_molecular_fraction_list[sel_id][i]:.6f}"
                            for sel_id in range(len(selection_groups))
                        ]
                    )
                    + "\n"
                )
                fout.write(line)

    if args.radius_gyration_largest is not None:
        radius_gyration_largest_filename = validate_extension(
            args.radius_gyration_largest, "xvg"
        )
        with open(radius_gyration_largest_filename, "w") as fout:
            fout.write("#Time(ns)    Rg(nm)    Rg1(nm)    Rg2(nm)    Rg3(nm)\n")
            for t, (rg, rg1, rg2, rg3) in zip(time_list, rg_largest_list):
                fout.write(
                    f"{t:.3f}    {rg / 10.0:.6f}    {rg1 / 10.0:.6f}    {rg2 / 10.0:.6f}    {rg3 / 10.0:.6f}\n"
                )

    if args.asphericity is not None:
        asphericity_filename = validate_extension(args.asphericity, "xvg")
        with open(asphericity_filename, "w") as fout:
            header = (
                "#Time(ns)    Mean    "
                + "    ".join(
                    [
                        f"Cluster{i + 1}"
                        for i in range(
                            max((len(a_list) for a_list in asphericity_list), default=0)
                        )
                    ]
                )
                + "\n"
            )
            fout.write(header)

            for i in range(len(time_list)):
                line = (
                    f"{time_list[i]:.3f}    "
                    + f"{_mean_or_nan(asphericity_list[i]):.6f}    "
                    + "    ".join(f"{v:.6f}" for v in asphericity_list[i])
                    + "\n"
                )
                fout.write(line)

    if args.ellipticity is not None:
        ellipticity_filename = validate_extension(args.ellipticity, "xvg")
        with open(ellipticity_filename, "w") as fout:
            header = (
                "#Time(ns)    Mean    "
                + "    ".join(
                    [
                        f"Cluster{i + 1}"
                        for i in range(
                            max((len(e_list) for e_list in ellipticity_list), default=0)
                        )
                    ]
                )
                + "\n"
            )
            fout.write(header)

            for i in range(len(time_list)):
                line = (
                    f"{time_list[i]:.3f}    "
                    + f"{_mean_or_nan(ellipticity_list[i]):.6f}    "
                    + "    ".join(f"{v:.6f}" for v in ellipticity_list[i])
                    + "\n"
                )
                fout.write(line)


assembly_commands = single_command("assembly", getargs_assembly, assembly, desc)
