# assembly calculation tool in DROPPS package by Yiming Tang @ Fudan
# Development started on Jan 26 2026

from argparse import ArgumentParser

from dropps.fileio.filename_control import validate_extension

from dropps.share.trajectory import trajectory_class
from MDAnalysis.transformations import unwrap
from tqdm import tqdm
from openmm import CustomNonbondedForce, unit

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from MDAnalysis.lib.nsgrid import FastNS


import numpy as np

prog = "assembly"
desc = '''This program calculate the time evoluted formation of molecular assemblies.'''

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
        d -= L * np.round(d / L)   # minimum image (orthorhombic)
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

    rg_principal = np.sqrt(eigvals)          # (Rg1, Rg2, Rg3)
    rg = float(np.sqrt(eigvals.sum()))       # sqrt(trace(S))

    return rg, rg_principal, S, eigvecs


def getargs_assembly(argv):

    parser = ArgumentParser(prog=prog, description=desc)

    parser.add_argument('-s', '--run-input', type=str, required=True, 
                    help="TPR file containing all information for a simulation run.")
    
    parser.add_argument('-f', '--input', type=str, required=True, 
                    help="XTC file which is taken as input trajectory.")
    
    parser.add_argument('-n', '--index', type=str, required=False,
                    help="Index file containing non-default groups.")
    
    parser.add_argument('-ref', '--reference-group', type=int,
                    help="Reference group of atoms to determine clusters.")
    
    parser.add_argument('-sel', '--selection-group', type=int, nargs='+',
                    help="Groups of atoms on which their number within each cluster will be printed")
    
    parser.add_argument('-pbc', '--treat-pbc', action='store_true', default=False,
                        help="Treat broken molecule at periodic boundaries.")
    
    parser.add_argument('-c', '--cutoff', type=float, default=0.7,
                        help="Cutoff distance for contact calculation, unit is nanometer.")
    
    parser.add_argument('-t', '--threashold', type=int, default=5,
                        help="Threashold number of contacts between two chains to define an assembly.")
    
    parser.add_argument('-b', '--start-time', type=int,
                        help="Time (ns) of the first frame to calculate.")

    parser.add_argument('-e', '--end-time', type=int,
                        help="Time (ns) of the last frame to calculate.")
    
    parser.add_argument('-dt', '--delta-time', type=int,
                        help="Intervals (ns) between two calculated frames.")

    parser.add_argument('-cn', '--cluster-number', type=str,
                        help="Number of clusters as a function of time.")
    
    parser.add_argument('-cs', '--cluster-size', type=str, 
                        help="Size of the largest clusters as a function of time.")
    
    parser.add_argument('-csd', '--cluster-size-distribution', type=str, 
                        help="Distribution of cluster sizes at each time frame.")
    
    parser.add_argument('-mf', '--molecule-fraction', type=str,
                        help="Molecular fraction of each selection group in large clusters.")
    
    parser.add_argument('-mfc', '--molecule-fraction-cutoff', type=int, default=10,
                        help="Cutoff size to define large clusters for molecular fraction calculation.")
    
    parser.add_argument('-rgl', '--radius-gyration-largest', type=str,
                        help="Radius of gyration of the largest cluster as a function of time.")
    
    parser.add_argument('-asp', '--asphericity', type=str,
                        help="Asphericity values of all large clusters as a function of time.")
    
    parser.add_argument('-elp', '--ellipticity', type=str,
                        help="Ellipticity values of all large clusters as a function of time.")
    
    
    args = parser.parse_args(argv)
    return args


def assembly(args):
    
    # load trajectory into memory

    try:
        trajectory = trajectory_class(args.run_input, args.index, args.input)
    except Exception as exc:
        print("## An exception occurred when trying to open trajectory file %s." % args.input)
        print(f"## Root cause: {exc}")
        quit()

    if args.treat_pbc is True:
        print(f"## Distance will be calculated using periodic boundary conditions.")
        trajectory.Universe.trajectory.add_transformations(unwrap(trajectory.Universe.atoms))
    else:
        print(f"## WARNING: Distance will be calculated without periodic boundary conditions.")


    # We treat time for analysis and generate frame for analysis

    start_frame, end_frame, interval_frame = trajectory.time2frame(args.start_time, args.end_time, args.delta_time)
    frame_list = list(range(start_frame, end_frame, interval_frame))

    # We treat groups
    
    if args.reference_group is None or args.selection_group is None:
        trajectory.index.print_all()

    if args.reference_group is not None:
        print(f"## Will use group {args.reference_group} as reference group to determine clusters.")
        reference_group, reference_group_name = trajectory.getSelection(f"group {args.reference_group}")
    else:
        reference_group, reference_group_name = trajectory.getSelection_interactive("reference group for cluster determination")

    if args.selection_group is not None:
        print(f"## Will use groups {','.join([f"{i}" for i in args.selection_group])} to print composition of clusters.")
        selection_groups = [trajectory.getSelection(f"group {gid}")[0] for gid in args.selection_group]
        selection_group_names = [f"group{i}" for i in args.selection_group]
    else:
        selection_groups, selection_group_names = trajectory.getSelection_interactive_multiple("selection groups for cluster composition output")

    # We now split the indices

    reference_chains = trajectory.index.splitch_indices(reference_group.indices)
    selection_chains_list = [trajectory.index.splitch_indices(group.indices) for group in selection_groups]

    print(f"## We now test reference and selection chains.")
    print(f"## There are {len(reference_chains)} reference chains.")
    reference_chain_length = [len(chain) for chain in reference_chains]
    if len(set(reference_chain_length)) != 1:
        print("## Reference chains have different lengths: ", set(reference_chain_length))
    else:
        print(f"## Each reference chain has length {reference_chain_length[0]}.")
    
    for i, selection_chains in enumerate(selection_chains_list):
        for sel_chain in selection_chains:
            if sel_chain not in reference_chains:
                print(f"## WARNING: Selection group {selection_group_names[i]} has chain not in reference chains.")
                quit()

    print(f"## Test pass, all selection chains are contained in reference chains.")

    # We now generate representative node list
    reference_chains_representatives = [min(chain) for chain in reference_chains]
    selection_chains_representatives_list = [[min(chain) for chain in selection_chains] for selection_chains in selection_chains_list]

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

    for ts in tqdm(trajectory.Universe.trajectory[start_frame:end_frame:interval_frame]):

        time_list.append(trajectory.Universe.trajectory.time / 1000)

        # We now build contact map between reference chains

        pos = all_atoms.positions

        box = ts.dimensions
        ns = FastNS(args.cutoff * 10.0, pos, box=box, pbc=args.treat_pbc)
        
        pairs = ns.self_search().get_pairs()

        if pairs.size == 0:
            continue

        gi = labels[pairs[:, 0]]
        gj = labels[pairs[:, 1]]

        # Bin pair counts into matrix quickly using a single bincount
        key = gi.astype(np.int64) * n_groups + gj.astype(np.int64)
        counts = np.bincount(key, minlength=n_groups * n_groups).reshape(n_groups, n_groups)

        # Because self_search returns each pair once (i<j), make symmetric if desired
        # For inter-group contacts, symmetry is usually what you want:
        mats = counts + counts.T
        np.fill_diagonal(mats, 0)

        if not np.allclose(mats, mats.T):
            raise ValueError("Contact matrix is not symmetric.")

        # We now determine clusters

        adjancy = (mats >= args.threashold)
        graph = csr_matrix(adjancy)
        n_components, new_labels = connected_components(graph, directed=False, return_labels=True)

        members = [[] for _ in range(n_components)]
        for idx, lab in enumerate(new_labels):
            members[lab].append(idx)

        # Sort components by size (desc), stable by label otherwise
        members.sort(key=len, reverse=True)
        sizes = [len(m) for m in members]

        cluster_number_list.append(n_components)
        largest_size_list.append(sizes[0] if len(sizes) > 0 else 0)
        size_distribution_list.append(sizes)

        # We now determine selection group compositions
        size_distribution_verbose = []
        for comp_id, comp in enumerate(members):
            comp_dict = {}
            for sel_id, selection_chains_representatives in enumerate(selection_chains_representatives_list):
                count = 0
                for chain_rep in selection_chains_representatives:
                    chain_idx = reference_chains_representatives.index(chain_rep)
                    if chain_idx in comp:
                        count +=1
                comp_dict[selection_group_names[sel_id]] = count
            size_distribution_verbose.append((len(comp), comp_dict))
        size_distribution_list_verbose.append(size_distribution_verbose)

        #print(sizes)
        #print(size_distribution_verbose)

        # We now calculate molecular fractions in large clusters
        for sel_id, selection_chains_representatives in enumerate(selection_chains_representatives_list):
            large_cluster_molecular_fraction = 0.0
            large_cluster_count = 0
            for comp in members:
                if len(comp) >= args.molecule_fraction_cutoff:
                    count = 0
                    for chain_rep in selection_chains_representatives:
                        chain_idx = reference_chains_representatives.index(chain_rep)
                        if chain_idx in comp:
                            count +=1
                    fraction = count / len(comp)
                    large_cluster_molecular_fraction += fraction
                    large_cluster_count += 1
            if large_cluster_count > 0:
                large_cluster_molecular_fraction /= large_cluster_count
            large_cluster_molecular_fraction_list[sel_id].append(large_cluster_molecular_fraction)
        
        # We now calculate radius of gyration of largest cluster
        if args.radius_gyration_largest is not None:
            largest_cluster = members[0]
            largest_cluster_atom_indices = []
            for chain_idx in largest_cluster:
                chain = reference_chains[chain_idx]
                largest_cluster_atom_indices.extend(chain)
            largest_cluster_atom_indices = np.asarray(largest_cluster_atom_indices, dtype=np.int64)
            largest_cluster_atoms = trajectory.Universe.atoms[largest_cluster_atom_indices]
            pos = largest_cluster_atoms.positions
            masses = largest_cluster_atoms.masses

            rg, (rg1, rg2, rg3), S, eigvecs = rg_tensor_principal_components(pos, masses, trajectory.Universe.dimensions)
            
            rg_largest_list.append((rg, rg1, rg2, rg3))
        
        if args.asphericity is not None or args.ellipticity is not None:
            # We now calculate asphericity of large clusters
            largest_cluster = members[0]
            asphericity_list_temp = []
            ellipticity_list_temp = []
            for comp in members:
                if len(comp) >= args.molecule_fraction_cutoff:
                    comp_atom_indices = []
                    for chain_idx in comp:
                        chain = reference_chains[chain_idx]
                        comp_atom_indices.extend(chain)
                    comp_atom_indices = np.asarray(comp_atom_indices, dtype=np.int64)
                    comp_atoms = trajectory.Universe.atoms[comp_atom_indices]
                    pos = comp_atoms.positions
                    masses = comp_atoms.masses

                    rg, (rg1, rg2, rg3), S, eigvecs = rg_tensor_principal_components(pos, masses, trajectory.Universe.dimensions)
                    
                    asphericity = ((rg1 - rg2)**2 + (rg2 - rg3)**2 + (rg3 - rg1)**2) / (2 * (rg1 + rg2 + rg3)**2)
                    ellipticity = rg1 / rg3
                    asphericity_list_temp.append(asphericity)
                    ellipticity_list_temp.append(ellipticity)

            asphericity_list.append(asphericity_list_temp)
            ellipticity_list.append(ellipticity_list_temp)
            
    # We now output results

    if args.cluster_number is not None:
        cluster_number_filename = validate_extension(args.cluster_number, 'xvg')
        with open(cluster_number_filename, 'w') as fout:
            fout.write("#Time(ns)    Num_Clusters\n")
            for t, ncl in zip(time_list, cluster_number_list):
                fout.write(f"{t:.3f}    {ncl}\n")
    
    if args.cluster_size is not None:
        cluster_size_filename = validate_extension(args.cluster_size, 'xvg')
        with open(cluster_size_filename, 'w') as fout:
            fout.write("#Time(ns)    Largest_Cluster_Size\n")
            for t, sz in zip(time_list, largest_size_list):
                fout.write(f"{t:.3f}    {sz}\n")

    if args.cluster_size_distribution is not None:

        size_distribution_dicts = [
            {size: sizes_of_frame.count(size)
                for size in set(sizes_of_frame)
            }
            for sizes_of_frame in size_distribution_list
        ]
        sizes = sorted(
            {k for d in size_distribution_dicts for k in d}
        )

        size_distribution_filename = validate_extension(args.cluster_size_distribution, 'xvg')
        with open(size_distribution_filename, 'w') as fout:
            header = "#Time(ns)    " + "    ".join([f"{sz}" for sz in sizes]) + "\n"
            fout.write(header)
            for t, dist_dict in zip(time_list, size_distribution_dicts):
                line = f"{t:.3f}    " + "    ".join([f"{dist_dict.get(sz, 0)}" for sz in sizes]) + "\n"
                fout.write(line)
    
    if args.molecule_fraction is not None:
        molecule_fraction_filename = validate_extension(args.molecule_fraction, 'xvg')
        with open(molecule_fraction_filename, 'w') as fout:
            header = "#Time(ns)    " + "    ".join(selection_group_names) + "\n"
            fout.write(header)
            for i in range(len(time_list)):
                line = f"{time_list[i]:.3f}    " + "    ".join([f"{large_cluster_molecular_fraction_list[sel_id][i]:.6f}" for sel_id in range(len(selection_groups))]) + "\n"
                fout.write(line)
    
    if args.radius_gyration_largest is not None:
        radius_gyration_largest_filename = validate_extension(args.radius_gyration_largest, 'xvg')
        with open(radius_gyration_largest_filename, 'w') as fout:
            fout.write("#Time(ns)    Rg(nm)    Rg1(nm)    Rg2(nm)    Rg3(nm)\n")
            for t, (rg, rg1, rg2, rg3) in zip(time_list, rg_largest_list):
                fout.write(f"{t:.3f}    {rg/10.0:.6f}    {rg1/10.0:.6f}    {rg2/10.0:.6f}    {rg3/10.0:.6f}\n")

    if args.asphericity is not None:
        asphericity_filename = validate_extension(args.asphericity, 'xvg')
        with open(asphericity_filename, 'w') as fout:
            header = "#Time(ns)    Mean    " + "    ".join([f"Cluster{i+1}" for i in range(max(len(a_list) for a_list in asphericity_list))]) + "\n"
            fout.write(header)

            for i in range(len(time_list)):
                line = (
                    f"{time_list[i]:.3f}    "
                    + f"{np.mean(asphericity_list[i]):.6f}    "
                    + "    ".join(f"{v:.6f}" for v in asphericity_list[i])
                    + "\n"
                )
                fout.write(line)
    
    if args.ellipticity is not None:
        ellipticity_filename = validate_extension(args.ellipticity, 'xvg')
        with open(ellipticity_filename, 'w') as fout:
            header = "#Time(ns)    Mean    " + "    ".join([f"Cluster{i+1}" for i in range(max(len(e_list) for e_list in ellipticity_list))]) + "\n"
            fout.write(header)

            for i in range(len(time_list)):
                line = (
                    f"{time_list[i]:.3f}    "
                    + f"{np.mean(ellipticity_list[i]):.6f}    "
                    + "    ".join(f"{v:.6f}" for v in ellipticity_list[i])
                    + "\n"
                )
                fout.write(line)


from dropps.share.command_class import single_command
assembly_commands = single_command("assembly", getargs_assembly, assembly, desc)