# contact map (next generation, using neighborhood searching) in DROPPS package by Yiming Tang @ Fudan
# Development started on Jan 29 2026

from argparse import ArgumentParser
import numpy as np
from tqdm import tqdm
from dropps.fileio.filename_control import validate_extension
from dropps.fileio.xpm_reader import write_xpm
from dropps.fileio.xvg_reader import write_xvg

from MDAnalysis.transformations import wrap

from dropps.share.trajectory import trajectory_class

from MDAnalysis.lib.nsgrid import FastNS

import warnings
from Bio import BiopythonDeprecationWarning
warnings.filterwarnings("ignore", category=BiopythonDeprecationWarning)

prog = "contact"
desc = '''This program calculate contact map in a residue-level resolution.'''

def remove_near_diagonal(mat, cutoff):
    mat = mat.copy()   # avoid modifying original
    n = mat.shape[0]

    i, j = np.ogrid[:n, :n]
    mask = np.abs(i - j) <= cutoff

    mat[mask] = 0
    return mat

def _build_lookup_fixed(chains, max_idx):

    """
    chains: list of equal-length lists (indices for each chain)
    returns:
      idx_to_chain: (max_idx+1,) -> chain id or -1
      idx_to_pos:   (max_idx+1,) -> local position within chain or -1
    """
    idx_to_chain = np.full(max_idx + 1, -1, dtype=np.int32)
    idx_to_pos   = np.full(max_idx + 1, -1, dtype=np.int32)

    for cid, idxs in enumerate(chains):

        idxs = np.asarray(idxs, dtype=np.int64)
        idxs = idxs[idxs <= max_idx]
        
        #idxs = np.asarray(idxs[np.asarray(idxs, dtype=np.int64) <= max_idx], dtype=np.int64)
        idx_to_chain[idxs] = cid
        idx_to_pos[idxs]   = np.arange(len(idxs), dtype=np.int32)

    return idx_to_chain, idx_to_pos


def _canonicalize_pairs(pairs, undirected=True, unique=True, drop_self=True):
    pairs = np.asarray(pairs, dtype=np.int64)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("pairs must be a (N,2) array-like of indices")

    if pairs.size == 0:
        return pairs

    if drop_self:
        pairs = pairs[pairs[:, 0] != pairs[:, 1]]

    if undirected:
        pairs = np.sort(pairs, axis=1)

    if unique and pairs.size:
        pairs = np.unique(pairs, axis=0)

    return pairs


def compute_residue_contact_maps_fixed(
    pairs,
    reference_chains,  # n_ref x L_ref (lists)
    selection_chains,  # n_sel x L_sel (lists) or None
    undirected=True,
    unique_pairs=True,
    drop_self_pairs=True,
    dtype=np.int64
):
    """
    Fast dense residue-residue contact maps for two molecule types with fixed per-type length.

    Returns a dict with:
      ref_sel_inter: (n_ref, n_sel, L_ref, L_sel)
      ref_ref_inter: (n_ref, n_ref, L_ref, L_ref)    [inter-chain only; diag chains zero]
      sel_sel_inter: (n_sel, n_sel, L_sel, L_sel)    [inter-chain only; diag chains zero]
      ref_intra:     (n_ref, L_ref, L_ref)
      sel_intra:     (n_sel, L_sel, L_sel)
    """
    n_ref = len(reference_chains)
    L_ref = len(reference_chains[0]) if n_ref else 0
    # Sanity: fixed lengths within each type
    if n_ref and any(len(ch) != L_ref for ch in reference_chains):
        raise ValueError("reference_chains are not fixed-length within type")

    if selection_chains is not None:
        n_sel = len(selection_chains)    
        L_sel = len(selection_chains[0]) if n_sel else 0
        # Sanity: fixed lengths within each type
        if n_sel and any(len(ch) != L_sel for ch in selection_chains):
            raise ValueError("selection_chains are not fixed-length within type")

    pairs = _canonicalize_pairs(pairs, undirected=undirected, unique=unique_pairs, drop_self=drop_self_pairs)

    # Allocate outputs
    ref_ref_inter = np.zeros((n_ref, n_ref, L_ref, L_ref), dtype=dtype)
    ref_intra     = np.zeros((n_ref, L_ref, L_ref), dtype=dtype)

    if selection_chains is not None:
        ref_sel_inter = np.zeros((n_ref, n_sel, L_ref, L_sel), dtype=dtype)    
        sel_sel_inter = np.zeros((n_sel, n_sel, L_sel, L_sel), dtype=dtype)    
        sel_intra     = np.zeros((n_sel, L_sel, L_sel), dtype=dtype)

    if pairs.size == 0:
        if selection_chains is None:
            return dict(
                ref_ref_inter=ref_ref_inter,
                ref_intra=ref_intra,
            )
        else:
            return dict(
                ref_sel_inter=ref_sel_inter,
                ref_ref_inter=ref_ref_inter,
                sel_sel_inter=sel_sel_inter,
                ref_intra=ref_intra,
                sel_intra=sel_intra,
            )

    a = pairs[:, 0]
    b = pairs[:, 1]
    max_idx = int(pairs.max())

    # Lookups (safe indexing: out-of-range -> -1)
    ref_chain, ref_pos = _build_lookup_fixed(reference_chains, max_idx) if n_ref else (np.array([-1], np.int32), np.array([-1], np.int32))
    if selection_chains is not None:
        sel_chain, sel_pos = _build_lookup_fixed(selection_chains, max_idx) if n_sel else (np.array([-1], np.int32), np.array([-1], np.int32))

    def safe(arr, idx):
        out = np.full(idx.shape, -1, dtype=np.int32)
        m = idx < arr.shape[0]
        out[m] = arr[idx[m]]
        return out

    rca = safe(ref_chain, a); rcb = safe(ref_chain, b)
    rpa = safe(ref_pos,   a); rpb = safe(ref_pos,   b)

    if selection_chains is not None:
        sca = safe(sel_chain, a); scb = safe(sel_chain, b)
        spa = safe(sel_pos,   a); spb = safe(sel_pos,   b)

    # (4) reference intra (same ref chain)
    m = (rca >= 0) & (rcb >= 0) & (rca == rcb)
    if np.any(m):
        np.add.at(ref_intra, (rca[m], rpa[m], rpb[m]), 1)
        if undirected:
            np.add.at(ref_intra, (rca[m], rpb[m], rpa[m]), 1)
    
    # (2) reference-reference inter (different ref chains)
    m = (rca >= 0) & (rcb >= 0) & (rca != rcb)
    if np.any(m):
        i = rca[m]; j = rcb[m]
        pi = rpa[m]; pj = rpb[m]
        np.add.at(ref_ref_inter, (i, j, pi, pj), 1)
        if undirected:
            np.add.at(ref_ref_inter, (j, i, pj, pi), 1)

    if selection_chains is not None:

        # (5) selection intra (same sel chain)
        m = (sca >= 0) & (scb >= 0) & (sca == scb)
        if np.any(m):
            np.add.at(sel_intra, (sca[m], spa[m], spb[m]), 1)
            if undirected:
                np.add.at(sel_intra, (sca[m], spb[m], spa[m]), 1)

        # (3) selection-selection inter (different sel chains)
        m = (sca >= 0) & (scb >= 0) & (sca != scb)
        if np.any(m):
            i = sca[m]; j = scb[m]
            pi = spa[m]; pj = spb[m]
            np.add.at(sel_sel_inter, (i, j, pi, pj), 1)
            if undirected:
                np.add.at(sel_sel_inter, (j, i, pj, pi), 1)

        # (1) reference-selection inter (one in ref, one in sel)
        # Case A: a in ref, b in sel
        m = (rca >= 0) & (scb >= 0)
        if np.any(m):
            np.add.at(ref_sel_inter, (rca[m], scb[m], rpa[m], spb[m]), 1)

        # Case B: a in sel, b in ref  -> store as (ref_chain_of_b, sel_chain_of_a)
        m = (sca >= 0) & (rcb >= 0)
        if np.any(m):
            np.add.at(ref_sel_inter, (rcb[m], sca[m], rpb[m], spa[m]), 1)

        # Ensure ref_ref_inter and sel_sel_inter are inter-chain only (zero diagonal chain blocks)
        if n_ref:
            for c in range(n_ref):
                ref_ref_inter[c, c, :, :] = 0
        if n_sel:
            for c in range(n_sel):
                sel_sel_inter[c, c, :, :] = 0
    
    if selection_chains is None:
        return dict(
            ref_ref_inter=ref_ref_inter,
            ref_intra=ref_intra,
        )
    else:
        return dict(
            ref_sel_inter=ref_sel_inter,
            ref_ref_inter=ref_ref_inter,
            sel_sel_inter=sel_sel_inter,
            ref_intra=ref_intra,
            sel_intra=sel_intra,
        )


def getargs_contact(argv):

    parser = ArgumentParser(prog=prog, description=desc)

    parser.add_argument('-s', '--run-input', type=str, required=True, 
                        help="TPR file containing all information for a simulation run.")
    
    parser.add_argument('-f', '--input', type=str, required=True, 
                    help="XTC file which is taken as input trajectory.")
    
    parser.add_argument('-n', '--index', type=str, required=False,
                        help="Index file containing non-default groups.")
    
    parser.add_argument('-ref', '--reference-group', type=str,
                        help="Reference group of atoms (x axis of the contact map).")
    
    parser.add_argument('-sel', '--selection-group', type=str,
                        help="Selection group of atoms (y axis of the contact map).")
    
    parser.add_argument('-cs', '--cutoff-scheme', type=str, choices=["global", "residue"], required=True,
                        help="Scheme for determining inter-residue contacts.")
    
    parser.add_argument('-c', '--cutoff', type=float, default=0.7,
                        help="Cutoff distance for contact calculation, unit is nanometer.")
    
    parser.add_argument('-cm', '--cutoff-multiplier', type=float, default=1.2,
                        help="Factor which is multiplied to sigma value for residue-wise contact cutoff.")
    
    parser.add_argument('-rd', '--remove-diagonal', type=int, default=2,
                        help="Number of diagonals (i - j <= abs(rd)) to remove in contact map.")
        
    parser.add_argument('-b', '--start-time', type=int,
                        help="Time (ns) of the first frame to calculate.")

    parser.add_argument('-e', '--end-time', type=int,
                        help="Time (ns) of the last frame to calculate.")

    parser.add_argument('-dt', '--delta-time', type=int,
                        help="Intervals (ns) between two calculated frames.")
    
    parser.add_argument('-pbc', '--treat-pbc', action='store_true', default=False,
                        help="Treat broken molecule at periodic boundaries.")
    
    # output map files
          
    parser.add_argument('-ors', '--output-inter-reference-selection', type=str,
                        help="DAT file to write inter-chain contact map between reference and selection groups.")
    
    parser.add_argument('-orr', '--output-inter-reference-reference', type=str,
                        help="DAT file to write inter-chain contact map between reference groups.")
    
    parser.add_argument('-oss', '--output-inter-selection-selection', type=str,
                        help="DAT file to write inter-chain contact map between selection groups.")
            
    parser.add_argument('-or', '--output-intra-reference', type=str,
                        help="DAT file to write intra-chain contact map between reference groups.")
    
    parser.add_argument('-os', '--output-intra-selection', type=str,
                        help="DAT file to write intra-chain contact map between selection groups.")
    
    # output time evolution files

    parser.add_argument('-otrs', '--output-time-inter-reference-selection', type=str,
                        help="XVG file to write time evolution of inter-chain contact numbers between reference and selection groups.")
    
    parser.add_argument('-otrr', '--output-time-inter-reference-reference', type=str,
                        help="XVG file to write time evolution of inter-chain contact numbers between reference groups.")
    
    parser.add_argument('-otss', '--output-time-inter-selection-selection', type=str,
                        help="XVG file to write time evolution of inter-chain contact numbers between selection groups.")
    
    parser.add_argument('-otr', '--output-time-intra-reference', type=str,
                        help="XVG file to write time evolution of intra-chain contact numbers between reference groups.")
    
    parser.add_argument('-ots', '--output-time-intra-selection', type=str,
                        help="XVG file to write time evolution of intra-chain contact numbers between selection groups.")
    
    # output schemes

    parser.add_argument('-intraavg', '--intra-average', action='store_true', default=False,
                        help="Output average intra-chain contact map instead of summing-up maps.")
        
    parser.add_argument('-otype', '--output-type', type=str, choices=["dat", "xpm", "xlsx"], default="dat",
                        help="Type of output files. Possible choice from dat, xpm, and xlsx")
    
    args = parser.parse_args(argv)
    return args

def contact(args):

    # load trajectory into memory

    try:
        trajectory = trajectory_class(args.run_input, args.index, args.input)
    except Exception as exc:
        print("## An exception occurred when trying to open trajectory file %s." % args.input)
        print(f"## Root cause: {exc}")
        quit()
    
    trajectory.Universe.trajectory.add_transformations(wrap(trajectory.Universe.atoms))

    #if args.treat_pbc is True:
    #    print(f"## Distance will be calculated using periodic boundary conditions.")
    #    trajectory.Universe.trajectory.add_transformations(unwrap(trajectory.Universe.atoms))
    #else:
    #    print(f"## WARNING: Distance will be calculated without periodic boundary conditions.")

    # We treat time for analysis and generate frame for analysis

    start_frame, end_frame, interval_frame = trajectory.time2frame(args.start_time, args.end_time, args.delta_time)

    #frame_list = list(range(start_frame, end_frame, interval_frame))

    # We now generate atom groups for calculations

    if args.reference_group is None or args.selection_group is None:
        trajectory.index.print_all()

    if args.reference_group is not None:
        print(f"## Will use group {args.reference_group} as reference group.")
        reference_group, reference_group_name = trajectory.getSelection(f"group {args.reference_group}")
    else:
        reference_group, reference_group_name = trajectory.getSelection_interactive("reference group")

    if args.selection_group is not None:
        print(f"## Will use group {args.selection_group} as selection group.")
        selection_group, selection_group_name = trajectory.getSelection(f"group {args.selection_group}")
    else:
        selection_group, selection_group_name = trajectory.getSelection_interactive("selection group (can be empty)")
    
    ref_length = len(reference_group.indices)
    sel_length = len(selection_group.indices)

    print(f"## Reference group '{reference_group_name}' has {ref_length} atoms.")
    print(f"## Selection group '{selection_group_name}' has {sel_length} atoms.")

    if ref_length == 0:
        print(f"ERROR: Reference group has zero atoms.")
        quit()
    if sel_length == 0:
        print(f"## WARNING: Selection group has zero atoms. Please make sure this is intended.")
        has_selection = False
    else:
        has_selection = True

    # We now split the indices and check chains

    reference_chains = trajectory.index.splitch_indices(reference_group.indices)
    chain_length_list_reference = [len(chain) for chain in reference_chains]
    if len(set(chain_length_list_reference)) != 1:
        print(f"ERROR: Chains in reference groups are not same in length.")
        quit()
    chain_length_reference = chain_length_list_reference[0]    
    print(f"## Processed {len(reference_chains)} chains of length {chain_length_reference} in reference group.")

    if has_selection:
        selection_chains = trajectory.index.splitch_indices(selection_group.indices)
        chain_length_list_selection = [len(chain) for chain in selection_chains]
        if len(set(chain_length_list_selection)) != 1:
            print(f"ERROR: Chains in selection groups are not same in length.")
            quit()
        chain_length_selection = chain_length_list_selection[0]
        print(f"## Processed {len(selection_chains)} chains of length {chain_length_selection} in selection group.")

        # We now test whether the two groups are identical or ovelapping

        c1 = np.array(reference_chains)
        c2 = np.array(selection_chains)

        if np.array_equal(c1, c2):
            print(f"## WARNING: Reference and selection groups are identical.") 
    
        cal_map_bool = any((
            args.output_inter_reference_selection is not None,
            args.output_inter_reference_reference is not None,
            args.output_inter_selection_selection is not None,
            args.output_intra_reference is not None,
            args.output_intra_selection is not None,
        ))
        cal_time_evo_bool = any((
            args.output_time_inter_reference_selection is not None,
            args.output_time_inter_reference_reference is not None,
            args.output_time_inter_selection_selection is not None,
            args.output_time_intra_reference is not None,
            args.output_time_intra_selection is not None,
        ))

    else:
        cal_map_bool = any((
            args.output_inter_reference_reference is not None,
            args.output_intra_reference is not None,
        ))
        cal_time_evo_bool = any((
            args.output_time_inter_reference_reference is not None,
            args.output_time_intra_reference is not None,
        ))
        if any((
            args.output_inter_reference_selection is not None,
            args.output_inter_selection_selection is not None,
            args.output_intra_selection is not None,
        )):
            print(f"ERROR: Selection group is empty, but output files for selection group are specified.")
            quit()
        if any((
            args.output_time_inter_reference_selection is not None,
            args.output_time_inter_selection_selection is not None,
            args.output_time_intra_selection is not None,
        )):
            print(f"ERROR: Selection group is empty, but output time evolution files for selection group are specified.")
            quit()

    if not cal_map_bool and not cal_time_evo_bool:
        print(f"## ERROR: No output files are specified, exiting.")
        quit()
    
    # We determine cutoff scheme and generate cutoff distance vector
    if args.cutoff_scheme == "global":
        print(f"## Will use a glocal cutoff of {args.cutoff} nm.")
        cutoff = args.cutoff * 10.0
    
    elif args.cutoff_scheme == "residue":
        print(f"## Will use a residue responsive cutoff will a sigma multiplier of {args.cutoff_multiplier}")
        cutoff_vector = np.array([args.cutoff_multiplier * sigma * 10.0 for sigma in trajectory.sigmas])
        cutoff = cutoff_vector.max()
    
    else:
        print(f"ERROR: Unknown cutoff scheme {args.cutoff_scheme}.")
        quit()
    
    # Start calculation
    print(f"## We will now start calculating.")

    # Create matrices and lists that will hold contact maps and time evolution data
    
    if cal_map_bool:
        print(f"## Contact maps will be calculated and outputted.")
        # We generate matrices to store contact maps
        contact_map_inter_rr = np.zeros((chain_length_reference, chain_length_reference)) 
        contact_map_intra_r = np.zeros((chain_length_reference, chain_length_reference)) 
        if has_selection:
            contact_map_inter_rs = np.zeros((chain_length_reference, chain_length_selection))         
            contact_map_inter_ss = np.zeros((chain_length_selection, chain_length_selection))         
            contact_map_intra_s = np.zeros((chain_length_selection, chain_length_selection))
    
    if cal_time_evo_bool:
        print(f"## Time evolution of contact numbers will be calculated and outputted.")
        # We generate lists to store time evolution data
        time_evo_inter_rr = list()
        time_evo_intra_r = list()
        if has_selection:
            time_evo_inter_rs = list()        
            time_evo_inter_ss = list()        
            time_evo_intra_s = list()
    
    # We start calculation

    print(f"## Calculating for requested frames.")

    if has_selection:
        search_atom_indices = np.sort(np.asarray(list(set(reference_group.indices).union(set(selection_group.indices))), dtype=np.int64))
        search_atoms = trajectory.Universe.atoms[search_atom_indices]
    else:
        search_atom_indices = np.sort(np.asarray(reference_group.indices, dtype=np.int64))
        search_atoms = trajectory.Universe.atoms[search_atom_indices]

    time_list = list()  

    print(f"## Neighborhood search will be performed with cutoff {cutoff/10.0} nm.")

    print(f"## Processing frames from {start_frame} to {end_frame} with interval {interval_frame}.")

    for ts in tqdm(trajectory.Universe.trajectory[start_frame:end_frame:interval_frame]):

        time_list.append(ts.time / 1000)  # in ns

        pos = search_atoms.positions
        box = ts.dimensions
        ns = FastNS(cutoff, pos, box=box, pbc=args.treat_pbc)

        # drop self-pairs (a,a)
        pairs = ns.self_search().get_pairs()
        pairs_atomid = search_atom_indices[pairs]     

        if args.cutoff_scheme == "residue":

            cutoffs = cutoff_vector[pairs_atomid].mean(axis=1)
            pair_distances = ns.self_search().get_pair_distances()
            mask = pair_distances <= cutoffs
            pairs_atomid = pairs_atomid[mask]

        maps = compute_residue_contact_maps_fixed(
            pairs_atomid,
            reference_chains,
            selection_chains if has_selection else None,
            undirected=True,
            unique_pairs=False,
            drop_self_pairs=True,
            dtype=np.int64,  # usually enough, saves RAM
        )

        ref_intra = maps["ref_intra"]           # (n_ref, L_ref, L_ref)
        ref_ref   = maps["ref_ref_inter"]       # (n_ref, n_ref, L_ref, L_ref) diag blocks are zero
        n_ref, L_ref = ref_intra.shape[0], ref_intra.shape[1]

        # ---------------- Mean maps ----------------
        # (4) Mean intra-chain map for reference (αSyn): (L_ref, L_ref)
        mean_ref_intra = ref_intra.mean(axis=0) if args.intra_average else ref_intra.sum(axis=0)
        mean_ref_intra = remove_near_diagonal(mean_ref_intra, args.remove_diagonal)

        # (2) Mean inter-chain map within reference type (αSyn–αSyn), averaged over ALL i != j
        mask_ref = ~np.eye(n_ref, dtype=bool)              # True for i != j
        blocks_ref_ref = ref_ref[mask_ref]                # (n_ref*(n_ref-1), L_ref, L_ref)
        mean_ref_ref_inter = (
            blocks_ref_ref.sum(axis=0)
            if blocks_ref_ref.size
            else np.zeros((L_ref, L_ref), dtype=ref_ref.dtype)
        )

        if has_selection:

            sel_intra = maps["sel_intra"]           # (n_sel, L_sel, L_sel)
            ref_sel   = maps["ref_sel_inter"]       # (n_ref, n_sel, L_ref, L_sel)
            sel_sel   = maps["sel_sel_inter"]       # (n_sel, n_sel, L_sel, L_sel) diag blocks are zero
            n_sel, L_sel = sel_intra.shape[0], sel_intra.shape[1]

            # (5) Mean intra-chain map for selection (Tau): (L_sel, L_sel)
            mean_sel_intra = sel_intra.mean(axis=0) if args.intra_average else sel_intra.sum(axis=0)
            mean_sel_intra = remove_near_diagonal(mean_sel_intra, args.remove_diagonal)

            # (1) Mean inter-type map (αSyn–Tau), averaged over all chain pairs: (L_ref, L_sel)
            mean_ref_sel = ref_sel.sum(axis=(0, 1))


            # (3) Mean inter-chain map within selection type (Tau–Tau), averaged over ALL i != j
            mask_sel = ~np.eye(n_sel, dtype=bool)              # True for i != j
            blocks_sel_sel = sel_sel[mask_sel]                # (n_sel*(n_sel-1), L_sel, L_sel)
            mean_sel_sel_inter = (
                blocks_sel_sel.sum(axis=0)
                if blocks_sel_sel.size
                else np.zeros((L_sel, L_sel), dtype=sel_sel.dtype)
            )

        # We add these matrices to the global contact maps

        if cal_map_bool:

            contact_map_inter_rr += mean_ref_ref_inter
            contact_map_intra_r += mean_ref_intra

            if has_selection:
                contact_map_inter_rs += mean_ref_sel
                contact_map_inter_ss += mean_sel_sel_inter
                contact_map_intra_s += mean_sel_intra
        
        if cal_time_evo_bool:

            time_evo_inter_rr.append(mean_ref_ref_inter.sum() / 2)
            time_evo_intra_r.append(mean_ref_intra.sum() / 2)

            if has_selection:

                time_evo_inter_rs.append(mean_ref_sel.sum())                
                time_evo_inter_ss.append(mean_sel_sel_inter.sum() / 2)                
                time_evo_intra_s.append(mean_sel_intra.sum() / 2)
        
    print(f"## Calculation completed.")

    # We now output results

    if cal_map_bool:
        if has_selection:
            output_dict = {
                args.output_inter_reference_selection: contact_map_inter_rs / len(time_list),
                args.output_inter_reference_reference: contact_map_inter_rr / len(time_list),
                args.output_inter_selection_selection: contact_map_inter_ss / len(time_list),
                args.output_intra_reference: contact_map_intra_r / len(time_list),
                args.output_intra_selection: contact_map_intra_s / len(time_list)
            }
        else:
            output_dict = {
                args.output_inter_reference_reference: contact_map_inter_rr / len(time_list),
                args.output_intra_reference: contact_map_intra_r / len(time_list)
            }

        for filename_raw in output_dict.keys():
            if filename_raw is None:
                continue
            filename = validate_extension(filename_raw, args.output_type)
            omatrix = output_dict[filename_raw]
            if args.output_type == "dat":
                np.savetxt(filename, omatrix, fmt="%.6f", delimiter="\t")
            if args.output_type == "xlsx":
                from pandas import DataFrame
                DataFrame(omatrix).to_excel(filename, index=False, header=False)
            if args.output_type == "xpm":
                write_xpm(omatrix, filename)
            
    if cal_time_evo_bool:
        xlabel = "Time (ns)"
        if has_selection:
            output_time_dict = {
                args.output_time_inter_reference_selection: (time_evo_inter_rs, f"inter-{reference_group_name}-{selection_group_name}" ),
                args.output_time_inter_reference_reference: (time_evo_inter_rr, f"inter-{reference_group_name}-{reference_group_name}"),
                args.output_time_inter_selection_selection: (time_evo_inter_ss, f"inter-{selection_group_name}-{selection_group_name}"),
                args.output_time_intra_reference: (time_evo_intra_r, f"intra-{reference_group_name}"),
                args.output_time_intra_selection: (time_evo_intra_s, f"intra-{selection_group_name}")
            }
        else:
            output_time_dict = {
                args.output_time_inter_reference_reference: (time_evo_inter_rr, f"inter-{reference_group_name}-{reference_group_name}"),
                args.output_time_intra_reference: (time_evo_intra_r, f"intra-{reference_group_name}")
            }

        for filename_raw in output_time_dict.keys():
            if filename_raw is None:
                continue
            filename = validate_extension(filename_raw, "xvg")
            ydata, legend = output_time_dict[filename_raw]
            ylabel = f"Contact number ({legend})"
            title = f"Time evolution of contact number ({legend})"
            write_xvg(filename, time_list, ydata, title=title, xlabel=xlabel,ylabel=ylabel,legends=[legend])


from dropps.share.command_class import single_command
contact_commands = single_command("contact", getargs_contact, contact, desc)
