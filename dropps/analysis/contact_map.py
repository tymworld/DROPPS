# contact map in DROPPS package by Yiming Tang @ Fudan
# Development started on July 18 2025

from argparse import ArgumentParser
from dropps.share.trajectory import trajectory_class
from dropps.fileio.filename_control import validate_extension
from dropps.fileio.xpm_reader import write_xpm

from tqdm import tqdm
import numpy as np

from MDAnalysis.analysis import distances

import warnings
from Bio import BiopythonDeprecationWarning

warnings.filterwarnings("ignore", category=BiopythonDeprecationWarning)

from MDAnalysis.transformations import unwrap

prog = "cmap"
desc = '''This program calculate contact map in a residue-level resolution.'''

def getargs_contactmap(argv):

    parser = ArgumentParser(prog=prog, description=desc)

    parser.add_argument('-s', '--run-input', type=str, required=True, 
                        help="TPR file containing all information for a simulation run.")
    
    parser.add_argument('-f', '--input', type=str, required=True, 
                    help="XTC file which is taken as input trajectory.")
    
    parser.add_argument('-n', '--index', type=str, required=False,
                        help="Index file containing non-default groups.")
    
    parser.add_argument('-ref', '--reference-group', type=int,
                        help="Reference group of atoms (x axis of the contact map).")
    
    parser.add_argument('-sel', '--selection-group', type=int,
                        help="Selection group of atoms (y axis of the contact map).")
    
    parser.add_argument('-c', '--cutoff', type=float, default=0.7,
                        help="Cutoff distance for contact calculation.")
    
    parser.add_argument('-b', '--start-time', type=int,
                        help="Time (ns) of the first frame to calculate.")

    parser.add_argument('-e', '--end-time', type=int,
                        help="Time (ns) of the last frame to calculate.")

    parser.add_argument('-dt', '--delta-time', type=int,
                        help="Intervals (ns) between two calculated frames.")
    
    parser.add_argument('-pbc', '--treat-pbc', action='store_true', default=False,
                        help="Treat broken molecule at periodic boundaries.")
          
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
    
    parser.add_argument('-otype', '--output-type', type=str, choices=["dat", "xpm", "xlsx"], default="dat",
                        help="Type of output files. Possible choice from dat, xpm, and xlsx")
    
    args = parser.parse_args(argv)
    return args


def compute_contact_maps(contact_map, reference_chains, selection_chains):

    identical = np.array_equal(np.array(reference_chains), np.array(selection_chains))

    # Assume all chains are the same length
    len_ref = len(reference_chains[0])
    len_sel = len(selection_chains[0])
    n_ref = len(reference_chains)
    n_sel = len(selection_chains)

    # Initialize outputs
    contact_ref_sel = np.zeros((len_ref, len_sel))
    contact_ref_ref = np.zeros((len_ref, len_ref))
    contact_sel_sel = np.zeros((len_sel, len_sel))
    intra_ref = np.zeros((len_ref, len_ref))
    intra_sel = np.zeros((len_sel, len_sel))
    
    # Inter-chain contact map between reference chains
    for i in range(n_ref):
        for j in range(i+1, n_ref):
            contact_ref_ref += contact_map[np.ix_(reference_chains[i], reference_chains[j])]
            contact_ref_ref += contact_map[np.ix_(reference_chains[j], reference_chains[i])]  # symmetry
    contact_ref_ref /= n_ref

    # Inter-chain contact map between selection chains
    for i in range(n_sel):
        for j in range(i+1, n_sel):
            contact_sel_sel += contact_map[np.ix_(selection_chains[i], selection_chains[j])]
            contact_sel_sel += contact_map[np.ix_(selection_chains[j], selection_chains[i])]
    contact_sel_sel /= n_sel

    # Intra-chain contact map of reference
    for chain in reference_chains:
        intra_ref += contact_map[np.ix_(chain, chain)]
    intra_ref /= n_ref

    # Intra-chain contact map of selection
    for chain in selection_chains:
        intra_sel += contact_map[np.ix_(chain, chain)]
    inter_sel /= n_sel

    # Inter-chain contact map between reference and selection
    if identical:
        contact_ref_sel = contact_ref_ref
    else:
        for ref_chain in reference_chains:
            for sel_chain in selection_chains:
                contact_ref_sel += contact_map[np.ix_(ref_chain, sel_chain)]

    return contact_ref_sel, contact_ref_ref, contact_sel_sel, intra_ref, intra_sel

def contactmap(args):

    output_parameters = [args.output_inter_reference_selection,
                         args.output_inter_reference_reference,
                         args.output_inter_selection_selection,
                         args.output_intra_reference,
                         args.output_intra_selection]
    
    if all(x is None for x in output_parameters):
        print("ERROR: No output file specified.")
        quit()

    # load trajectory into memory

    try:
        trajectory = trajectory_class(args.run_input, args.index, args.input)
    except:
        print("## An exception occurred when trying to open trajectory file %s." % args.input)
        quit()

    if args.treat_pbc is True:
        print(f"## Distance will be calculated using periodic boundary conditions.")
        trajectory.Universe.trajectory.add_transformations(unwrap(trajectory.Universe.atoms))
    else:
        print(f"## WARNING: Distance will be calculated without periodic boundary conditions.")

    # We treat time for analysis and generate frame for analysis

    start_frame, end_frame, interval_frame = trajectory.time2frame(args.start_time, args.end_time, args.delta_time)

    frame_list = list(range(start_frame, end_frame, interval_frame))

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
        selection_group, selection_group_name = trajectory.getSelection_interactive("selection group")
    
    # We now split the indices

    reference_chains = trajectory.index.splitch_indices(reference_group.indices)
    selection_chains = trajectory.index.splitch_indices(selection_group.indices)

    # We check chains
    chain_length_list_reference = [len(chain) for chain in reference_chains]
    chain_length_list_selection = [len(chain) for chain in selection_chains]

    
    if len(set(chain_length_list_reference)) != 1:
        print(f"ERROR: Chains in reference groups are not same in length.")
        quit()
    if len(set(chain_length_list_selection)) != 1:
        print(f"ERROR: Chains in selection groups are not same in length.")
        quit()
    
    chain_length_reference = chain_length_list_reference[0]
    chain_length_selection = chain_length_list_selection[0]
    print(f"## Processed {len(reference_chains)} chains of length {chain_length_reference} in reference group.")
    print(f"## Processed {len(selection_chains)} chains of length {chain_length_selection} in selection group.")

    # We now test whether the two groups are identical or ovelapping

    c1 = np.array(reference_chains)
    c2 = np.array(selection_chains)

    if np.array_equal(c1, c2):
        print(f"## WARNING: Reference and selection groups are identical.")
    
    # We start calculation

    print(f"## We will not start calculating.")

    contact_map = np.zeros((trajectory.num_atoms(), trajectory.num_atoms()))

    print(f"## Raw data will contains contact maps between {trajectory.num_atoms()}*{trajectory.num_atoms()} pairs.")
    print(f"## Cutoff distance for contacts set as {args.cutoff}")
    print(f"## Calculating for requested frames.")

    for frame_index in tqdm(frame_list):
        distance_map = distances.distance_array(trajectory.Universe.trajectory[frame_index], 
                                                trajectory.Universe.trajectory[frame_index], trajectory.Universe.dimensions)
        contact_map_temp = (distance_map < args.cutoff).astype(int)
        contact_map += contact_map_temp

    contact_map = contact_map / len(frame_list) 

    print(f"## Raw {trajectory.num_atoms()}*{trajectory.num_atoms()} contact map calculation completed.")

    contact_ref_sel, contact_ref_ref, contact_sel_sel, intra_ref, intra_sel = \
        compute_contact_maps(contact_map, reference_chains, selection_chains)
        
    if args.output_inter_reference_selection is not None:
        if not np.array_equal(c1, c2):     
            print(f"## IMPORTANT NOTE: Inter-chain contacts between reference and selection will not be normalized.")
    
    output_dict = {
        args.output_inter_reference_selection: contact_ref_sel,
        args.output_inter_reference_reference: contact_ref_ref,
        args.output_inter_selection_selection: contact_sel_sel,
        args.output_intra_reference: intra_ref,
        args.output_intra_selection: intra_sel
    }

    output_dict = {key:output_dict[key] for key in output_dict.keys() if key is not None}

    print(f"## Generating output files.")

    for filename_raw in output_dict.keys():
        filename = validate_extension(filename_raw, args.output_type)
        omatrix = output_dict[filename_raw]
        if args.output_type == "dat":
            np.savetxt(filename, omatrix, fmt="%.3f", delimiter="\t")
        if args.output_type == "xlsx":
            from pandas import DataFrame
            DataFrame(omatrix).to_excel(filename, index=False, header=False)
        if args.output_type == "xpm":
            write_xpm(omatrix, filename)

from dropps.share.command_class import single_command
contactmap_commands = single_command("cmap", getargs_contactmap, contactmap, desc)
