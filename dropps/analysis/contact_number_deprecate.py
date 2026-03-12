# Contact number calculation tool in DROPPS package by Yiming Tang @ Fudan
# Development started on July 21 2025

from argparse import ArgumentParser
from dropps.share.trajectory import trajectory_class
from dropps.fileio.filename_control import validate_extension
from dropps.fileio.xvg_reader import write_xvg

from MDAnalysis.analysis import distances

import numpy as np
from tqdm import tqdm
from itertools import product

import warnings
from Bio import BiopythonDeprecationWarning

warnings.filterwarnings("ignore", category=BiopythonDeprecationWarning)

from MDAnalysis.transformations import unwrap

prog = "contact"
desc = '''This program calculate the intra-chain contact numbers between two groups.'''

def getargs_contact_number(argv):

    parser = ArgumentParser(prog=prog, description=desc)

    parser.add_argument('-s', '--run-input', type=str, required=True, 
                        help="TPR file containing all information for a simulation run.")
    
    parser.add_argument('-f', '--input', type=str, required=True, 
                    help="XTC file which is taken as input trajectory.")
    
    parser.add_argument('-n', '--index', type=str, required=False,
                        help="Index file containing non-default groups.")
    
    parser.add_argument('-ref', '--reference', type=int, nargs='+',
                        help="Reference groups contaning atoms to calculate contact from.")
    
    parser.add_argument('-sel', '--selection', type=int, nargs='+',
                        help="Selection groups contaning atoms to calculate contact to.")
    
    parser.add_argument('-t', '--type', type=str, choices=["inter", "intra"], default="inter",
                        help="Calculate inter-chain or intra-chain interactions.")
    
    parser.add_argument('-c', '--cutoff', type=float, default=0.7,
                        help="Cutoff distance for contact calculation.")
    
    parser.add_argument('-ne', '--number-of-exclusion', type=int, default=0,
                        help="Contacts between residues connected by what number of bonds will be ignored.")
    
    parser.add_argument('-b', '--start-time', type=int,
                        help="Time (ns) of the first frame to calculate.")

    parser.add_argument('-e', '--end-time', type=int,
                        help="Time (ns) of the last frame to calculate.")

    parser.add_argument('-dt', '--delta-time', type=int,
                        help="Intervals (ns) between two calculated frames.")
    
    parser.add_argument('-pbc', '--treat-pbc', action='store_true', default=False,
                        help="Treat broken molecule at periodic boundaries.")

    parser.add_argument('-o', '--output', type=str, required=True,
                        help="Output of contact number as a function of time.")
    
    args = parser.parse_args(argv)
    return args

def contact_number(args):

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

    if args.reference is not None:
        print(f"## Will use group {args.reference} to calculate contact from.")
        reference = trajectory.getSelection(f"group {args.reference}")
        reference_name = f"group{args.reference}"
    else:
        trajectory.index.print_all()
        reference, reference_name = trajectory.getSelection_interactive("contact from this group")
    
    if args.selection is not None:
        print(f"## Will use group {args.selection} to calculate contact to.")
        selection = trajectory.getSelection(f"group {args.selection}")
        selection_name = f"group{args.selection}"
    else:
        trajectory.index.print_all()
        selection, selection_name = trajectory.getSelection_interactive("contact to this group")
    
    c1 = reference.indices
    c2 = selection.indices

    if np.array_equal(c1, c2) and np.intersect1d(c1, c2).size > 0:
        print(f"ERROR: Reference and selection groups are overlapping but not identical. Please check.")
        quit()
    
    if args.type == "intra":
        if np.array_equal(c1, c2):
            print(f"ERROR: Reference and selection groups are identical. Intra-chain contact numbers will be meaningless.")
            quit()
    
    # We now split the indices
    reference_splitchains = trajectory.index.splitch_indices(reference.indices)
    selection_splitchains = trajectory.index.splitch_indices(selection.indices)

    combined_indices = sorted(list(set(reference.indices.tolist() + selection.indices.tolist())))
    new_indices = list(range(len(combined_indices)))
    old2new_indices = {combined_indices[i]: new_indices[i] 
                       for i in range(len(combined_indices))}
    new2old_indices = {new_indices[i]: combined_indices[i]
                       for i in range(len(combined_indices))}
    
    reference_splitchains_new = [[old2new_indices[i] for i in chain] for chain in reference_splitchains]
    selection_splitchains_new = [[old2new_indices[i] for i in chain] for chain in selection_splitchains]

    if len(set([len(chain) for chain in reference_splitchains_new])) > 1:
        print(f"ERROR: At least two chains in reference group has different number of residues.")
        quit()

    if len(set([len(chain) for chain in selection_splitchains_new])) > 1:
        print(f"ERROR: At least two chains in selection group has different number of residues.")
        quit() 
    
    n_residue_per_chain_reference = len(reference_splitchains_new[0])
    n_residue_per_chain_selection = len(selection_splitchains_new[0])
    n_chain_reference = len(reference_splitchains_new)
    n_chain_selection = len(selection_splitchains_new)

    if args.type == "intra" and n_chain_reference != n_chain_selection:
        print(f"ERROR: Number of chains in reference and selection not equal.")
        quit()

    print(f"## Identified {n_residue_per_chain_reference} residues within each of the {n_chain_reference} chains in reference.")
    print(f"## Identified {n_residue_per_chain_selection} residues within each of the {n_chain_selection} chains in selection.")

    print(f"## Residues spliited by at most {args.number_of_exclusion} bonds will be excluded from calculations.")

    # We start calculation

    #contact_map = np.zeros((len(combined_indices), len(combined_indices)))

    print(f"## Raw data will contains contact maps between {len(combined_indices)}*{len(combined_indices)} pairs.")
    print(f"## Cutoff distance for contacts set as {args.cutoff}")
    print(f"## Calculating for requested frames.")

    calculation_atomGroup = trajectory.Universe.atoms[combined_indices]

    inter_chain_contact_number = list()
    intra_chain_contact_number = list()

    time_list = list()
    print(f"## Will calculating {len(frame_list)} frames with {len(frame_list)} iteractions listed below.")

    for ts_index, ts in tqdm(enumerate(trajectory.Universe.trajectory[frame_list])):

        time_list.append(ts.time / 1000)

        distance_map = distances.distance_array(calculation_atomGroup, calculation_atomGroup, trajectory.Universe.dimensions)

        # We calculate inter-chain contact numbers

        inter_chain_number_temp = sum(
            np.sum(distance_map[np.ix_(ref, sel)])
            for i, ref in enumerate(reference_splitchains_new)
            for j, sel in enumerate(selection_splitchains_new)
            if i != j
            )
        
        # We calculate intra-chain contact numbers

        intra_chain_number_temp = sum(
            distance_map[i, j]
            for chain in reference_splitchains_new
            for i, j in product(chain, repeat=2)
            if abs(trajectory.get_resID(new2old_indices[i]) - trajectory.get_resID(new2old_indices[j])) > args.number_of_exclusion
            )

        intra_chain_number_temp = 0
        for chain_id in range(len(reference_splitchains_new)):
            for atom_index_reference in reference_splitchains_new[chain_id]:
                for atom_index_selection in reference_splitchains_new[chain_id]:
                    if abs(trajectory.get_resID(new2old_indices[atom_index_reference]) - trajectory.get_resID(new2old_indices[atom_index_selection])) \
                        > args.number_of_exclusion:
                        intra_chain_number_temp += distance_map[atom_index_reference][atom_index_selection]
        
        inter_chain_contact_number.append(inter_chain_number_temp)
        intra_chain_contact_number.append(intra_chain_number_temp)
    
    print(f"## Calculation ended.")
    print(f"## We will write time evolutions of contact numbers to files.")

    if args.type == "intra":
        print(f"WARNING: Intra-chain contacts will be normalized by dividing chain number.")
    else:
        print(f"WARNING: Inter-chain contacts will NOT be normalized.")

    output_filename = validate_extension(args.output, "xvg")

    contact_arrays = np.array(inter_chain_contact_number) if args.type == "inter" else np.array(intra_chain_contact_number) / n_chain_reference
    
    xlabel = "Time (ns)"
    ylabel = "Contact number"
    title = "Inter-chain contact number (not normalized)" if args.type == "inter" else "Intra-chain contact number"

    write_xvg(output_filename, time_list, contact_arrays, title=title,
              xlabel=xlabel, ylabel=ylabel)



from dropps.share.command_class import single_command
contact_number_commands = single_command("contact", getargs_contact_number, contact_number, desc)



