# extract tool in DROPPS package by Yiming Tang @ Fudan
# Development started on Sep 2 2025

from argparse import ArgumentParser
from dropps.share.trajectory import trajectory_class
from dropps.fileio.pdb_reader import write_pdb
from dropps.fileio.filename_control import validate_extension

from math import floor

import numpy as np

from MDAnalysis.analysis import lineardensity as lin

from MDAnalysis.transformations import unwrap
prog = "extract"
desc = '''This program extract a single frame from a HPS trajectory.'''


def getargs_extract(argv):

    parser = ArgumentParser(prog=prog, description=desc)

    parser.add_argument('-s', '--run-input', type=str, required=True, 
                        help="TPR file containing all information for a simulation run.")

    parser.add_argument('-f', '--input', type=str, required=True, 
                    help="XTC file which is taken as input trajectory.")
    
    parser.add_argument('-o', '--output', type=str, required=True, 
                    help="PDB file which to write the single frame.")

    parser.add_argument('-n', '--index', type=str, required=False,
                        help="Index file containing non-default groups.")
    
    #parser.add_argument('-sel', '--selection', type=int, nargs='+',
    #                    help="Groups of atoms to output")
    
    parser.add_argument('-selfit', '--selection-fit', type=int,
                        help="Group of atoms on which the dense phase is determined.")
    
    parser.add_argument('-pbc', '--pbc', type=str, choices=["mol", "atom"], default="mol",
                        help="Mol means make all molecules whole, atom means put all atoms in the box.")
    
    parser.add_argument('-c', '--center', type=str, choices=["x", "y", "z"], 
                        help="Center the dense phase at specified coordinates before extraction.")
    
    parser.add_argument('-dpt', '--dense-phase-threshold', default=0.5, type=float,
                        help="Threshold for dense phase which is a multiplier of the highest density.")
    
    parser.add_argument('-t', '--time', type=int,
                        help="Time (ns) of the frame to extract.")
    
    parser.add_argument('-fm', '--frame', type=int,
                        help="Frame index to extract.")

    args = parser.parse_args(argv)

    return args

def extract(args):

    # load trajectory into memory

    try:
        trajectory = trajectory_class(args.run_input, args.index, args.input)
    except Exception as exc:
        print("## An exception occurred when trying to open trajectory file %s." % args.input)
        print(f"## Root cause: {exc}")
        quit()
    
    if args.time is None and args.frame is None:
        print("ERROR: No frame is specified.")
        quit()
    
    if args.time is not None and args.frame is not None:
        print(f"ERROR: Only one of time and frame can be specified.")
        quit()
    
    if args.frame is not None:
        start_frame = args.frame
        end_frame = args.frame
        center_interval_frame = 0
    else:
        start_frame, end_frame, center_interval_frame = trajectory.time2frame(args.time, args.time, 0)

    
    ts = trajectory.Universe.trajectory[start_frame]
    coordinates = trajectory.Universe.atoms.positions
    box_size = trajectory.Universe.dimensions[0:3] / 10.0

    x = coordinates[:,0]/ 10.0
    y = coordinates[:,1]/ 10.0
    z = coordinates[:,2]/ 10.0

    # We calculate density profile, if centering is requested
    if args.center is not None:
        print(f"## Will use frame {start_frame} to {end_frame} with recentering interval of {center_interval_frame}.")
        print(f"## Will center dense phase along {args.center}-axis.")
    
        if args.selection_fit is None:
            trajectory.index.print_all()
            if args.selection_fit is not None:
                print(f"## Will use group {args.selection_fit} for dense phase determination.")
                fit_group, fit_group_name = trajectory.getSelection(f"group {args.selection_fit}")
            else:
                fit_group, fit_group_name = trajectory.getSelection_interactive("group for dense phase determination")
        
        mass_density_for_center_analyzer = lin.LinearDensity(fit_group, grouping='atoms', binsize=0.5).run(start=start_frame, stop=end_frame+1 )
        axis = getattr(args, "axis", args.center)
        component = getattr(mass_density_for_center_analyzer.results, axis)
        field = "mass_density"

        mass_density_for_center = np.array(getattr(component, field))

        max_density = np.max(mass_density_for_center)
        dense_phase_threshold = args.dense_phase_threshold
        threshold_density = dense_phase_threshold * max_density

        dense_bins = mass_density_for_center > threshold_density

        # Find the continuous dense phase regions considering periodic boundary conditions
        extended_dense_bins = np.concatenate([dense_bins, dense_bins])  # Extend for periodicity
        dense_region_indices = np.where(extended_dense_bins)[0]

        # Find the largest continuous region in the dense bins
        dense_phases = []
        current_phase = [dense_region_indices[0]]
        for idx in dense_region_indices[1:]:
            if idx == current_phase[-1] + 1:
                current_phase.append(idx)
            else:
                dense_phases.append(current_phase)
                current_phase = [idx]
        dense_phases.append(current_phase)
    
        # Map back to the original bins and choose the largest region
        num_bins = len(mass_density_for_center)
        print(f"## There are {num_bins} bins along {axis}-axis.")
        largest_dense_phase = max(dense_phases, key=len)
        dense_phase_center = floor(np.mean(largest_dense_phase) % num_bins) * 0.5  # bin size is 0.5
        

        # Now recenter the trajectory
        
        print(f"## Dense phase threshold set at {dense_phase_threshold:.2f} of max density {max_density:.4f} (i.e., {threshold_density:.4f}).")
        print(f"## Dense phase located at {len(largest_dense_phase)} bins center at {dense_phase_center:.2f} Å along {axis}-axis.")

        axis_index = {"x":0, "y":1, "z":2}[axis]

        axis_max = box_size[axis_index] * 10.0
        axis_min = 0.0
        axis_middle = (axis_max + axis_min) / 2.0

        print(f"## box is the box axis is {axis_min / 10.0:.2f} to {axis_max / 10.0:.2f} nm, middle at {axis_middle / 10.0:.2f} nm.") 
        print(f"## Dense phase center located at {dense_phase_center / 10.0:.2f} nm along {axis}-axis.")

        axis_new_coor = (coordinates[:,axis_index] - dense_phase_center + axis_middle)/10.0

        if axis == "x":
            x = axis_new_coor
        elif axis == "y":
            y = axis_new_coor
        elif axis == "z":
            z = axis_new_coor

    else:
        print(f"## No centering requested, will extract frame directly.")
    
    # Centering ended here

    if args.pbc == "mol":
        print(f"## Molecules will be made whole.")
    elif args.pbc == "atom":
        print(f"## Atoms will be put inside the simulation box.")
        def apply_pbc(coords, box_size):
            return [coord % box_size if coord >= 0 else (coord % box_size + box_size) % box_size
                    for i, coord in enumerate(coords)]
        x = apply_pbc(x, box_size[0])
        y = apply_pbc(y, box_size[1])
        z = apply_pbc(z, box_size[2])

    pdb_raw = trajectory.tpr.pdb_raw

    output_name = validate_extension(args.output, "pdb")

    write_pdb(output_name, box_size, pdb_raw.record_names, pdb_raw.serial_numbers,
              pdb_raw.atom_names, pdb_raw.residue_names, pdb_raw.chain_IDs,
              pdb_raw.residue_sequence_numbers, x, y, z, 
              pdb_raw.occupancys, pdb_raw.bfactors, pdb_raw.elements,
              pdb_raw.molecule_length_list)


from dropps.share.command_class import single_command
extract_commands = single_command("extract", getargs_extract, extract, desc)
