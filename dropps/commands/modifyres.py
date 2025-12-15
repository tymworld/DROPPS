# modifyres tool in DROPPS package by Yiming Tang @ Fudan
# Development started on Nov 10 2025

from argparse import ArgumentParser
from copy import deepcopy
from dropps.fileio.itp_reader import read_itp, write_itp, Atomtype, Atom, Bond, Angle
from dropps.fileio.pdb_reader import read_pdb, write_pdb, phrase_pdb_atoms
from dropps.fileio.filename_control import validate_extension
from dropps.share.forcefield import getff, forcefield_list

from openmm.unit import kilojoule_per_mole, nanometer, atomic_mass_unit, degree, radian

from pathlib import Path

import re
import os
import numpy as np

prog = "modifyres"
desc = '''This program add residue modifications for an itp and a pdb file.'''

def find_new_atom_position(coordinates, bond_length, number, n_samples=5000):
    """
    Find a new atom position that:
    (1) Lies at a distance `bond_length` from coordinates[number - 1]
    (2) Is as far as possible from all other atoms in `coordinates`.

    Parameters
    ----------
    coordinates : unit.Quantity of shape (N, 3)
        Coordinates of N atoms, with length units (e.g., nanometers).
    bond_length : unit.Quantity
        Desired bond length (e.g., 0.14 * unit.nanometers).
    number : int
        The atom index (1-based) to which the new atom will be bonded.
    n_samples : int
        Number of random directions to sample on the sphere.

    Returns
    -------
    new_position : unit.Quantity of shape (3,)
        The optimal new atom coordinate in same unit as input.
    """

    # Convert to numpy array in consistent unit (e.g., nanometers)
    #coord_unit = coordinates.unit
    coords = np.array(coordinates)
    r0 = coords[number - 1]

    # Random directions uniformly distributed on the sphere
    phi = np.random.uniform(0, 2 * np.pi, n_samples)
    costheta = np.random.uniform(-1, 1, n_samples)
    theta = np.arccos(costheta)
    directions = np.stack([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta)
    ], axis=1)

    # Candidate points at fixed bond length
    r_candidates = r0 + bond_length * directions

    # Compute minimum distance to all existing atoms for each candidate
    min_distances = np.min(
        np.linalg.norm(r_candidates[:, None, :] - coords[None, :, :], axis=-1),
        axis=1
    )

    # Exclude the reference atom from penalty (since it's bonded)
    d_to_ref = np.linalg.norm(r_candidates - r0, axis=1)
    mask = np.abs(d_to_ref - bond_length) < 1e-6
    min_distances[~mask] = -np.inf

    # Pick the candidate with the largest minimum distance to other atoms
    best_idx = np.argmax(min_distances)
    best_position = r_candidates[best_idx]

    return best_position

def getargs_modifyres(argv):

    parser = ArgumentParser(prog=prog, description=desc)

    parser.add_argument('-ip', '--input-topology', type=str, required=True, 
                        help="Input itp file.")
    
    parser.add_argument('-if', '--input-structure', type=str, required=True, 
                        help="Input pdb file.")

    parser.add_argument('-op', '--output-topology', type=str, required=True, 
                        help="Output itp file with modification added.")

    parser.add_argument('-of', '--output-structure', type=str, required=True, 
                        help="Output pdb file with modification added.")
    
    parser.add_argument('-ff', '--forcefield', choices=forcefield_list, help="Forcefield selection")

    parser.add_argument('-m', '--modifications', type=str, nargs="+", required=True,
                        help="modifications, format: original+number+modified, eg: S129SMP")

    args = parser.parse_args(argv)

    return args

def modifyres(args):

    # We first get forcefield information

    current_file_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    forcefields_dir = Path(current_file_dir) / "share" / "forcefields"
    files1 = list(forcefields_dir.glob("*.ff"))
    cwd = Path.cwd()
    files2 = list(cwd.glob("*.ff"))

    # Get base names for conflict checking
    names1 = set(f.name for f in files1)
    names2 = set(f.name for f in files2)

    # Check for conflicts
    conflicts = names1 & names2
    if conflicts:
        print("ERROR: Conflict detected! The following .ff file(s) exist in both system and working directory:")
        for name in conflicts:
            print(f"  {name}")
        quit()

    # Combine and make a list of paths
    all_files = files1 + files2

    # Print basename list
    if not all_files:
        print("ERROR: No forcefields found.")
        quit()
    else:
        print("## Available forcefields:")
        for idx, f in enumerate(all_files, start=1):
            print(f"{idx}: {f.name}")

        # Let user select

        if args.forcefield is not None:
            filenames = [file for file in all_files if os.path.basename(file) == args.forcefield + ".ff"]

            if len(filenames) == 0:
                print(f"ERROR: Unknown forcefield {args.forcefield}.")
                quit()
            selected_file_path = Path(filenames[0])
        else:

            while True:
                try:
                    choice = int(input("Select a file by index: "))
                    if 1 <= choice <= len(all_files):
                        break
                    else:
                        print("Invalid choice. Try again.")
                except ValueError:
                    print("Please enter a valid integer.")

            selected_file_path = all_files[choice - 1]
        print(f"## Selected forcefield: {selected_file_path.name}")

        # Save path in parameter
        parameter_file_path = selected_file_path

    forcefield = getff(parameter_file_path)

    print(f"## Forcefield successfully processed.")

    # We now get information for modification

    output_itp_name = validate_extension(args.output_topology, "itp")
    output_pdb_name = validate_extension(args.output_structure, "pdb")

    input_itp_name = args.input_topology
    input_pdb_name = args.input_structure

    # We first read the itp and pdb file
    topology_raw = read_itp(input_itp_name)
    print("## Input topology has been readed and loaded to memory.")
    structure_raw, box = read_pdb(input_pdb_name)
    print("## Input structure has been readed and loaded to memory.")
    print(f"################################################################################")
    print(f"## WARNING: This program can only be runned on unmodified protein topologies. ##")
    print(f"################################################################################")  

    # We now process and check the modification list
    modifications = []
    for modification in args.modifications:
        try:
            original, number, modified = re.match(r"([A-Za-z]+)(\d+)([A-Za-z]+)", modification).groups()
            number = int(number)

            if number == 1 or number == len(topology_raw.atoms):
                print(f"ERROR: Currently, cannot add modification to terminal residue.")
                print(f"ERROR: This will be fixed in a future version.")
                quit()

            if not original in forcefield.abbr:
                print(f"ERROR: Residue {original} not recognized in forcefield.")
                quit()
            
            if not modified in forcefield.modifications.keys():
                print(f"ERROR: Modified residue {modified} not recognized in forcefield.")
                quit()
            
            if topology_raw.atoms[number - 1].abbr != original:
                print(f"ERROR: Residue {number} in input topology is {topology_raw.atoms[number - 1]} instead of {original}")
                quit()
            
            modifications.append([original, number, modified])

        except:
            print(f"ERROR: Cannot process modification {modification}.")
            quit()

    # We now generate a list of new atom types.

    topology_new = deepcopy(topology_raw)
    modified_types = list(set([modification[2] for modification in modifications]))

    # We first generate backbone modifications
    for modified_type in modified_types:

        abbr_m = modified_type + "B"
        name_m = forcefield.modifications[modified_type][2] + "_BB"
        sigma_m = forcefield.modifications[modified_type][3][0]
        lambda_m = forcefield.modifications[modified_type][4][0]
        T0_m = topology_raw.atomtypes[topology_raw.typelist.index(forcefield.modifications[modified_type][0])].T0
        T1_m = topology_raw.atomtypes[topology_raw.typelist.index(forcefield.modifications[modified_type][0])].T1
        T2_m = topology_raw.atomtypes[topology_raw.typelist.index(forcefield.modifications[modified_type][0])].T2

        if len(abbr_m) > 4:
            print("ERROR: Atom abbr raw length cannot exceed 3.")
            quit()

        print(f"## Will add atom type {abbr_m}, name: {name_m}, sigma: {sigma_m}, lambda: {lambda_m}, T0/1/2: {T0_m}/{T1_m}/{T2_m}")
        
        topology_new.atomtypes.append(Atomtype(abbr_m, name_m, sigma_m, lambda_m,
                                               T0_m, T1_m, T2_m))
        topology_new.typelist.append(abbr_m)
    
    # We next generate sidechain modifications
    for modified_type in modified_types:

        abbr_m = modified_type + "S"
        name_m = forcefield.modifications[modified_type][2] + "_SC"
        sigma_m = forcefield.modifications[modified_type][3][1]
        lambda_m = forcefield.modifications[modified_type][4][1]
        T0_m = topology_raw.atomtypes[topology_raw.typelist.index(forcefield.modifications[modified_type][0])].T0
        T1_m = topology_raw.atomtypes[topology_raw.typelist.index(forcefield.modifications[modified_type][0])].T1
        T2_m = topology_raw.atomtypes[topology_raw.typelist.index(forcefield.modifications[modified_type][0])].T2

        if len(abbr_m) > 4:
            print("ERROR: Atom abbr raw length cannot exceed 3.")
            quit()
        
        print(f"## Will add atom type {abbr_m}, name: {name_m}, sigma: {sigma_m}, lambda: {lambda_m}, T0/1/2: {T0_m}/{T1_m}/{T2_m}")
        
        topology_new.atomtypes.append(Atomtype(abbr_m, name_m, sigma_m, lambda_m,
                                               T0_m, T1_m, T2_m))
        topology_new.typelist.append(abbr_m)
    
    # We now start modification.

    atom_number_now = len(topology_raw.atoms)

    for [original, number, modified] in modifications:

        print(f"## Will modify residue {original}{number} to {modified}.")
        [ff_original, ff_abbr, ff_aa, ff_sigmas, ff_lambdas, ff_masses,\
          ff_charges, ff_bond_length, ff_bond_k, ff_angle_thetas, ff_angle_ks]\
          = forcefield.modifications[modified]
        
        original_index = number
        original_left_index = original_index - 1
        original_right_index = original_index + 1

        new_index = atom_number_now + 1
        atom_number_now += 1

        # Double check
        if not topology_new.atoms[original_index - 1].abbr == original:
            print(f"ERROR: Atom index {original_index} of the protein is not {original}")
            quit()

        # Add atoms        
        topology_new.atoms[original_index - 1] = Atom(ff_abbr + 'B', ff_aa + "_BB",
                                                      ff_aa, topology_raw.atoms[original_index - 1].residueid,
                                                      ff_masses[0], ff_charges[0])
        
        print(f"## Modified original atom {topology_raw.atoms[original_index - 1].abbr}{original_index} to {topology_new.atoms[original_index - 1].abbr}")

        topology_new.atoms.append(Atom(ff_abbr + 'S', ff_aa + "SC",
                                       ff_aa, topology_raw.atoms[original_index - 1].residueid,
                                       ff_masses[1], ff_charges[1]))
        
        # Double check
        if not len(topology_new.atoms) == new_index:
            print(f"ERROR: Length of new topology is not correct. Please check code.")
            quit()

        print(f"## Added new atom {new_index} {topology_new.atoms[new_index - 1].abbr}")

        # Add bonds
        topology_new.bonds.append(Bond(original_index - 1, new_index - 1, 
                                       ff_bond_length * nanometer, ff_bond_k * kilojoule_per_mole / nanometer ** 2))
        print(f"## Add bond from {original_index} to {new_index}.")

        # Add angles
        if topology_new.angles is None:
            topology_new.angles = list()
        topology_new.angles.append(Angle(original_left_index - 1, original_index - 1, new_index - 1,
                                         ff_angle_thetas[0] * degree, 
                                         ff_angle_ks[0] * kilojoule_per_mole / radian ** 2))
        print(f"## Add angle {original_left_index}-{original_index}-{new_index}, theta={ff_angle_thetas[0]}, k={ff_angle_ks[0]}")
        topology_new.angles.append(Angle(original_right_index - 1, original_index - 1, new_index - 1,
                                         ff_angle_thetas[1] * degree,
                                           ff_angle_ks[1] * kilojoule_per_mole / radian ** 2))
        print(f"## Add angle {original_right_index}-{original_index}-{new_index}, theta={ff_angle_thetas[1]}, k={ff_angle_ks[1]}")
        
    # We now write output itp file.

    write_itp(output_itp_name, topology_new)

    print(f"## Modified itp written to file {output_itp_name}") 

    # We now generate coordinates for new atom and treat structure file
    #structure_raw, box

    structure_new = deepcopy(structure_raw)

    coordinates = [[atom['x'], atom['y'], atom['z']] for atom in structure_raw]
    coordinates_in_nanometer = [[atom['x'].value_in_unit(nanometer), 
                                 atom['y'].value_in_unit(nanometer), 
                                 atom['z'].value_in_unit(nanometer)] 
                                 for atom in structure_raw]
    # Double check
    if not len(coordinates) == len(topology_raw.atoms):
        print(f"ERROR: Length of topology and structure file not match.")
        quit()
    
    for [original, number, modified] in modifications:
        original_coordinate = coordinates[number - 1]
        bond_length_nanometer = forcefield.modifications[modified][7]
        new_coordinate = find_new_atom_position(coordinates_in_nanometer, bond_length_nanometer, number)
        new_coordinate = [value * nanometer for value in new_coordinate]

        [ff_original, ff_abbr, ff_aa, ff_sigmas, ff_lambdas, ff_masses,\
          ff_charges, ff_bond_length, ff_bond_k, ff_angle_thetas, ff_angle_ks]\
          = forcefield.modifications[modified]

        print(f"## New atom {modified}S will be placed around {original}{number} "
              + f"({original_coordinate[0]}, {original_coordinate[1]}, {original_coordinate[2]}), "
              + f"at ({new_coordinate[0]}, {new_coordinate[1]}, {new_coordinate[2]})")
        
        structure_new[number - 1]["name"] = ff_abbr + "B"

        structure_new.append({
                    "serial": len(coordinates) + 1,
                    "name": ff_abbr + "S",
                    "resname": structure_new[number - 1]["resname"],
                    "chain": structure_new[number - 1]["chain"],
                    "resseq": structure_new[number - 1]["resseq"],
                    "x": new_coordinate[0],
                    "y": new_coordinate[1],
                    "z": new_coordinate[2],
                    "occupancy": structure_new[number - 1]["occupancy"],
                    "bfactor": structure_new[number - 1]["bfactor"],
                    "element": structure_new[number - 1]["element"],
                })
        
        coordinates.append(new_coordinate)
        coordinates_in_nanometer.append([new_coordinate[0].value_in_unit(nanometer),
                                         new_coordinate[1].value_in_unit(nanometer),
                                         new_coordinate[2].value_in_unit(nanometer)])
    
    new_pdb_data = phrase_pdb_atoms(structure_new, box)
    write_pdb(output_pdb_name, new_pdb_data.box_size, new_pdb_data.record_names,
              new_pdb_data.serial_numbers, new_pdb_data.atom_names, new_pdb_data.residue_names,
              new_pdb_data.chain_IDs, new_pdb_data.residue_sequence_numbers, new_pdb_data.x_nms,
              new_pdb_data.y_nms, new_pdb_data.z_nms, new_pdb_data.occupancys,
              new_pdb_data.bfactors, new_pdb_data.elements, new_pdb_data.molecule_length_list)
    

    print(f"## Modified pdb written to file {output_pdb_name}") 
    



from dropps.share.command_class import single_command
modifyres_commands = single_command("modifyres", getargs_modifyres, modifyres, desc)