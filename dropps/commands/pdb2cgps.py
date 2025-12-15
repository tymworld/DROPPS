# seq2hoomd tool in cgps.ng package by Yiming Tang @ Fudan
# Development started on June 6 2025

from argparse import ArgumentParser
import re
from tqdm import tqdm

from dropps.share.forcefield import getff, forcefield_list
from dropps.fileio.pdb_reader import distance
import random
import math
from pathlib import Path
import os

def pdb2cgps(args):

    first_residue_index = args.residue_index
    sequence_length = len(args.sequence)

    # Get sequence for this protein/molecule

    print(f"## Raw sequence: {args.sequence}")
    sequence_abbreviation = list(args.sequence)

    if args.post_translational_modification is not None:
        for ptm in args.post_translational_modification:
            match = re.match(r'([A-Za-z]+)(\d+)([A-Za-z]+)', ptm)
            if not match:
                print("ERROR: Cannot process post translational modification " + ptm + ".")
                quit()
            original = match.group(1)
            residue_number = int(match.group(2))
            if residue_number < first_residue_index or residue_number > first_residue_index + sequence_length - 1:
                print("ERROR: Mutated residue %d out of range of %d to %d." 
                    % (residue_number, first_residue_index, first_residue_index + sequence_length - 1))
                quit()
            mutant = match.group(3)[0] + match.group(3)[1:]
            print("## Residue %s%d will be mutated to %s." % (original, residue_number, mutant))
            if sequence_abbreviation[residue_number - first_residue_index] != original:
                print("ERROR When processing post translational modification %s, residue %d is not %s." % (ptm, residue_number, original))
                quit()
            else:
                sequence_abbreviation[residue_number - first_residue_index] = mutant

    if args.charged_NTD:
        print("## N terminal will be patched by an additional positive charge.")
        sequence_abbreviation[0] += '_N'
    else:
        print("## N terminal (mainchain) will be neutral.")

    if args.charged_CTD:
        print("## C terminal will be patched by an additional negative charge.")
        sequence_abbreviation[-1] += '_C' 
    else:
        print("## C terminal (mainchain) will be neutral.")

    print("## Sequence: "+ ','.join(sequence_abbreviation))

    # Define file for structure and topology generation

    degree_extend = float(args.degree_extend)
    if degree_extend < 0 or degree_extend > 1:
        print("ERROR: Degree of extend should be a float within 0 and 1.")
        quit()

    if args.output_conformation is not None:
        conformation_file_prefix = args.output_conformation[0:-4] if ".pdb" in args.output_conformation else args.output_conformation
        
        if args.number == 1:
            conformation_file_name_list = [conformation_file_prefix + ".pdb"]
        else:
            conformation_file_name_list = [conformation_file_prefix + "_%d.pdb" % (i+1) for i in range(args.number)]

        print(f"## Will generate conformation file at " + ', '.join(conformation_file_name_list))

    if args.output_topology is not None:

        topology_file_prefix = args.output_topology[0:-4] if ".itp" in args.output_topology else args.output_topology
        topology_file_name = topology_file_prefix + ".itp"

        print(f"## Will generate topology file at " + topology_file_name)

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

    def generate_chain_conformation_single():

        print(f"## Start generating chain conformation.")
        # chain_succeed is a flag for success of a single chain containing numbers of beads.

        chain_succeed = False

        while not chain_succeed:
            # We first get coordinate for the first amino acid
            beads = [[0, 0, 0]]

            # We get coordinate for the second amino acid
            theta = random.uniform(0, math.pi)
            phi = random.uniform(0, 2 * math.pi)

            atom1 = sequence_abbreviation[0]
            atom2 = sequence_abbreviation[1]
            bondtype = forcefield.abbr2bondtypeindex[f"{atom1}-{atom2}"]
            bondname = forcefield.bondtypes[bondtype]
            bond_length = forcefield.bond2param[bondname]['length'] 
            
            temp_x = bond_length * math.sin(theta) * math.cos(phi)
            temp_y = bond_length * math.sin(theta) * math.sin(phi)
            temp_z = bond_length * math.cos(theta)
            beads.append([temp_x, temp_y, temp_z])

            breaked = False
            # We get coordinate for the remaining amino acids
            for i in tqdm(range(2, len(sequence_abbreviation))):

                if breaked:
                    break

                last_vector = [beads[i-1][0] - beads[i-2][0],
                                beads[i-1][1] - beads[i-2][1],
                                beads[i-1][2] - beads[i-2][2]]
                
                atom1 = sequence_abbreviation[i-1]
                atom2 = sequence_abbreviation[i]
                bondtype = forcefield.abbr2bondtypeindex[f"{atom1}-{atom2}"]
                bondname = forcefield.bondtypes[bondtype]
                bond_length = forcefield.bond2param[bondname]['length'] 

                succeed_single_bead = False
                try_time_single_bead = 0

                while not succeed_single_bead and try_time_single_bead < 50:
                    # We generate a random bead coordinate
                    theta = random.uniform(0, math.pi)
                    phi = random.uniform(0, 2 * math.pi)
                    this_vector = [bond_length * math.sin(theta) * math.cos(phi),
                                    bond_length * math.sin(theta) * math.sin(phi),
                                    bond_length * math.cos(theta)]
                    result_vector = [last_vector[j] * degree_extend + this_vector[j] * (1-degree_extend)
                                        for j in range(3)]
                    result_vector_length = distance(result_vector, [0, 0, 0])
                    result_vector = [j/result_vector_length * bond_length for j in result_vector]
                    temp_coordinate = [beads[i-1][j] + result_vector[j] for j in range(3)]
                    # We check if this bead has overlap with previous beads
                    has_overlap = False
                    for exist_bead in beads[0:-2]:
                        if distance(exist_bead, temp_coordinate) < bond_length * 1.2:
                            has_overlap = True

                    rg_too_big = False
                    if max(temp_coordinate) > args.radius or min(temp_coordinate) < - args.radius:
                        rg_too_big = True

                    # If there is no overlap, we add this bead
                    if (not has_overlap) and (not rg_too_big):
                        beads.append(temp_coordinate)
                        succeed_single_bead = True
                    try_time_single_bead += 1

                if try_time_single_bead >= 50:
                    breaked = True
            
            if len(beads) == len(sequence_abbreviation):
                chain_succeed = True
            
        print("## Successfully generating one chain conformation.")
        return beads

    # We read structure file if given

    if args.input_pdb is None:
        specify_structure_pdb = False
    else:
        specify_structure_pdb = True

        if len(args.input_pdb) < 5 or args.input_pdb[-4:] != ".pdb":
            print(f"## ERROR: The input structure file {args.input_pdb} is not a pdb file.")
            quit()

        try:
            structure_pdb_file = open(args.input_pdb, 'r')
            print(f"## Open structure pdb file {args.input_pdb} which is assumed as all-atom configuration.")
        except:
            print("## An exception occurred when trying to open structure pdb file %s." % args.input_pdb)
            quit()
        
        CA_coordinates = [[float(line[30:38])/10.0, float(line[38:46])/10.0, float(line[46:54])/10.0]
                        for line in structure_pdb_file.readlines()
                        if len(line.strip().split()) >= 8 and line.strip().split()[2] == "CA"]
        
        #print(CA_coordinates)
        
        if len(CA_coordinates) != sequence_length:
            print(f"## ERROR: There are {len(CA_coordinates)} CA atoms in {args.input_pdb} but sequence has length of {sequence_length}.")
            quit()
        
    # We first generate conformations for this chain and write to PDB file.
    # Remember PDB file is in angstrom format instead of nanometer.

    if args.output_conformation is not None:
        for conformation_file in conformation_file_name_list:
            print(f"## Generating conformation for {conformation_file}.")

            box_size = args.radius * 2
            atom_positions = CA_coordinates if specify_structure_pdb else generate_chain_conformation_single()

            with open(conformation_file, 'w') as pdb_file:
                pdb_file.write(f"CRYST1{box_size*10:9.3f}{box_size*10:9.3f}{box_size*10:9.3f}  90.00  90.00  90.00 P 1           1\n")

                for i, (abbr, pos) in enumerate(zip(sequence_abbreviation, atom_positions), 1):
                    x, y, z = pos

                    # PDB atom line: fixed-width format
                    # Columns: https://www.wwpdb.org/documentation/file-format-content/format33/sect9.html#ATOM
                    pdb_file.write(
                        "{:<6s}{:>5d} {:<4s} {:>3s} {:1s}{:>4d}    "
                        "{:>8.3f}{:>8.3f}{:>8.3f}{:6.2f}{:6.2f}          {:>2s}\n".format(
                            "ATOM",      # Record name
                            i + first_residue_index - 1,           # Atom serial number
                            abbr[:4],    # Atom name, left aligned, max 4 chars
                            forcefield.abbr2aa[abbr][:3],       # Residue name
                            "A",         # Chain ID
                            i,           # Residue sequence number
                            (x + box_size / 2)*10, (y + box_size / 2)*10, (z + box_size / 2)*10,     # Coordinates
                            1.00, 0.00,  # Occupancy, B-factor
                            abbr[-1] if len(abbr) == 1 else abbr[0]  # Element symbol (fallback)
                        )
                    )

                pdb_file.write("END\n")


    # We next generate topology for this chain and write to itp file.

    if args.output_topology is not None:
        with open(topology_file_name, 'w') as itp_file:

            print(f"## Generating topology to {topology_file_name}.")

            # Write forcefield and information lines
            print(f"## The molecule will be named as {args.output_name} in topology file.")
            itp_file.write(f"[ moleculetype ]\n; molname  nrexcl\n")
            itp_file.write(f"{args.output_name}     1\n\n")

            # Write atom type information
            itp_file.write(f"[ atomtypes ]\n")
            itp_file.write(f"; atom-abbr     atom-name  sigma   lambda   T0               T1              T2\n")
            for abbr in set(sequence_abbreviation):
                itp_file.write(f"  {abbr:12s}  {forcefield.abbr2aa[abbr][:3]:7s}   "
                            + f"{forcefield.abbr2sigma[abbr]:>6.3f}   {forcefield.abbr2lambda[abbr]:.3f}   "
                            + f"{forcefield.abbr2tempcoff[abbr][0]:>13.9f}   "
                            + f"{forcefield.abbr2tempcoff[abbr][1]:>13.9f}   "
                            + f"{forcefield.abbr2tempcoff[abbr][2]:>13.9f}   \n")
            itp_file.write("\n")

            # Write atom information
            itp_file.write(f"[ atoms ]\n")
            itp_file.write(f"; id   atom-abbr  atom-name     residue  resid  mass    charge\n")
            qtot = 0
            for index, abbr in enumerate(sequence_abbreviation):
                qtot += forcefield.abbr2charge[abbr]
                itp_file.write(f"  {index + 1:<3d}  {abbr:<8s}   {forcefield.abbr2aa[abbr]:12s}  {forcefield.abbr2aa[abbr][:3]:7s}  {index + first_residue_index:<5d}  "
                            + f"{forcefield.abbr2mass[abbr]:>6.2f}  {forcefield.abbr2charge[abbr]:5.2f}   ; qtot {qtot}\n")
            itp_file.write("\n")

            # Write bond information
            itp_file.write(f"[ bonds ]\n")
            itp_file.write(f"; ai   aj   r0    k\n")

            for index in range(len(sequence_abbreviation) - 1):

                bondparameters = forcefield.bond2param[
                    forcefield.abbr2bondtype[f"{sequence_abbreviation[index]}-{sequence_abbreviation[index + 1]}"]
                    ]
                itp_file.write(f"  {(index + 1):<3d}  {(index + 2):<3d}  {bondparameters["length"]:.2f}  {bondparameters["k"]}\n")
            itp_file.write("\n")

prog = "seq2cgps"
desc = '''This program create pdb and top files for a given sequence.'''

def getargs_pdb2cgps(argv):

    # Command line argument parser

    parser = ArgumentParser(prog=prog, description=desc)

    parser.add_argument('-s', '--sequence', type=str, help='Protein sequence', required=True)
    parser.add_argument('-f', '--input-pdb', type=str, help="PDB file containing structure information of a chain.")
    parser.add_argument('-ri', '--residue-index', type=int, help='Residue number of the first residue',
                        default=1)
    parser.add_argument('-ptm', '--post-translational-modification', type=str, nargs="+", 
                        help='Post translational modification to add, format: original+number+modified, eg: S129S')
    parser.add_argument('-r', '--radius', type=float, help='Maximum radius of gyration of generated chain', 
                        default=2.0)
    parser.add_argument('-n', '--number', type=int, help='Number of conformations to generate',
                        default=1)
    parser.add_argument('-e', '--degree-extend', type=float, help="Degree of extend for generated chain (0~1)",
                        default = 0.5)
    parser.add_argument('-ff', '--forcefield', choices=forcefield_list, help="Forcefield selection")
    parser.add_argument('-oc', '--output-conformation', type=str, help="File prefix to write output configuration",
                        required=False)
    parser.add_argument('-op', '--output-topology', type=str, help="File prefix to write topology file",
                        required=False)
    parser.add_argument('-on', '--output-name', type=str, default="MOL", help="Molecule name.")
    parser.add_argument('-cNTD', '--charged-NTD', action='store_true', default=False, 
                        help="Whether N terminal is patched by an additional positive charge.")
    parser.add_argument('-cCTD', '--charged-CTD', action='store_true', default=False, 
                        help="Whether C terminal is patched by an additional negative charge.")

    args = parser.parse_args(argv)

    return args

from dropps.share.command_class import single_command
pdb2cgps_commands = single_command("pdb2dps", getargs_pdb2cgps, pdb2cgps, desc)
