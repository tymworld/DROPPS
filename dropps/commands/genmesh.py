# genmesh tool in cgps.ng package by Yiming Tang @ Fudan
# Development started on June 6 2025

from argparse import ArgumentParser
import random
import numpy as np
import itertools
from dropps.fileio.pdb_reader import read_pdb, write_pdb
import itertools
import string
from dropps.fileio.itp_reader import read_itp

from openmm.unit import nanometer

def genmesh(args):

    if len(args.structure) != len(args.topology) or len(args.structure) != len(args.number):
        print("ERROR: Number of structure, topology, and number not match.")
        quit()


    # We first get all pdb files
    if len(args.structure) != len(args.number):
        print("ERROR: Number of structure files and structure numbers are not equal.")
        quit()
    structure_number = len(args.structure)
    print(f"## Will process {structure_number} structure files.")

    try:
        molecule_list = [read_pdb(args.structure[index])[0] for index in range(structure_number)]
        molecule_number_list = args.number
    except:
        print("## An exception occurred when trying to open structure file.")
        quit()

    for molecule_index in range(structure_number):
        print("## Molecule %d contains %d residues, will be inserted for %d times." \
            % ((molecule_index + 1), len(molecule_list[molecule_index]), molecule_number_list[molecule_index]))

    ## We now generate the mesh
    molecule_number = np.sum(molecule_number_list)

    if args.mesh is not None:
        mesh = args.mesh
        if mesh[0] * mesh[1] * mesh[2] < molecule_number:
            print(f"ERROR: The required mesh {mesh[0]}*{mesh[1]}*{mesh[2]} cannot contain {molecule_number} molecules.")
            quit()
    else:
        mesh_x_y_z = int(np.ceil(np.power(molecule_number, 1/3)))
        mesh = [mesh_x_y_z, mesh_x_y_z, mesh_x_y_z]

    print(f"## The mesh for {molecule_number} molecule insertion will be {mesh[0]}*{mesh[1]}*{mesh[2]}.")

    # We first get the radius of each molecule

    diameter = np.ceil(
        np.max([
            [np.max([atom["x"].value_in_unit(nanometer) for atom in molecule]) - np.min([atom["x"].value_in_unit(nanometer) for atom in molecule]) for molecule in molecule_list],
            [np.max([atom["y"].value_in_unit(nanometer) for atom in molecule]) - np.min([atom["y"].value_in_unit(nanometer) for atom in molecule]) for molecule in molecule_list],
            [np.max([atom["z"].value_in_unit(nanometer) for atom in molecule]) - np.min([atom["z"].value_in_unit(nanometer) for atom in molecule]) for molecule in molecule_list]
        ])
    )

    print(f"## The maximum diameter for all configurations is {diameter}.")
    print(f"## The minimun distance between wach two configuration is {args.gap}.")
    distance_between_mesh_point = np.ceil(diameter + args.gap)
    print(f"## The distance between each two adjancy mesh point is {distance_between_mesh_point}.")

    # We now determine the size of the box

    box_size_mesh = [int(np.ceil(mesh[index] * distance_between_mesh_point + args.gap)) for index in range(3)]
    print(f"## The box size determined by mesh is larger than {box_size_mesh[0]}*{box_size_mesh[1]}*{box_size_mesh[2]}")
    box_size = [float(max(box_size_mesh[0], args.minimum_x)),
                float(max(box_size_mesh[1], args.minimum_y)),
                float(max(box_size_mesh[2], args.minimum_z))]
    print(f"## The box size is set as {box_size[0]}*{box_size[1]}*{box_size[2]}")

    # We then generate a list for mesh point (centroid of each molecule)

    mesh_points_raw = np.array([
        [diameter / 2 + (diameter + args.gap) * x_index,
        diameter / 2 + (diameter + args.gap) * y_index,
        diameter / 2 + (diameter + args.gap) * z_index]
        for x_index in range(mesh[0])
        for y_index in range(mesh[1])
        for z_index in range(mesh[2])
    ])


    if args.shuffle:
        mesh_points_raw = np.random.permutation(mesh_points_raw)
        print("## Insertion will be performed on a randomly shuffuled mesh.")
    else:
        print("## Insertion will be performed on left-to-right, down-to-up mode.")

    mesh_points = mesh_points_raw - np.mean(mesh_points_raw, axis=0)

    # We now generate information for all atoms

    record_name = list()
    serial_number = list()
    atom_name = list()
    residue_name = list()
    chain_ID = list()
    residue_sequence_number = list()
    x_nm = list()
    y_nm = list()
    z_nm = list()
    occupancy = list()
    b_factor = list()
    element_symbol = list()

    molecule_index = 0
    atom_index = 1

    labels = [''.join(p) for n in range(1, 3) for p in itertools.product(string.ascii_uppercase, repeat=n)]

    molecule_length_list = list()
    for molecule_type_id in range(len(molecule_list)):
        molecule = molecule_list[molecule_type_id]
        molecule_number = molecule_number_list[molecule_type_id]

        for molecule_id in range(molecule_number):

            molecule_length_list.append(len(molecule))
            mesh_center = mesh_points[molecule_index]
            print(f"## Will write molecule {molecule_type_id + 1} (replica {molecule_id + 1}) on mesh point {mesh_center}.")

            for resindex, atom in enumerate(molecule):
                record_name.append("ATOM")
                serial_number.append(atom_index)
                atom_name.append(atom["name"])
                residue_name.append(atom["resname"])
                chain_ID.append(labels[0])
                residue_sequence_number.append(resindex + 1)
                x_nm.append(atom["x"].value_in_unit(nanometer) + mesh_center[0] + box_size[0] / 2) * nanometer    
                y_nm.append(atom["y"].value_in_unit(nanometer) + mesh_center[1] + box_size[0] / 2) * nanometer
                z_nm.append(atom["z"].value_in_unit(nanometer) + mesh_center[2] + box_size[0] / 2)   * nanometer 
                occupancy.append(1)
                b_factor.append(0)
                element_symbol.append(atom["element"])   

                atom_index += 1
            labels = labels[1:]
            molecule_index += 1

    # We now write pdb file
    conformation_file_prefix = args.output_conformation[0:-4] if ".pdb" in args.output_conformation else args.output_conformation
    conformation_file_name = conformation_file_prefix + ".pdb"

    write_pdb(conformation_file_name, box_size, record_name, serial_number, atom_name, residue_name,
            chain_ID, residue_sequence_number, x_nm, y_nm, z_nm, occupancy, b_factor, element_symbol, molecule_length_list)

    # We now write topology file

    output_topology_file_prefix = args.output_topology[0:-4] if ".top" in args.output_topology else args.output_topology
    output_topology_filename = output_topology_file_prefix + ".top"

    topologies = [read_itp(filename) for filename in args.topology]
    molecule_names = [top.molecule_name for top in topologies]

    output_topology_file = open(output_topology_filename, 'w')

    output_topology_file.write(f"\n[ itp files ]")

    topology_filename_seen = set()
    topology_filename_remove_duplicate = [x for x in args.topology if not (x in topology_filename_seen or topology_filename_seen.add(x))]

    for filename in topology_filename_remove_duplicate:
        output_topology_file.write(f"\n{filename}")
    output_topology_file.write(f"\n")

    output_topology_file.write(f"\n[ system ]")
    output_topology_file.write(f"\n# molecule   number")

    if len(set(molecule_names)) < len(molecule_names):
        print("###############################################################")
        print("## WARNING: Duplicate molecule name found in topology files. ##")
        print("## These files will be merged.                               ##")
        print("## Please make sure nothing is wrong.                        ##")
        print("###############################################################")

    new_molecule_name = []
    new_molecule_number_list = []

    prev_name = None
    accumulated = 0

    for name, num in zip(molecule_names, molecule_number_list):
        if name == prev_name:
            accumulated += num
        else:
            if prev_name is not None:
                new_molecule_name.append(prev_name)
                new_molecule_number_list.append(accumulated)
            prev_name = name
            accumulated = num

    # Don't forget the last group
    if prev_name is not None:
        new_molecule_name.append(prev_name)
        new_molecule_number_list.append(accumulated)

    for molecule_id in range(len(new_molecule_name)):
        output_topology_file.write(f"\n  {new_molecule_name[molecule_id]:<9s}  {new_molecule_number_list[molecule_id]}")

    output_topology_file.close()

prog = "genmesh"
desc = '''This program generate mesh for a given set of molecules.'''

def getargs_genmesh(argv):

    parser = ArgumentParser(prog=prog, description=desc)

    parser.add_argument('-f', '--structure', nargs='+', type=str, required=True, 
                        help="PDB files which are taken as input. Allow multiple file.")
    parser.add_argument('-p', '--topology', nargs='+', type=str, required=True, 
                        help="ITP files which are taken as input. Allow multiple file (one for each pdb file).")
    parser.add_argument('-n', '--number', type=int, nargs='+', required=True,
                        help="Number of molecules to add. Allow multiple input (one number for each pdb file).")
    parser.add_argument('-g', '--gap', type=float, default=1,
                        help="Minimum distance between each two molecules.")
    parser.add_argument('-mesh', '--mesh', type=int, nargs=3, 
                        help="Three integers representing the size of the mesh in x/y/z direction. If not specified, mesh will be guessed.")
    #parser.add_argument('-gmesh', '--guess-mesh', type=bool, default=False,
    #                    help="Whether let the program guess the size of the mesh in x/y/z direction.")
    parser.add_argument('-bt', '--box-type', type=str, choices=["xy", "cubic", "anosotropy"], default="anosotropy",
                        help="The type of the size of box.")
    parser.add_argument('-mx', '--minimum-x', type=float, help="Minimum length of the box in the x direction. Zero is as small as posible.", default=0)
    parser.add_argument('-my', '--minimum-y', type=float, help="Minimum length of the box in the y direction. Zero is as small as posible.", default=0)
    parser.add_argument('-mz', '--minimum-z', type=float, help="Minimum length of the box in the z direction. Zero is as small as posible.", default=0)
    parser.add_argument('-s', '--shuffle', action='store_true', default=False, 
                        help="Whether the molecules to input are shuffled before insertion.")

    parser.add_argument('-oc', '--output-conformation', type=str, help="File prefix to write output configuration",
                        default="system.pdb",
                        required=True)
    parser.add_argument('-op', '--output-topology', type=str, help="File prefix to write output configuration",
                        default="system.top",
                        required=True)

    args = parser.parse_args(argv)
    
    return args

from dropps.share.command_class import single_command
genmesh_commands = single_command("genmesh", getargs_genmesh, genmesh, desc)

