# editconf tool in CGPS.ng package by Yiming Tang @ Fudan
# Development started on June 6 2025

from dropps.share.argument_parser import ArgumentParser
from copy import deepcopy
from dropps.share.pbc import unwrap_pbc
from dropps.fileio.pdb_reader import read_pdb, write_pdb, phrase_pdb_atoms
import numpy as np
from openmm.unit import nanometer

def editconf(args):

    output_file_prefix = args.output[0:-4] if ".pdb" in args.output else args.output
    output_file_name = output_file_prefix + ".pdb"

    # We first get the pdb file

    atoms, box = read_pdb(args.structure)
    pdb_data = phrase_pdb_atoms(atoms, box)

    # We test whether each dimension should be treated
    raw_box = box
    raw_x = raw_box[0]
    raw_y = raw_box[1]
    raw_z = raw_box[2]

    if args.x_axis is None and args.multiply_x_axis is None:
        expand_x = False
        output_x = raw_x
    else:
        expand_x = True
        if args.x_axis is not None and args.multiply_x_axis is not None:
            print("ERROR: x axis length and x axis multiplier cannot be specified together.")
            quit()
        if args.x_axis is not None:
            output_x = args.x_axis * nanometer
        else:
            output_x = args.multiply_x_axis * raw_x
        
    if args.y_axis is None and args.multiply_y_axis is None:
        expand_y = False
        output_y = raw_y
    else:
        expand_y = True
        if args.y_axis is not None and args.multiply_y_axis is not None:
            print("ERROR: y axis length and y axis multiplier cannot be specified together.")
            quit()
        if args.y_axis is not None:
            output_y = args.y_axis * nanometer
        else:
            output_y = args.multiply_y_axis * raw_y

    if args.z_axis is None and args.multiply_z_axis is None:
        expand_z = False
        output_z = raw_z
    else:
        expand_z = True
        if args.z_axis is not None and args.multiply_z_axis is not None:
            print("ERROR: z axis length and z axis multiplier cannot be specified together.")
            quit()
        if args.z_axis is not None:
            output_z = args.z_axis * nanometer
        else:
            output_z = args.multiply_z_axis * raw_z


    # We now treat pbc

    coordinates = np.array([pdb_data.x_nms, pdb_data.y_nms, pdb_data.z_nms]).transpose()
    pbc_treated_coordinates = unwrap_pbc(atoms, box)
    new_coordinates = deepcopy(coordinates)

    if expand_x:
        if output_x <= raw_x:
            print(f"ERROR: Cannot expand x axis from {raw_x} to a smaller/equal value of {output_x}.")
            quit()
        else:
            new_coordinates[:,0] = pbc_treated_coordinates[:,0]
            if not args.treat_pbc_x:
                print("## PBC for x axis treated although input says don't.")
            else:
                print("## PBC for x axis treated.")
    elif args.treat_pbc_x:
        new_coordinates[:,0] = pbc_treated_coordinates[:,0]
        print("## PBC for x axis treated.")

    if expand_y:
        if output_y <= raw_y:
            print("ERROR: Cannot expand y axis from %d to a smaller/equal value of %d." % (raw_y, output_y))
            quit()
        else:
            new_coordinates[:,1] = pbc_treated_coordinates[:,1]
            coordinate = deepcopy(new_coordinates)
            if not args.treat_pbc_y:
                print("## PBC for y axis treated although input says don't.")
            else:
                print("## PBC for y axis treated.")
    elif args.treat_pbc_y:
        new_coordinates[:,1] = pbc_treated_coordinates[:,1]
        print("## PBC for y axis treated.")
            
    if expand_z:
        if output_z <= raw_z:
            print("ERROR: Cannot expand z axis from %d to a smaller/equal value of %d." % (raw_z, output_z))
            quit()
        else:

            new_coordinates[:,2] = pbc_treated_coordinates[:,2]
            if not args.treat_pbc_z:
                print("## PBC for z axis treated although input says don't.")
            else:
                print("## PBC for z axis treated.")
    elif args.treat_pbc_z:
        new_coordinates[:,2] = pbc_treated_coordinates[:,2]
        print("## PBC for z axis treated.")

    # We now generate a new frame

    x_trans = (output_x - raw_x) / 2
    y_trans = (output_y - raw_y) / 2
    z_trans = (output_z - raw_z) / 2

    write_pdb(output_file_name, [output_x.value_in_unit(nanometer), output_y.value_in_unit(nanometer), output_z.value_in_unit(nanometer)],
            pdb_data.record_names,
            pdb_data.serial_numbers,
            pdb_data.atom_names,
            pdb_data.residue_names,
            pdb_data.chain_IDs,
            pdb_data.residue_sequence_numbers,
            new_coordinates[:,0] + x_trans.value_in_unit(nanometer),
            new_coordinates[:,1] + y_trans.value_in_unit(nanometer),
            new_coordinates[:,2] + z_trans.value_in_unit(nanometer),
            pdb_data.occupancys,
            pdb_data.bfactors,
            pdb_data.elements,
            pdb_data.molecule_length_list
            )

    print(f"## Ouput PDB file write to {output_file_name}.")

prog = "editconf"
desc = "Resize a PDB simulation box and optionally unwrap coordinates along selected axes."

def getargs_editconf(argv):

    parser = ArgumentParser(prog=prog, description=desc)

    parser.add_argument('-f', '--structure', type=str, required=True, 
                        help="Input structure file (.pdb).")

    parser.add_argument('-o', '--output', type=str, required=True, 
                        help="Output structure file (.pdb); the extension is added if omitted.")

    x_size = parser.add_mutually_exclusive_group()
    x_size.add_argument('-x', '--x-axis', type=int,
                        help="New box length along x, in nm.")
    x_size.add_argument('-mx', '--multiply-x-axis', type=int,
                        help="Integer multiplier applied to the current x box length.")

    y_size = parser.add_mutually_exclusive_group()
    y_size.add_argument('-y', '--y-axis', type=int,
                        help="New box length along y, in nm.")
    y_size.add_argument('-my', '--multiply-y-axis', type=int,
                        help="Integer multiplier applied to the current y box length.")

    z_size = parser.add_mutually_exclusive_group()
    z_size.add_argument('-z', '--z-axis', type=int,
                        help="New box length along z, in nm.")
    z_size.add_argument('-mz', '--multiply-z-axis', type=int,
                        help="Integer multiplier applied to the current z box length.")

    parser.add_argument('-px', '--treat-pbc-x', action='store_true', default=False, 
                        help="Unwrap coordinates across periodic boundaries along x.")
    parser.add_argument('-py', '--treat-pbc-y', action='store_true', default=False, 
                        help="Unwrap coordinates across periodic boundaries along y.")
    parser.add_argument('-pz', '--treat-pbc-z', action='store_true', default=False, 
                        help="Unwrap coordinates across periodic boundaries along z.")


    args = parser.parse_args(argv)
    return args

from dropps.share.command_class import single_command
editconf_commands = single_command("editconf", getargs_editconf, editconf, desc)
