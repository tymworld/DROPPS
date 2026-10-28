# generate elestic (genelestic) tool in CGPS.ng package by Yiming Tang @ Fudan
# Development started on June 8 2025

from dropps.share.argument_parser import ArgumentParser
from copy import deepcopy
import numpy as np

from openmm.unit import nanometer, kilojoule_per_mole

from dropps.fileio.pdb_reader import read_pdb
from dropps.fileio.itp_reader import read_itp, write_itp, Bond

def genelastic(args):
    output_file_prefix = args.output[0:-4] if ".itp" in args.output else args.output
    output_file_name = output_file_prefix + ".itp"

    # We first get PDB files and read coordinates

    try:
        atoms, box = read_pdb(args.structure)
        print(f"## Open structure file {args.structure} which contains {len(atoms)} atoms.")
    except Exception as exc:
        print("## An exception occurred when trying to open structure file %s." % args.structure)
        print(f"## Root cause: {exc}")
        quit()

    # We next get topology

    if args.topology[-3:] != "itp":
        print("ERROR: Only itp file can be treated as input.")
        quit()

    try:
        topology = read_itp(args.topology)
        print(f"## Open topology file {args.topology} which contains {len(topology.atoms)} atoms.")
    except Exception as exc:
        print("## An exception occurred when trying to open topology file %s." % args.topology)
        print(f"## Root cause: {exc}")
        quit()

    # We now get elastic profile file

    try:
        elastic_cluster_list = [[int(number) - 1 for number in line.strip().split()] for line in open(args.elastic_residues, 'r') if len(line) > 1]
    except Exception as exc:
        print("## An exception occurred when trying to open elastic-residue file %s." % args.elastic_residues)
        print(f"## Root cause: {exc}")
        quit()

    print(f"## Reference structure contains {len(elastic_cluster_list)} clusters.")
    print(f"## Will generate elastic bonds between residue within {args.elastic_lower:.2f} ~ {args.elastic_upper:.2f} nm.")
    print(f"## Elastic bond constant will be {args.elastic_force_constant:.2f} kJ / nm ^ 2")

    # We now create elastic bond profiles
    exist_bonds = [[bond.a1, bond.a2] for bond in topology.bonds]

    added_bonds = list()

    addded_bond_number = 0

    for cluster in elastic_cluster_list:
        for atomid_1 in cluster:
            for atomid_2 in cluster:
                if any(np.array_equal(pair, [atomid_1, atomid_2]) or np.array_equal(pair, [atomid_2, atomid_1])
                    for pair in exist_bonds):
                    continue

                if any(np.array_equal(pair, [atomid_1, atomid_2]) or np.array_equal(pair, [atomid_2, atomid_1])
                    for pair in added_bonds):
                    continue
                
                if atomid_1 == atomid_2:
                    continue

                coor_1 = np.array([atoms[atomid_1]["x"], atoms[atomid_1]["y"], atoms[atomid_1]["z"]])
                coor_2 = np.array([atoms[atomid_2]["x"], atoms[atomid_2]["y"], atoms[atomid_2]["z"]])
                distance = np.linalg.norm(coor_1 - coor_2)

                if distance > args.elastic_lower * nanometer and distance < args.elastic_upper * nanometer:

                    bond_length = distance
                    bond_k = args.elastic_force_constant * kilojoule_per_mole / nanometer ** 2

                    topology.bonds.append(Bond(atomid_1, atomid_2, bond_length, bond_k))
                    added_bonds.append([atomid_1, atomid_2])
                    print(f"## Adding bond (k={bond_k}) between atom {atomid_1 + 1} and {atomid_2 + 1} with distance {bond_length}.")
                    addded_bond_number += 1

    print(f"## A total number of {addded_bond_number} bonds have been added.")

    # We now save the new topology file.

    write_itp(output_file_name, topology)

prog = "genelastic"
desc = "Add a distance-based elastic network to an ITP topology."

def getargs_genelastic(argv):

    parser = ArgumentParser(prog=prog, description=desc)

    parser.add_argument('-f', '--structure', type=str, required=True, 
                        help="Input reference structure file (.pdb).")

    parser.add_argument('-p', '--topology', type=str, required=True, 
                        help="Input topology file (.itp).")

    parser.add_argument('-o', '--output', type=str, required=True, 
                        help="Output topology file (.itp); the extension is added if omitted.")

    parser.add_argument('-r', '-er', '--elastic-residues', type=str, required=True,
                        help="Input text file listing one elastic-network bead group per line, using 1-based bead indices.")

    parser.add_argument('-ef', '--elastic-force-constant', type=float, default=5000,
                        help="Elastic-bond force constant, in kJ mol^-1 nm^-2.")

    parser.add_argument('-el', '--elastic-lower', type=float, default=0.5,
                        help="Lower distance cutoff for elastic bonds, in nm.")

    parser.add_argument('-eu', '--elastic-upper', type=float, default=0.9,
                        help="Upper distance cutoff for elastic bonds, in nm.")

    args = parser.parse_args(argv)
    return args

from dropps.share.command_class import single_command
genelastic_commands = single_command("genelastic", getargs_genelastic, genelastic, desc)
