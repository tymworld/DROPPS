# pdb2bond tool that add conect record to a pdb file in DROPPS package by Yiming Tang @ Fudan
# Development started on Jan 31 2026

from argparse import ArgumentParser

from dropps.fileio.pdb_reader import read_pdb, write_pdbData, phrase_pdb_atoms
from dropps.fileio.itp_reader import read_itp
from dropps.fileio.tpr_reader import read_tpr

from collections import defaultdict
import numpy as np

prog = "pdb2bond"
desc = '''This program add conect record to a pdb file.'''

def getargs_pdb2bond(argv):

    parser = ArgumentParser(prog=prog, description=desc)

    parser.add_argument('-s', '--run-input', type=str, 
                        help="TPR file containing all information for a simulation run.")
    
    parser.add_argument('-p', '--topology', type=str,
                        help="ITP file containing information for a simulation system")
    
    parser.add_argument('-f', '--input', type=str, required=True, 
                    help="PDB file which is taken as input.")
    
    parser.add_argument('-o', '--output', type=str, required=True,
                    help="PDB file to write output.")
    
    parser.add_argument('-noa', '--not-obmit-atom-type', action='store_true', default=False,
                        help="By default, atom type field is obmitted and all atom treated as Carbon. This option disables that behavior.")
    
    args = parser.parse_args(argv)
    return args

def pdb2bond(args):

    # We test input topology and load

    if not (args.topology is None) ^ (args.run_input is None):
        print(f"ERROR: One and only one of tpr and itp file can be specified.")
        quit()
    
    if args.topology is not None:

        try:
            topology = read_itp(args.topology)

        except Exception as exc:
            print(f"ERROR: Cannot process {args.topology} as a single molecule topology (ITP) file.")
            print(f"ERROR: Root cause: {exc}")
            quit()
        
        topology_atom_number = len(topology.atoms)
        bond_list = [[bond.a1, bond.a2] for bond in topology.bonds]
    
    elif args.run_input is not None:

        try:
            topology = read_tpr(args.run_input).mdtopology

        except Exception as exc:
            print(f"ERROR: Cannot process {args.run_input} as a system topology (TPR) file.")
            print(f"ERROR: Root cause: {exc}")
            quit()
        
        topology_atom_number = topology.getNumAtoms()
        bond_list = [[bond[0].index, bond[1].index] for bond in topology.bonds()]

    
    if len(bond_list) == 0:
        print(f"ERROR: Get no bond from input ITP or TPR files.")
        quit()
    
    # We now expand the bond list to be write-ready

    all_beads = np.unique(bond_list)

    adj = defaultdict(list)
    for a, b in bond_list:
        if a == b:
            continue  # ignore self-bonds (optional)
        adj[a].append(b)
        adj[b].append(a)
    
    bead_to_bonded = {bead: np.unique(adj.get(bead, ())) for bead in all_beads}
        
    # We now read pdb file
    pdb_atoms, pdb_box = read_pdb(args.input)
    pdb_data = phrase_pdb_atoms(pdb_atoms, pdb_box)
    pdb_atom_number = len(pdb_atoms)
    if pdb_atom_number != topology_atom_number:
        print(f"ERROR: Number of atoms in pdb file ({pdb_atom_number}) does not equal to that in topology ({topology_atom_number}).")
        quit()
    
    # We now write

    if not args.not_obmit_atom_type:
        pdb_data.elements = ["C"] * len(pdb_data.elements)

    try:
        write_pdbData(args.output ,pdb_data, bead_to_bonded)
    except Exception as exc:
        print(f"ERROR: Cannot write to pdb file {args.output}")
        print(f"ERROR: Root cause: {exc}")
        quit()
        
    print(f"PDB file written to {args.output}")


from dropps.share.command_class import single_command
pdb2bond_commands = single_command("pdb2bond", getargs_pdb2bond, pdb2bond, desc)

    
