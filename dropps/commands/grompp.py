# mdrun tool in CGPS package by Yiming Tang @ Fudan
# Development started on June 6 2025

import os
import sys
import pickle

import numpy as np

from openmm.unit import nanometer, kilojoule_per_mole, moles, liter, picosecond, kelvin, bar, nanosecond

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from argparse import ArgumentParser
from dropps.share.forcefield import forcefield_list
from dropps.share.parameters import getparameter

from dropps.share.build_system import build_system

from dropps.fileio.pdb_reader import read_pdb, write_pdb, phrase_pdb_atoms

def grompp(args):

    if len(args.output) < 5 or args.output[-4:] != ".tpr":
        output_tpr = args.output + ".tpr"
    else:
        output_tpr = args.output

    parameters = getparameter(args.parameter)
    mdsystem, mdtopology, positions, ITP_Topology_list = build_system(args.structure, args.topology, getparameter(args.parameter))

    atoms_pdb_raw,box_raw = read_pdb(args.structure)
    PDB_raw = phrase_pdb_atoms(atoms_pdb_raw,box_raw)

    runtime_files = {
        "parameters": parameters,
        "mdsystem": mdsystem,
        "mdtopology": mdtopology,
        "positions": positions,
        "pdb_raw": PDB_raw,
        "ITP_list": ITP_Topology_list
    }

    with open(output_tpr, "wb") as f:
        pickle.dump(runtime_files, f)

    print(f"## Write run time files to {output_tpr}.")

prog = "mdrun"
desc = '''This program initates and runs an molecular dynamics simulation.'''

def getargs_grompp(argv):

    parser = ArgumentParser(prog=prog, description=desc)

    parser.add_argument('-f', '--structure', type=str, required=True, 
                        help="GSD file containing the initial configuration of the system.")
    parser.add_argument('-p', '--topology', type=str, required=True, 
                        help="GSD file containing the initial configuration of the system.")
    parser.add_argument('-m', '--parameter', type=str, required=True,
                        help="Parameter file (mdp) for this simulation.")
    parser.add_argument('-o', '--output', type=str, required=True,
                        help="Path to output tpr files.")

    args = parser.parse_args(argv)
    return args

from dropps.share.command_class import single_command
grompp_commands = single_command("grompp", getargs_grompp, grompp, desc)
