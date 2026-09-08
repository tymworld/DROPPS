# pdb2bond tool that add conect record to a pdb file in DROPPS package by Yiming Tang @ Fudan
# Development started on Jan 31 2026

from argparse import SUPPRESS

from dropps.share.argument_parser import ArgumentParser
from dropps.share.command_class import single_command

from dropps.fileio.pdb_reader import read_pdb, write_pdbData, phrase_pdb_atoms
from dropps.fileio.itp_reader import read_itp
from dropps.fileio.tpr_reader import read_tpr

from collections import defaultdict
import os
import re
import numpy as np

prog = "pdb2bond"
desc = "Add PDB CONECT records from a DROPPS run file or system topology."


def getargs_pdb2bond(argv):
    parser = ArgumentParser(prog=prog, description=desc)

    topology_source = parser.add_mutually_exclusive_group(
        required=True, section="input"
    )
    topology_source.add_argument(
        "-s",
        "--run-input",
        type=str,
        metavar="FILE",
        help="Input DROPPS run file (.tpr) containing system bonds.",
    )
    topology_source.add_argument(
        "-p",
        "--topology",
        type=str,
        metavar="FILE",
        help="Input system topology file (.top) containing system bonds.",
    )

    parser.add_argument(
        "-f", "--input", type=str, required=True, help="Input structure file (.pdb)."
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Output structure file (.pdb) containing CONECT records.",
    )

    parser.add_argument(
        "-noa",
        "--preserve-atom-types",
        dest="preserve_atom_types",
        action="store_true",
        default=False,
        help="Preserve input element fields instead of writing every bead as carbon.",
    )
    parser.add_argument(
        "--not-obmit-atom-type",
        dest="preserve_atom_types",
        action="store_true",
        default=SUPPRESS,
        help=SUPPRESS,
    )

    args = parser.parse_args(argv)
    return args


def _read_top(top_file):
    with open(top_file, "r") as f:
        lines = f.readlines()

    sections = defaultdict(list)
    current_section = None

    for line in lines:
        line = line.split(";")[0].split("#")[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            current_section = line.strip("[]").strip().lower()
            continue
        if current_section:
            sections[current_section].append(line)

    if "itp files" not in sections or len(sections["itp files"]) == 0:
        raise ValueError(f"Cannot find [ itp files ] section in TOP file {top_file}.")
    if "system" not in sections or len(sections["system"]) == 0:
        raise ValueError(f"Cannot find [ system ] section in TOP file {top_file}.")

    top_dir = os.path.dirname(os.path.abspath(top_file))
    itp_by_molecule_name = {}
    for itp_path_raw in sections["itp files"]:
        itp_filename = re.split(r"\s+", itp_path_raw.strip())[0]
        itp_path = (
            itp_filename
            if os.path.isabs(itp_filename)
            else os.path.join(top_dir, itp_filename)
        )
        itp_topology = read_itp(itp_path)
        molecule_name = itp_topology.molecule_name
        if molecule_name in itp_by_molecule_name:
            raise ValueError(
                f"Duplicate molecule name '{molecule_name}' in [ itp files ] section of {top_file}."
            )
        itp_by_molecule_name[molecule_name] = itp_topology

    molecule_with_number = []
    for line in sections["system"]:
        parts = re.split(r"\s+", line.strip())
        if len(parts) < 2:
            raise ValueError(
                f"Cannot parse line '{line}' in [ system ] section of TOP file {top_file}."
            )
        molecule_name = parts[0]
        molecule_number = int(parts[1])
        molecule_with_number.append([molecule_name, molecule_number])

    topology_atom_number = 0
    bond_list = []
    atom_id_addition = 0

    for molecule_name, molecule_number in molecule_with_number:
        if molecule_name not in itp_by_molecule_name:
            raise ValueError(
                f"Molecule '{molecule_name}' in [ system ] does not have a matching ITP in [ itp files ]."
            )
        this_itp = itp_by_molecule_name[molecule_name]
        this_atom_number = len(this_itp.atoms)
        this_bond_list = (
            []
            if this_itp.bonds is None
            else [[bond.a1, bond.a2] for bond in this_itp.bonds]
        )
        for _ in range(molecule_number):
            topology_atom_number += this_atom_number
            for bond_a1, bond_a2 in this_bond_list:
                bond_list.append(
                    [bond_a1 + atom_id_addition, bond_a2 + atom_id_addition]
                )
            atom_id_addition += this_atom_number

    return topology_atom_number, bond_list


def pdb2bond(args):
    # We test input topology and load

    if not (args.topology is None) ^ (args.run_input is None):
        print("ERROR: One and only one of tpr and top file can be specified.")
        quit()

    if args.topology is not None:
        try:
            topology_atom_number, bond_list = _read_top(args.topology)

        except Exception as exc:
            print(
                f"ERROR: Cannot process {args.topology} as a system topology (TOP) file."
            )
            print(f"ERROR: Root cause: {exc}")
            quit()

    elif args.run_input is not None:
        try:
            topology = read_tpr(args.run_input).mdtopology

        except Exception as exc:
            print(
                f"ERROR: Cannot process {args.run_input} as a system topology (TPR) file."
            )
            print(f"ERROR: Root cause: {exc}")
            quit()

        topology_atom_number = topology.getNumAtoms()
        bond_list = [[bond[0].index, bond[1].index] for bond in topology.bonds()]

    if len(bond_list) == 0:
        print("ERROR: Get no bond from input TOP or TPR files.")
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
        print(
            f"ERROR: Number of atoms in pdb file ({pdb_atom_number}) does not equal to that in topology ({topology_atom_number})."
        )
        quit()

    # We now write

    if not args.preserve_atom_types:
        pdb_data.elements = ["C"] * len(pdb_data.elements)

    try:
        write_pdbData(args.output, pdb_data, bead_to_bonded)
    except Exception as exc:
        print(f"ERROR: Cannot write to pdb file {args.output}")
        print(f"ERROR: Root cause: {exc}")
        quit()

    print(f"PDB file written to {args.output}")


pdb2bond_commands = single_command("pdb2bond", getargs_pdb2bond, pdb2bond, desc)
