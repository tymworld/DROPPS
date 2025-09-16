import math
from openmm.unit import nanometer
from dataclasses import dataclass
from typing import List

def distance(bead1, bead2):
    temp_distance = math.sqrt(pow(bead1[0] - bead2[0], 2)
                              + pow(bead1[1] - bead2[1], 2)
                              + pow(bead1[2] - bead2[2], 2))
    return temp_distance

def read_pdb(pdb_path):
    """
    Reads a coarse-grained PDB file and returns a list of atom information.
    Each atom is represented as a dictionary with keys:
    - serial
    - name
    - resname
    - chain
    - resseq
    - x, y, z
    - occupancy
    - bfactor
    - element
    """
    atoms = []
    box = None

    print(f"## Reading pdb file {pdb_path}.")

    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith("CRYST1"):
                # Columns 7–15: a, 16–24: b, 25–33: c (box lengths)
                lx = float(line[6:15]) / 10.0
                ly = float(line[15:24]) / 10.0
                lz = float(line[24:33]) / 10.0
                box = [lx, ly, lz] * nanometer
                print(f"## Box size of {pdb_path} is {lx} * {ly} * {lz} nm^3.")

            elif line.startswith("ATOM"):
                atom = {
                    "serial": int(line[6:11]),
                    "name": line[12:16].strip(),
                    "resname": line[17:20].strip(),
                    "chain": line[21].strip(),
                    "resseq": int(line[22:26]),
                    "x": float(line[30:38]) / 10.0 * nanometer,
                    "y": float(line[38:46]) / 10.0 * nanometer,
                    "z": float(line[46:54]) / 10.0 * nanometer,
                    "occupancy": float(line[54:60]),
                    "bfactor": float(line[60:66]),
                    "element": line[76:78].strip()
                }
                atoms.append(atom)
        print(f"## Phrased {len(atoms)} atoms from pdb file {pdb_path}.")

    return atoms, box


@dataclass
class PDBData:
    record_names: List[str]
    serial_numbers: List[int]
    atom_names: List[str]
    residue_names: List[str]
    chain_IDs: List[str]
    residue_sequence_numbers: List[int]
    x_nms: List[float]
    y_nms: List[float]
    z_nms: List[float]
    occupancys: List[float]
    bfactors: List[float]
    elements: List[str]
    molecule_list: List[List[int]]
    molecule_length_list: List[int]
    box_size: List[float]

def phrase_pdb_atoms(atoms, box):
    box_size = box.value_in_unit(nanometer)
    record_names = ["ATOM"] * len(atoms)
    serial_numbers = [atom["serial"] for atom in atoms]
    atom_names = [atom["name"] for atom in atoms]
    residue_names = [atom["resname"] for atom in atoms]
    chain_IDs = [atom["chain"] for atom in atoms]
    residue_sequence_numbers = [atom["resseq"] for atom in atoms]
    x_nms = [atom["x"].value_in_unit(nanometer) for atom in atoms]
    y_nms = [atom["y"].value_in_unit(nanometer) for atom in atoms]
    z_nms = [atom["z"].value_in_unit(nanometer) for atom in atoms]
    occupancys = [atom["occupancy"] for atom in atoms]
    bfactors = [atom["bfactor"] for atom in atoms]
    elements = [atom["element"] for atom in atoms]

    molecule_list = []
    current_group = [0]  # Start with the first index

    for i in range(1, len(chain_IDs)):
        if chain_IDs[i] == chain_IDs[i - 1]:
            current_group.append(i)
        else:
            molecule_list.append(current_group)
            current_group = [i]  # Start a new group

    # Don't forget to append the last group
    molecule_list.append(current_group)

    molecule_length_list = [len(molecule) for molecule in molecule_list]

    return PDBData(
        record_names=record_names,
        serial_numbers=serial_numbers,
        atom_names=atom_names,
        residue_names=residue_names,
        chain_IDs=chain_IDs,
        residue_sequence_numbers=residue_sequence_numbers,
        x_nms=x_nms,
        y_nms=y_nms,
        z_nms=z_nms,
        occupancys=occupancys,
        bfactors=bfactors,
        elements=elements,
        molecule_list=molecule_list,
        molecule_length_list=molecule_length_list,
        box_size=box_size
    )


def write_pdb(pdb_file_name, box_size, record_name, serial_number, atom_name, residue_name, chain_ID, residue_sequence_number, 
              x_nm, y_nm, z_nm, occupancy, b_factor, element_symbol, molecule_length_list):
    
    with open(pdb_file_name, 'w') as pdb_file:

        
        pdb_file.write(f"CRYST1{box_size[0] * 10:9.3f}{box_size[1] * 10:9.3f}{box_size[2] * 10:9.3f}  90.00  90.00  90.00 P 1           1\n")
        TER_list = [sum(molecule_length_list[0: (i + 1)]) - 1 for i in range(len(molecule_length_list))]

        for index in range(len(atom_name)):

            # PDB atom line: fixed-width format
            # Columns: https://www.wwpdb.org/documentation/file-format-content/format33/sect9.html#ATOM
            pdb_file.write(
                "{:<6s}{:>5d} {:<4s} {:>3s} {:1s}{:>4d}    "
                "{:>8.3f}{:>8.3f}{:>8.3f}{:6.2f}{:6.2f}          {:>2s}\n".format(
                    record_name[index],      # Record name
                    serial_number[index],           # Atom serial number
                    atom_name[index],    # Atom name, left aligned, max 4 chars
                    residue_name[index],       # Residue name
                    chain_ID[index][-1],         # Chain ID
                    residue_sequence_number[index],           # Residue sequence number
                    x_nm[index] * 10.0, y_nm[index] * 10.0, z_nm[index] * 10.0,     # Coordinates
                    occupancy[index], b_factor[index],  # Occupancy, B-factor
                    element_symbol[index]  # Element symbol (fallback)
                )
            )
            if index in TER_list:
                pdb_file.write("TER\n")

        pdb_file.write("END\n")
