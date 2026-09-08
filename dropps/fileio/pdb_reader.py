import math
import operator
from openmm.unit import nanometer
from dataclasses import dataclass

# from __future__ import annotations
from typing import Dict, Iterable, List, Sequence


_HY36_WIDTH = 5
_HY36_DECIMAL_LIMIT = 10**_HY36_WIDTH
_HY36_BLOCK_SIZE = 26 * 36 ** (_HY36_WIDTH - 1)
_HY36_UPPER_DIGITS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_HY36_LOWER_DIGITS = _HY36_UPPER_DIGITS.lower()


def _encode_base36(value, digits, width):
    encoded = []
    for _ in range(width):
        value, remainder = divmod(value, 36)
        encoded.append(digits[remainder])
    if value:
        raise ValueError("Value does not fit in the requested base-36 width.")
    return "".join(reversed(encoded))


def _decode_base36(value, digits):
    digit_values = {character: index for index, character in enumerate(digits)}
    decoded = 0
    for character in value:
        try:
            digit = digit_values[character]
        except KeyError as exc:
            raise ValueError(f"Invalid hybrid-36 digit {character!r}.") from exc
        decoded = decoded * 36 + digit
    return decoded


def atom_id_to_written_id(atom_id):
    """Encode a non-negative atom serial in the PDB five-column hybrid-36 form."""

    atom_id = operator.index(atom_id)
    if atom_id < 0:
        raise ValueError("Atom ID must be non-negative.")
    if atom_id < _HY36_DECIMAL_LIMIT:
        return f"{atom_id:>{_HY36_WIDTH}d}"

    offset = atom_id - _HY36_DECIMAL_LIMIT
    base36_offset = 10 * 36 ** (_HY36_WIDTH - 1)
    if offset < _HY36_BLOCK_SIZE:
        return _encode_base36(
            offset + base36_offset,
            _HY36_UPPER_DIGITS,
            _HY36_WIDTH,
        )

    offset -= _HY36_BLOCK_SIZE
    if offset < _HY36_BLOCK_SIZE:
        return _encode_base36(
            offset + base36_offset,
            _HY36_LOWER_DIGITS,
            _HY36_WIDTH,
        )

    maximum = _HY36_DECIMAL_LIMIT + 2 * _HY36_BLOCK_SIZE - 1
    raise ValueError(
        f"Atom ID {atom_id} exceeds the largest five-column hybrid-36 "
        f"serial ({maximum})."
    )


def written_id_to_atom_id(written_id):
    """Decode a decimal or standard hybrid-36 PDB atom serial."""

    field = str(written_id)
    if len(field) > _HY36_WIDTH:
        raise ValueError(f"PDB atom serial {field!r} is wider than five columns.")
    field = field.rjust(_HY36_WIDTH)
    first = field[0]
    if first in {" ", "-"} or first.isdigit():
        return int(field)
    if first in _HY36_UPPER_DIGITS[10:]:
        return (
            _decode_base36(field, _HY36_UPPER_DIGITS)
            - 10 * 36 ** (_HY36_WIDTH - 1)
            + _HY36_DECIMAL_LIMIT
        )
    if first in _HY36_LOWER_DIGITS[10:]:
        return (
            _decode_base36(field, _HY36_LOWER_DIGITS)
            + 16 * 36 ** (_HY36_WIDTH - 1)
            + _HY36_DECIMAL_LIMIT
        )
    raise ValueError(f"Invalid five-column PDB atom serial {field!r}.")


def distance(bead1, bead2):
    temp_distance = math.sqrt(
        pow(bead1[0] - bead2[0], 2)
        + pow(bead1[1] - bead2[1], 2)
        + pow(bead1[2] - bead2[2], 2)
    )
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

    with open(pdb_path, "r") as f:
        for line in f:
            if line.startswith("CRYST1"):
                # DROPPS currently builds only diagonal OpenMM box vectors.
                # Rejecting tilted cells here prevents silently changing their
                # geometry into an orthorhombic box with the same lengths.
                lx = float(line[6:15]) / 10.0
                ly = float(line[15:24]) / 10.0
                lz = float(line[24:33]) / 10.0
                alpha = float(line[33:40])
                beta = float(line[40:47])
                gamma = float(line[47:54])
                lengths = (lx, ly, lz)
                angles = (alpha, beta, gamma)
                if not all(math.isfinite(value) and value > 0 for value in lengths):
                    raise ValueError(
                        f"PDB CRYST1 box lengths must be finite and positive: "
                        f"{lengths}."
                    )
                if not all(math.isfinite(value) for value in angles):
                    raise ValueError(f"PDB CRYST1 box angles must be finite: {angles}.")
                if not all(abs(value - 90.0) <= 1.0e-3 for value in angles):
                    raise ValueError(
                        "DROPPS currently supports only orthorhombic PDB boxes; "
                        f"CRYST1 angles are alpha={alpha}, beta={beta}, "
                        f"gamma={gamma} degrees."
                    )
                box = [lx, ly, lz] * nanometer
                print(f"## Box size of {pdb_path} is {lx} * {ly} * {lz} nm^3.")

            elif line.startswith("ATOM"):
                atom = {
                    "serial": written_id_to_atom_id(line[6:11]),
                    "name": line[12:16].strip(),
                    "resname": line[17:20].strip(),
                    "chain": line[21].strip(),
                    "resseq": int(line[22:26]),
                    "x": float(line[30:38]) / 10.0 * nanometer,
                    "y": float(line[38:46]) / 10.0 * nanometer,
                    "z": float(line[46:54]) / 10.0 * nanometer,
                    "occupancy": float(line[54:60]),
                    "bfactor": float(line[60:66]),
                    "element": line[76:78].strip(),
                }
                atoms.append(atom)
        print(f"## Phrased {len(atoms)} atoms from pdb file {pdb_path}.")

    if box is None:
        raise ValueError(f"PDB file {pdb_path} has no CRYST1 periodic box record.")
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
        box_size=box_size,
    )


def write_pdbData(pdb_file_name, data: PDBData, bond_list=None):
    write_pdb(
        pdb_file_name,
        data.box_size,
        data.record_names,
        data.serial_numbers,
        data.atom_names,
        data.residue_names,
        data.chain_IDs,
        data.residue_sequence_numbers,
        data.x_nms,
        data.y_nms,
        data.z_nms,
        data.occupancys,
        data.bfactors,
        data.elements,
        data.molecule_length_list,
        bond_list,
    )


def write_pdb(
    pdb_file_name,
    box_size,
    record_name,
    serial_number,
    atom_name,
    residue_name,
    chain_ID,
    residue_sequence_number,
    x_nm,
    y_nm,
    z_nm,
    occupancy,
    b_factor,
    element_symbol,
    molecule_length_list,
    bond_list=None,
):
    with open(pdb_file_name, "w") as pdb_file:
        pdb_file.write(
            f"CRYST1{box_size[0] * 10:9.3f}{box_size[1] * 10:9.3f}{box_size[2] * 10:9.3f}  90.00  90.00  90.00 P 1           1\n"
        )
        TER_list = [
            sum(molecule_length_list[0 : (i + 1)]) - 1
            for i in range(len(molecule_length_list))
        ]

        for index in range(len(atom_name)):
            # PDB atom line: fixed-width format
            # Columns: https://www.wwpdb.org/documentation/file-format-content/format33/sect9.html#ATOM
            pdb_file.write(
                "{:<6s}{:>5s} {:<4s} {:>3s} {:1s}{:>4d}    "
                "{:>8.3f}{:>8.3f}{:>8.3f}{:6.2f}{:6.2f}          {:>2s}\n".format(
                    record_name[index],  # Record name
                    atom_id_to_written_id(serial_number[index]),  # Atom serial number
                    atom_name[index],  # Atom name, left aligned, max 4 chars
                    residue_name[index],  # Residue name
                    (chain_ID[index] or " ")[-1],  # Chain ID
                    residue_sequence_number[index],  # Residue sequence number
                    x_nm[index] * 10.0,
                    y_nm[index] * 10.0,
                    z_nm[index] * 10.0,  # Coordinates
                    occupancy[index],
                    b_factor[index],  # Occupancy, B-factor
                    element_symbol[index],  # Element symbol (fallback)
                )
            )
            if index in TER_list:
                pdb_file.write("TER\n")

        if bond_list is not None:
            for ln in conect_lines_from_bond_dict(bond_list):
                pdb_file.write(f"{ln}\n")

        pdb_file.write("END\n")


def conect_lines_from_bond_dict(
    bond_list: Dict[int, Sequence[int]],
    *,
    sort_atoms: bool = True,
    sort_bonded: bool = True,
) -> List[str]:
    """
    Convert a 0-based adjacency dict {atom_index: [connected_atom_indices...]}
    into PDB CONECT records (1-based serials), with up to 4 bonded atoms per line.

    Returns a list of lines WITHOUT trailing newline.
    """
    # Choose deterministic order if requested
    atoms: Iterable[int] = sorted(bond_list) if sort_atoms else bond_list.keys()

    lines: List[str] = []
    for a0 in atoms:
        bonded0 = list(bond_list.get(a0, []))
        if sort_bonded:
            bonded0.sort()

        a_serial = a0 + 1  # 1-based serial in PDB

        # Chunk bonded atoms into groups of 4 per CONECT line
        for i in range(0, len(bonded0), 4):
            chunk = bonded0[i : i + 4]
            # Build fixed-width fields: "CONECT" + 5*integer(5 cols each)
            fields = ["CONECT", atom_id_to_written_id(a_serial)] + [
                atom_id_to_written_id(b + 1) for b in chunk
            ]
            line = "".join(fields).ljust(31)  # pad to at least through col 31
            lines.append(line.rstrip())  # keep clean; PDB readers usually accept this
    return lines
