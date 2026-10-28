# editconf tool in CGPS.ng package by Yiming Tang @ Fudan
# Development started on June 6 2025

from dropps.share.argument_parser import ArgumentParser
from dropps.share.command_class import single_command
from copy import deepcopy
from dropps.share.pbc import unwrap_pbc
from dropps.fileio.pdb_reader import read_pdb, write_pdb, phrase_pdb_atoms
from dropps.fileio.filename_control import validate_extension
import numpy as np
from openmm.unit import nanometer


def editconf(args):
    atoms, box = read_pdb(args.structure)
    if not atoms:
        raise ValueError("Input structure contains no ATOM records.")
    pdb_data = phrase_pdb_atoms(atoms, box)
    raw_lengths = np.asarray(box.value_in_unit(nanometer), dtype=float)
    direct_lengths = (args.x_axis, args.y_axis, args.z_axis)
    multipliers = (
        args.multiply_x_axis,
        args.multiply_y_axis,
        args.multiply_z_axis,
    )
    treat_pbc = (args.treat_pbc_x, args.treat_pbc_y, args.treat_pbc_z)

    target_lengths = raw_lengths.copy()
    expanded = [False, False, False]
    axis_names = "xyz"
    for axis, (direct, multiplier) in enumerate(zip(direct_lengths, multipliers)):
        if direct is None and multiplier is None:
            continue
        candidate = (
            float(direct)
            if direct is not None
            else raw_lengths[axis] * float(multiplier)
        )
        if not np.isfinite(candidate) or candidate <= raw_lengths[axis]:
            raise ValueError(
                f"New {axis_names[axis]} box length must be finite and larger "
                f"than the current {raw_lengths[axis]:g} nm."
            )
        target_lengths[axis] = candidate
        expanded[axis] = True

    coordinates = np.array([pdb_data.x_nms, pdb_data.y_nms, pdb_data.z_nms]).transpose()
    pbc_treated_coordinates = unwrap_pbc(atoms, box)
    new_coordinates = deepcopy(coordinates)

    for axis, (must_unwrap, was_expanded) in enumerate(zip(treat_pbc, expanded)):
        if must_unwrap or was_expanded:
            new_coordinates[:, axis] = pbc_treated_coordinates[:, axis]
            reason = "requested" if must_unwrap else "required by box expansion"
            print(f"## Unwrapped the {axis_names[axis]} axis ({reason}).")

    translation = 0.5 * (target_lengths - raw_lengths)
    new_coordinates += translation
    output_file_name = validate_extension(args.output, "pdb")

    write_pdb(
        output_file_name,
        target_lengths,
        pdb_data.record_names,
        pdb_data.serial_numbers,
        pdb_data.atom_names,
        pdb_data.residue_names,
        pdb_data.chain_IDs,
        pdb_data.residue_sequence_numbers,
        new_coordinates[:, 0],
        new_coordinates[:, 1],
        new_coordinates[:, 2],
        pdb_data.occupancys,
        pdb_data.bfactors,
        pdb_data.elements,
        pdb_data.molecule_length_list,
    )

    print(f"## Wrote resized PDB file to {output_file_name}.")


prog = "editconf"
desc = (
    "Resize a PDB simulation box and optionally unwrap coordinates along selected axes."
)


def getargs_editconf(argv):
    parser = ArgumentParser(prog=prog, description=desc)

    parser.add_argument(
        "-f",
        "--structure",
        type=str,
        required=True,
        help="Input structure file (.pdb).",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Output structure file (.pdb); the extension is added if omitted.",
    )

    x_size = parser.add_mutually_exclusive_group()
    x_size.add_argument(
        "-x", "--x-axis", type=float, help="New box length along x, in nm."
    )
    x_size.add_argument(
        "-mx",
        "--multiply-x-axis",
        type=float,
        help="Multiplier applied to the current x box length.",
    )

    y_size = parser.add_mutually_exclusive_group()
    y_size.add_argument(
        "-y", "--y-axis", type=float, help="New box length along y, in nm."
    )
    y_size.add_argument(
        "-my",
        "--multiply-y-axis",
        type=float,
        help="Multiplier applied to the current y box length.",
    )

    z_size = parser.add_mutually_exclusive_group()
    z_size.add_argument(
        "-z", "--z-axis", type=float, help="New box length along z, in nm."
    )
    z_size.add_argument(
        "-mz",
        "--multiply-z-axis",
        type=float,
        help="Multiplier applied to the current z box length.",
    )

    parser.add_argument(
        "-px",
        "--treat-pbc-x",
        action="store_true",
        default=False,
        help="Unwrap coordinates across periodic boundaries along x.",
    )
    parser.add_argument(
        "-py",
        "--treat-pbc-y",
        action="store_true",
        default=False,
        help="Unwrap coordinates across periodic boundaries along y.",
    )
    parser.add_argument(
        "-pz",
        "--treat-pbc-z",
        action="store_true",
        default=False,
        help="Unwrap coordinates across periodic boundaries along z.",
    )

    args = parser.parse_args(argv)
    return args


editconf_commands = single_command("editconf", getargs_editconf, editconf, desc)
