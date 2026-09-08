# addangle tool in CGPS.ng package by Yiming Tang @ Fudan
# Development started on June 11 2025

from dropps.share.argument_parser import ArgumentParser
from dropps.share.command_class import single_command
from pathlib import Path
import math

from dropps.fileio.itp_reader import read_itp, write_itp, Angle
from openmm.unit import degree, kilojoule_per_mole, radian


def _read_angle_records(path):
    records = []
    seen_centers = set()
    with open(path, encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            line = raw_line.split("#", 1)[0].split(";", 1)[0].strip()
            if not line:
                continue
            fields = line.split()
            if len(fields) != 3:
                raise ValueError(
                    f"Angle file line {line_number} must contain exactly "
                    "BEAD_ID ANGLE_DEG FORCE_CONSTANT."
                )
            try:
                center = int(fields[0])
                angle_degrees = float(fields[1])
                force_constant = float(fields[2])
            except ValueError as exc:
                raise ValueError(
                    f"Angle file line {line_number} contains a non-numeric value."
                ) from exc
            if center in seen_centers:
                raise ValueError(
                    f"Angle center residue {center} is defined more than once."
                )
            if not math.isfinite(angle_degrees) or not 0.0 < angle_degrees < 180.0:
                raise ValueError(
                    f"Angle on line {line_number} must be between 0 and 180 degrees."
                )
            if not math.isfinite(force_constant) or force_constant <= 0.0:
                raise ValueError(
                    f"Force constant on line {line_number} must be positive and finite."
                )
            seen_centers.add(center)
            records.append((center, angle_degrees, force_constant))
    if not records:
        raise ValueError(f"Angle file {path!r} contains no restraint records.")
    return records


def addangle(args):
    input_itp = Path(args.input_topology)
    if input_itp.suffix.lower() != ".itp":
        raise ValueError(f"Input topology {input_itp} must use the .itp extension.")

    output_itp = Path(args.output_topology)
    if output_itp.suffix.lower() != ".itp":
        output_itp = output_itp.with_suffix(".itp")

    topology = read_itp(input_itp)
    if len(topology.atoms) < 3:
        raise ValueError(
            "Angle restraints require a topology with at least three beads."
        )
    print(f"## Loaded topology {input_itp} with {len(topology.atoms)} beads.")

    records = _read_angle_records(args.angle_list)
    residue_to_atom = {}
    for atom_index, atom in enumerate(topology.atoms):
        if atom.residueid in residue_to_atom:
            raise ValueError(
                f"Residue ID {atom.residueid} occurs more than once; "
                "addangle requires one bead per residue."
            )
        residue_to_atom[atom.residueid] = atom_index

    if topology.angles is None:
        topology.angles = []
    terminal_residues = {
        topology.atoms[0].residueid,
        topology.atoms[-1].residueid,
    }
    added = 0
    for center_residue, angle_degrees, force_constant in records:
        if center_residue not in residue_to_atom:
            raise ValueError(
                f"Angle center residue {center_residue} is absent from the topology."
            )
        center_atom = residue_to_atom[center_residue]
        if center_residue in terminal_residues or center_atom in {
            0,
            len(topology.atoms) - 1,
        }:
            print(
                f"## WARNING: Ignoring terminal angle center residue {center_residue}."
            )
            continue

        atom_ids = (center_atom - 1, center_atom, center_atom + 1)
        topology.angles.append(
            Angle(
                *atom_ids,
                angle_degrees * degree,
                force_constant * kilojoule_per_mole / radian**2,
            )
        )
        print(
            f"## Added angle at residue {center_residue}: beads "
            f"{atom_ids[0] + 1}-{atom_ids[1] + 1}-{atom_ids[2] + 1}, "
            f"theta={angle_degrees:g} degree, k={force_constant:g} "
            "kJ mol^-1 rad^-2."
        )
        added += 1

    if added == 0:
        raise ValueError("No non-terminal angle restraints were added.")
    write_itp(output_itp, topology)
    print(f"## Wrote {added} angle restraint(s) to {output_itp}.")


prog = "addangle"
desc = "Add angle restraints to an ITP topology."


def getargs_addangle(argv):
    parser = ArgumentParser(prog=prog, description=desc)

    parser.add_argument(
        "-ip",
        "--input-topology",
        type=str,
        required=True,
        help="Input topology file (.itp).",
    )

    parser.add_argument(
        "-op",
        "--output-topology",
        type=str,
        required=True,
        help="Output topology file (.itp) with angle restraints; the extension is added if omitted.",
    )

    parser.add_argument(
        "-al",
        "--angle-list",
        type=str,
        required=True,
        help="Input text file with one 'BEAD_ID ANGLE_DEG FORCE_CONSTANT' record per line; bead IDs are 1-based.",
    )

    args = parser.parse_args(argv)

    return args


addangle_commands = single_command("addangle", getargs_addangle, addangle, desc)
