# generate elestic (genelestic) tool in CGPS.ng package by Yiming Tang @ Fudan
# Development started on June 8 2025

from dropps.share.argument_parser import ArgumentParser
from dropps.share.command_class import single_command
from itertools import combinations
from pathlib import Path
import math

import numpy as np

from openmm.unit import nanometer, kilojoule_per_mole

from dropps.fileio.pdb_reader import read_pdb
from dropps.fileio.itp_reader import read_itp, write_itp, Bond


def _read_elastic_groups(path, atom_count):
    groups = []
    with open(path, encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            line = raw_line.split("#", 1)[0].split(";", 1)[0].strip()
            if not line:
                continue
            try:
                group = [int(value) - 1 for value in line.split()]
            except ValueError as exc:
                raise ValueError(
                    f"Elastic-group line {line_number} contains a non-integer bead ID."
                ) from exc
            if len(group) < 2:
                raise ValueError(
                    f"Elastic-group line {line_number} must contain at least two beads."
                )
            if len(group) != len(set(group)):
                raise ValueError(
                    f"Elastic-group line {line_number} contains duplicate bead IDs."
                )
            invalid = [index + 1 for index in group if not 0 <= index < atom_count]
            if invalid:
                raise ValueError(
                    f"Elastic-group line {line_number} contains out-of-range bead "
                    f"IDs: {invalid}; valid IDs are 1-{atom_count}."
                )
            groups.append(group)
    if not groups:
        raise ValueError(f"Elastic-group file {path!r} contains no groups.")
    return groups


def genelastic(args):
    if Path(args.topology).suffix.lower() != ".itp":
        raise ValueError("Elastic-network topology input must use the .itp extension.")
    if not math.isfinite(args.elastic_lower) or args.elastic_lower < 0.0:
        raise ValueError("Elastic lower cutoff must be non-negative and finite.")
    if (
        not math.isfinite(args.elastic_upper)
        or args.elastic_upper <= args.elastic_lower
    ):
        raise ValueError(
            "Elastic upper cutoff must be finite and larger than the lower cutoff."
        )
    if (
        not math.isfinite(args.elastic_force_constant)
        or args.elastic_force_constant <= 0.0
    ):
        raise ValueError("Elastic force constant must be positive and finite.")

    output_path = Path(args.output)
    if output_path.suffix.lower() != ".itp":
        output_path = output_path.with_suffix(".itp")

    atoms, _ = read_pdb(args.structure)
    topology = read_itp(args.topology)
    if len(atoms) != len(topology.atoms):
        raise ValueError(
            f"Reference structure contains {len(atoms)} beads, but topology "
            f"defines {len(topology.atoms)}."
        )
    groups = _read_elastic_groups(args.elastic_residues, len(atoms))

    print(f"## Reference structure contains {len(groups)} elastic group(s).")
    print(
        f"## Adding nonbonded bead pairs within {args.elastic_lower:g}-"
        f"{args.elastic_upper:g} nm at k={args.elastic_force_constant:g} "
        "kJ mol^-1 nm^-2."
    )

    existing_pairs = {
        tuple(sorted((bond.a1, bond.a2))) for bond in (topology.bonds or [])
    }
    added_pairs = set()
    coordinates_nm = np.asarray(
        [
            [
                atom["x"].value_in_unit(nanometer),
                atom["y"].value_in_unit(nanometer),
                atom["z"].value_in_unit(nanometer),
            ]
            for atom in atoms
        ],
        dtype=float,
    )
    if topology.bonds is None:
        topology.bonds = []

    for group in groups:
        for atom_1, atom_2 in combinations(group, 2):
            pair = tuple(sorted((atom_1, atom_2)))
            if pair in existing_pairs or pair in added_pairs:
                continue
            distance_nm = float(
                np.linalg.norm(coordinates_nm[atom_1] - coordinates_nm[atom_2])
            )
            if args.elastic_lower < distance_nm < args.elastic_upper:
                topology.bonds.append(
                    Bond(
                        atom_1,
                        atom_2,
                        distance_nm * nanometer,
                        args.elastic_force_constant * kilojoule_per_mole / nanometer**2,
                    )
                )
                added_pairs.add(pair)

    write_itp(output_path, topology)
    print(f"## Wrote {len(added_pairs)} new elastic bond(s) to {output_path}.")


prog = "genelastic"
desc = "Add a distance-based elastic network to an ITP topology."


def getargs_genelastic(argv):
    parser = ArgumentParser(prog=prog, description=desc)

    parser.add_argument(
        "-f",
        "--structure",
        type=str,
        required=True,
        help="Input reference structure file (.pdb).",
    )

    parser.add_argument(
        "-p", "--topology", type=str, required=True, help="Input topology file (.itp)."
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Output topology file (.itp); the extension is added if omitted.",
    )

    parser.add_argument(
        "-er",
        "--elastic-residues",
        type=str,
        required=True,
        help="Input text file listing one elastic-network bead group per line, using 1-based bead indices.",
    )

    parser.add_argument(
        "-ef",
        "--elastic-force-constant",
        type=float,
        default=5000,
        help="Elastic-bond force constant, in kJ mol^-1 nm^-2.",
    )

    parser.add_argument(
        "-el",
        "--elastic-lower",
        type=float,
        default=0.5,
        help="Lower distance cutoff for elastic bonds, in nm.",
    )

    parser.add_argument(
        "-eu",
        "--elastic-upper",
        type=float,
        default=0.9,
        help="Upper distance cutoff for elastic bonds, in nm.",
    )

    args = parser.parse_args(argv)
    return args


genelastic_commands = single_command("genelastic", getargs_genelastic, genelastic, desc)
