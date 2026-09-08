# seq2hoomd tool in dps package by Yiming Tang @ Fudan
# Development started on June 6 2025

from dropps.share.argument_parser import ArgumentParser
from dropps.share.command_class import single_command
import re
from tqdm import tqdm
import itertools

from dropps.share.forcefield import getff, forcefield_list
from dropps.fileio.pdb_reader import distance
import random
import math
from pathlib import Path
import os

import numpy as np


def _read_ca_coordinates(path):
    """Read finite C-alpha coordinates from the first PDB model, in nm."""

    coordinates = []
    model_started = False
    with open(path, encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            record = line[0:6].strip()
            if record == "MODEL":
                if model_started:
                    break
                model_started = True
                continue
            if record == "ENDMDL":
                break
            if record not in {"ATOM", "HETATM"}:
                continue
            if line[12:16].strip() != "CA":
                continue
            altloc = line[16:17]
            if altloc not in {"", " ", "A"}:
                continue
            try:
                coordinate = [
                    float(line[30:38]) / 10.0,
                    float(line[38:46]) / 10.0,
                    float(line[46:54]) / 10.0,
                ]
            except (ValueError, IndexError) as exc:
                raise ValueError(
                    f"Malformed C-alpha coordinates in {path!r} on line {line_number}."
                ) from exc
            if not np.all(np.isfinite(coordinate)):
                raise ValueError(
                    f"Non-finite C-alpha coordinates in {path!r} on line {line_number}."
                )
            coordinates.append(coordinate)

    if not coordinates:
        raise ValueError(f"Input PDB {path!r} contains no C-alpha atoms.")
    values = np.asarray(coordinates, dtype=float)
    center = 0.5 * (values.min(axis=0) + values.max(axis=0))
    return values - center


def _write_coulomb_global(itp_file, forcefield):
    itp_file.write("[ coulomb_global ]\n")
    if (
        getattr(forcefield, "relative_permittivity_mode", "constant")
        == "temperature_dependent"
    ):
        coeffs = forcefield.relative_permittivity_coeffs
        itp_file.write(
            "; relative_permittivity(T) = k_-1/T + k0 + k1*T + k2*T^2 + k3*T^3\n"
        )
        itp_file.write(
            f"  {coeffs[0]:.12g} {coeffs[1]:.12g} {coeffs[2]:.12g} {coeffs[3]:.12g} {coeffs[4]:.12g}\n\n"
        )
    else:
        itp_file.write("; relative_permittivity\n")
        itp_file.write(f"  {forcefield.relative_permittivity:.8f}\n\n")


def _write_simulation_settings(itp_file, forcefield):
    simulation_settings = getattr(forcefield, "simulation_settings", [])
    if len(simulation_settings) == 0:
        return

    itp_file.write("[ simulation_setting ]\n")
    itp_file.write("; name value type_at_NVT type_at_NPT\n")
    for setting in simulation_settings:
        itp_file.write(
            f"  {setting['name']}  {setting['value']}  {setting['nvt_policy']}  {setting['npt_policy']}\n"
        )
    itp_file.write("\n")


def pdb2cgps(args):
    args.sequence = "".join(args.sequence.split())
    if not args.sequence:
        raise ValueError("Sequence must contain at least one residue or nucleotide.")
    if args.output_conformation is None and args.output_topology is None:
        raise ValueError(
            "At least one output is required: --output-conformation or "
            "--output-topology."
        )
    if not math.isfinite(args.radius) or args.radius <= 0.0:
        raise ValueError("Conformation radius must be a positive finite number.")
    if args.number <= 0:
        raise ValueError("Number of conformations must be a positive integer.")
    if args.max_attempts <= 0:
        raise ValueError("Maximum conformation attempts must be positive.")
    if not math.isfinite(args.degree_extend) or not 0.0 <= args.degree_extend <= 1.0:
        raise ValueError("Degree of extension must be between 0 and 1.")

    rng = random.Random(args.seed)

    first_residue_index = args.residue_index
    sequence_length = len(args.sequence)

    # Get sequence for this protein/molecule

    print(f"## Raw sequence: {args.sequence}")
    sequence_abbreviation = list(args.sequence)

    if args.post_translational_modification is not None:
        for ptm in args.post_translational_modification:
            match = re.fullmatch(r"([A-Za-z]+)(\d+)([A-Za-z]+)", ptm)
            if not match:
                print(
                    "ERROR: Cannot process post translational modification " + ptm + "."
                )
                quit()
            original = match.group(1)
            residue_number = int(match.group(2))
            if (
                residue_number < first_residue_index
                or residue_number > first_residue_index + sequence_length - 1
            ):
                print(
                    "ERROR: Mutated residue %d out of range of %d to %d."
                    % (
                        residue_number,
                        first_residue_index,
                        first_residue_index + sequence_length - 1,
                    )
                )
                quit()
            mutant = match.group(3)[0] + match.group(3)[1:]
            print(
                "## Residue %s%d will be mutated to %s."
                % (original, residue_number, mutant)
            )
            if sequence_abbreviation[residue_number - first_residue_index] != original:
                print(
                    "ERROR When processing post translational modification %s, residue %d is not %s."
                    % (ptm, residue_number, original)
                )
                quit()
            else:
                sequence_abbreviation[residue_number - first_residue_index] = mutant

    if args.charged_NTD:
        print("## N terminal will be patched by an additional positive charge.")
        sequence_abbreviation[0] += "_N"
    else:
        print("## N terminal (mainchain) will be neutral.")

    if args.charged_CTD:
        print("## C terminal will be patched by an additional negative charge.")
        sequence_abbreviation[-1] += "_C"
    else:
        print("## C terminal (mainchain) will be neutral.")

    print("## Sequence: " + ",".join(sequence_abbreviation))

    # Define file for structure and topology generation

    degree_extend = float(args.degree_extend)

    if args.output_conformation is not None:
        conformation_path = Path(args.output_conformation)
        conformation_file_prefix = str(
            conformation_path.with_suffix("")
            if conformation_path.suffix.lower() == ".pdb"
            else conformation_path
        )

        if args.number == 1:
            conformation_file_name_list = [conformation_file_prefix + ".pdb"]
        else:
            conformation_file_name_list = [
                conformation_file_prefix + "_%d.pdb" % (i + 1)
                for i in range(args.number)
            ]

        print(
            "## Will generate conformation file at "
            + ", ".join(conformation_file_name_list)
        )

    if args.output_topology is not None:
        topology_path = Path(args.output_topology)
        topology_file_prefix = str(
            topology_path.with_suffix("")
            if topology_path.suffix.lower() == ".itp"
            else topology_path
        )
        topology_file_name = topology_file_prefix + ".itp"

        print("## Will generate topology file at " + topology_file_name)

    current_file_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    forcefields_dir = Path(current_file_dir) / "share" / "forcefields"
    files1 = sorted(
        path for path in forcefields_dir.glob("*.ff") if not path.name.startswith("._")
    )
    cwd = Path.cwd()
    files2 = sorted(path for path in cwd.glob("*.ff") if not path.name.startswith("._"))

    # Get base names for conflict checking
    names1 = set(f.name for f in files1)
    names2 = set(f.name for f in files2)

    # Check for conflicts
    conflicts = names1 & names2
    if conflicts:
        print(
            "ERROR: Conflict detected! The following .ff file(s) exist in both system and working directory:"
        )
        for name in conflicts:
            print(f"  {name}")
        quit()

    # Combine and make a list of paths
    all_files = files1 + files2

    # Print basename list
    if not all_files:
        print("ERROR: No forcefields found.")
        quit()
    else:
        print("## Available forcefields:")
        for idx, f in enumerate(all_files, start=1):
            print(f"{idx}: {f.name}")

        # Let user select

        if args.forcefield is not None:
            filenames = [
                file
                for file in all_files
                if os.path.basename(file) == args.forcefield + ".ff"
            ]

            if len(filenames) == 0:
                print(f"ERROR: Unknown forcefield {args.forcefield}.")
                quit()
            selected_file_path = Path(filenames[0])
        else:
            while True:
                try:
                    choice = int(input("Select a file by index: "))
                    if 1 <= choice <= len(all_files):
                        break
                    else:
                        print("Invalid choice. Try again.")
                except ValueError:
                    print("Please enter a valid integer.")

            selected_file_path = all_files[choice - 1]
        print(f"## Selected forcefield: {selected_file_path.name}")

        # Save path in parameter
        parameter_file_path = selected_file_path

    forcefield = getff(parameter_file_path)

    missing_residues = sorted(
        abbreviation
        for abbreviation in set(sequence_abbreviation)
        if abbreviation not in forcefield.abbr2aa
    )
    if missing_residues:
        raise ValueError(
            f"Force field {selected_file_path.stem} does not define residue "
            f"type(s): {', '.join(missing_residues)}."
        )

    print("## Forcefield successfully processed.")

    def generate_chain_conformation_single():
        print("## Start generating chain conformation.")
        # chain_succeed is a flag for success of a single chain containing numbers of beads.

        if len(sequence_abbreviation) == 1:
            return [[0.0, 0.0, 0.0]]

        for conformation_attempt in range(1, args.max_attempts + 1):
            # We first get coordinate for the first amino acid
            beads = [[0, 0, 0]]

            # We get coordinate for the second amino acid
            theta = rng.uniform(0, math.pi)
            phi = rng.uniform(0, 2 * math.pi)

            atom1 = sequence_abbreviation[0]
            atom2 = sequence_abbreviation[1]
            bondtype = forcefield.abbr2bondtypeindex[f"{atom1}-{atom2}"]
            bondname = forcefield.bondtypes[bondtype]
            bond_length = forcefield.bond2param[bondname]["length"]

            temp_x = bond_length * math.sin(theta) * math.cos(phi)
            temp_y = bond_length * math.sin(theta) * math.sin(phi)
            temp_z = bond_length * math.cos(theta)
            beads.append([temp_x, temp_y, temp_z])

            breaked = False
            # We get coordinate for the remaining amino acids
            for i in tqdm(range(2, len(sequence_abbreviation))):
                if breaked:
                    break

                last_vector = [
                    beads[i - 1][0] - beads[i - 2][0],
                    beads[i - 1][1] - beads[i - 2][1],
                    beads[i - 1][2] - beads[i - 2][2],
                ]

                atom1 = sequence_abbreviation[i - 1]
                atom2 = sequence_abbreviation[i]
                bondtype = forcefield.abbr2bondtypeindex[f"{atom1}-{atom2}"]
                bondname = forcefield.bondtypes[bondtype]
                bond_length = forcefield.bond2param[bondname]["length"]

                succeed_single_bead = False
                try_time_single_bead = 0

                while not succeed_single_bead and try_time_single_bead < 50:
                    # We generate a random bead coordinate
                    theta = rng.uniform(0, math.pi)
                    phi = rng.uniform(0, 2 * math.pi)
                    this_vector = [
                        bond_length * math.sin(theta) * math.cos(phi),
                        bond_length * math.sin(theta) * math.sin(phi),
                        bond_length * math.cos(theta),
                    ]
                    result_vector = [
                        last_vector[j] * degree_extend
                        + this_vector[j] * (1 - degree_extend)
                        for j in range(3)
                    ]
                    result_vector_length = distance(result_vector, [0, 0, 0])
                    result_vector = [
                        j / result_vector_length * bond_length for j in result_vector
                    ]
                    temp_coordinate = [
                        beads[i - 1][j] + result_vector[j] for j in range(3)
                    ]
                    # We check if this bead has overlap with previous beads
                    has_overlap = False
                    for exist_bead in beads[0:-2]:
                        if distance(exist_bead, temp_coordinate) < bond_length * 1.2:
                            has_overlap = True

                    rg_too_big = False
                    if (
                        max(temp_coordinate) > args.radius
                        or min(temp_coordinate) < -args.radius
                    ):
                        rg_too_big = True

                    # If there is no overlap, we add this bead
                    if (not has_overlap) and (not rg_too_big):
                        beads.append(temp_coordinate)
                        succeed_single_bead = True
                    try_time_single_bead += 1

                if try_time_single_bead >= 50:
                    breaked = True

            if len(beads) == len(sequence_abbreviation):
                print(
                    "## Successfully generated one chain conformation "
                    f"after {conformation_attempt} attempt(s)."
                )
                return beads

        raise RuntimeError(
            "Could not generate a non-overlapping conformation within "
            f"{args.max_attempts} attempts. Increase --radius, reduce "
            "--degree-extend, or increase --max-attempts."
        )

    # We read structure file if given

    if args.input_pdb is None:
        specify_structure_pdb = False
    else:
        specify_structure_pdb = True

        if len(args.input_pdb) < 5 or args.input_pdb[-4:] != ".pdb":
            print(
                f"## ERROR: The input structure file {args.input_pdb} is not a pdb file."
            )
            quit()

        CA_coordinates = _read_ca_coordinates(args.input_pdb)
        print(
            f"## Read and centered the first-model C-alpha trace from {args.input_pdb}."
        )

        if len(CA_coordinates) != sequence_length:
            print(
                f"## ERROR: There are {len(CA_coordinates)} CA atoms in {args.input_pdb} but sequence has length of {sequence_length}."
            )
            quit()

    # We first generate conformations for this chain and write to PDB file.
    # Remember PDB file is in angstrom format instead of nanometer.

    if args.output_conformation is not None:
        for conformation_file in conformation_file_name_list:
            print(f"## Generating conformation for {conformation_file}.")

            atom_positions = (
                CA_coordinates
                if specify_structure_pdb
                else generate_chain_conformation_single()
            )
            coordinate_span = np.ptp(np.asarray(atom_positions, dtype=float), axis=0)
            box_size = max(args.radius * 2, float(np.max(coordinate_span)) + 1.0)

            with open(conformation_file, "w", encoding="utf-8") as pdb_file:
                pdb_file.write(
                    f"CRYST1{box_size * 10:9.3f}{box_size * 10:9.3f}{box_size * 10:9.3f}  90.00  90.00  90.00 P 1           1\n"
                )

                for i, (abbr, pos) in enumerate(
                    zip(sequence_abbreviation, atom_positions), 1
                ):
                    x, y, z = pos

                    # PDB atom line: fixed-width format
                    # Columns: https://www.wwpdb.org/documentation/file-format-content/format33/sect9.html#ATOM
                    pdb_file.write(
                        "{:<6s}{:>5d} {:<4s} {:>3s} {:1s}{:>4d}    "
                        "{:>8.3f}{:>8.3f}{:>8.3f}{:6.2f}{:6.2f}          {:>2s}\n".format(
                            "ATOM",  # Record name
                            i,  # Atom serial number
                            abbr[:4],  # Atom name, left aligned, max 4 chars
                            forcefield.abbr2aa[abbr][:3],  # Residue name
                            "A",  # Chain ID
                            i + first_residue_index - 1,  # Residue sequence number
                            (x + box_size / 2) * 10,
                            (y + box_size / 2) * 10,
                            (z + box_size / 2) * 10,  # Coordinates
                            1.00,
                            0.00,  # Occupancy, B-factor
                            abbr[-1]
                            if len(abbr) == 1
                            else abbr[0],  # Element symbol (fallback)
                        )
                    )

                pdb_file.write("END\n")

    # We next generate topology for this chain and write to itp file.

    if args.output_topology is not None:
        with open(topology_file_name, "w", encoding="utf-8") as itp_file:
            if (
                forcefield.function_type_LJ == "Ashbaugh-Hatch"
                and forcefield.function_type_Coulomb == "Debye-Huckel"
            ):
                print(f"## Generating topology to {topology_file_name}.")

                # Write forcefield and information lines
                print(
                    f"## The molecule will be named as {args.output_name} in topology file."
                )

                itp_file.write("[ moleculetype ]\n; molname  nrexcl\n")
                itp_file.write(f"{args.output_name}     1\n\n")

                # Write function type
                itp_file.write("[ function-type ]\n")
                itp_file.write("; LJ_function   Coulomb_function\n")
                itp_file.write(
                    f"  {forcefield.function_type_LJ}   {forcefield.function_type_Coulomb}\n\n"
                )

                _write_coulomb_global(itp_file, forcefield)
                _write_simulation_settings(itp_file, forcefield)

                # Write atom type information
                itp_file.write("[ atomtypes ]\n")
                itp_file.write(
                    "; atom-abbr     atom-name  sigma   lambda   T0               T1              T2\n"
                )
                for abbr in dict.fromkeys(sequence_abbreviation):
                    itp_file.write(
                        f"  {abbr:12s}  {forcefield.abbr2aa[abbr][:3]:7s}   "
                        + f"{forcefield.abbr2sigma[abbr]:>6.3f}   {forcefield.abbr2lambda[abbr]:.3f}   "
                        + f"{forcefield.abbr2tempcoff[abbr][0]:>13.9f}   "
                        + f"{forcefield.abbr2tempcoff[abbr][1]:>13.9f}   "
                        + f"{forcefield.abbr2tempcoff[abbr][2]:>13.9f}   \n"
                    )
                itp_file.write("\n")

                itp_file.write("[ nonbond_global ]\n")
                itp_file.write("; epsilon(kJ/mol)\n")
                itp_file.write(f"  {forcefield.epsilon:.8f}\n\n")

                # Write atom information
                itp_file.write("[ atoms ]\n")
                itp_file.write(
                    "; id   atom-abbr  atom-name     residue  resid  mass    charge\n"
                )
                qtot = 0
                for index, abbr in enumerate(sequence_abbreviation):
                    qtot += forcefield.abbr2charge[abbr]
                    itp_file.write(
                        f"  {index + 1:<3d}  {abbr:<8s}   {forcefield.abbr2aa[abbr]:12s}  {forcefield.abbr2aa[abbr][:3]:7s}  {index + first_residue_index:<5d}  "
                        + f"{forcefield.abbr2mass[abbr]:>6.2f}  {forcefield.abbr2charge[abbr]:5.2f}   ; qtot {qtot}\n"
                    )
                itp_file.write("\n")

                # Write bond information
                itp_file.write("[ bonds ]\n")
                itp_file.write("; ai   aj   r0/nm k/(kJ/mol)/nm^2\n")

                for index in range(len(sequence_abbreviation) - 1):
                    bondparameters = forcefield.bond2param[
                        forcefield.abbr2bondtype[
                            f"{sequence_abbreviation[index]}-{sequence_abbreviation[index + 1]}"
                        ]
                    ]
                    itp_file.write(
                        f"  {(index + 1):<3d}  {(index + 2):<3d}  {bondparameters['length']:.2f}  {bondparameters['k']}\n"
                    )
                itp_file.write("\n")

            elif (
                forcefield.function_type_LJ == "Wang-Frenkel"
                and forcefield.function_type_Coulomb == "Debye-Huckel"
            ):
                print(f"## Generating topology to {topology_file_name}.")

                # Write forcefield and information lines
                print(
                    f"## The molecule will be named as {args.output_name} in topology file."
                )

                itp_file.write("[ moleculetype ]\n; molname  nrexcl\n")
                itp_file.write(f"{args.output_name}     1\n\n")

                # Write function type
                itp_file.write("[ function-type ]\n")
                itp_file.write("; LJ_function   Coulomb_function\n")
                itp_file.write(
                    f"  {forcefield.function_type_LJ}   {forcefield.function_type_Coulomb}\n\n"
                )

                _write_coulomb_global(itp_file, forcefield)
                _write_simulation_settings(itp_file, forcefield)

                # Write atom type information
                itp_file.write("[ atomtypes ]\n")
                itp_file.write("; atom-abbr     atom-name \n")
                all_wf_types = sorted(set(forcefield.abbr))
                for abbr in all_wf_types:
                    itp_file.write(
                        f"  {abbr:12s}  {forcefield.abbr2aa[abbr][:3]:7s}   \n"
                    )
                itp_file.write("\n")

                # Write sigma, lambda, epsilon information
                itp_file.write("[ nonbond_params ]\n")
                itp_file.write("; atom-abbr-1  atom-abbr-2   sigma   mu   epsilon\n")
                for abbr_1, abbr_2 in itertools.combinations_with_replacement(
                    all_wf_types, 2
                ):
                    pair = f"{abbr_1}-{abbr_2}"
                    pair_rev = f"{abbr_2}-{abbr_1}"
                    if pair in forcefield.abbrs2sigma:
                        sigma = forcefield.abbrs2sigma[pair]
                        mu = forcefield.abbrs2mu[pair]
                        epsilon = forcefield.abbrs2epsilon[pair]
                    elif pair_rev in forcefield.abbrs2sigma:
                        sigma = forcefield.abbrs2sigma[pair_rev]
                        mu = forcefield.abbrs2mu[pair_rev]
                        epsilon = forcefield.abbrs2epsilon[pair_rev]
                    else:
                        print(
                            f"ERROR: Missing Wang-Frenkel nonbonded parameter for pair {pair}."
                        )
                        quit()
                    itp_file.write(
                        f"  {abbr_1:12s}  {abbr_2:12s}   {sigma:>6.3f}   {mu:.6f}   {epsilon:.6f}\n"
                    )
                itp_file.write("\n")

                # Write atom information
                itp_file.write("[ atoms ]\n")
                itp_file.write(
                    "; id   atom-abbr  atom-name     residue  resid  mass    charge\n"
                )
                qtot = 0
                for index, abbr in enumerate(sequence_abbreviation):
                    qtot += forcefield.abbr2charge[abbr]
                    itp_file.write(
                        f"  {index + 1:<3d}  {abbr:<8s}   {forcefield.abbr2aa[abbr]:12s}  {forcefield.abbr2aa[abbr][:3]:7s}  {index + first_residue_index:<5d}  "
                        + f"{forcefield.abbr2mass[abbr]:>6.2f}  {forcefield.abbr2charge[abbr]:5.2f}   ; qtot {qtot}\n"
                    )
                itp_file.write("\n")

                # Write bond information
                itp_file.write("[ bonds ]\n")
                itp_file.write("; ai   aj   r0/nm k/(kJ/mol)/nm^2\n")

                for index in range(len(sequence_abbreviation) - 1):
                    bondparameters = forcefield.bond2param[
                        forcefield.abbr2bondtype[
                            f"{sequence_abbreviation[index]}-{sequence_abbreviation[index + 1]}"
                        ]
                    ]
                    itp_file.write(
                        f"  {(index + 1):<3d}  {(index + 2):<3d}  {bondparameters['length']:.2f}  {bondparameters['k']}\n"
                    )
                itp_file.write("\n")

            else:
                print(
                    f"## ERROR: In pdb2dps, Unknown forcefield function types: LJ-{forcefield.function_type_LJ}, Coulomb-{forcefield.function_type_Coulomb}."
                )
                quit()


prog = "pdb2dps"
desc = "Generate coarse-grained PDB conformations and an ITP topology from a protein sequence."


def getargs_pdb2cgps(argv):
    # Command line argument parser

    parser = ArgumentParser(prog=prog, description=desc)

    parser.add_argument(
        "-s", "--sequence", type=str, help="One-letter protein sequence.", required=True
    )
    parser.add_argument(
        "-f",
        "--input-pdb",
        type=str,
        help="Optional all-atom PDB file (.pdb) used to initialize C-alpha coordinates.",
    )
    parser.add_argument(
        "-ri",
        "--residue-index",
        type=int,
        help="Residue number assigned to the first bead.",
        default=1,
    )
    parser.add_argument(
        "-ptm",
        "--post-translational-modification",
        type=str,
        nargs="+",
        help="Post-translational modifications in ORIGINAL+NUMBER+MODIFIED form, for example S129SMP.",
    )
    parser.add_argument(
        "-r",
        "--radius",
        type=float,
        help="Maximum radius of gyration for a generated conformation, in nm.",
        default=2.0,
    )
    parser.add_argument(
        "-n",
        "--number",
        type=int,
        help="Number of conformations to generate.",
        default=1,
    )
    parser.add_argument(
        "-e",
        "--degree-extend",
        type=float,
        help="Chain-extension fraction between 0 and 1.",
        default=0.5,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1215,
        help="Random seed used for reproducible conformation generation.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=10000,
        help="Maximum whole-chain construction attempts before failing.",
    )
    parser.add_argument(
        "-ff",
        "--forcefield",
        choices=forcefield_list,
        help="Force field to use; if omitted, prompt interactively.",
    )
    parser.add_argument(
        "-oc",
        "--output-conformation",
        type=str,
        help="Output PDB path or prefix; multiple conformations use numbered .pdb files.",
        required=False,
    )
    parser.add_argument(
        "-op",
        "--output-topology",
        type=str,
        help="Output topology file (.itp); the extension is added if omitted.",
        required=False,
    )
    parser.add_argument(
        "-on",
        "--output-name",
        type=str,
        default="MOL",
        help="Molecule name written to the ITP topology.",
    )
    parser.add_argument(
        "-cNTD",
        "--charged-NTD",
        action="store_true",
        default=False,
        help="Add a positive charge patch to the N terminus.",
    )
    parser.add_argument(
        "-cCTD",
        "--charged-CTD",
        action="store_true",
        default=False,
        help="Add a negative charge patch to the C terminus.",
    )

    args = parser.parse_args(argv)

    return args


pdb2dps_commands = single_command("pdb2dps", getargs_pdb2cgps, pdb2cgps, desc)
