# genmesh tool in dps package by Yiming Tang @ Fudan
# Development started on June 6 2025

from dropps.share.argument_parser import ArgumentParser
from dropps.share.command_class import single_command
import numpy as np
from dropps.fileio.pdb_reader import read_pdb, write_pdb
import string
from dropps.fileio.itp_reader import read_itp
from dropps.fileio.filename_control import validate_extension

from openmm.unit import nanometer


def _chain_id_for_molecule(molecule_index):
    """Return a valid one-column PDB chain ID without a molecule-count limit."""

    if molecule_index < 0:
        raise ValueError("Molecule index must be non-negative.")
    return string.ascii_uppercase[molecule_index % len(string.ascii_uppercase)]


def _forcefield_function_types(topologies):
    """Return the common LJ and Coulomb function types or reject a mixture."""

    if not topologies:
        raise ValueError("At least one topology is required.")
    lj_types = sorted({topology.function_type_LJ for topology in topologies})
    coulomb_types = sorted({topology.function_type_Coulomb for topology in topologies})
    if len(lj_types) != 1 or len(coulomb_types) != 1:
        raise ValueError(
            "Different forcefield function types found in input topology files: "
            f"LJ={lj_types}, Coulomb={coulomb_types}."
        )
    return lj_types[0], coulomb_types[0]


def _molecule_center_nm(molecule):
    """Return the center of a molecule's coordinate bounding box in nm."""
    if not molecule:
        raise ValueError("Input structures must contain at least one atom.")
    coordinates = np.asarray(
        [
            [
                atom["x"].value_in_unit(nanometer),
                atom["y"].value_in_unit(nanometer),
                atom["z"].value_in_unit(nanometer),
            ]
            for atom in molecule
        ],
        dtype=float,
    )
    return 0.5 * (coordinates.min(axis=0) + coordinates.max(axis=0))


def _merge_molecule_counts(names, counts):
    """Merge duplicate topology molecule names while preserving first order."""
    merged = {}
    for name, count in zip(names, counts):
        merged[name] = merged.get(name, 0) + count
    return list(merged), list(merged.values())


def _apply_box_type(box_size, box_type):
    """Apply the requested orthorhombic box geometry."""

    normalized = "anisotropy" if box_type == "anosotropy" else box_type
    x_length, y_length, z_length = map(float, box_size)
    if normalized == "cubic":
        side = max(x_length, y_length, z_length)
        return [side, side, side]
    if normalized == "xy":
        lateral = max(x_length, y_length)
        return [lateral, lateral, z_length]
    if normalized == "anisotropy":
        return [x_length, y_length, z_length]
    raise ValueError(f"Unknown box type {box_type!r}.")


def genmesh(args):
    if len(args.structure) != len(args.topology) or len(args.structure) != len(
        args.number
    ):
        raise ValueError(
            "Numbers of structure files, topology files, and molecule counts "
            "must match."
        )
    if not args.structure:
        raise ValueError("At least one molecule type is required.")
    if any(number <= 0 for number in args.number):
        raise ValueError("Every molecule count must be a positive integer.")
    if not np.isfinite(args.gap) or args.gap < 0.0:
        raise ValueError("Mesh gap must be a non-negative finite number.")
    minimum_lengths = (args.minimum_x, args.minimum_y, args.minimum_z)
    if any(not np.isfinite(length) or length < 0.0 for length in minimum_lengths):
        raise ValueError("Minimum box lengths must be non-negative finite numbers.")
    if args.mesh is not None and any(size <= 0 for size in args.mesh):
        raise ValueError("Mesh dimensions must be positive integers.")

    # We first get all pdb files
    if len(args.structure) != len(args.number):
        raise ValueError("Number of structure files and structure counts must match.")
    structure_number = len(args.structure)
    print(f"## Will process {structure_number} structure files.")

    try:
        molecule_list = [read_pdb(path)[0] for path in args.structure]
        topologies = [read_itp(path) for path in args.topology]
        molecule_number_list = args.number
    except (OSError, RuntimeError, ValueError) as exc:
        raise RuntimeError("Could not read a structure or topology file.") from exc

    for structure_path, topology_path, molecule, topology in zip(
        args.structure, args.topology, molecule_list, topologies
    ):
        if len(molecule) != len(topology.atoms):
            raise ValueError(
                f"Structure {structure_path!r} contains {len(molecule)} beads, "
                f"but topology {topology_path!r} defines {len(topology.atoms)}."
            )

    molecule_centers = [_molecule_center_nm(molecule) for molecule in molecule_list]

    for molecule_index in range(structure_number):
        print(
            "## Molecule %d contains %d residues, will be inserted for %d times."
            % (
                (molecule_index + 1),
                len(molecule_list[molecule_index]),
                molecule_number_list[molecule_index],
            )
        )

    ## We now generate the mesh
    molecule_number = np.sum(molecule_number_list)

    if args.mesh is not None:
        mesh = args.mesh
        if mesh[0] * mesh[1] * mesh[2] < molecule_number:
            raise ValueError(
                f"The required mesh {mesh[0]}*{mesh[1]}*{mesh[2]} cannot "
                f"contain {molecule_number} molecules."
            )
    else:
        mesh_x_y_z = int(np.ceil(np.power(molecule_number, 1 / 3)))
        mesh = [mesh_x_y_z, mesh_x_y_z, mesh_x_y_z]

    print(
        f"## The mesh for {molecule_number} molecule insertion will be {mesh[0]}*{mesh[1]}*{mesh[2]}."
    )

    # We first get the radius of each molecule

    if args.non_cubic_molecule:
        print(
            "## Non-cubic molecule option is set. The maximum diameter among x/y/z direction will be used for mesh generation."
        )
        diameter_x = np.max(
            [
                np.max([atom["x"].value_in_unit(nanometer) for atom in molecule])
                - np.min([atom["x"].value_in_unit(nanometer) for atom in molecule])
                for molecule in molecule_list
            ]
        )
        diameter_y = np.max(
            [
                np.max([atom["y"].value_in_unit(nanometer) for atom in molecule])
                - np.min([atom["y"].value_in_unit(nanometer) for atom in molecule])
                for molecule in molecule_list
            ]
        )
        diameter_z = np.max(
            [
                np.max([atom["z"].value_in_unit(nanometer) for atom in molecule])
                - np.min([atom["z"].value_in_unit(nanometer) for atom in molecule])
                for molecule in molecule_list
            ]
        )

        distance_between_mesh_point_x = diameter_x + args.gap
        distance_between_mesh_point_y = diameter_y + args.gap
        distance_between_mesh_point_z = diameter_z + args.gap

        print(
            f"The maximum diameter among all molecules is {diameter_x}, {diameter_y}, {diameter_z} in x/y/z direction respectively."
        )
        print(f"The minimun distance between wach two configuration is {args.gap}.")
        print(
            f"## The distance between each two adjancy mesh point is {distance_between_mesh_point_x}, {distance_between_mesh_point_y}, {distance_between_mesh_point_z} in x/y/z direction respectively."
        )

        box_size_mesh = [
            int(np.ceil(mesh[0] * distance_between_mesh_point_x + args.gap)),
            int(np.ceil(mesh[1] * distance_between_mesh_point_y + args.gap)),
            int(np.ceil(mesh[2] * distance_between_mesh_point_z + args.gap)),
        ]

        print(
            f"## The box size determined by mesh is larger than {box_size_mesh[0]}*{box_size_mesh[1]}*{box_size_mesh[2]}"
        )
        box_size = [
            float(max(box_size_mesh[0], args.minimum_x)),
            float(max(box_size_mesh[1], args.minimum_y)),
            float(max(box_size_mesh[2], args.minimum_z)),
        ]
        print(f"## The box size is set as {box_size[0]}*{box_size[1]}*{box_size[2]}")

        mesh_points_raw = np.array(
            [
                [
                    diameter_x / 2 + (diameter_x + args.gap) * x_index,
                    diameter_y / 2 + (diameter_y + args.gap) * y_index,
                    diameter_z / 2 + (diameter_z + args.gap) * z_index,
                ]
                for x_index in range(mesh[0])
                for y_index in range(mesh[1])
                for z_index in range(mesh[2])
            ]
        )

    else:
        diameter = np.ceil(
            np.max(
                [
                    [
                        np.max(
                            [atom["x"].value_in_unit(nanometer) for atom in molecule]
                        )
                        - np.min(
                            [atom["x"].value_in_unit(nanometer) for atom in molecule]
                        )
                        for molecule in molecule_list
                    ],
                    [
                        np.max(
                            [atom["y"].value_in_unit(nanometer) for atom in molecule]
                        )
                        - np.min(
                            [atom["y"].value_in_unit(nanometer) for atom in molecule]
                        )
                        for molecule in molecule_list
                    ],
                    [
                        np.max(
                            [atom["z"].value_in_unit(nanometer) for atom in molecule]
                        )
                        - np.min(
                            [atom["z"].value_in_unit(nanometer) for atom in molecule]
                        )
                        for molecule in molecule_list
                    ],
                ]
            )
        )

        print(f"## The maximum diameter for all configurations is {diameter}.")
        print(f"## The minimun distance between wach two configuration is {args.gap}.")
        distance_between_mesh_point = diameter + args.gap
        print(
            f"## The distance between each two adjancy mesh point is {distance_between_mesh_point}."
        )

        # We now determine the size of the box

        box_size_mesh = [
            int(np.ceil(mesh[index] * distance_between_mesh_point + args.gap))
            for index in range(3)
        ]
        print(
            f"## The box size determined by mesh is larger than {box_size_mesh[0]}*{box_size_mesh[1]}*{box_size_mesh[2]}"
        )
        box_size = [
            float(max(box_size_mesh[0], args.minimum_x)),
            float(max(box_size_mesh[1], args.minimum_y)),
            float(max(box_size_mesh[2], args.minimum_z)),
        ]
        print(f"## The box size is set as {box_size[0]}*{box_size[1]}*{box_size[2]}")

        # We then generate a list for mesh point (centroid of each molecule)

        mesh_points_raw = np.array(
            [
                [
                    diameter / 2 + (diameter + args.gap) * x_index,
                    diameter / 2 + (diameter + args.gap) * y_index,
                    diameter / 2 + (diameter + args.gap) * z_index,
                ]
                for x_index in range(mesh[0])
                for y_index in range(mesh[1])
                for z_index in range(mesh[2])
            ]
        )

    box_size = _apply_box_type(box_size, args.box_type)
    print(
        f"## Applied {('anisotropy' if args.box_type == 'anosotropy' else args.box_type)} "
        f"box geometry: {box_size[0]}*{box_size[1]}*{box_size[2]} nm."
    )

    if args.shuffle:
        rng = np.random.default_rng(args.seed)
        mesh_points_raw = rng.permutation(mesh_points_raw)
        print(f"## Insertion mesh shuffled reproducibly with seed {args.seed}.")
    else:
        print("## Insertion will be performed on left-to-right, down-to-up mode.")

    mesh_points = mesh_points_raw - np.mean(mesh_points_raw, axis=0)

    # We now generate information for all atoms

    record_name = list()
    serial_number = list()
    atom_name = list()
    residue_name = list()
    chain_ID = list()
    residue_sequence_number = list()
    x_nm = list()
    y_nm = list()
    z_nm = list()
    occupancy = list()
    b_factor = list()
    element_symbol = list()

    molecule_index = 0
    atom_index = 1

    molecule_length_list = list()
    for molecule_type_id in range(len(molecule_list)):
        molecule = molecule_list[molecule_type_id]
        molecule_center = molecule_centers[molecule_type_id]
        molecule_number = molecule_number_list[molecule_type_id]

        for molecule_id in range(molecule_number):
            molecule_length_list.append(len(molecule))
            mesh_center = mesh_points[molecule_index]
            chain_id = _chain_id_for_molecule(molecule_index)
            print(
                f"## Will write molecule {molecule_type_id + 1} (replica {molecule_id + 1}) on mesh point {mesh_center}."
            )

            for resindex, atom in enumerate(molecule):
                record_name.append("ATOM")
                serial_number.append(atom_index)
                atom_name.append(atom["name"])
                residue_name.append(atom["resname"])
                chain_ID.append(chain_id)
                residue_sequence_number.append(resindex + 1)
                x_nm.append(
                    atom["x"].value_in_unit(nanometer)
                    - molecule_center[0]
                    + mesh_center[0]
                    + box_size[0] / 2
                ) * nanometer
                y_nm.append(
                    atom["y"].value_in_unit(nanometer)
                    - molecule_center[1]
                    + mesh_center[1]
                    + box_size[1] / 2
                ) * nanometer
                z_nm.append(
                    atom["z"].value_in_unit(nanometer)
                    - molecule_center[2]
                    + mesh_center[2]
                    + box_size[2] / 2
                ) * nanometer
                occupancy.append(1)
                b_factor.append(0)
                element_symbol.append(atom["element"])

                atom_index += 1
            molecule_index += 1

    # We now write pdb file
    conformation_file_name = validate_extension(args.output_conformation, "pdb")

    write_pdb(
        conformation_file_name,
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
    )

    # We now write topology file

    output_topology_filename = validate_extension(args.output_topology, "top")

    # We test forcefield functions for these itp files
    forcefield_function_type_LJ, forcefield_function_type_Coulomb = (
        _forcefield_function_types(topologies)
    )
    print(
        "## All input topology files use forcefield function types: "
        f"LJ-{forcefield_function_type_LJ}, "
        f"Coulomb-{forcefield_function_type_Coulomb}."
    )

    molecule_names = [top.molecule_name for top in topologies]

    if len(set(molecule_names)) < len(molecule_names):
        print("###############################################################")
        print("## WARNING: Duplicate molecule name found in topology files. ##")
        print("## These files will be merged.                               ##")
        print("## Please make sure nothing is wrong.                        ##")
        print("###############################################################")

    new_molecule_name, new_molecule_number_list = _merge_molecule_counts(
        molecule_names,
        molecule_number_list,
    )

    topology_filename_remove_duplicate = list(dict.fromkeys(args.topology))
    with open(output_topology_filename, "w", encoding="utf-8") as output_topology_file:
        output_topology_file.write("[ itp files ]\n")
        for filename in topology_filename_remove_duplicate:
            output_topology_file.write(f"{filename}\n")

        output_topology_file.write("\n[ system ]\n")
        output_topology_file.write("# molecule   number\n")
        for molecule_name, molecule_number in zip(
            new_molecule_name, new_molecule_number_list
        ):
            output_topology_file.write(f"  {molecule_name:<9s}  {molecule_number}\n")


prog = "genmesh"
desc = "Pack one or more molecule types onto a three-dimensional simulation-box mesh."


def getargs_genmesh(argv):
    parser = ArgumentParser(prog=prog, description=desc)

    parser.add_argument(
        "-f",
        "--structure",
        nargs="+",
        type=str,
        required=True,
        help="Input structure files (.pdb), one per molecule type.",
    )
    parser.add_argument(
        "-p",
        "--topology",
        nargs="+",
        type=str,
        required=True,
        help="Input topology files (.itp), in the same order as --structure.",
    )
    parser.add_argument(
        "-n",
        "--number",
        type=int,
        nargs="+",
        required=True,
        help="Molecule counts, one integer for each input structure.",
    )
    parser.add_argument(
        "-g",
        "--gap",
        type=float,
        default=1,
        help="Minimum gap between neighboring molecules, in nm.",
    )
    parser.add_argument(
        "-mesh",
        "--mesh",
        type=int,
        nargs=3,
        help="Mesh dimensions NX NY NZ; if omitted, choose a cubic mesh automatically.",
    )
    # parser.add_argument('-gmesh', '--guess-mesh', type=bool, default=False,
    #                    help="Whether let the program guess the size of the mesh in x/y/z direction.")
    parser.add_argument(
        "-bt",
        "--box-type",
        type=str,
        choices=["xy", "cubic", "anisotropy", "anosotropy"],
        default="anisotropy",
        help="Box geometry; 'anosotropy' remains as a deprecated spelling alias.",
    )
    parser.add_argument(
        "-mx",
        "--minimum-x",
        type=float,
        help="Minimum box length along x, in nm; zero selects the smallest fitted length.",
        default=0,
    )
    parser.add_argument(
        "-my",
        "--minimum-y",
        type=float,
        help="Minimum box length along y, in nm; zero selects the smallest fitted length.",
        default=0,
    )
    parser.add_argument(
        "-mz",
        "--minimum-z",
        type=float,
        help="Minimum box length along z, in nm; zero selects the smallest fitted length.",
        default=0,
    )
    parser.add_argument(
        "-s",
        "--shuffle",
        action="store_true",
        default=False,
        help="Shuffle molecule types before inserting them on the mesh.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1215,
        help="Random seed used when --shuffle is enabled.",
    )

    parser.add_argument(
        "-ncm",
        "--non-cubic-molecule",
        action="store_true",
        default=False,
        help="Use separate molecular extents along x, y, and z instead of one cubic extent.",
    )

    parser.add_argument(
        "-oc",
        "--output-conformation",
        type=str,
        help="Output packed structure file (.pdb); the extension is added if omitted.",
        default="system.pdb",
    )
    parser.add_argument(
        "-op",
        "--output-topology",
        type=str,
        help="Output system topology file (.top); the extension is added if omitted.",
        default="system.top",
    )

    args = parser.parse_args(argv)

    return args


genmesh_commands = single_command("genmesh", getargs_genmesh, genmesh, desc)
