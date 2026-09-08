from pathlib import Path
from collections import defaultdict
from dropps.fileio.itp_reader import read_itp
from dropps.fileio.pdb_reader import read_pdb
import openmm
import openmm.app
from openmm.unit import nanometer, kilojoule_per_mole, radian, dimensionless
import math


from dropps.hp.constants import kappa_coefficient, k0

import numpy as np


def _read_system_topology(top_file):
    """Read and validate the manuscript-defined TOP file sections."""

    top_path = Path(top_file).resolve()
    sections = defaultdict(list)
    current_section = None
    try:
        lines = top_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"Cannot read system topology {top_file!r}: {exc}") from exc

    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.split(";", 1)[0].split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            current_section = (
                line[1:-1].strip().lower().replace("-", "_").replace(" ", "_")
            )
            continue
        if current_section is None:
            raise ValueError(
                f"System topology line {line_number} occurs before any section."
            )
        sections[current_section].append((line_number, line))

    for required in ("itp_files", "system"):
        if not sections.get(required):
            raise ValueError(
                f"System topology must contain a non-empty "
                f"[ {required.replace('_', ' ')} ] section."
            )

    itp_paths = []
    for line_number, entry in sections["itp_files"]:
        fields = entry.split()
        if len(fields) != 1:
            raise ValueError(
                f"System topology line {line_number} must contain one ITP path."
            )
        itp_path = Path(fields[0])
        if not itp_path.is_absolute():
            itp_path = top_path.parent / itp_path
        itp_paths.append(itp_path.resolve())

    if len(itp_paths) != len(set(itp_paths)):
        raise ValueError("The [ itp files ] section contains duplicate paths.")

    loaded_topologies = [read_itp(path) for path in itp_paths]
    topology_by_name = {}
    for topology, path in zip(loaded_topologies, itp_paths):
        if topology.molecule_name in topology_by_name:
            raise ValueError(
                f"Molecule name {topology.molecule_name!r} is defined by more "
                "than one ITP file."
            )
        topology_by_name[topology.molecule_name] = topology

    molecule_counts = []
    seen_molecules = set()
    for line_number, entry in sections["system"]:
        fields = entry.split()
        if len(fields) != 2:
            raise ValueError(
                f"System topology line {line_number} must contain MOLECULE_NAME COUNT."
            )
        molecule_name, raw_count = fields
        if molecule_name in seen_molecules:
            raise ValueError(
                f"Molecule {molecule_name!r} occurs more than once in [ system ]."
            )
        try:
            count = int(raw_count)
        except ValueError as exc:
            raise ValueError(
                f"Molecule count on line {line_number} must be an integer."
            ) from exc
        if count <= 0:
            raise ValueError(f"Molecule count on line {line_number} must be positive.")
        if molecule_name not in topology_by_name:
            raise ValueError(
                f"Molecule {molecule_name!r} in [ system ] has no matching ITP."
            )
        seen_molecules.add(molecule_name)
        molecule_counts.append((molecule_name, count))

    unused = sorted(set(topology_by_name).difference(seen_molecules))
    if unused:
        raise ValueError(
            "ITP molecule definitions are not used in [ system ]: " + ", ".join(unused)
        )
    return molecule_counts, itp_paths, loaded_topologies, topology_by_name


def _build_exclusion_pairs(topology_list, addition_per_molecule):
    """Build unique per-molecule 1-2/1-3 exclusions."""
    exclusion_pairs = set()

    for topology, addition in zip(topology_list, addition_per_molecule):
        nrexcl = topology.nrexcl
        if nrexcl not in (1, 2):
            raise ValueError(
                f"Only 1-2 and 1-3 exclusions are supported; "
                f"molecule {topology.molecule_name} has nrexcl={nrexcl}."
            )

        neighbors = defaultdict(set)
        for bond in topology.bonds or []:
            atom_1 = bond.a1 + addition
            atom_2 = bond.a2 + addition
            exclusion_pairs.add(tuple(sorted((atom_1, atom_2))))
            neighbors[atom_1].add(atom_2)
            neighbors[atom_2].add(atom_1)

        if nrexcl == 2:
            for atom_1, direct_neighbors in neighbors.items():
                for middle_atom in direct_neighbors:
                    for atom_3 in neighbors[middle_atom]:
                        if atom_1 != atom_3:
                            exclusion_pairs.add(tuple(sorted((atom_1, atom_3))))

    return [list(pair) for pair in sorted(exclusion_pairs)]


def _validate_molecule_topology(topology):
    """Validate bead references and bonded parameters before OpenMM creation."""

    atom_count = len(topology.atoms)
    known_types = {atomtype.abbr for atomtype in topology.atomtypes}
    missing_types = sorted({atom.abbr for atom in topology.atoms} - known_types)
    if missing_types:
        raise ValueError(
            f"Topology {topology.molecule_name!r} uses undefined atom type(s): "
            + ", ".join(missing_types)
        )

    for bond_number, bond in enumerate(topology.bonds or [], start=1):
        if bond.a1 == bond.a2 or not all(
            0 <= atom_index < atom_count for atom_index in (bond.a1, bond.a2)
        ):
            raise ValueError(
                f"Topology {topology.molecule_name!r} bond {bond_number} has "
                f"invalid bead indices {bond.a1 + 1}, {bond.a2 + 1}."
            )
        length_nm = float(bond.r0.value_in_unit(nanometer))
        force_constant = float(bond.k.value_in_unit(kilojoule_per_mole / nanometer**2))
        if not math.isfinite(length_nm) or length_nm <= 0.0:
            raise ValueError(
                f"Topology {topology.molecule_name!r} bond {bond_number} has "
                "a non-positive or non-finite equilibrium length."
            )
        if not math.isfinite(force_constant) or force_constant <= 0.0:
            raise ValueError(
                f"Topology {topology.molecule_name!r} bond {bond_number} has "
                "a non-positive or non-finite force constant."
            )

    for angle_number, angle in enumerate(topology.angles or [], start=1):
        indices = (angle.a1, angle.a2, angle.a3)
        if len(set(indices)) != 3 or not all(
            0 <= atom_index < atom_count for atom_index in indices
        ):
            raise ValueError(
                f"Topology {topology.molecule_name!r} angle {angle_number} "
                f"has invalid bead indices {[index + 1 for index in indices]}."
            )
        angle_degrees = float(angle.theta_in_degree.value_in_unit(openmm.unit.degree))
        force_constant = float(angle.k.value_in_unit(kilojoule_per_mole / radian**2))
        if not math.isfinite(angle_degrees) or not 0.0 < angle_degrees < 180.0:
            raise ValueError(
                f"Topology {topology.molecule_name!r} angle {angle_number} "
                "must be between 0 and 180 degrees."
            )
        if not math.isfinite(force_constant) or force_constant <= 0.0:
            raise ValueError(
                f"Topology {topology.molecule_name!r} angle {angle_number} has "
                "a non-positive or non-finite force constant."
            )


def _merge_atomtypes(loaded_topologies):
    """Merge atom types while rejecting incompatible duplicate abbreviations."""

    merged = {}
    sources = {}
    for topology in loaded_topologies:
        source_name = getattr(topology, "molecule_name", "unknown molecule")
        for atomtype in topology.atomtypes:
            signature = tuple(sorted(vars(atomtype).items()))
            if atomtype.abbr in merged:
                previous_signature = tuple(sorted(vars(merged[atomtype.abbr]).items()))
                if signature != previous_signature:
                    raise ValueError(
                        f"Conflicting definition of atom type {atomtype.abbr!r} "
                        f"between {sources[atomtype.abbr]!r} and "
                        f"{source_name!r}: {dict(previous_signature)} != "
                        f"{dict(signature)}."
                    )
                continue
            merged[atomtype.abbr] = atomtype
            sources[atomtype.abbr] = source_name
    return merged


def combine_topologies(topology_list):
    """
    topology_list: list of objects, each has dicts:
      - top.types2sigma, top.types2mu, top.types2epsilon
    Each dict maps "type1-type2" -> value
    Returns: typelist_nb, sigma_matrix, mu_matrix, epsilon_matrix
    """

    def split_key(k: str):
        a, b = k.split("-", 1)
        return a.strip(), b.strip()

    def norm_pair(a: str, b: str):
        return (a, b) if a <= b else (b, a)

    # -------- (1) collect all types --------
    types_set = set()
    for top in topology_list:
        for d in (top.types2sigma, top.types2mu, top.types2epsilon):
            for k in d.keys():
                a, b = split_key(k)
                types_set.add(a)
                types_set.add(b)

    typelist_nb = sorted(types_set)
    n = len(typelist_nb)
    idx = {t: i for i, t in enumerate(typelist_nb)}

    # -------- helper: merge dicts with conflict checks (A-B same as B-A) --------
    def merge_param(topology_list, attr_name: str):
        merged = {}  # (minType,maxType) -> value
        for top in topology_list:
            d = getattr(top, attr_name)
            for k, v in d.items():
                a, b = split_key(k)
                p = norm_pair(a, b)
                if p in merged:
                    if merged[p] != v:
                        raise ValueError(
                            f"Conflict in {attr_name} for pair {p[0]}-{p[1]}: "
                            f"{merged[p]} vs {v}"
                        )
                else:
                    merged[p] = v
        return merged

    sigma_map = merge_param(topology_list, "types2sigma")
    mu_map = merge_param(topology_list, "types2mu")
    epsilon_map = merge_param(topology_list, "types2epsilon")

    # -------- (2)(3)(4) build symmetric matrices --------
    def build_matrix(pmap, name: str, fill=np.nan):
        M = np.full((n, n), fill, dtype=float)
        for (a, b), v in pmap.items():
            i, j = idx[a], idx[b]
            M[i, j] = float(v)
            M[j, i] = float(v)
        # optional: ensure diagonal exists if present as "A-A"
        # (already handled by norm_pair)
        return M

    sigma_matrix = build_matrix(sigma_map, "sigma")
    mu_matrix = build_matrix(mu_map, "mu")
    epsilon_matrix = build_matrix(epsilon_map, "epsilon")

    missing_pairs = []
    for i, a in enumerate(typelist_nb):
        for j in range(i, n):
            b = typelist_nb[j]
            missing_terms = []
            if np.isnan(sigma_matrix[i, j]):
                missing_terms.append("sigma")
            if np.isnan(mu_matrix[i, j]):
                missing_terms.append("mu")
            if np.isnan(epsilon_matrix[i, j]):
                missing_terms.append("epsilon")
            if len(missing_terms) > 0:
                missing_pairs.append(f"{a}-{b} ({','.join(missing_terms)})")

    if len(missing_pairs) > 0:
        preview = ", ".join(missing_pairs[:20])
        suffix = (
            ""
            if len(missing_pairs) <= 20
            else f" ... and {len(missing_pairs) - 20} more"
        )
        raise ValueError(
            "Missing Wang-Frenkel nonbonded parameters across loaded topologies. "
            f"Missing pairs: {preview}{suffix}"
        )

    return typelist_nb, sigma_matrix, mu_matrix, epsilon_matrix


# Example:
# typelist_nb, sigma_matrix, mu_matrix, epsilon_matrix = combine_topologies(topology_list)


def _normalize_setting_name(name):
    return name.strip().replace("-", "_").lower()


def _convert_expected_value(raw_value, actual_value):
    if isinstance(actual_value, bool):
        raw_lower = raw_value.strip().lower()
        if raw_lower == "true":
            return True
        if raw_lower == "false":
            return False
        raise ValueError("must be True or False for boolean parameter")
    if isinstance(actual_value, int) and not isinstance(actual_value, bool):
        return int(raw_value)
    if isinstance(actual_value, float):
        return float(raw_value)
    if isinstance(actual_value, str):
        return raw_value.strip().replace("-", "_")
    return type(actual_value)(raw_value)


def _is_value_match(expected_value, actual_value):
    if isinstance(expected_value, float) and isinstance(actual_value, float):
        return math.isclose(expected_value, actual_value, rel_tol=1e-9, abs_tol=1e-12)
    return expected_value == actual_value


def _validate_simulation_settings(loaded_topologies, parameters):
    all_settings = []
    seen = set()
    for topology in loaded_topologies:
        for setting in getattr(topology, "simulation_settings", []):
            key = (
                setting["name"],
                setting["value"],
                setting["nvt_policy"],
                setting["npt_policy"],
            )
            if key not in seen:
                seen.add(key)
                all_settings.append(setting)

    if len(all_settings) == 0:
        return

    key_lookup = {_normalize_setting_name(k): k for k in parameters.keys()}
    is_npt = parameters.get("pcoulp", False) is True
    mode_name = "NPT" if is_npt else "NVT"

    for setting in all_settings:
        policy = setting["npt_policy"] if is_npt else setting["nvt_policy"]
        if policy == "none":
            continue

        normalized_name = _normalize_setting_name(setting["name"])
        if normalized_name not in key_lookup:
            if policy == "forced":
                raise ValueError(
                    f"[simulation-setting] Forced {mode_name} setting "
                    f"{setting['name']!r} does not exist in MDP parameters."
                )
            print(
                f"WARNING: [simulation-setting] RECOMMENDED {mode_name} setting '{setting['name']}' "
                "does not exist in mdp parameters."
            )
            continue

        parameter_key = key_lookup[normalized_name]
        actual_value = parameters[parameter_key]

        try:
            expected_value = _convert_expected_value(setting["value"], actual_value)
        except Exception as exc:
            raise ValueError(
                f"[simulation-setting] Cannot parse expected value "
                f"{setting['value']!r} for MDP key {parameter_key!r}: {exc}"
            ) from exc

        if _is_value_match(expected_value, actual_value):
            continue

        if policy == "forced":
            raise ValueError(
                f"[simulation-setting] Forced {mode_name} setting mismatch for "
                f"{parameter_key!r}: expected {expected_value!r}, but MDP has "
                f"{actual_value!r}."
            )
        if policy == "recommended":
            print(
                f"WARNING: [simulation-setting] VERY CLEAR WARNING: recommended {mode_name} setting mismatch "
                f"for '{parameter_key}': expected '{expected_value}', but mdp has '{actual_value}'."
            )


def build_system(pdb_file, top_file, parameters):
    print("######## Start of system building ########")
    molecule_with_number, itp_paths, loaded_topologies, topology_by_name = (
        _read_system_topology(top_file)
    )

    forcefield_function_type_LJ = sorted(
        {top.function_type_LJ for top in loaded_topologies}
    )
    forcefield_function_type_Coulomb = sorted(
        {top.function_type_Coulomb for top in loaded_topologies}
    )
    if (
        len(forcefield_function_type_LJ) != 1
        or len(forcefield_function_type_Coulomb) != 1
    ):
        raise ValueError(
            "All input topology files must use the same interaction functions; "
            f"found LJ={forcefield_function_type_LJ}, "
            f"Coulomb={forcefield_function_type_Coulomb}."
        )
    forcefield_function_type_LJ = forcefield_function_type_LJ[0]
    forcefield_function_type_Coulomb = forcefield_function_type_Coulomb[0]
    print(
        "## All input topology files use forcefield function types: "
        f"LJ-{forcefield_function_type_LJ}, "
        f"Coulomb-{forcefield_function_type_Coulomb}."
    )

    _validate_simulation_settings(loaded_topologies, parameters)
    for topology in loaded_topologies:
        _validate_molecule_topology(topology)

    print("## The following ITP files are loaded:")
    for topology, itp_path in zip(loaded_topologies, itp_paths):
        print(f"##   {topology.molecule_name}: {itp_path}")

    invalid_nrexcl = [top for top in loaded_topologies if top.nrexcl not in (1, 2)]
    if invalid_nrexcl:
        details = ", ".join(
            f"{top.molecule_name}={top.nrexcl}" for top in invalid_nrexcl
        )
        raise ValueError(f"Only 1-2 and 1-3 exclusions are supported; found {details}.")

    nrexcl_summary = ", ".join(
        f"{top.molecule_name}:1-{1 + top.nrexcl}" for top in loaded_topologies
    )
    print(f"## Non-bonded exclusions by molecule: {nrexcl_summary}.")

    # We get a combined list of atomtypes
    atomtypes_raw = [atype for top in loaded_topologies for atype in top.atomtypes]
    try:
        atomtypes_dict = _merge_atomtypes(loaded_topologies)
    except ValueError as exc:
        raise ValueError(f"Cannot combine topology atom types: {exc}") from exc

    print(
        f"## There are {len(atomtypes_raw)} atom types, and {len(atomtypes_dict)} when removing duplicates."
    )
    print(f"## Read atom types: {','.join(sorted(atomtypes_dict))}")

    # We now get a combined list of topology
    topology_list = []
    for molecule_name, molecule_number in molecule_with_number:
        topology_list.extend([topology_by_name[molecule_name]] * molecule_number)
    print(f"## Topology list contains {len(topology_list)} topologies.")

    # We now get a combined list of atoms
    atoms = list()
    atom_to_chainID = list()
    for chain_id, topology in enumerate(topology_list):
        if not topology.atoms:
            raise ValueError(
                f"Topology for molecule {topology.molecule_name!r} contains no atoms."
            )
        atoms.extend(topology.atoms)
        atom_to_chainID.extend([chain_id] * len(topology.atoms))

    invalid_particles = [
        index + 1
        for index, atom in enumerate(atoms)
        if (
            not math.isfinite(float(atom.mass))
            or float(atom.mass) <= 0.0
            or not math.isfinite(float(atom.charge))
        )
    ]
    if invalid_particles:
        raise ValueError(
            "Topology contains non-positive/non-finite masses or non-finite "
            f"charges at bead(s): {invalid_particles[:20]}."
        )
    total_charge = sum(float(atom.charge) for atom in atoms)

    print(f"## Atom list contains {len(atoms)} atoms.")

    print(f"## System net charge: {total_charge:.6g} e.")
    if abs(total_charge) > 1.0e-6:
        print(
            "## NOTE: The system is not charge-neutral; screened Debye-Huckel "
            "electrostatics permits this configuration."
        )

    # We next read pdb file

    pdb_information, box = read_pdb(pdb_file)
    if len(pdb_information) != len(atoms):
        raise ValueError(
            f"PDB has {len(pdb_information)} beads while TOP has {len(atoms)}."
        )

    serials = [atom["serial"] for atom in pdb_information]
    if len(serials) != len(set(serials)):
        raise ValueError("PDB atom serial numbers must be unique.")
    name_mismatches = [
        index + 1
        for index, (pdb_atom, topology_atom) in enumerate(zip(pdb_information, atoms))
        if pdb_atom["name"] != topology_atom.abbr[:4]
    ]
    if name_mismatches:
        preview = ", ".join(str(index) for index in name_mismatches[:20])
        raise ValueError(
            "PDB bead order/type does not match the expanded TOP/ITP topology; "
            f"mismatch at bead IDs: {preview}."
        )

    pdb_coordinates_nm = np.asarray(
        [
            [
                atom["x"].value_in_unit(nanometer),
                atom["y"].value_in_unit(nanometer),
                atom["z"].value_in_unit(nanometer),
            ]
            for atom in pdb_information
        ],
        dtype=float,
    )
    if not np.all(np.isfinite(pdb_coordinates_nm)):
        raise ValueError("PDB contains non-finite coordinates.")
    box_lengths_nm = np.asarray(box.value_in_unit(nanometer), dtype=float)
    tolerance_nm = 1.0e-6
    outside = np.flatnonzero(
        np.any(pdb_coordinates_nm < -tolerance_nm, axis=1)
        | np.any(pdb_coordinates_nm > box_lengths_nm + tolerance_nm, axis=1)
    )
    if outside.size:
        preview = ", ".join(str(index + 1) for index in outside[:20])
        raise ValueError(
            "PDB bead coordinates must lie inside the periodic box; outside "
            f"bead IDs: {preview}. Use dps trjconv/editconf to wrap them first."
        )

    # We now build system and topology

    mdsystem = openmm.System()
    mdtopology = openmm.app.Topology()
    positions = []

    atoms_topology = list()

    for index, atom in enumerate(atoms):
        mdsystem.addParticle(atom.mass)

        if index == 0 or atom_to_chainID[index] > atom_to_chainID[index - 1]:
            chain = mdtopology.addChain()

        this_residue = mdtopology.addResidue(atom.residuename, chain)

        atoms_topology.append(
            mdtopology.addAtom(
                atom.abbr, openmm.app.Element.getBySymbol("He"), this_residue
            )
        )

        positions.append(
            [
                pdb_information[index]["x"],
                pdb_information[index]["y"],
                pdb_information[index]["z"],
            ]
        )

    mdtopology.setPeriodicBoxVectors(
        [
            openmm.Vec3(box[0], 0 * nanometer, 0 * nanometer),
            openmm.Vec3(0 * nanometer, box[1], 0 * nanometer),
            openmm.Vec3(0 * nanometer, 0 * nanometer, box[2]),
        ]
    )

    print(
        "## Congratulations! The structure has been successfully phrased (but not yet added to simulation)."
    )
    print(f"## Structure phrased to positions with {len(positions)} points.")

    # We add harmonic bonds to the system

    if parameters["bondtype"] not in ["constraint", "bond"]:
        print(f"## ERROR: cannot process bondtype {parameters['bondtype']}.")
        quit()

    print(f"## Bond: Bonds will be treated as {parameters['bondtype']}.")

    bead_number_per_molecule = [len(top.atoms) for top in topology_list]
    addition_per_molecule = [0]
    addition_per_molecule.extend(
        [
            sum(bead_number_per_molecule[0:id])
            for id in range(1, len(bead_number_per_molecule))
        ]
    )

    if parameters["bondtype"] == "bond":
        bond_force = openmm.HarmonicBondForce()
        bond_force.setUsesPeriodicBoundaryConditions(True)
        bond_force.setForceGroup(0)

        for id, top in enumerate(topology_list):
            addition = addition_per_molecule[id]
            for bond in top.bonds or []:
                mdtopology.addBond(
                    atoms_topology[bond.a1 + addition],
                    atoms_topology[bond.a2 + addition],
                    None,
                )
                bond_force.addBond(
                    bond.a1 + addition, bond.a2 + addition, bond.r0, bond.k
                )
                # print(f"{bond.a1 + addition}, {bond.a2 + addition}, {bond.r0 * nanometer}, {bond.k * kilojoule_per_mole / nanometer ** 2}")

        mdsystem.addForce(bond_force)

        print(f"## There are altogether {bond_force.getNumBonds()} bonds.")

    else:
        for id, top in enumerate(topology_list):
            addition = addition_per_molecule[id]
            for bond in top.bonds or []:
                mdtopology.addBond(
                    atoms_topology[bond.a1 + addition],
                    atoms_topology[bond.a2 + addition],
                    None,
                )
                mdsystem.addConstraint(bond.a1 + addition, bond.a2 + addition, bond.r0)

        print(f"## There are altogether {mdsystem.getNumConstraints()} bonds.")

    print("## Bond: Bonded interaction added to system.")

    # We add harmonic angles to the system

    if any(topology.angles is not None for topology in topology_list):
        angle_force = openmm.HarmonicAngleForce()
        angle_force.setUsesPeriodicBoundaryConditions(True)
        angle_force.setForceGroup(0)

        bead_number_per_molecule = [len(top.atoms) for top in topology_list]
        addition_per_molecule = [0]
        addition_per_molecule.extend(
            [
                sum(bead_number_per_molecule[0:id])
                for id in range(1, len(bead_number_per_molecule))
            ]
        )

        for id, top in enumerate(topology_list):
            addition = addition_per_molecule[id]
            if top.angles is not None:
                for angle in top.angles:
                    angle_force.addAngle(
                        angle.a1 + addition,
                        angle.a2 + addition,
                        angle.a3 + addition,
                        angle.theta_in_degree.value_in_unit(radian),
                        angle.k.value_in_unit(kilojoule_per_mole / radian**2),
                    )

        mdsystem.addForce(angle_force)

        print(
            f"## There are altogether {angle_force.getNumAngles()} angles added to system."
        )

    else:
        print("ANGLES: No angles will be added to system.")

    # We build an exclusion list

    print("## Determining exclusions using each molecule's nrexcl setting.")

    exclusion_list = _build_exclusion_pairs(topology_list, addition_per_molecule)

    print(f"## Exclusion list contains {len(exclusion_list)} pairs.")

    # We add LJ and coulomb interaction
    print(
        f"## Nonbonded options: shift_lj = {parameters['shift_lj']}, shift_coul = {parameters['shift_coul']}."
    )

    lj_cutoff_scheme = parameters.get("cutoff_scheme_lj", "static")
    required_cutoffs_nm = []
    if lj_cutoff_scheme not in ["static", "dynamic"]:
        print(
            f"ERROR: Unknown cutoff_scheme_lj '{lj_cutoff_scheme}'. Allowed values: static, dynamic."
        )
        quit()
    if lj_cutoff_scheme == "dynamic" and parameters["cutoff_lj_multi"] <= 0:
        print("ERROR: cutoff_lj_multi must be > 0 when cutoff_scheme_lj is dynamic.")
        quit()

    if parameters["vdwtype"] == "pLJ":
        if forcefield_function_type_LJ != "Ashbaugh-Hatch":
            print(
                "## ERROR: MDP file comanding pLJ for vdwtype but topology is not Ashbaugh-Hatch type."
            )
            quit()

        epsilon_list = list(set([top.epsilon for top in loaded_topologies]))
        if len(epsilon_list) != 1:
            print(
                "ERROR: Different Ashbaugh-Hatch epsilon values found in input topology files."
            )
            quit()
        epsilon_lj = epsilon_list[0]

        if parameters["shift_lj"]:
            if lj_cutoff_scheme == "dynamic":
                lj_force = openmm.CustomNonbondedForce("""
                step(cutoff - r)*(ah - ah_cut);
                ah = step(rc_min - r)*(lj + (1 - lambda)*epsilon) + step(r - rc_min)*(lambda*lj);
                ah_cut = step(rc_min - cutoff)*(lj_cut + (1 - lambda)*epsilon) + step(cutoff - rc_min)*(lambda*lj_cut);
                lj = 4*epsilon*((sigma/r)^12 - (sigma/r)^6);
                lj_cut = 4*epsilon*((sigma/cutoff)^12 - (sigma/cutoff)^6);
                rc_min = 2^(1/6)*sigma;
                cutoff = cutoff_lj_multi * sigma;
                sigma = 0.5*(sigma1 + sigma2);
                lambda = 0.5*(lambda_base1 + lambda_base2 + lambda_T01 + lambda_T02 + (lambda_T11 + lambda_T12)*temperature + (lambda_T21 + lambda_T22)*temperature^2);
                """)
            else:
                lj_force = openmm.CustomNonbondedForce("""
                step(cutoff - r)*(ah - ah_cut);
                ah = step(rc_min - r)*(lj + (1 - lambda)*epsilon) + step(r - rc_min)*(lambda*lj);
                ah_cut = step(rc_min - cutoff)*(lj_cut + (1 - lambda)*epsilon) + step(cutoff - rc_min)*(lambda*lj_cut);
                lj = 4*epsilon*((sigma/r)^12 - (sigma/r)^6);
                lj_cut = 4*epsilon*((sigma/cutoff)^12 - (sigma/cutoff)^6);
                rc_min = 2^(1/6)*sigma;
                sigma = 0.5*(sigma1 + sigma2);
                lambda = 0.5*(lambda_base1 + lambda_base2 + lambda_T01 + lambda_T02 + (lambda_T11 + lambda_T12)*temperature + (lambda_T21 + lambda_T22)*temperature^2);
                """)
        else:
            if lj_cutoff_scheme == "dynamic":
                lj_force = openmm.CustomNonbondedForce("""
                step(cutoff - r)*ah;
                ah = step(rc_min - r)*(lj + (1 - lambda)*epsilon) + step(r - rc_min)*(lambda*lj);
                lj = 4*epsilon*((sigma/r)^12 - (sigma/r)^6);
                rc_min = 2^(1/6)*sigma;
                cutoff = cutoff_lj_multi * sigma;
                sigma = 0.5*(sigma1 + sigma2);
                lambda = 0.5*(lambda_base1 + lambda_base2 + lambda_T01 + lambda_T02 + (lambda_T11 + lambda_T12)*temperature + (lambda_T21 + lambda_T22)*temperature^2);
                """)
            else:
                lj_force = openmm.CustomNonbondedForce("""
                step(rc_min - r)*(lj + (1 - lambda)*epsilon) + step(r - rc_min)*(lambda*lj);
                lj = 4*epsilon*((sigma/r)^12 - (sigma/r)^6);
                rc_min = 2^(1/6)*sigma;
                sigma = 0.5*(sigma1 + sigma2);
                lambda = 0.5*(lambda_base1 + lambda_base2 + lambda_T01 + lambda_T02 + (lambda_T11 + lambda_T12)*temperature + (lambda_T21 + lambda_T22)*temperature^2);
                """)

        print("## LJ: LJ interaction will be calculated using the following equation:")
        print(f"## {lj_force.getEnergyFunction()}")

        lj_force.addPerParticleParameter("sigma")
        lj_force.addPerParticleParameter("lambda_base")
        lj_force.addPerParticleParameter("lambda_T0")
        lj_force.addPerParticleParameter("lambda_T1")
        lj_force.addPerParticleParameter("lambda_T2")
        lj_force.addGlobalParameter(
            "epsilon", defaultValue=epsilon_lj * kilojoule_per_mole
        )
        lj_force.addGlobalParameter(
            "temperature", defaultValue=parameters["production_temperature"]
        )
        if lj_cutoff_scheme == "dynamic":
            lj_force.addGlobalParameter(
                "cutoff_lj_multi", defaultValue=parameters["cutoff_lj_multi"]
            )
        elif parameters["shift_lj"]:
            lj_force.addGlobalParameter(
                "cutoff", defaultValue=parameters["cutoff_lj"] * nanometer
            )

        for i in range(lj_force.getNumGlobalParameters()):
            print(
                f"## LJ: Global parameter for LJ interaction, {lj_force.getGlobalParameterName(i)}, "
                + f"set as {lj_force.getGlobalParameterDefaultValue(i)}."
            )

        parameter_string = ",".join(
            [
                f"{lj_force.getPerParticleParameterName(i)}"
                for i in range(lj_force.getNumPerParticleParameters())
            ]
        )
        print(
            f"## LJ: LJ interaction contains {lj_force.getNumPerParticleParameters()} per particle parameters: {parameter_string}."
        )

        # Adding exclusions to LJ interaction.
        for [a1, a2] in exclusion_list:
            lj_force.addExclusion(a1, a2)

        print(
            f"## LJ: LJ interaction contains {lj_force.getNumExclusions()} exclusion pairs."
        )

        if lj_cutoff_scheme == "dynamic":
            particle_sigmas = [
                atomtypes_dict[atom.abbr].sigma
                for top in topology_list
                for atom in top.atoms
            ]
            max_sigma = max(particle_sigmas)
            lj_neighbor_cutoff = parameters["cutoff_lj_multi"] * max_sigma
            lj_force.setCutoffDistance(lj_neighbor_cutoff * nanometer)
            required_cutoffs_nm.append(float(lj_neighbor_cutoff))
        else:
            lj_force.setCutoffDistance(parameters["cutoff_lj"] * nanometer)
            required_cutoffs_nm.append(float(parameters["cutoff_lj"]))
        lj_force.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)

        if lj_cutoff_scheme == "dynamic":
            print(
                f"## LJ: Dynamic cutoff enabled. Pair cutoff is sigma*{parameters['cutoff_lj_multi']}. Neighbor-list cutoff set to {lj_force.getCutoffDistance()}."
            )
        else:
            print(
                f"## LJ: Cutoff for LJ interaction set to static value of {lj_force.getCutoffDistance()}."
            )

        for id, top in enumerate(topology_list):
            for atom in top.atoms:
                atomtype = atomtypes_dict[atom.abbr]
                lj_force.addParticle(
                    [
                        atomtype.sigma,
                        atomtype.mylambda,
                        atomtype.T0,
                        atomtype.T1,
                        atomtype.T2,
                    ]
                )

        lj_force.setForceGroup(1)

        mdsystem.addForce(lj_force)

        print(
            f"## LJ interaction added to system with {lj_force.getNumParticles()} particles."
        )

    elif parameters["vdwtype"] == "MPiPi":
        if forcefield_function_type_LJ != "Wang-Frenkel":
            print(
                "## ERROR: MDP file commanding MPiPi for vdwtype but topology is not Wang–Frenkel type."
            )
            quit()

        if lj_cutoff_scheme != "dynamic" or not math.isclose(
            parameters["cutoff_lj_multi"], 3.0, rel_tol=0.0, abs_tol=1.0e-12
        ):
            raise ValueError(
                "MPiPi/Wang-Frenkel interactions require "
                "cutoff-scheme-lj = dynamic and cutoff-lj-multi = 3 so the "
                "pair potential terminates at R = 3 sigma."
            )

        if parameters["shift_lj"]:
            if lj_cutoff_scheme == "dynamic":
                lj_force = openmm.CustomNonbondedForce("""
                step(cutoff - r) * (wf - wf_cut);
                wf = epsilon * alpha * part1 * part2 ^ (2 * nu);
                wf_cut = epsilon * alpha * part1_cut * part2_cut ^ (2 * nu);
                part1 = (sigma / r) ^ (2 * mu) - 1;
                part2 = (R / r) ^ (2 * mu) - 1;
                part1_cut = (sigma / cutoff) ^ (2 * mu) - 1;
                part2_cut = (R / cutoff) ^ (2 * mu) - 1;
                alpha = 2 * nu * (R / sigma) ^ (2 * mu) * (upper / lower) ^ (2 * nu + 1);
                upper = 2 * nu + 1;
                lower = 2 * nu * ((R / sigma) ^ (2 * mu) - 1);
                R = RScale * sigma;
                cutoff = cutoff_lj_multi * sigma;
                epsilon = eps_unit * epsilon_Table(type1,type2);
                sigma = sigma_unit * sigma_Table(type1,type2);
                mu = mu_unit * mu_Table(type1,type2)
                """)
            else:
                lj_force = openmm.CustomNonbondedForce("""
                step(cutoff - r) * (wf - wf_cut);
                wf = epsilon * alpha * part1 * part2 ^ (2 * nu);
                wf_cut = epsilon * alpha * part1_cut * part2_cut ^ (2 * nu);
                part1 = (sigma / r) ^ (2 * mu) - 1;
                part2 = (R / r) ^ (2 * mu) - 1;
                part1_cut = (sigma / cutoff) ^ (2 * mu) - 1;
                part2_cut = (R / cutoff) ^ (2 * mu) - 1;
                alpha = 2 * nu * (R / sigma) ^ (2 * mu) * (upper / lower) ^ (2 * nu + 1);
                upper = 2 * nu + 1;
                lower = 2 * nu * ((R / sigma) ^ (2 * mu) - 1);
                R = RScale * sigma;
                epsilon = eps_unit * epsilon_Table(type1,type2);
                sigma = sigma_unit * sigma_Table(type1,type2);
                mu = mu_unit * mu_Table(type1,type2)
                """)
        else:
            if lj_cutoff_scheme == "dynamic":
                lj_force = openmm.CustomNonbondedForce("""
                step(cutoff - r) * wf;
                wf = epsilon * alpha * part1 * part2 ^ (2 * nu);
                part1 = (sigma / r) ^ (2 * mu) - 1;
                part2 = (R / r) ^ (2 * mu) - 1;
                alpha = 2 * nu * (R / sigma) ^ (2 * mu) * (upper / lower) ^ (2 * nu + 1);
                upper = 2 * nu + 1;
                lower = 2 * nu * ((R / sigma) ^ (2 * mu) - 1);
                R = RScale * sigma;
                cutoff = cutoff_lj_multi * sigma;
                epsilon=eps_unit * epsilon_Table(type1,type2);
                sigma=sigma_unit * sigma_Table(type1,type2);
                mu=mu_unit * mu_Table(type1,type2)
                """)
            else:
                lj_force = openmm.CustomNonbondedForce("""
                epsilon * alpha * part1 * part2 ^ (2 * nu);
                part1 = (sigma / r) ^ (2 * mu) - 1;
                part2 = (R / r) ^ (2 * mu) - 1;
                alpha = 2 * nu * (R / sigma) ^ (2 * mu) * (upper / lower) ^ (2 * nu + 1);
                upper = 2 * nu + 1;
                lower = 2 * nu * ((R / sigma) ^ (2 * mu) - 1);
                R = RScale * sigma;
                epsilon=eps_unit * epsilon_Table(type1,type2);
                sigma=sigma_unit * sigma_Table(type1,type2);
                mu=mu_unit * mu_Table(type1,type2)
                """)

        print("## LJ: LJ interaction will be calculated using the following equation:")
        print(f"## {lj_force.getEnergyFunction()}")

        lj_force.addGlobalParameter("eps_unit", defaultValue=1 * kilojoule_per_mole)
        lj_force.addGlobalParameter("sigma_unit", defaultValue=1 * nanometer)
        lj_force.addGlobalParameter("mu_unit", defaultValue=1 * dimensionless)

        lj_force.addGlobalParameter("RScale", defaultValue=3)
        lj_force.addGlobalParameter("nu", defaultValue=1)
        if lj_cutoff_scheme == "dynamic":
            lj_force.addGlobalParameter(
                "cutoff_lj_multi", defaultValue=parameters["cutoff_lj_multi"]
            )
        elif parameters["shift_lj"]:
            lj_force.addGlobalParameter(
                "cutoff", defaultValue=parameters["cutoff_lj"] * nanometer
            )

        # lj_force.addPerParticleParameter("epsilon")
        # lj_force.addPerParticleParameter("sigma")
        # lj_force.addPerParticleParameter("mu")

        lj_force.addPerParticleParameter("type")

        for i in range(lj_force.getNumGlobalParameters()):
            print(
                f"## LJ: Global parameter for LJ interaction, {lj_force.getGlobalParameterName(i)}, "
                + f"set as {lj_force.getGlobalParameterDefaultValue(i)}."
            )

        parameter_string = ",".join(
            [
                f"{lj_force.getPerParticleParameterName(i)}"
                for i in range(lj_force.getNumPerParticleParameters())
            ]
        )
        print(
            f"## LJ: LJ interaction contains {lj_force.getNumPerParticleParameters()} per particle parameters: {parameter_string}."
        )

        # Adding exclusions to LJ interaction.
        for [a1, a2] in exclusion_list:
            lj_force.addExclusion(a1, a2)

        print(
            f"## LJ: LJ interaction contains {lj_force.getNumExclusions()} exclusion pairs."
        )

        try:
            typelist_nb, sigma_matrix, mu_matrix, epsilon_matrix = combine_topologies(
                topology_list
            )
        except ValueError as exc:
            print(f"ERROR: {exc}")
            quit()
        if lj_cutoff_scheme == "dynamic":
            max_sigma = float(np.nanmax(sigma_matrix))
            if not np.isfinite(max_sigma) or max_sigma <= 0:
                print(
                    f"ERROR: Invalid max sigma value ({max_sigma}) for dynamic LJ cutoff."
                )
                quit()
            lj_neighbor_cutoff = parameters["cutoff_lj_multi"] * max_sigma
            lj_force.setCutoffDistance(lj_neighbor_cutoff * nanometer)
            required_cutoffs_nm.append(float(lj_neighbor_cutoff))
        else:
            lj_force.setCutoffDistance(parameters["cutoff_lj"] * nanometer)
            required_cutoffs_nm.append(float(parameters["cutoff_lj"]))
        lj_force.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)

        if lj_cutoff_scheme == "dynamic":
            print(
                f"## LJ: Dynamic cutoff enabled. Pair cutoff is sigma*{parameters['cutoff_lj_multi']}. Neighbor-list cutoff set to {lj_force.getCutoffDistance()}."
            )
        else:
            print(
                f"## LJ: Cutoff for LJ interaction set to static value of {lj_force.getCutoffDistance()}."
            )

        ntypes = len(typelist_nb)
        print(f"## Will add tabulated function for LJ interaction with {ntypes} types.")
        print(f"## Type list: {','.join(typelist_nb)}")
        lj_force.addTabulatedFunction(
            "epsilon_Table",
            openmm.Discrete2DFunction(ntypes, ntypes, epsilon_matrix.flatten()),
        )
        lj_force.addTabulatedFunction(
            "sigma_Table",
            openmm.Discrete2DFunction(ntypes, ntypes, sigma_matrix.flatten()),
        )
        lj_force.addTabulatedFunction(
            "mu_Table", openmm.Discrete2DFunction(ntypes, ntypes, mu_matrix.flatten())
        )

        for id, top in enumerate(topology_list):
            for atom in top.atoms:
                lj_force.addParticle([typelist_nb.index(atom.abbr)])

        mdsystem.addForce(lj_force)

        print(
            f"## LJ interaction added to system with {lj_force.getNumParticles()} particles."
        )

    else:
        print("## CRITICAL WARNING: LJ interaction will not be calculated.")
        quit()

    if parameters["coulombtype"] == "yukawa":
        if forcefield_function_type_Coulomb != "Debye-Huckel":
            print(
                f"## ERROR: MDP file commanding yukawa for coulombtype but topology is not Debye-Huckel type but {forcefield_function_type_Coulomb}."
            )
            quit()

        if parameters["shift_coul"]:
            coulomb_force = openmm.CustomNonbondedForce("""
            step(cutoff - r) * k / D * q1 * q2 * (exp(-r/debye) / r - exp(-cutoff/debye) / cutoff);
            debye = debye_coefficient*sqrt(D*temperature/salt_conc);
            D = dielectric_km1/temperature + dielectric_k0 + dielectric_k1*temperature + dielectric_k2*temperature^2 + dielectric_k3*temperature^3
            """)
        else:
            coulomb_force = openmm.CustomNonbondedForce("""
            k/D*q1*q2*exp(-r/debye)/r;
            debye = debye_coefficient*sqrt(D*temperature/salt_conc);
            D = dielectric_km1/temperature + dielectric_k0 + dielectric_k1*temperature + dielectric_k2*temperature^2 + dielectric_k3*temperature^3
            """)

        print(
            "## Coulomb: Coulomb interaction will be calculated using the following equation:"
        )
        print(f"## {coulomb_force.getEnergyFunction()}")

        coulomb_force.addGlobalParameter("k", k0)
        rp_signatures = []
        for top in loaded_topologies:
            mode = getattr(top, "relative_permittivity_mode", "constant")
            if mode == "temperature_dependent":
                coeffs = tuple(getattr(top, "relative_permittivity_coeffs", []))
                rp_signatures.append(("temperature_dependent", coeffs))
            else:
                rp_signatures.append(("constant", float(top.relative_permittivity)))

        rp_unique = list(set(rp_signatures))
        if len(rp_unique) != 1:
            print(
                "ERROR: Different relative_permittivity settings found in input topology files."
            )
            quit()

        rp_mode, rp_value = rp_unique[0]
        if rp_mode == "temperature_dependent":
            k_minus_1, k0_rp, k1_rp, k2_rp, k3_rp = rp_value
            print(
                "## Coulomb: Relative permittivity follows the topology's temperature-dependent model."
            )
        else:
            k_minus_1, k0_rp, k1_rp, k2_rp, k3_rp = (0.0, rp_value, 0.0, 0.0, 0.0)
            print(f"## Coulomb: Relative permittivity uses constant D={rp_value}.")

        def dielectric_at(temperature):
            return (
                k_minus_1 / temperature
                + k0_rp
                + k1_rp * temperature
                + k2_rp * temperature**2
                + k3_rp * temperature**3
            )

        ramp_temperatures = np.linspace(
            parameters["initial_temperature"],
            parameters["production_temperature"],
            1001,
        )
        dielectric_values = np.asarray(
            [dielectric_at(value) for value in ramp_temperatures]
        )
        if not np.all(np.isfinite(dielectric_values)) or np.any(dielectric_values <= 0):
            bad_index = int(
                np.flatnonzero(
                    ~np.isfinite(dielectric_values) | (dielectric_values <= 0)
                )[0]
            )
            raise ValueError(
                "Relative permittivity must remain finite and positive over "
                f"the configured temperature ramp; D({ramp_temperatures[bad_index]:.6g} K)="
                f"{dielectric_values[bad_index]:.6g}."
            )

        coulomb_force.addGlobalParameter(
            "temperature", parameters["production_temperature"]
        )
        coulomb_force.addGlobalParameter("dielectric_km1", k_minus_1)
        coulomb_force.addGlobalParameter("dielectric_k0", k0_rp)
        coulomb_force.addGlobalParameter("dielectric_k1", k1_rp)
        coulomb_force.addGlobalParameter("dielectric_k2", k2_rp)
        coulomb_force.addGlobalParameter("dielectric_k3", k3_rp)
        if parameters["shift_coul"]:
            coulomb_force.addGlobalParameter(
                "cutoff", parameters["cutoff_coul"] * nanometer
            )

        salt_conc = parameters["salt_conc"]
        coulomb_force.addGlobalParameter("salt_conc", salt_conc)
        coulomb_force.addGlobalParameter("debye_coefficient", kappa_coefficient)
        production_temperature = parameters["production_temperature"]
        production_dielectric = dielectric_at(production_temperature)
        debye_length = (
            math.sqrt(production_dielectric * production_temperature / salt_conc)
            * kappa_coefficient
            * nanometer
        )

        print(
            f"## Coulomb: Salt concentration set as {salt_conc} M; at {production_temperature} K the Debye length is {debye_length}."
        )

        for i in range(coulomb_force.getNumGlobalParameters()):
            print(
                f"## Coulomb: Global parameter for Coulomb interaction, {coulomb_force.getGlobalParameterName(i)}, "
                + f"set as {coulomb_force.getGlobalParameterDefaultValue(i)}."
            )

        coulomb_force.addPerParticleParameter("q")

        parameter_string = ",".join(
            [
                f"{coulomb_force.getPerParticleParameterName(i)}"
                for i in range(coulomb_force.getNumPerParticleParameters())
            ]
        )
        print(
            f"## Coulomb: Coulomb interaction contains {coulomb_force.getNumPerParticleParameters()} per particle parameters: {parameter_string}."
        )

        # Adding exclusions to Coulomb interaction.
        for [a1, a2] in exclusion_list:
            coulomb_force.addExclusion(a1, a2)

        print(
            f"## Coulomb: Coulomb interaction contains {coulomb_force.getNumExclusions()} exclusion pairs."
        )

        coulomb_force.setCutoffDistance(parameters["cutoff_coul"] * nanometer)
        required_cutoffs_nm.append(float(parameters["cutoff_coul"]))
        coulomb_force.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)

        print(
            f"## Coulomb: Cutoff for Coulomb interaction set to static value of {coulomb_force.getCutoffDistance()}."
        )

        for id, top in enumerate(topology_list):
            for atom in top.atoms:
                coulomb_force.addParticle([atom.charge])

        coulomb_force.setForceGroup(2)
        mdsystem.addForce(coulomb_force)

        print(
            f"## Coulomb interaction added to system with {coulomb_force.getNumParticles()} particles."
        )

    else:
        print("## WARNING: Coulomb interaction will not be calculated.")

    print(f"## The mdsystem contains {mdsystem.getNumForces()} types of forces.")

    box_lengths_nm = np.asarray(box.value_in_unit(nanometer), dtype=float)
    maximum_cutoff_nm = max(required_cutoffs_nm, default=0.0)
    half_shortest_box_nm = 0.5 * float(np.min(box_lengths_nm))
    if maximum_cutoff_nm >= half_shortest_box_nm - 1.0e-12:
        raise ValueError(
            f"Largest nonbonded cutoff ({maximum_cutoff_nm:g} nm) must be "
            "strictly smaller than half the shortest box length "
            f"({half_shortest_box_nm:g} nm). Increase the box or reduce the cutoff."
        )

    # --- Periodic box ---
    mdsystem.setDefaultPeriodicBoxVectors(
        openmm.Vec3(box[0], 0, 0), openmm.Vec3(0, box[1], 0), openmm.Vec3(0, 0, box[2])
    )
    print(f"## Simulation box set to {box[0]} * {box[1]} * {box[2]} nm.")

    ITP_Topology_list = topology_list

    print(
        "## Note: This is the end of building system. Will return mdsystem and positions."
    )
    print("######## End of system building ########")

    return mdsystem, mdtopology, positions, ITP_Topology_list
