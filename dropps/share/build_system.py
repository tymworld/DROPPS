import re
from collections import defaultdict
from dropps.fileio.itp_reader import read_itp
from dropps.fileio.pdb_reader import read_pdb
import openmm
import openmm.app
from openmm.unit import nanometer, kilojoule_per_mole, moles, liter, kilocalorie_per_mole, radian
import math
import copy

if __name__ == "__main__":

    from os.path import dirname, realpath, sep, pardir
    import sys
    import json
    import os

    # Get the absolute path of the current script
    current_path = os.path.abspath(__file__)

    # Get the parent directory
    parent_dir = os.path.dirname(current_path)       # current .py file's directory
    father_dir = os.path.dirname(parent_dir)         # parent of that directory

    # Add to Python path
    sys.path.append(father_dir)

from dropps.hp.constants import kappa_coefficient, relative_permittivity, k0

def build_system(pdb_file, top_file, parameters):

    print("######## Start of system building ########")

    # We first read topology

    with open(top_file, 'r') as f:
        lines = f.readlines()

        sections = defaultdict(list)
        current_section = None

        for line in lines:
            line = line.strip().split('#')[0]
            if not line or line.startswith(';'):
                continue
            if line.startswith('[') and line.endswith(']'):
                current_section = line.strip('[]').strip().lower()
            elif current_section:
                sections[current_section].append(line)

    
        molecule_with_number = [re.split(r'\s+', line.split(";")[0].strip()) for line in sections["system"]]

        loaded_topologies = [read_itp(itp_filename) for itp_filename in sections["itp files"]]
        molecule_number_list = [int(molecule[1]) for molecule in molecule_with_number]
        print("## The following ITP files are read and loaded.")
        for itp_filename in sections["itp files"]:
            print(f"## Molecule name {read_itp(itp_filename).molecule_name} phrased from file {itp_filename}.")
    
    nrexcl = min([top.nrexcl for top in loaded_topologies])
    if nrexcl not in [1,2]:
        print(f"ERROR: Only 1-2 and 1-3 exclusions are supported.")
        quit()

    print(f"## Non-bonded interaction will be calculated without 1-{1+nrexcl} interaction.")
    
    # We get a combined list of atomtypes
    atomtypes_raw = [atype for top in loaded_topologies for atype in top.atomtypes]
    atomtypes_dict = dict()
    for atomtype in atomtypes_raw:
        atomtypes_dict[atomtype.abbr] = atomtype
    
    print(f"## There are {len(atomtypes_raw)} atom types, and {len(atomtypes_dict)} when removing duplicates.")
    print(f"## Read atom types: {','.join(atomtypes_dict.keys())}")
    
    # We now get a combined list of topology
    topology_list = list()
    for molecule_name, molecule_number in molecule_with_number:
        this_topology = [top for top in loaded_topologies if top.molecule_name == molecule_name][0]
        for molecule_index in range(int(molecule_number)):
            topology_list.append(this_topology)
    print(f"## Topology list contains {len(topology_list)} topologies.")
    
    # We now get a combined list of atoms
    atoms = list()
    atom_to_chainID = list()
    chainID = 0

    mol_index = 0

    for mol_type_index in range(len(molecule_number_list)):
        for mol_number in range(molecule_number_list[mol_type_index]):
            atoms.extend(topology_list[mol_index].atoms)
            atom_to_chainID.extend([chainID] * len(topology_list[mol_index].atoms))
            chainID += 1
            mol_index += 1
    
    print(f"## Atom list contains {len(atoms)} atoms.")

    print(f"## Congratulations! The topology has been successfully phrased.")
                
    # We next read pdb file

    pdb_information, box = read_pdb(pdb_file)
    if len(pdb_information) != len(atoms):
        print(f"ERROR: PDB has {len(pdb_information)} beads while TOP has {len(atoms)}.")
        quit()
    
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
        
        atoms_topology.append(mdtopology.addAtom(atom.abbr, openmm.app.Element.getBySymbol("He"), this_residue))

        positions.append([pdb_information[index]["x"], pdb_information[index]["y"], pdb_information[index]["z"]])
    
    
    mdtopology.setPeriodicBoxVectors([openmm.Vec3(box[0], 0 * nanometer, 0 * nanometer),
        openmm.Vec3(0* nanometer, box[1], 0* nanometer),
        openmm.Vec3(0* nanometer, 0* nanometer, box[2])])
    
    print(f"## Congratulations! The structure has been successfully phrased (but not yet added to simulation).")
    print(f"## Structure phrased to positions with {len(positions)} points.")
    
    # We add harmonic bonds to the system    

    if parameters["bondtype"] not in ["constraint", "bond"]:
        print(f"## ERROR: cannot process bondtype {parameters["bondtype"]}.")
        quit()
    
    print(f"## Bond: Bonds will be treated as {parameters["bondtype"]}.")

    if parameters["bondtype"] == "bond":

        bond_force = openmm.HarmonicBondForce()
        bond_force.setUsesPeriodicBoundaryConditions(True)
        bond_force.setForceGroup(0)

        bead_number_per_molecule = [len(top.atoms) for top in topology_list]
        addition_per_molecule = [0]
        addition_per_molecule.extend([sum(bead_number_per_molecule[0:id]) for id in range(1, len(bead_number_per_molecule))])

        for id, top in enumerate(topology_list):
            addition = addition_per_molecule[id]
            for bond in top.bonds:
                mdtopology.addBond(atoms_topology[bond.a1 + addition], atoms_topology[bond.a2 + addition], None)
                bond_force.addBond(bond.a1 + addition, bond.a2 + addition, bond.r0, bond.k)
                #print(f"{bond.a1 + addition}, {bond.a2 + addition}, {bond.r0 * nanometer}, {bond.k * kilojoule_per_mole / nanometer ** 2}")
        
        mdsystem.addForce(bond_force)

        # We build a bonded list for adding non-bonded exclusions

        bond_list = [[bond_force.getBondParameters(id)[0], bond_force.getBondParameters(id)[1]] for id in range(bond_force.getNumBonds())]
        bond_list_copy = [[bond_force.getBondParameters(id)[0], bond_force.getBondParameters(id)[1]] for id in range(bond_force.getNumBonds())]
        print(f"## There are altogether {len(bond_list)} bonds.")
    
    else:
        bead_number_per_molecule = [len(top.atoms) for top in topology_list]
        addition_per_molecule = [0]
        addition_per_molecule.extend([sum(bead_number_per_molecule[0:id]) for id in range(1, len(bead_number_per_molecule))])

        for id, top in enumerate(topology_list):
            addition = addition_per_molecule[id]
            for bond in top.bonds:
                mdtopology.addBond(bond.a1 + addition, bond.a2 + addition, None)
                mdsystem.addConstraint(bond.a1 + addition, bond.a2 + addition, bond.r0)
        
        # We build a bonded list for adding non-bonded exclusions

        bond_list = [[mdsystem.getConstraintParameters(id)[0], mdsystem.getConstraintParameters(id)[1]] for id in range(mdsystem.getNumConstraints())]
        bond_list_copy = [[mdsystem.getConstraintParameters(id)[0], mdsystem.getConstraintParameters(id)[1]] for id in range(mdsystem.getNumConstraints())]
    
        print(f"## There are altogether {len(bond_list)} bonds.")

    print(f"## Bond: Bonded interaction added to system.")
    
    # We add harmonic angles to the system 

    if any(topology.angles is not None for topology in topology_list):
        angle_force = openmm.HarmonicAngleForce()
        angle_force.setUsesPeriodicBoundaryConditions(True)
        angle_force.setForceGroup(0)

        bead_number_per_molecule = [len(top.atoms) for top in topology_list]
        addition_per_molecule = [0]
        addition_per_molecule.extend([sum(bead_number_per_molecule[0:id]) for id in range(1, len(bead_number_per_molecule))])

        for id, top in enumerate(topology_list):
            addition = addition_per_molecule[id]
            if top.angles is not None:

                for angle in top.angles:

                    angle_force.addAngle(angle.a1 + addition, angle.a2 + addition, angle.a3 + addition, 
                                         angle.theta_in_degree.value_in_unit(radian)
                                         , angle.k.value_in_unit(kilojoule_per_mole / radian**2))
        
        mdsystem.addForce(angle_force)

        print(f"## There are altogether {angle_force.getNumAngles()} angles added to system.")

    else:
        print(f"ANGLES: No angles will be added to system.")


    # We build an exclusion list

    print(f"## Determining exclusions based on a nrexcl of {nrexcl}.")

    exclusion_list = copy.deepcopy(bond_list_copy)

    if nrexcl == 2:
        # Step 1: Convert bond list to a set of frozensets for fast lookup
        bond_set = set(frozenset((a, b)) for a, b in bond_list)

        # Step 2: Precompute neighbors of each particle
        neighbors = defaultdict(set)
        for a, b in bond_set:
            neighbors[a].add(b)
            neighbors[b].add(a)

        num_particles = mdsystem.getNumParticles()

        # Step 3: Loop through pairs
        for id1 in range(num_particles):
            for id3 in range(id1 + 1, num_particles):
                bonded = False
                # Check if id1 and id3 are connected via a common neighbor id2
                for id2 in neighbors[id1]:
                    if id3 in neighbors[id2]:
                        bonded = True
                        break
                if bonded:
                    exclusion_list.append([id1, id3])
    
    print(f"## Exclusion list contains {len(exclusion_list)} pairs.")

    # We add LJ and coulomb interaction

    if parameters["vdwtype"] == "pLJ":

        lj_force = openmm.CustomNonbondedForce("""
        step(rc - r)*(lj + (1 - lambda)*epsilon) + step(r - rc)*(lambda*lj);
        lj = 4*epsilon*((sigma/r)^12 - (sigma/r)^6);
        rc = 2^(1/6)*sigma;
        sigma = 0.5*(sigma1 + sigma2);
        lambda = 0.5*(lambda1 + lambda2);
        """)

        print(f"## LJ: LJ interaction will be calculated using the following equation:")
        print(f"## {lj_force.getEnergyFunction()}")

        lj_force.addPerParticleParameter("sigma")
        lj_force.addPerParticleParameter("lambda")
        lj_force.addGlobalParameter("epsilon", defaultValue=0.2 * kilocalorie_per_mole)

        for i in range(lj_force.getNumGlobalParameters()):
            print(f"## LJ: Global parameter for LJ interaction, {lj_force.getGlobalParameterName(i)}, "\
                + f"set as {lj_force.getGlobalParameterDefaultValue(i)}.")
            
        parameter_string = ','.join([f"{lj_force.getPerParticleParameterName(i)}" for i in range(lj_force.getNumPerParticleParameters())])    
        print(f"## LJ: LJ interaction contains {lj_force.getNumPerParticleParameters()} per particle parameters: {parameter_string}.")

        # Adding exclusions to LJ interaction.
        for [a1, a2] in exclusion_list:
            lj_force.addExclusion(a1, a2)
        
        print(f"## LJ: LJ interaction contains {lj_force.getNumExclusions()} exclusion pairs.")
            
        lj_force.setCutoffDistance(parameters["cutoff_lj"] * nanometer)
        lj_force.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)

        print(f"## LJ: Cutoff for LJ interaction set to static value of {lj_force.getCutoffDistance()}.")

    else:
        print(f"## WARNING: LJ interaction will not be calculated.")

    if parameters["coulombtype"] == "yukawa":

        coulomb_force = openmm.CustomNonbondedForce("""
        k/D*q1*q2*exp(-r/debye)/r
        """)

        print(f"## Coulomb: Coulomb interaction will be calculated using the following equation:")
        print(f"## {coulomb_force.getEnergyFunction()}")

        coulomb_force.addGlobalParameter("k", k0)
        coulomb_force.addGlobalParameter("D", relative_permittivity)

        salt_conc = parameters["salt_conc"]
        temperature = parameters["production_temperature"]
        debye_length = math.sqrt(relative_permittivity * temperature / salt_conc) * kappa_coefficient * nanometer
        coulomb_force.addGlobalParameter("debye", debye_length / nanometer)

        print(f"## Coulomb: Salt concentration set as {salt_conc} M, temperature set as {temperature} K, thus debye length is {debye_length}.")

        for i in range(coulomb_force.getNumGlobalParameters()):
            print(f"## Coulomb: Global parameter for Coulomb interaction, {coulomb_force.getGlobalParameterName(i)}, "\
                  + f"set as {coulomb_force.getGlobalParameterDefaultValue(i)}.")

        coulomb_force.addPerParticleParameter("q")

        parameter_string = ','.join([f"{coulomb_force.getPerParticleParameterName(i)}" for i in range(coulomb_force.getNumPerParticleParameters())])    
        print(f"## Coulomb: Coulomb interaction contains {coulomb_force.getNumPerParticleParameters()} per particle parameters: {parameter_string}.")
      
        # Adding exclusions to Coulomb interaction.
        for [a1, a2] in exclusion_list:
            coulomb_force.addExclusion(a1, a2)
        
        print(f"## Coulomb: Coulomb interaction contains {coulomb_force.getNumExclusions()} exclusion pairs.")
        
        coulomb_force.setCutoffDistance(parameters["cutoff_coul"] * nanometer)
        coulomb_force.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)

        print(f"## Coulomb: Cutoff for Coulomb interaction set to static value of {coulomb_force.getCutoffDistance()}.")

    else:
        print(f"## WARNING: Coulomb interaction will not be calculated.")
    
    if parameters["vdwtype"] == "pLJ":

        for id, top in enumerate(topology_list):
            for atom in top.atoms:
                mylambda = atomtypes_dict[atom.abbr].mylambda + atomtypes_dict[atom.abbr].T0 \
                      + atomtypes_dict[atom.abbr].T1 * temperature \
                      + atomtypes_dict[atom.abbr].T2 * temperature * temperature

                lj_force.addParticle([atomtypes_dict[atom.abbr].sigma, mylambda])

        lj_force.setForceGroup(1)
        
        mdsystem.addForce(lj_force)

        print(f"## LJ interaction added to system with {lj_force.getNumParticles()} particles.")

    if parameters["coulombtype"] == "yukawa":

        for id, top in enumerate(topology_list):
            for atom in top.atoms:
                coulomb_force.addParticle([atom.charge])
        
        coulomb_force.setForceGroup(2)
        mdsystem.addForce(coulomb_force)

        print(f"## Coulomb interaction added to system with {coulomb_force.getNumParticles()} particles.")
    
    print(f"## The mdsystem contains {mdsystem.getNumForces()} types of forces.")

    # --- Periodic box ---
    mdsystem.setDefaultPeriodicBoxVectors(
        openmm.Vec3(box[0], 0, 0),
        openmm.Vec3(0, box[1], 0),
        openmm.Vec3(0, 0, box[2])
    )
    print(f"## Simulation box set to {box[0]} * {box[1]} * {box[2]} nm.")

    ITP_Topology_list = topology_list

    print(f"## Note: This is the end of building system. Will return mdsystem and positions.")
    print("######## End of system building ########")

    return mdsystem, mdtopology, positions, ITP_Topology_list

# Example usage
if __name__ == "__main__":

    from os.path import dirname, realpath, sep, pardir
    import sys
    import json
    import os

    if len(sys.argv) < 2:
        print("Usage: python build_system.py <path_to_itp_file>")
        sys.exit(1)

    pdb_path = sys.argv[1]
    top_path = sys.argv[2]

    from parameters import getparameter

    result = build_system(pdb_path, top_path, getparameter("/Users/TYM-work/Repository/py_cgps.ng/src/share/templates/md.mdp"))

