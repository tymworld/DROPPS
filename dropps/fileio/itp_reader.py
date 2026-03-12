import re
from collections import defaultdict

from openmm.unit import kilojoule_per_mole, nanometer, atomic_mass_unit, degree, radian

class Atomtype_AH_DH():
    def __init__(self, abbr, name, sigma, my_lambda, T0, T1, T2):
        self.abbr = abbr
        self.name = name
        self.sigma = sigma
        self.mylambda = my_lambda
        self.T0 = T0
        self.T1 = T1
        self.T2 = T2

class Atomtype_WF_DH():
    def __init__(self, abbr, name):
        self.abbr = abbr
        self.name = name

class Atom():
    def __init__(self, abbr, name, residuename, residueid, mass, charge):
        self.abbr = abbr
        self.name = name
        self.residuename = residuename
        self.residueid = int(residueid)
        self.mass = mass
        self.charge = charge

class Bond():
    def __init__(self, a1, a2, r0, k):
        # Within this structure, a1, a2 start with 0.
        self.a1 = a1
        self.a2 = a2
        self.r0 = r0
        self.k = k

class Angle():
    def __init__(self, a1, a2, a3, theta_in_degree, k):
        # Within this structure, a1, a2, a3 start with 0.
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        self.theta_in_degree = theta_in_degree
        self.k = k

class ITPTopology():
    def __init__(self, data):

        self.function_type_LJ = data["function_type_LJ"]
        self.function_type_Coulomb = data["function_type_Coulomb"]

        if self.function_type_LJ == "Ashbaugh-Hatch" and self.function_type_Coulomb == "Debye-Huckel":

            self.molecule_name = data["moleculetype"][0]
            self.nrexcl = int(data["moleculetype"][1])

            self.typelist = [atomtype["abbr"] for atomtype in data["atomtypes"]]

            self.atomtypes = [Atomtype_AH_DH(singletype["abbr"], singletype["name"],singletype["sigma"],singletype["lambda"],
                                    singletype["T0"], singletype["T1"], singletype["T2"])
                            for singletype in data["atomtypes"]]
            
            self.type2sigma = {atomtype.abbr: atomtype.sigma for atomtype in self.atomtypes}
            self.type2mylambda = {atomtype.abbr: atomtype.mylambda for atomtype in self.atomtypes}
            
            self.atoms = [Atom(atom["abbr"],atom["name"],atom["residue"],atom["residueid"],atom["mass"],atom["charge"])
                for atom in data["atoms"]
            ]

            if "bonds" in data.keys():
                self.bonds = [Bond(bond["i"] - 1, bond["j"] - 1, bond["length"], bond["force"])
                    for bond in data["bonds"]
                ]
            else:
                self.bonds = None
            
            if "angles" in data.keys():
                self.angles = [Angle(angle["i"] - 1, angle["j"] - 1, angle["k"] - 1, angle["theta_in_degree"], angle["force"])
                    for angle in data["angles"]
                ]
            else:
                self.angles = None
        
        elif self.function_type_LJ == "Wang-Frenkel" and self.function_type_Coulomb == "Debye-Huckel":

            self.molecule_name = data["moleculetype"][0]
            self.nrexcl = int(data["moleculetype"][1])

            self.typelist = [atomtype["abbr"] for atomtype in data["atomtypes"]]

            self.atomtypes = [Atomtype_WF_DH(singletype["abbr"], singletype["name"])
                for singletype in data["atomtypes"]]
            
            self.types2sigma = {f"{nonbonded["atom_abbr_1"]}-{nonbonded["atom_abbr_2"]}": nonbonded["sigma"] for nonbonded in data["nonbond_params"]}
            self.types2mu = {f"{nonbonded["atom_abbr_1"]}-{nonbonded["atom_abbr_2"]}": nonbonded["mu"] for nonbonded in data["nonbond_params"]}
            self.types2epsilon = {f"{nonbonded["atom_abbr_1"]}-{nonbonded["atom_abbr_2"]}": nonbonded["epsilon"] for nonbonded in data["nonbond_params"]}

            # We test 
            for abbr1 in self.typelist:
                for abbr2 in self.typelist:
                    pair = f"{abbr1}-{abbr2}"
                    if pair not in self.types2sigma:
                        pair_rev = f"{abbr2}-{abbr1}"
                        if pair_rev not in self.types2sigma:
                            print(f"ERROR: In ITPTopology, Missing nonbonded parameter for pair {pair} in nonbond_params section.")
                            quit()

            self.atoms = [Atom(atom["abbr"],atom["name"],atom["residue"],atom["residueid"],atom["mass"],atom["charge"])
                for atom in data["atoms"]
            ]

            if "nonbond_params" in data.keys():
                self.nonbond_params = data["nonbond_params"]
            else:
                self.nonbond_params = None

            if "bonds" in data.keys():
                self.bonds = [Bond(bond["i"] - 1, bond["j"] - 1, bond["length"], bond["force"])
                    for bond in data["bonds"]
                ]
            else:
                self.bonds = None
            
            if "angles" in data.keys():
                self.angles = [Angle(angle["i"] - 1, angle["j"] - 1, angle["k"] - 1, angle["theta_in_degree"], angle["force"])
                    for angle in data["angles"]
                ]
            else:
                self.angles = None

        else:
            print(f"ERROR: In ITPTopology, Unknown forcefield function types: LJ-{self.function_type_LJ}, Coulomb-{self.function_type_Coulomb}.")
            quit()


def read_itp(file_path):
    """
    Parse a GROMACS-style .itp file and return structured data.

    Returns a dictionary with keys:
    - 'moleculetype': [molname, nrexcl]
    - 'atomtypes': list of {abbr, name, sigma, lambda}
    - 'atoms': list of {id, abbr, name, residue, mass, charge}
    - 'bonds': list of {i, j, type, params}
    """

    with open(file_path, 'r') as f:
        lines = f.readlines()

    sections = defaultdict(list)
    current_section = None

    for line in lines:
        line = line.strip().split('#')[0]
        if not line or line.startswith(';'):
            continue
        if line.startswith('[') and line.endswith(']'):
            current_section = line.strip('[]').strip().lower().replace("-", "_") 
        elif current_section:
            sections[current_section].append(line)

    parsed_data = {}

    # Parse function type
    if "function_type" in sections:
        parsed_data["function_type_LJ"] = sections["function_type"][0].strip().split()[0]
        parsed_data["function_type_Coulomb"] = sections["function_type"][0].strip().split()[1]
    else:
        print(f"ERROR: The itp file {file_path} may not include function type. ")
        print(f"       You maybe using a new DROPPS version with an old itp file.")
        quit()
    
    
    # Parse moleculetype

    if "moleculetype" in sections and sections["moleculetype"]:
        parsed_data["moleculetype"] = sections["moleculetype"][0].split()
    
    if parsed_data["function_type_LJ"] == "Ashbaugh-Hatch" and parsed_data["function_type_Coulomb"] == "Debye-Huckel":

        # Parse atomtypes
        if "atomtypes" in sections:
            parsed_data["atomtypes"] = []
            for line in sections["atomtypes"]:
                parts = re.split(r'\s+', line)
                if len(parts) >= 4:
                    parsed_data["atomtypes"].append({
                        "abbr": parts[0],
                        "name": parts[1],
                        "sigma": float(parts[2]),
                        "lambda": float(parts[3]),
                        "T0": float(parts[4]),
                        "T1": float(parts[5]),
                        "T2": float(parts[6])
                    })

    elif parsed_data["function_type_LJ"] == "Wang-Frenkel" and parsed_data["function_type_Coulomb"] == "Debye-Huckel":
        
        # Parse atomtypes
        if "atomtypes" in sections:
            parsed_data["atomtypes"] = []
            for line in sections["atomtypes"]:
                parts = re.split(r'\s+', line)
                if len(parts) >= 1:
                    parsed_data["atomtypes"].append({
                        "abbr": parts[0],
                        "name": parts[1]
                    })
        
        if "nonbond_params" in sections:
            parsed_data["nonbond_params"] = []
            for line in sections["nonbond_params"]:
                parts = re.split(r'\s+', line)
                if len(parts) >= 5:
                    parsed_data["nonbond_params"].append({
                        "atom_abbr_1": parts[0],
                        "atom_abbr_2": parts[1],
                        "sigma": float(parts[2]),
                        "mu": float(parts[3]),
                        "epsilon": float(parts[4])
                    })
    else:
        print(f"ERROR: In read_itp, Unknown forcefield function types: LJ-{parsed_data['function_type_LJ']}, Coulomb-{parsed_data['function_type_Coulomb']}.")
        quit()

    # Parse atoms
    if "atoms" in sections:
        parsed_data["atoms"] = []
        for line in sections["atoms"]:
            line = line.split(";")[0].strip()
            parts = re.split(r'\s+', line)
            if len(parts) >= 6:
                parsed_data["atoms"].append({
                    "id": int(parts[0]),
                    "abbr": parts[1],
                    "name": parts[2],
                    "residue": parts[3],
                    "residueid": parts[4],
                    "mass": float(parts[5]),
                    "charge": float(parts[6])
                })

    # Parse bonds
    if "bonds" in sections:
        parsed_data["bonds"] = []
        for line in sections["bonds"]:
            line = line.split(";")[0].strip()
            parts = re.split(r'\s+', line)
            if len(parts) >= 4:
                bond_info = {
                    "i": int(parts[0]),
                    "j": int(parts[1]),
                    "length": float(parts[2]) * nanometer,
                    "force": float(parts[3]) * kilojoule_per_mole / nanometer ** 2
                }
                parsed_data["bonds"].append(bond_info)

    # Parse angles
    if "angles" in sections:
        parsed_data["angles"] = []
        for line in sections["angles"]:
            line = line.split(";")[0].strip()
            parts = re.split(r'\s+', line)
            if len(parts) >= 4:
                angle_info = {
                    "i": int(parts[0]),
                    "j": int(parts[1]),
                    "k": int(parts[2]),
                    "theta_in_degree": float(parts[3]) * degree,
                    "force": float(parts[4]) * kilojoule_per_mole / radian ** 2
                }
                parsed_data["angles"].append(angle_info)

    return ITPTopology(parsed_data)

def write_itp(file_path, topology:ITPTopology):

    with open(file_path, 'w') as itp_file:

        print(f"## Generating topology to {file_path}.")

        # Write forcefield and information lines
        print(f"## The molecule will be named as {topology.molecule_name} in topology file.")
        itp_file.write(f"[ moleculetype ]\n; molname  nrexcl\n")
        itp_file.write(f"{topology.molecule_name}     1\n\n")

        # Write function type
        itp_file.write(f"[ function-type ]\n")
        itp_file.write(f"; LJ_function   Coulomb_function\n")
        itp_file.write(f"  {topology.function_type_LJ}   {topology.function_type_Coulomb}\n\n")

        # Write atom type information
        itp_file.write(f"[ atomtypes ]\n")

        if topology.function_type_LJ == "Wang-Frenkel":

            itp_file.write(f"; atom-abbr     atom-name\n")
            for atomtype in topology.atomtypes:
                itp_file.write(f"  {atomtype.abbr:12s}  {atomtype.name:7s}\n")
            itp_file.write("\n")

            itp_file.write(f"[ nonbond_params ]\n")
            itp_file.write(f"; atom-abbr-1  atom-abbr-2   sigma   mu      epsilon\n")
            for pair, sigma in topology.types2sigma.items():
                mu = topology.types2mu[pair]
                epsilon = topology.types2epsilon[pair]
                abbr1, abbr2 = pair.split("-")
                itp_file.write(f"  {abbr1:12s}  {abbr2:12s}  {sigma:>6.3f}  {mu:>6.3f}  {epsilon:>13.6f}\n")
            itp_file.write("\n")


        elif topology.function_type_LJ == "Ashbaugh-Hatch":

            itp_file.write(f"; atom-abbr     atom-name  sigma   lambda   T0               T1              T2\n")
            for atomtype in topology.atomtypes:
                itp_file.write(f"  {atomtype.abbr:12s}  {atomtype.name:7s}   "
                           + f"{atomtype.sigma:>6.3f}   {atomtype.mylambda:.3f}    "
                           + f"{atomtype.T0:>13.9f}   "
                           + f"{atomtype.T1:>13.9f}   "
                           + f"{atomtype.T2:>13.9f}   \n")
            itp_file.write("\n")

        # Write atom information
        itp_file.write(f"[ atoms ]\n")
        itp_file.write(f"; id   atom-abbr  atom-name     residue  resid  mass    charge\n")
        qtot = 0
        for index, atom in enumerate(topology.atoms):
            qtot +=atom.charge

            itp_file.write(f"  {index + 1:<3d}  {atom.abbr:<8s}   {atom.name:12s}  "
                           + f"{atom.residuename:7s}  {atom.residueid:<5d}  "
                           + f"{atom.mass:>6.2f}  {atom.charge:5.2f}   ; qtot {qtot}\n")
        itp_file.write("\n")


        # Write bond information

        if topology.bonds is not None:
            itp_file.write(f"[ bonds ]\n")
            itp_file.write(f"; ai   aj   r0/nm k/(kJ/mol)\n")

            for bond in topology.bonds:
                itp_file.write(f"  {(bond.a1 + 1):<3d}  {(bond.a2 + 1):<3d}  {bond.r0.value_in_unit(nanometer):.2f}  {bond.k.value_in_unit(kilojoule_per_mole/nanometer**2)}\n")
            itp_file.write("\n")

        # Write angle information
        if topology.angles is not None:
            itp_file.write(f"[ angles ]\n")
            itp_file.write(f"; ai   aj   ak   theta/degree k/(kJ/mol)/rad^2\n")

            for angle in topology.angles:
                itp_file.write(f"  {(angle.a1 + 1):<3d}  {(angle.a2 + 1):<3d}  {(angle.a3 + 1):<3d}"\
                             + f"  {angle.theta_in_degree.value_in_unit(degree):.2f}"\
                             + f"       {angle.k.value_in_unit(kilojoule_per_mole/radian**2)}\n")
            itp_file.write("\n")
        

# Example usage
if __name__ == "__main__":
    import sys
    import json


    if len(sys.argv) < 2:
        print("Usage: python itp_reader.py <path_to_itp_file>")
        sys.exit(1)

    file_path = sys.argv[1]
    result = read_itp(file_path)
    #print(json.dumps(result, indent=2))

