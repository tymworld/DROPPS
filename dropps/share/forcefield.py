import glob
from copy import deepcopy
import os
from pathlib import Path 
import re
import numpy as np

forcefields_directory = Path(os.path.dirname(os.path.abspath(__file__)), "forcefields")
forcefields_files = glob.glob(f"{forcefields_directory}/*.ff")
pattern = r'(.*?)\.ff'
forcefield_list = [os.path.basename(re.search(pattern, path).group(1)) for path in forcefields_files]

def _parse_relative_permittivity_from_sections(sections):
    # Backward-compatible default
    model = "constant"
    value = 80.0
    coeffs = None

    if "relative_permittivity" not in sections or len(sections["relative_permittivity"]) == 0:
        return model, value, coeffs

    line = sections["relative_permittivity"][0].strip()
    parts = line.split()

    # Try pure numeric input first:
    # 1 number  => constant
    # 5 numbers => temperature-dependent: k_-1, k0, k1, k2, k3
    try:
        nums = [float(x) for x in parts]
        if len(nums) == 1:
            return "constant", nums[0], None
        if len(nums) == 5:
            return "temperature_dependent", None, nums
    except Exception:
        pass

    # Also support labeled forms for convenience.
    keyword = parts[0].lower()
    if keyword == "constant" and len(parts) >= 2:
        try:
            return "constant", float(parts[1]), None
        except Exception:
            pass
    if keyword in ("temperature-dependent", "temperature_dependent", "td") and len(parts) >= 6:
        try:
            return "temperature_dependent", None, [float(x) for x in parts[1:6]]
        except Exception:
            pass

    print("ERROR: Cannot parse [ relative-permittivity ] section in forcefield file. Use 1 number (constant) or 5 numbers (k_-1 k0 k1 k2 k3).")
    quit()

class forcefield_top_WF_DH:
    def __init__(self, abbr, abbr2aa, abbr2original, abbrs2epsilon, abbrs2sigma, abbrs2mu, 
                 abbr2mass, abbr2charge, relative_permittivity_mode, relative_permittivity, relative_permittivity_coeffs,
                 bondtypes, abbr2bondtypeindex, bond2param, simulation_settings):
        self.function_type_LJ = "Wang-Frenkel"
        self.function_type_Coulomb = "Debye-Huckel"
        self.abbr = abbr
        self.abbr2aa = abbr2aa
        self.abbr2original = abbr2original
        self.abbrs2epsilon = abbrs2epsilon
        self.abbrs2sigma = abbrs2sigma
        self.abbrs2mu = abbrs2mu
        self.abbr2mass = abbr2mass
        self.abbr2charge = abbr2charge
        self.relative_permittivity_mode = relative_permittivity_mode
        self.relative_permittivity = relative_permittivity
        self.relative_permittivity_coeffs = relative_permittivity_coeffs
        self.bondtypes = bondtypes
        self.abbr2bondtypeindex = abbr2bondtypeindex
        self.abbr2bondtype = {abbr:self.bondtypes[self.abbr2bondtypeindex[abbr]] for abbr in self.abbr2bondtypeindex.keys()}
        self.bond2param = bond2param
        self.simulation_settings = simulation_settings

class forcefield_top_AH_DH:
    def __init__(self, abbr, abbr2aa, abbr2original, abbr2sigma, abbr2lambda, epsilon, abbr2mass, abbr2charge, abbr2tempcoff,
                 relative_permittivity_mode, relative_permittivity, relative_permittivity_coeffs,
                 bondtypes, abbr2bondtypeindex, bond2param, modifications, simulation_settings):
        self.function_type_LJ = "Ashbaugh-Hatch"
        self.function_type_Coulomb = "Debye-Huckel"
        self.abbr = abbr
        self.abbr2aa = abbr2aa
        self.abbr2original = abbr2original
        self.abbr2sigma = abbr2sigma
        self.abbr2lambda = abbr2lambda
        self.epsilon = epsilon
        self.abbr2mass = abbr2mass
        self.abbr2charge = abbr2charge
        self.abbr2tempcoff = abbr2tempcoff
        self.relative_permittivity_mode = relative_permittivity_mode
        self.relative_permittivity = relative_permittivity
        self.relative_permittivity_coeffs = relative_permittivity_coeffs
        self.bondtypes = bondtypes
        self.abbr2bondtypeindex = abbr2bondtypeindex
        self.abbr2bondtype = {abbr:self.bondtypes[self.abbr2bondtypeindex[abbr]] for abbr in self.abbr2bondtypeindex.keys()}
        self.bond2param = bond2param
        self.modifications = modifications
        self.simulation_settings = simulation_settings


def _parse_simulation_settings_from_sections(sections):
    settings = []
    if "simulation_setting" not in sections:
        return settings

    allowed_policy = {"none", "forced", "recommended"}
    for line in sections["simulation_setting"]:
        parts = line.strip().split()
        if len(parts) != 4:
            print("ERROR: Cannot parse [ simulation-setting ] in forcefield file. Each line must have 4 values:")
            print("       name value type_at_NVT type_at_NPT")
            quit()

        name, value, nvt_policy, npt_policy = parts
        nvt_policy = nvt_policy.lower()
        npt_policy = npt_policy.lower()
        if nvt_policy not in allowed_policy or npt_policy not in allowed_policy:
            print("ERROR: Invalid policy in [ simulation-setting ] section.")
            print("       The 3rd and 4th values must be one of: none, forced, recommended.")
            quit()

        settings.append(
            {
                "name": name,
                "value": value,
                "nvt_policy": nvt_policy,
                "npt_policy": npt_policy,
            }
        )
    return settings

def read_and_split_sections(filename):
    sections = {}
    current_section = None
    current_lines = []

    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()  # Remove leading and trailing whitespace
            if not line or line[0] == '#':  # Skip blank lines
                continue

            # Check if the line starts a new section (i.e., it's of the form [section-name])
            if line.startswith("[") and line.endswith("]"):
                # If we are currently tracking a section, save it
                if current_section:
                    sections[current_section] = current_lines

                # Start a new section
                current_section = line[1:-1].strip().replace("-", "_")  # Remove the brackets from the section title
                current_lines = []
            else:
                # Add line to the current section
                current_lines.append(line)

        # Don't forget to save the last section
        if current_section:
            sections[current_section] = current_lines

    return sections

def getff(parameter_file_path):
    ff_path = parameter_file_path
    sections = read_and_split_sections(ff_path)

    LJ_function = sections['function_type'][0].strip().split()[0]
    Coulomb_function = sections['function_type'][0].strip().split()[1]

    if LJ_function == "Ashbaugh-Hatch" and Coulomb_function == "Debye-Huckel":
        return getff_AH_DH(sections)
    elif LJ_function == "Wang-Frenkel" and Coulomb_function == "Debye-Huckel":
        return getff_WF_DH(sections)
    else:
        print(f"ERROR: In forcefield, Unknown forcefield function types: LJ-{LJ_function}, Coulomb-{Coulomb_function}.")
        quit()

def getff_WF_DH(sections):

    # Initialize containers
    abbr = []
    abbr2aa = {}
    abbr2original = {}
    abbrs2epsilon = {}
    abbrs2sigma = {}
    abbrs2mu = {}
    abbr2mass = {}
    abbr2charge = {}
    relative_permittivity_mode, relative_permittivity, relative_permittivity_coeffs = _parse_relative_permittivity_from_sections(sections)
    simulation_settings = _parse_simulation_settings_from_sections(sections)

    # Check for 'residue_type' section
    if 'residue_type' not in sections:
        print(f"ERROR: The forcefield file may not include any residues.")
        quit()
    
    # Iterate over lines in the 'residue_type' section to get residue information
    for line in sections['residue_type']:
        parts = line.strip().split()
        line_abbr = parts[0]
        line_aa = parts[1]
        line_mass = float(parts[2])
        line_charge = float(parts[3])

        abbr.append(line_abbr)
        abbr2aa[line_abbr] = deepcopy(line_aa)
        abbr2original[line_abbr] = deepcopy(line_aa)
        abbr2mass[line_abbr] = deepcopy(line_mass)
        abbr2charge[line_abbr] = deepcopy(line_charge)

    # Iterate over lines in 'residue_ptm' section to get PTM residue information

    if 'residue_ptm' in sections:
        for line in sections['residue_ptm']:
            parts = line.strip().split()
            line_abbr = parts[0]
            line_aa = parts[1]
            line_mass = float(parts[2])
            line_charge = float(parts[3])

            abbr.append(line_abbr)
            abbr2aa[line_abbr] = deepcopy(line_aa)
            abbr2original[line_abbr] = deepcopy(line_aa).split('_')[0]
            abbr2mass[line_abbr] = deepcopy(line_mass)
            abbr2charge[line_abbr] = deepcopy(line_charge)

    # Iterate over each residue in abbr to get N and C terminal patch version
    
    for abbr_single in deepcopy(abbr):

        abbr.append(abbr_single + '_N')
        abbr2aa[abbr_single + '_N'] = abbr2aa[abbr_single] + '_NTD'
        abbr2original[abbr_single + '_N'] = abbr2original[abbr_single]
        abbr2mass[abbr_single + '_N'] = abbr2mass[abbr_single]
        abbr2charge[abbr_single + '_N'] = abbr2charge[abbr_single] + 1.0

        abbr.append(abbr_single + '_C')
        abbr2aa[abbr_single + '_C'] = abbr2aa[abbr_single] + '_CTD'
        abbr2original[abbr_single + '_C'] = abbr2original[abbr_single]
        abbr2mass[abbr_single + '_C'] = abbr2mass[abbr_single]
        abbr2charge[abbr_single + '_C'] = abbr2charge[abbr_single] - 1.0
    
    # We now get parameters for sigma, epslion, and mu
    # First we test whether the parameters in forcefield file make sense.
    if 'residue_sigma' not in sections.keys() or 'residue_epsilon' not in sections.keys() or 'residue_mu' not in sections.keys():
        print(f"ERROR: The forcefield file may not include sigma, epsilon, or mu sections.")
        quit()

    epsilon_residue_list = sections['residue_epsilon'][0].strip().split()
    sigma_residue_list = sections['residue_sigma'][0].strip().split()
    mu_residue_list = sections['residue_mu'][0].strip().split()

    epsilon_matrix = np.array([np.fromstring(row.strip(), sep=' ')
                               for row in sections['residue_epsilon'][1:]])
    
    sigma_matrix = np.array([np.fromstring(row.strip(), sep=' ')
                               for row in sections['residue_sigma'][1:]])
    
    mu_matrix = np.array([np.fromstring(row.strip(), sep=' ')
                               for row in sections['residue_mu'][1:]])
    
    if epsilon_matrix.shape != (len(epsilon_residue_list), len(epsilon_residue_list)):
        print(f"ERROR: The size of epsilon matrix does not match the number of residues in residue_epsilon section.")
        quit()
    
    if sigma_matrix.shape != (len(sigma_residue_list), len(sigma_residue_list)):
        print(f"ERROR: The size of sigma matrix does not match the number of residues in residue_sigma section.")
        quit()
    
    if mu_matrix.shape != (len(mu_residue_list), len(mu_residue_list)):
        print(f"ERROR: The size of mu matrix does not match the number of residues in residue_mu section.")
        quit()

    def _base_residue(name: str) -> str:
        return name[:-2] if name.endswith(("_N", "_C")) else name
    
    existence_test = all(
        (_base_residue(a) in epsilon_residue_list and
         _base_residue(a) in sigma_residue_list and
         _base_residue(a) in mu_residue_list)
        for a in abbr
    )
        
    if not existence_test:
        print(f"ERROR: Some residues in residue_type section do not have corresponding parameters in residue_sigma, residue_epsilon, or residue_mu sections.")
        quit()
    
    if not np.allclose(epsilon_matrix, epsilon_matrix.T):
        print(f"ERROR: The epsilon matrix is not symmetric.")
        quit()
    
    if not np.allclose(sigma_matrix, sigma_matrix.T):
        print(f"ERROR: The sigma matrix is not symmetric.")
        quit()
    
    if not np.allclose(mu_matrix, mu_matrix.T):
        print(f"ERROR: The mu matrix is not symmetric.")
        quit()
    
    # We now set parameters for abbrs2epsilon, abrs2sigma, and abrs2mu

    for r1 in abbr:
        for r2 in abbr:
            abbrs2epsilon[f"{r1}-{r2}"] = epsilon_matrix[
                epsilon_residue_list.index(_base_residue(r1)),
                epsilon_residue_list.index(_base_residue(r2))
            ]
            abbrs2sigma[f"{r1}-{r2}"] = sigma_matrix[
                sigma_residue_list.index(_base_residue(r1)),
                sigma_residue_list.index(_base_residue(r2))
            ]
            abbrs2mu[f"{r1}-{r2}"] = mu_matrix[
                mu_residue_list.index(_base_residue(r1)),
                mu_residue_list.index(_base_residue(r2))
            ]
    
    # Now we treat bond types
    if len(sections['bond']) % 4 != 0 :
        print(f"ERROR: The bond lines in forcefield file not a duplicate of 3.")
        quit()
    
    bond_type_number = round(len(sections['bond'])/4)
    bond2param = dict()

    for bond_type_index in range(bond_type_number):
        bond_name = sections['bond'][bond_type_index * 4]
        bond_length = float(sections['bond'][bond_type_index * 4 + 1].split()[0])
        bond_k = float(sections['bond'][bond_type_index * 4 + 1].split()[1])

        bond_terminal_1_atoms_raw = sections['bond'][bond_type_index * 4 + 2].split()
        bond_terminal_1_atoms_NTD = [atom + "_N" for atom in bond_terminal_1_atoms_raw]
        bond_terminal_1_atoms_CTD = [atom + "_C" for atom in bond_terminal_1_atoms_raw]
        bond_terminal_2_atoms_raw = sections['bond'][bond_type_index * 4 + 3].split()
        bond_terminal_2_atoms_NTD = [atom + "_N" for atom in bond_terminal_2_atoms_raw]
        bond_terminal_2_atoms_CTD = [atom + "_C" for atom in bond_terminal_2_atoms_raw]

        bond_terminal_1_atoms = deepcopy(bond_terminal_1_atoms_raw)
        bond_terminal_1_atoms.extend(bond_terminal_1_atoms_NTD)
        bond_terminal_1_atoms.extend(bond_terminal_1_atoms_CTD)

        bond_terminal_2_atoms = deepcopy(bond_terminal_2_atoms_raw)
        bond_terminal_2_atoms.extend(bond_terminal_2_atoms_NTD)
        bond_terminal_2_atoms.extend(bond_terminal_2_atoms_CTD)

        bond2param[bond_name] = dict(name=bond_name, length = bond_length, k = bond_k, 
                                     atom_1 = bond_terminal_1_atoms, atom_2 = bond_terminal_2_atoms)
 
    bondtypes = list(bond2param.keys())
    abbr2bondtypeindex = dict()

    for bond_type_index in range(len(bondtypes)):
        bond_type = bondtypes[bond_type_index]
        abbr2bondtype_temp = {f"{atom1}-{atom2}" : bond_type_index 
                              for atom1 in bond2param[bond_type]['atom_1']
                              for atom2 in bond2param[bond_type]['atom_2']}
        abbr2bondtypeindex.update(abbr2bondtype_temp)
    
    return forcefield_top_WF_DH(abbr, abbr2aa, abbr2original, abbrs2epsilon, abbrs2sigma, abbrs2mu, \
                 abbr2mass, abbr2charge, relative_permittivity_mode, relative_permittivity, relative_permittivity_coeffs,
                 bondtypes, abbr2bondtypeindex, bond2param, simulation_settings)


def getff_AH_DH(sections):
    #if forcefield_name not in forcefield_list:
    #    print("ERROR: Unknown forcefield %s." % forcefield_name)
    #    quit()
    
    #ff_path = Path(forcefields_directory, f"ff_{forcefield_name}.dat")
    #ff_path = parameter_file_path
    #sections = read_and_split_sections(ff_path)

    # Initialize containers
    abbr = []
    abbr2aa = {}
    abbr2original = {}
    abbr2sigma = {}
    abbr2lambda = {}
    abbr2mass = {}
    abbr2charge = {}
    abbr2tempcoff = {}
    modifications = {}
    epsilon = None
    relative_permittivity_mode = "constant"
    relative_permittivity = 80.0
    relative_permittivity_coeffs = None
    simulation_settings = _parse_simulation_settings_from_sections(sections)

    # Check whether there is temperature coefficients
    temperature_coeff = dict()
    if "temperature_dependent" in sections.keys():
        for temperature_coeff_line in sections["temperature_dependent"]:
            coeff_0 = float(temperature_coeff_line.strip().split()[0])
            coeff_1 = float(temperature_coeff_line.strip().split()[1])
            coeff_2 = float(temperature_coeff_line.strip().split()[2])
            for aa in temperature_coeff_line.strip().split()[3].strip().split(','):
                temperature_coeff[aa] = [coeff_0, coeff_1, coeff_2]
    

    # Check for 'residue_type' section
    if 'residue_type' not in sections:
        print(f"ERROR: The forcefield file may not include any residues.")
        quit()

    if "residue_epsilon" not in sections or len(sections["residue_epsilon"]) == 0:
        print("ERROR: Ashbaugh-Hatch forcefield file must include [ residue-epsilon ] with a single epsilon value in kJ/mol.")
        quit()
    try:
        epsilon = float(sections["residue_epsilon"][0].split()[0])
    except Exception:
        print("ERROR: Cannot parse epsilon in [ residue-epsilon ] section for Ashbaugh-Hatch forcefield.")
        quit()

    relative_permittivity_mode, relative_permittivity, relative_permittivity_coeffs = _parse_relative_permittivity_from_sections(sections)

    # Iterate over lines in the 'residue_type' section to get residue information
    for line in sections['residue_type']:
        parts = line.strip().split()
        line_abbr = parts[0]
        line_aa = parts[1]
        line_sigma = float(parts[2])
        line_lambda = float(parts[3])
        line_mass = float(parts[4])
        line_charge = float(parts[5])

        abbr.append(line_abbr)
        abbr2aa[line_abbr] = deepcopy(line_aa)
        abbr2original[line_abbr] = deepcopy(line_aa)
        abbr2sigma[line_abbr] = deepcopy(line_sigma)
        abbr2lambda[line_abbr] = deepcopy(line_lambda)
        abbr2mass[line_abbr] = deepcopy(line_mass)
        abbr2charge[line_abbr] = deepcopy(line_charge)

        if line_abbr in temperature_coeff.keys():
            abbr2tempcoff[line_abbr] = [deepcopy(temperature_coeff[line_abbr][0]),
                                        deepcopy(temperature_coeff[line_abbr][1]), 
                                        deepcopy(temperature_coeff[line_abbr][2])]
        else:
            #print(f"## No temperature coefficient for residue {line_abbr} which will be set as 0,0,0.")
            abbr2tempcoff[line_abbr] = [0,0,0]
    
    # Iterate over lines in the 'residue_ptm' section to get residue information
    if 'residue_ptm' in sections:
        for line in sections['residue_ptm']:
            parts = line.strip().split()
            line_abbr = parts[0]
            line_aa = parts[1]
            line_sigma = float(parts[2])
            line_lambda = float(parts[3])
            line_mass = float(parts[4])
            line_charge = float(parts[5])

            abbr.append(line_abbr)
            abbr2aa[line_abbr] = deepcopy(line_aa)
            abbr2original[line_abbr] = deepcopy(line_aa).split('_')[0]
            abbr2sigma[line_abbr] = deepcopy(line_sigma)
            abbr2lambda[line_abbr] = deepcopy(line_lambda)
            abbr2mass[line_abbr] = deepcopy(line_mass)
            abbr2charge[line_abbr] = deepcopy(line_charge)
            if line_abbr in temperature_coeff.keys():
                abbr2tempcoff[line_abbr] = [deepcopy(temperature_coeff[line_abbr][0]),
                    deepcopy(temperature_coeff[line_abbr][1]), 
                    deepcopy(temperature_coeff[line_abbr][2])]
            else:
                #print(f"## No temperature coefficient for residue {line_abbr} which will be set as 0,0,0.")
                abbr2tempcoff[line_abbr] = [0,0,0]

    # Iterate over lines in the 'residue_modification' section to get modification information
    if 'residue_modification' in sections:
        for line in sections['residue_modification']:
            parts = line.strip().split()
            line_original = parts[0]
            line_abbr = parts[1]
            line_aa = parts[2]
            line_sigmas = [float(number) for number in parts[3].split(',')]
            line_lambdas = [float(number) for number in parts[4].split(',')]
            line_masses = [float(number) for number in parts[5].split(',')]
            line_charges = [float(number) for number in parts[6].split(',')]
            line_bond_length = float(parts[7])
            line_bond_k = float(parts[8])
            line_angle_thetas = [float(number) for number in parts[9].split(',')]
            line_angle_ks = [float(number) for number in parts[10].split(',')]
            modifications[line_abbr] = [line_original, line_abbr, line_aa,
                                        line_sigmas, line_lambdas, line_masses,
                                        line_charges, line_bond_length, line_bond_k,
                                        line_angle_thetas, line_angle_ks]
    
    # Iterate over each residue in abbr to get N and C terminal patch version
    for abbr_single in deepcopy(abbr):

        abbr.append(abbr_single + '_N')
        abbr2aa[abbr_single + '_N'] = abbr2aa[abbr_single] + '_NTD'
        abbr2original[abbr_single + '_N'] = abbr2original[abbr_single]
        abbr2sigma[abbr_single + '_N'] = abbr2sigma[abbr_single]
        abbr2lambda[abbr_single + '_N'] = abbr2lambda[abbr_single]
        abbr2mass[abbr_single + '_N'] = abbr2mass[abbr_single]
        abbr2charge[abbr_single + '_N'] = abbr2charge[abbr_single] + 1.0

        abbr.append(abbr_single + '_C')
        abbr2aa[abbr_single + '_C'] = abbr2aa[abbr_single] + '_CTD'
        abbr2original[abbr_single + '_C'] = abbr2original[abbr_single]
        abbr2sigma[abbr_single + '_C'] = abbr2sigma[abbr_single]
        abbr2lambda[abbr_single + '_C'] = abbr2lambda[abbr_single]
        abbr2mass[abbr_single + '_C'] = abbr2mass[abbr_single]
        abbr2charge[abbr_single + '_C'] = abbr2charge[abbr_single] - 1.0
        abbr2tempcoff[abbr_single + '_N'] = [deepcopy(abbr2tempcoff[abbr_single][0]),
                deepcopy(abbr2tempcoff[abbr_single][1]), 
                deepcopy(abbr2tempcoff[abbr_single][2])]
        abbr2tempcoff[abbr_single + '_C'] = [deepcopy(abbr2tempcoff[abbr_single][0]),
                deepcopy(abbr2tempcoff[abbr_single][1]), 
                deepcopy(abbr2tempcoff[abbr_single][2])]



    # Now we treat bond types
    if len(sections['bond']) % 4 != 0 :
        print(f"ERROR: The bond lines in forcefield file not a duplicate of 3.")
        quit()
    
    bond_type_number = round(len(sections['bond'])/4)
    bond2param = dict()

    for bond_type_index in range(bond_type_number):
        bond_name = sections['bond'][bond_type_index * 4]
        bond_length = float(sections['bond'][bond_type_index * 4 + 1].split()[0])
        bond_k = float(sections['bond'][bond_type_index * 4 + 1].split()[1])

        bond_terminal_1_atoms_raw = sections['bond'][bond_type_index * 4 + 2].split()
        bond_terminal_1_atoms_NTD = [atom + "_N" for atom in bond_terminal_1_atoms_raw]
        bond_terminal_1_atoms_CTD = [atom + "_C" for atom in bond_terminal_1_atoms_raw]
        bond_terminal_2_atoms_raw = sections['bond'][bond_type_index * 4 + 3].split()
        bond_terminal_2_atoms_NTD = [atom + "_N" for atom in bond_terminal_2_atoms_raw]
        bond_terminal_2_atoms_CTD = [atom + "_C" for atom in bond_terminal_2_atoms_raw]

        bond_terminal_1_atoms = deepcopy(bond_terminal_1_atoms_raw)
        bond_terminal_1_atoms.extend(bond_terminal_1_atoms_NTD)
        bond_terminal_1_atoms.extend(bond_terminal_1_atoms_CTD)

        bond_terminal_2_atoms = deepcopy(bond_terminal_2_atoms_raw)
        bond_terminal_2_atoms.extend(bond_terminal_2_atoms_NTD)
        bond_terminal_2_atoms.extend(bond_terminal_2_atoms_CTD)

        bond2param[bond_name] = dict(name=bond_name, length = bond_length, k = bond_k, 
                                     atom_1 = bond_terminal_1_atoms, atom_2 = bond_terminal_2_atoms)
 
    bondtypes = list(bond2param.keys())
    abbr2bondtypeindex = dict()

    for bond_type_index in range(len(bondtypes)):
        bond_type = bondtypes[bond_type_index]
        abbr2bondtype_temp = {f"{atom1}-{atom2}" : bond_type_index 
                              for atom1 in bond2param[bond_type]['atom_1']
                              for atom2 in bond2param[bond_type]['atom_2']}
        abbr2bondtypeindex.update(abbr2bondtype_temp)


    return forcefield_top_AH_DH(abbr, abbr2aa, abbr2original, abbr2sigma, abbr2lambda, epsilon,
                          abbr2mass, abbr2charge, abbr2tempcoff,
                          relative_permittivity_mode, relative_permittivity, relative_permittivity_coeffs,
                          bondtypes, abbr2bondtypeindex, bond2param, modifications, simulation_settings)
    

if __name__ == '__main__':
    ff = getff("/Users/TYM-work/Repository/DROPPS/dropps/share/forcefields/MPiPi.ff")
    print(ff.abbrs2epsilon)
    
