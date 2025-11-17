import glob
from copy import deepcopy
import os
from pathlib import Path 
import re

forcefields_directory = Path(os.path.dirname(os.path.abspath(__file__)), "forcefields")
forcefields_files = glob.glob(f"{forcefields_directory}/*.ff")
pattern = r'(.*?)\.ff'
forcefield_list = [os.path.basename(re.search(pattern, path).group(1)) for path in forcefields_files]

class forcefield_top:
    def __init__(self, abbr, abbr2aa, abbr2original, abbr2sigma, abbr2lambda, abbr2mass, abbr2charge, abbr2tempcoff,
                 bondtypes, abbr2bondtypeindex, bond2param, modifications):
        self.abbr = abbr
        self.abbr2aa = abbr2aa
        self.abbr2original = abbr2original
        self.abbr2sigma = abbr2sigma
        self.abbr2lambda = abbr2lambda
        self.abbr2mass = abbr2mass
        self.abbr2charge = abbr2charge
        self.abbr2tempcoff = abbr2tempcoff
        self.bondtypes = bondtypes
        self.abbr2bondtypeindex = abbr2bondtypeindex
        self.abbr2bondtype = {abbr:self.bondtypes[self.abbr2bondtypeindex[abbr]] for abbr in self.abbr2bondtypeindex.keys()}
        self.bond2param = bond2param
        self.modifications = modifications

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
    #if forcefield_name not in forcefield_list:
    #    print("ERROR: Unknown forcefield %s." % forcefield_name)
    #    quit()
    
    #ff_path = Path(forcefields_directory, f"ff_{forcefield_name}.dat")
    ff_path = parameter_file_path
    sections = read_and_split_sections(ff_path)

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
        print(f"ERROR: The forcefield file {ff_path} may not include any residues.")
        quit()

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
        print(f"ERROR: The bond lines in forcefield file {ff_path} not a duplicate of 3.")
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


    return forcefield_top(abbr, abbr2aa, abbr2original, abbr2sigma, abbr2lambda, 
                          abbr2mass, abbr2charge, abbr2tempcoff,
                          bondtypes, abbr2bondtypeindex, bond2param, modifications)
    

if __name__ == '__main__':
    ff = getff("HPS", 310)
    print(ff.abbr2aa)
    
