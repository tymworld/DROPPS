from dropps.fileio.pdb_reader import read_pdb, write_pdb
from itertools import groupby
from openmm.unit import nanometer
import numpy as np
from copy import copy,deepcopy

#def cal_distance(vector_1: np.array, vector_2: np.array):
#    return np.sqrt(np.sum(np.pow(vector_2 - vector_1, 2), axis=1)/3)

def cal_distance(vector_1: np.ndarray, vector_2: np.ndarray):
    return np.linalg.norm(vector_2 - vector_1, axis=1) / np.sqrt(3)

def find_nearest(last: np.array, this: np.array, box):
    x = box[0].value_in_unit(nanometer)
    y = box[1].value_in_unit(nanometer)
    z = box[2].value_in_unit(nanometer)

    possible_additions = np.array([[timex * x, timey * y, timez * z] \
                                   for timex in [-1,0,1] for timey in [-1,0,1] for timez in [-1,0,1]])
    distance = cal_distance(np.array(this) + np.array(possible_additions), last)
    min_index = distance.argmin()
    return deepcopy(this + possible_additions[min_index])

def unwrap_pbc(atoms, box):

    print(f"## PBCTREATER: Altogether there are {len(atoms)} atoms (input).")
    
    chainID_string = [atom["chain"] for atom in atoms]

    molecule_list = []
    current_group = [0]  # Start with the first index

    for i in range(1, len(chainID_string)):
        if chainID_string[i] == chainID_string[i - 1]:
            current_group.append(i)
        else:
            molecule_list.append(current_group)
            current_group = [i]  # Start a new group

    molecule_list.append(current_group)

    print("## PBCTREATER: Built molecule list for %d molecules." % len(molecule_list))
    print(f"## PBCTREATER: Altogether there are {sum([len(molecule)for molecule in molecule_list])} atoms (output).")

    # We now unwrap each molecule

    treated_coordinates = list()
    raw_coordinates = np.array(
        [[atom["x"].value_in_unit(nanometer), atom["y"].value_in_unit(nanometer), atom["z"].value_in_unit(nanometer)] for atom in atoms]
    )

    for molecule in molecule_list:

        treating_atom = molecule[0]
        treated_coordinates.append(raw_coordinates[treating_atom])
        treating_atom += 1
        

        while(treating_atom <= molecule[-1]):

            treated_single_atom =  find_nearest(treated_coordinates[treating_atom - 1], raw_coordinates[treating_atom], box)
            treated_coordinates.append(treated_single_atom)
            treating_atom += 1
    
    treated_coordinates = np.array(treated_coordinates)
    print("## PBC treated %s coordinates to %s coordinates." % (raw_coordinates.shape, treated_coordinates.shape))
    
    return treated_coordinates

