from openmm.app import Topology
from openmm import System
from collections import defaultdict

if __name__ == '__main__':
    import sys, os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dropps.fileio.tpr_reader import read_tpr
from dropps.fileio.itp_reader import ITPTopology
from typing import List
from copy import deepcopy

from itertools import chain

import re


## IMPORTANT: In DROPPS indexing, all atom index starts with 0.

class index_singleGroup():
    def __init__(self, name, indices, sorted=False):

        self.name = deepcopy(name.replace(" ", "_"))
        self.indices = deepcopy(indices)
        if sorted:
            self.indices.sort()
        self.length = len(self.indices)

class indexGroups():
    
    def __init__(self, mdtopology: Topology, itp_list: List[ITPTopology]):
        print(f"\n## Reading tpr file and loading topology")
        self.atomNumber = mdtopology.getNumAtoms()        
        self.itp_list = itp_list

        print(f"## Going to load {len(self.itp_list)} itp topology instances.")
        print(f"## The topology contains {self.atomNumber} atoms within {len(self.itp_list)} molecules.")

        print(f"## Analysing Proteins...")
        
        # We now build list for properties of all atoms.
        all_props = [
            (itp.molecule_name, atom.name, atom.abbr, atom.residueid,
            atom.residuename, atom.charge, atom.mass)
            for itp in itp_list for atom in itp.atoms
        ]

        self.atom_molnames, self.atom_names, self.atom_abbrs, self.atom_resids, \
        self.atom_resnames, self.atom_charges, self.atom_masses = zip(*all_props)

        self.atom_chainids = [chainid for chainid, chain in enumerate(itp_list)
                              for _ in chain.atoms]

        # We now build initial list for index groups
        self.initilize()
        self.validate()
    
    def initilize(self):
        self.index_groups = list()
        self.index_groups.append(index_singleGroup("System", list(range(0, self.atomNumber)), sorted=True))

        for molname in set(self.atom_molnames):
            ids = [i for i, name in enumerate(self.atom_molnames) if name == molname]
            self.index_groups.append(index_singleGroup(molname, ids))
    
    def write_ndx(self, ndx_filename):
        if ".ndx" not in ndx_filename:
            ndx_filename += ".ndx"
        with open(ndx_filename, 'w') as f:
            for index_group in self.index_groups:
                f.write(f"[ {index_group.name} ]\n")
                f.write(indices2string(index_group.indices))
                f.write("\n\n")
    
    def load_ndx(self, ndx_filename):
        with open(ndx_filename, 'r') as f:
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
            
            index_groups = list()
            
            for section_name in sections.keys():
                index_groups.append(index_singleGroup(section_name, string2indices(sections[section_name])))
        self.index_groups = deepcopy(index_groups)
        self.validate()
    
    def validate(self):
        validated = True
        for i, group in enumerate(self.index_groups):
            indices = group.indices
            seen = set()
            for idx in indices:
                if not (0 <= idx < self.atomNumber):
                    print(f"ERROR: Invalid index {idx} found in group {i}. Must be in range 0 to {self.atomNumber - 1}.")
                    validated = False
                if idx in seen:
                    print(f"ERROR: Duplicate index {idx} found in group {i}.")
                    validated = False
                else:
                    seen.add(idx)
        if not validated:
            quit()
    
    def print_all(self):
        print(f"\n")
        print(f"    ###########################################################################")
        print(f"    #### Index containing {len(self.index_groups):>3d} groups for {self.atomNumber:>7d} atoms."
              + f"                    ####")
        print(f"    #### |--------------------|---------------------|---------------------|####")
        for id, group in enumerate(self.index_groups): 
            name_formatted = group.name if len(group.name) <= 20 else group.name[:17] + '...'
            print(f"    #### | Index group {id:<6d} | {name_formatted:<20s}| {len(group.indices):>7d} atoms.      |####")
        print(f"    #### |--------------------|---------------------|---------------------|####")
        print(f"    ###########################################################################")
    
    def __print_chains_verbose(self):
        print(f"#### This index contains {self.atomNumber} atoms within {len(self.itp_list)} molecules.")
        print(f"#### {len(set(self.atom_molnames)):>3d} molecule types:  {','.join(set(self.atom_molnames))}")
        print(f"#### {len(set(self.atom_names)):>3d} Atom names:      {','.join(set(self.atom_names))}")
        print(f"#### {len(set(self.atom_abbrs)):>3d} atom abbrs:      {','.join(set(self.atom_abbrs))}")
        print(f"#### {len(set(self.atom_resnames)):>3d} atom resnames:   {','.join(set(self.atom_abbrs))}")
        grouped_indices, chainid_list = self.__splitch_indices(list(range(self.atomNumber)))

        string_list = [f"{chainid_list[chainID]}({self.atom_resids[grouped_indices[chainID][0]]}-{self.atom_resids[grouped_indices[chainID][-1]]})"
            for chainID in range(len(chainid_list))
        ]

        print(f"#### ChainID(ResID): {';'.join(string_list)}")
    
    def __splitch_indices(self, indices):

        # Use defaultdict to group indices by their chainid
        chainid_to_indices = defaultdict(list)
        for idx in indices:
            chainid = self.atom_chainids[idx]
            chainid_to_indices[chainid].append(idx)

        grouped_indices = list(chainid_to_indices.values())
        chainid_list = list(chainid_to_indices.keys())

        return grouped_indices, chainid_list
    
    def splitch_indices(self, indices):
        return self.__splitch_indices(indices)[0]
    
    def __splitch(self, groupID):
        grouped_indices, chainid_list = self.__splitch_indices(self.index_groups[groupID].indices)
        print(f"#### Splitting group {groupID} ({self.index_groups[groupID].name}) into {len(grouped_indices)} groups.")
        for newgroupID in range(len(grouped_indices)):
            newGroupName = f"{self.index_groups[groupID].name}_Ch{chainid_list[newgroupID]}"
            newGroupIndices = grouped_indices[newgroupID]
            self.index_groups.append(index_singleGroup(newGroupName, newGroupIndices))
        self.validate()

    def __rename(self, groupID, newname):
        if groupID >= len(self.index_groups) or groupID < 0:
            print(f"ERROR: Group {groupID} not exist.")
            return()
        print(f"#### Naming group {groupID} ({self.index_groups[groupID].name}) to {newname}")
        self.index_groups[groupID].name = deepcopy(newname)
    
    def showhelp(self):
        help_lines = [
            "",
            "Enter  :  list groups   'p': print groups   'l': Verbose list of chains   'q': save and quit",
            "'h'    :  show help     'name' nr:  name group    'splitch nr': split group nr by chains ",
            "'mol'  name :  select molecule name      'group' nr   : select group index",
            "'res'  name :  select residue name       'index' nr   : select atom index (starts with 0)",
            "'abbr' name :  select atom abbreviation  'resid' nr   : select residue index",
            "                                         'chainid nr' : select chain index"
                ]
        print('\n'.join(help_lines))

    def __phraseSingleSelection(self, single_selection_string):
        
        command2list = {"mol": self.atom_molnames,
                        "name": self.atom_names,
                        "abbr": self.atom_abbrs,
                        "res": self.atom_resnames,

                        "index": list(range(self.atomNumber)),
                        "resid": self.atom_resids,
                        "chainid": self.atom_chainids}
        
        if len(single_selection_string.split(" ")) < 2:
            print(f"Cannot process single selection string {single_selection_string}.")
            raise(TypeError)
                
        command = single_selection_string.split(" ")[0].strip()
        rangestring = ''.join(single_selection_string.split(" ")[1:]).strip()

        if command == "group":
            targets = parse_number_range(rangestring)
            index_in_groups = [self.index_groups[id].indices for id in targets]
            match = list(set(chain.from_iterable(index_in_groups)))
            print(f"## Find {len(match):>4d} matches for {command:<10s} {rangestring}.")
            return match

        if command in ["mol", "name", "abbr", "res"]:
            
            sources = command2list[command]
            targets = parse_string_range(rangestring)
        
        elif command in "index, resid, chainid":
            sources = command2list[command]
            targets = parse_number_range(rangestring)
        
        else:
            print(f"Unknown command {command}.")
        
        match = [i for i, val in enumerate(sources) if val in targets]
        print(f"## Find {len(match):>4d} matches for {command:<10s} {rangestring}.")

        return match

    def phraseSelection(self, selection_string):
        
        def parse_command(cmd):
            return set(self.__phraseSingleSelection(cmd.strip()))

        tokens = re.findall(r'\(|\)|&|\||[^&|()]+', selection_string)

        def evaluate(tokens):
            stack, ops = [], []
            precedence = {'&': 2, '|': 1}

            def apply_op():
                r, l = stack.pop(), stack.pop()
                stack.append(l & r if ops.pop() == '&' else l | r)

            for token in tokens:
                token = token.strip()
                if not token:
                    continue
                if token == '(':
                    ops.append(token)
                elif token == ')':
                    while ops and ops[-1] != '(':
                        apply_op()
                    ops.pop()
                elif token in precedence:
                    while ops and ops[-1] in precedence and precedence[ops[-1]] >= precedence[token]:
                        apply_op()
                    ops.append(token)
                else:
                    stack.append(parse_command(token))

            while ops:
                apply_op()

            return stack[0] if stack else set()

        result_indices = sorted(evaluate(tokens))
        print(f"## The resultant group will contains {len(result_indices)} atoms.")

        return result_indices

    def __append_selection(self, selection_string):

        result_indices = self.phraseSelection(selection_string)

        if len(result_indices) > 0:
            #group_name = selection_string.replace(" ","").replace("&","-").replace("|","|").replace("(","B").replace(")","b")
            group_name = selection_string.replace(" ","")
            self.index_groups.append(index_singleGroup(group_name, result_indices))

    def command(self, command):

        singleCommandDict = {"p": self.print_all, "l": self.__print_chains_verbose, "h":self.showhelp}
        actionDict = {"splitch": self.__splitch, "name": self.__rename}

        try:

            command_split = command.split(" ")
            first_command = command_split[0]
            
            if first_command in singleCommandDict.keys():
                singleCommandDict[first_command]()

            elif first_command in actionDict.keys():
                if first_command == "splitch":
                    if len(command_split) < 2:
                        print("ERROR: Split chain for which group?")
                        return
                    self.__splitch(int(command.split(" ")[1]))

                elif first_command == "name":
                    if len(command_split) < 3:
                        print("ERROR: Name what group to what?")
                        return
                    self.__rename(int(command.split(" ")[1]), command.split(" ")[2])
            
            else:
                # Preparing to phrase selection string
                self.__append_selection(command)
                    
        except:
            print(f"ERROR: Error in processing command \"{command}\".")


def indices2string(indices: List[int], field_width = 6):
    lines = []
    for i in range(0, len(indices), 10):
        line = "".join(f"{num:{field_width}d}" for num in indices[i:i+10])
        lines.append(line)
    return "\n".join(lines)

def string2indices(indices_lines):
    numbers = []
    for line in indices_lines:

        if not line.strip():
            continue

        for token in line.split():
            try:
                num = int(token)
                numbers.append(num)
            except ValueError:
                continue
    return numbers

def parse_number_range(s):
    s = s.replace(" ", "")  # Remove all spaces
    result = []
    parts = s.split(",")
    for part in parts:
        if "-" in part:
            start, end = part.split("-")
            result.extend(range(int(start), int(end) + 1))
        else:
            result.append(int(part))
    return result

def parse_string_range(s):
    s = s.replace(" ", "")  # Remove all spaces
    result = s.split(",")
    return result


if __name__ == '__main__':
    tpr = read_tpr(sys.argv[1])
    myid = indexGroups(tpr.mdtopology, tpr.itp_list)
    myid.load_ndx("haha.ndx")
    myid.print_all()
    




