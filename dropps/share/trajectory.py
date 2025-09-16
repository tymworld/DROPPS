import MDAnalysis as mda
import numpy as np
from openmm.unit import nanometer, picosecond, nanosecond

from dropps.fileio.tpr_reader import read_tpr
from dropps.share.indexing import indexGroups

def printcolor(text, success):
    if success:
        print(f"\033[32m{text}\033[0m")
    else:
        print(f"\033[5;34;46m{text}\033[0m")

class trajectory_class():

    def __init__(self, tpr_path, index_path = None, xtc_path = None):

        self.tpr = read_tpr(tpr_path)
        self.topology = self.tpr.mdtopology

        print(f"## Loaded topology from {tpr_path}.")

        positions_array = np.array(
            [[coor.value_in_unit(nanometer) for coor in atom]for atom in self.tpr.positions])
               
        self.Universe = mda.Universe(self.tpr.mdtopology, positions_array)

        # We generate charge lists

        charges = [atom.charge for itp in self.tpr.itp_list for atom in itp.atoms]
        masses = [atom.mass for itp in self.tpr.itp_list for atom in itp.atoms]

        self.Universe.add_TopologyAttr("charges", charges)
        self.Universe.add_TopologyAttr("masses", masses)
        
        print(f"## Generating Universe from topology and coordinates loaded from {tpr_path}.")

        self.index = indexGroups(self.tpr.mdtopology, self.tpr.itp_list) 
        print(f"## Index initilized by {tpr_path}.")

        if index_path is not None:
            self.index.load_ndx(index_path)
            print(f"## Known index entries loaded from {index_path}.")
        
        if xtc_path is not None:
            self.Universe.load_new(xtc_path)
            print(f"## Loaded trajectory file {xtc_path} into Universe.")
        
        # We generate information for each atoms.
        self.id2charge = charges
        self.id2masses = masses
        
        chain_length_list = [len(itp.atoms) for itp in self.tpr.itp_list]
        self.id2chainID = [i for i, length in enumerate(chain_length_list) for _ in range(length)]
        self.id2resID = [atom.residueid for itp in self.tpr.itp_list for atom in itp.atoms]
    
    def get_chainID(self, index):
        if index < 0 or index > self.num_atoms():
            print(f"ERROR: System of {self.num_atoms()} atoms doesn't contains atom index {index}.")
            quit()
        return self.id2chainID[index]
    
    def get_resID(self, index):
        if index < 0 or index > self.num_atoms():
            print(f"ERROR: System of {self.num_atoms()} atoms doesn't contains atom index {index}.")
            quit()
        return self.id2resID[index]
    
    def is_terminal(self, index):
        return (
            index == 0 or index == self.num_atoms() - 1 or
            self.get_chainID(index - 1) != self.get_chainID(index) or
            self.get_chainID(index + 1) != self.get_chainID(index)
        )
    
    def num_frames(self):
        return len(self.Universe.trajectory)
    
    def num_atoms(self):
        return len(self.Universe.atoms)
    
    def num_chains(self):
        return self.topology.getNumChains()
    
    def num_bonds(self):
        return len(self.Universe.bonds)
    
    def time_init(self):
        return self.Universe.trajectory[0].time * picosecond
    
    def time_end(self):
        return self.Universe.trajectory[-1].time * picosecond
    
    def time_step(self):
        return self.Universe.trajectory.dt * picosecond
    
    def time2frame(self, time_start = None, time_end = None, time_step = None):

        start_time = time_start * nanosecond if time_start is not None else self.time_init()
        end_time = time_end * nanosecond if time_end is not None else self.time_end()

        if time_step is None:
            interval_time = self.time_step()
        else:
            interval_time = time_step * nanosecond

        if start_time < self.time_init() or end_time > self.time_end():
            print(f"ERROR: Trajectory containing {self.time_init()} - {self.time_end()} "
                + f"while you demanding {start_time} - {end_time}.")
            print(f"YOU MUST BE KIDDING ME.")
            quit()

        start_frame = int((start_time - self.time_init()) / self.time_step())
        end_frame = int((end_time - self.time_init()) / self.time_step())
        interval_frame = int(interval_time / self.time_step())
        interval_frame = interval_frame if interval_frame > 1 else 1

        print(f"## Will use time {start_time} to {end_time} with interval of {interval_time}.")
        print(f"## Will use frame {start_frame} to {end_frame} with interval of {interval_frame}.") 

        return start_frame, end_frame, interval_frame
    
    def getSelection(self, selection_string, help = "analyzing group"):
        if selection_string.isdigit():
            indices = self.index.phraseSelection("group " + selection_string)
        else:         
            indices = self.index.phraseSelection(selection_string)
            
        selection = self.Universe.atoms[indices]
        print(f"## Selected {len(selection)} atoms for {help}.\n")
        selection_name = selection_string.replace(" ","")
        return selection, selection_name
    
    def getSelection_interactive(self, help = "analyzing group"):
        print(f"> Please enter one selection for {help}.")
        selection_string = input("> Selection: ").strip()
        return self.getSelection(selection_string)
    
    def getSelection_interactive_multiple(self, help="analyzing group"):
        selections = []
        selection_names = []
        print(f"> Please enter selections for {help}. Type 'q' or press Enter to finish.")
        
        while True:
            selection_string = input("> Selection: ").strip()
            if selection_string.lower() == 'q' or selection_string == '':
                break
            try:
                selection, name = self.getSelection(selection_string)
                selections.append(selection)
                selection_names.append(name)
            except Exception as e:
                print(f"Error processing selection: {e}")
            
        print(f"## Get {len(selections)} selections for {help}.")
        
        return selections, selection_names

    
        




