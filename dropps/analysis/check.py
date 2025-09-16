# check tool in DROPPS package by Yiming Tang @ Fudan
# Development started on July 17 2025

from argparse import ArgumentParser
from dropps.share.trajectory import trajectory_class
from dropps.fileio.tpr_reader import read_tpr

prog = "check"
desc = '''This program check tpr, xtc, and ndx files for data analysis.'''

def getargs_check(argv):

    parser = ArgumentParser(prog=prog, description=desc)

    parser.add_argument('-s', '--run-input', type=str, required=True, 
                        help="TPR file containing all information for a simulation run.")
    parser.add_argument('-f', '--trajectory', type=str, required=False, 
                        help="XTC file containing trajectory for a simulation run.")
    parser.add_argument('-n', '--index', type=str, required=False,
                        help="Load existing index file from this file.")

    args = parser.parse_args(argv)
    return args

def check(args):

    trajectory = trajectory_class(args.run_input, args.index, args.trajectory)
    print(f"\n#################### CHECKING ####################")

    print(f"## The system contains {trajectory.num_atoms()} atoms in {trajectory.num_chains()} chains.")
    print(f"## The system contains {trajectory.num_bonds()} bonds.")

    print(f"## The trajectory contains {trajectory.num_frames()} frames.")
    print(f"## The time of the first, last frame is "
          + f"{trajectory.Universe.trajectory[0].time / 1000}, {trajectory.Universe.trajectory[-1].time/ 1000} ns.")
    
    print(f"##################################################")
    print(f"## The initial system now contains {len(trajectory.index.index_groups)} index groups.")
    print(f"## Trying to select the first index group...")
    trajectory.getSelection("group 0")

    print(f"## Ready for data analysis.")
    print(f"#################### CHECKED  ####################")

    trajectory.index.print_all()
    while(True):
        trajectory.getSelection_interactive()
    
    print("hello")

from dropps.share.command_class import single_command
check_commands = single_command("check", getargs_check, check, desc)

