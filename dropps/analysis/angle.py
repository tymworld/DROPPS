# angle calculation tool in DROPPS package by Yiming Tang @ Fudan
# Development started on July 20 2025

from argparse import ArgumentParser
from dropps.share.trajectory import trajectory_class
from dropps.fileio.filename_control import validate_extension
from dropps.fileio.xvg_reader import write_xvg

from tqdm import tqdm
import numpy as np

import warnings
from Bio import BiopythonDeprecationWarning

warnings.filterwarnings("ignore", category=BiopythonDeprecationWarning)

from MDAnalysis.transformations import unwrap
prog = "angle"
desc = '''This program calculate the profile of angles along amino acid sequence of a protein.'''

def getargs_angle(argv):

    parser = ArgumentParser(prog=prog, description=desc)

    parser.add_argument('-s', '--run-input', type=str, required=True, 
                        help="TPR file containing all information for a simulation run.")
    
    parser.add_argument('-f', '--input', type=str, required=True, 
                    help="XTC file which is taken as input trajectory.")
    
    parser.add_argument('-n', '--index', type=str, required=False,
                        help="Index file containing non-default groups.")
    
    parser.add_argument('-sel', '--selection', type=int,
                        help="Reference group containing vertex atoms. Chains will be spliited and terminals will be ignored.")
    
    parser.add_argument('-b', '--start-time', type=int,
                        help="Time (ns) of the first frame to calculate.")

    parser.add_argument('-e', '--end-time', type=int,
                        help="Time (ns) of the last frame to calculate.")

    parser.add_argument('-dt', '--delta-time', type=int,
                        help="Intervals (ns) between two calculated frames.")
    
    parser.add_argument('-pbc', '--treat-pbc', action='store_true', default=False,
                        help="Treat broken molecule at periodic boundaries.")

    parser.add_argument('-ot', '--output-time', type=str,
                        help="Output of all angles (chain averaged) as a function of time.")
    
    parser.add_argument('-or', '--output-residue', type=str,
                        help="Output of all angles (time averaged) as a function of residue index.")
    
    parser.add_argument('-ors', '--output-residue-statistic', type=str,
                        help="Statistic of all angles as a function of residue index (mean and std).")
    
    parser.add_argument('-ov', '--output-verbose', type=str,
                        help="Verbose output of all angles.")
    
    args = parser.parse_args(argv)
    return args
    
def angle(args):

    # load trajectory into memory

    try:
        trajectory = trajectory_class(args.run_input, args.index, args.input)
    except:
        print("## An exception occurred when trying to open trajectory file %s." % args.input)
        quit()
    
    if args.treat_pbc is True:
        print(f"## Distance will be calculated using periodic boundary conditions.")
        trajectory.Universe.trajectory.add_transformations(unwrap(trajectory.Universe.atoms))
    else:
        print(f"## WARNING: Distance will be calculated without periodic boundary conditions.")

    # We treat time for analysis and generate frame for analysis

    start_frame, end_frame, interval_frame = trajectory.time2frame(args.start_time, args.end_time, args.delta_time)

    frame_list = list(range(start_frame, end_frame, interval_frame))

    # We now generate atom groups for calculations

    if args.selection is not None:
        print(f"## Will use group {args.selection} as selection group.")
        selection, selection_name = trajectory.getSelection(f"group {args.selection}")
    else:
        trajectory.index.print_all()
        selection, selection_name = trajectory.getSelection_interactive("selection group")

    # We now split the indices

    selection_chains = trajectory.index.splitch_indices(selection.indices)

    # We check chains
    chain_length_list = [len(chain) for chain in selection_chains]
    
    if len(set(chain_length_list)) != 1:
        print(f"ERROR: Chains in selection groups are not same in length.")
        quit()
    
    print(f"## Processed {len(selection_chains)} chains of length {len(selection_chains[0])} in selection group.")

    # We now validate selections and get chain groups
    chains_without_terminal = [[atom for atom in chain if not trajectory.is_terminal(atom)] for chain in selection_chains]
    resIDs = [[trajectory.get_resID(atom) for atom in chain] for chain in chains_without_terminal]
    all_resID_identical = all(resIDs[0] == lst for lst in resIDs[1:])

    if not all_resID_identical:
        print(f"ERROR: The chains within your selection {selection_name} don't possess same residue IDs.")
        quit()

    print(f"## Removing terminal residues, we will process {len(chains_without_terminal)} chains each with {len(chains_without_terminal[0])} angles.")
    print(f"## The residue indices of the center atoms are {','.join([f"{i}" for i in resIDs[0]])}")

    angle_number_per_chain = len(chains_without_terminal[0])
    chain_number = len(chains_without_terminal)

    angle_centers = [item for sublist in chains_without_terminal for item in sublist]
    angle_triplets = np.array([[item - 1, item, item + 1] for item in angle_centers])

    print(f"## Start calculating the time evolution of {len(angle_triplets)} angles.")

    angles = np.zeros((len(frame_list), len(angle_triplets)))

    time_list = list()

    for ts_index, ts in tqdm(enumerate(trajectory.Universe.trajectory[frame_list])):

        time_list.append(trajectory.Universe.trajectory.time / 1000)
        pos1 = trajectory.Universe.atoms[angle_triplets[:, 0]].positions
        pos2 = trajectory.Universe.atoms[angle_triplets[:, 1]].positions
        pos3 = trajectory.Universe.atoms[angle_triplets[:, 2]].positions

        # Compute vectors
        v1 = pos1 - pos2
        v2 = pos3 - pos2

        # Normalize
        v1_norm = v1 / np.linalg.norm(v1, axis=1)[:, np.newaxis]
        v2_norm = v2 / np.linalg.norm(v2, axis=1)[:, np.newaxis]

        # Dot product and angle
        cos_theta = np.sum(v1_norm * v2_norm, axis=1)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)  # avoid numerical issues
        theta = np.arccos(cos_theta) * 180.0 / np.pi  # convert to degrees

        angles[ts_index, :] = theta
    
    print(f"## Calculation ended.")
    print(f"## We will perform statistics and write to files.")

    if args.output_verbose is not None:
        verbose_filename = validate_extension(args.output_verbose, "xvg")
        legends = [f"Chain {trajectory.get_chainID(chain[0])} Residue {resid}" for chain in chains_without_terminal for resid in resIDs[0]]
        write_xvg(verbose_filename, time_list, angles.transpose(), 
                  title="Angles", xlabel="Time (ns)", ylabel="Angle (degree)",
                  legends=legends)
        print(f"## Calculated and written verbose angle profiles to {verbose_filename}.")
    
    angles_reshaped = angles.reshape(angles.shape[0], chain_number, angle_number_per_chain)
    
    if args.output_time is not None:
        time_filename = validate_extension(args.output_time, "xvg")
        legends = [f"Residue {resid}" for resid in resIDs[0]]
        write_xvg(time_filename, time_list, np.mean(angles_reshaped, axis=1).transpose(),
                  title="Averaged angles", xlabel="Time (ns)", ylabel="Angle (degree)",
                  legends=legends)
        print(f"## Calculated and written averaged angle for each time point and each residue to {time_filename}.")
    
    if args.output_residue is not None:
        residue_filename = validate_extension(args.output_residue, "xvg")
        legends = [f"Chain {trajectory.get_chainID(chain[0])}" for chain in chains_without_terminal]
        write_xvg(residue_filename, resIDs[0], np.mean(angles_reshaped, axis=0),
                  title="Averaged angles", xlabel="Residue index", ylabel="Angle (degree)",
                  legends=legends)
        print(f"## Calculated and written averaged angle for each frame and each chain to {residue_filename}.")
    
    if args.output_residue_statistic is not None:
        statistic_filename = validate_extension(args.output_residue_statistic, "xvg")
        angles_per_residue = angles_reshaped.transpose(2, 0, 1).reshape(angle_number_per_chain, -1)

        mean = angles_per_residue.mean(axis=1)
        std = angles_per_residue.std(axis=1)
        mean_std = np.array([mean, std])
        legends = ["Mean", "Standard error"]
        write_xvg(statistic_filename, resIDs[0], mean_std,
            title="Mean and std for angles", xlabel="Residue index", ylabel="Angle (degree)",
            legends=legends)
        print(f"## Calculated and written statistic for each angle to {statistic_filename}.")
        
    print(f"## All file written.")


from dropps.share.command_class import single_command
angle_commands = single_command("angle", getargs_angle, angle, desc)




