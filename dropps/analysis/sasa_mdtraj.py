# SASA tool in DROPPS package by Yiming Tang @ Fudan
# Development started on Feb 3 2026

from argparse import ArgumentParser
from dropps.share.trajectory import trajectory_class
import mdtraj
from dropps.fileio.pdb_reader import write_pdbData

prog = "sasa"
desc = '''This program calculate solvent accesible surface area (SASA).'''

def getargs_sasa(argv):

    parser = ArgumentParser(prog=prog, description=desc)

    parser.add_argument('-s', '--run-input', type=str, required=True, 
                        help="TPR file containing all information for a simulation run.")

    parser.add_argument('-f', '--input', type=str, required=True, 
                    help="XTC file which is taken as input trajectory.")
    
    parser.add_argument('-b', '--start-time', type=int,
                        help="Time (ns) of the first frame to calculate.")

    parser.add_argument('-e', '--end-time', type=int,
                        help="Time (ns) of the last frame to calculate.")

    parser.add_argument('-dt', '--delta-time', type=int,
                        help="Intervals (ns) between two calculated frames.")
    
    parser.add_argument('-o', '--output', type=str, required=False, 
                        help="XVG file to write verbose SASA as a function of time.")
    
    args = parser.parse_args(argv)

    return args


def sasa(args):
       
    # load trajectory into memory

    try:
        trajectory = trajectory_class(args.run_input, None, args.input)
    except:
        print("## An exception occurred when trying to open trajectory file %s." % args.input)
        quit()
    
    start_frame, end_frame, interval_frame = trajectory.time2frame(args.start_time, args.end_time, args.delta_time)

    
    temp_pdb = "temp_sasa.pdb"
    write_pdbData(temp_pdb, trajectory.tpr.pdb_raw)

    print(f"## Temporary PDB file for MDtraj SASA calculation written to {temp_pdb}")

    mdtraj_universe = mdtraj.load(args.input, top=temp_pdb)
    print(f"## Trajectory loaded into MDtraj universe with {mdtraj_universe.n_frames} frames and {mdtraj_universe.n_atoms} atoms.")

    print(f"## Starting SASA calculation from frame {start_frame} to frame {end_frame} with interval {interval_frame}.")
    
    sasa_analysis = mdtraj.shrake_rupley(mdtraj_universe[start_frame:end_frame:interval_frame])
    print(sasa_analysis.shape)
    

from dropps.share.command_class import single_command
sasa_commands = single_command(prog, getargs_sasa, sasa, desc)