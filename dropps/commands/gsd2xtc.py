# gsd2xtc tool in DROPPS package by Yiming Tang @ Fudan
# Development started on Nov 17 2025

from argparse import ArgumentParser

import MDAnalysis as mda
from MDAnalysis.coordinates import XTC
import numpy as np

from dropps.fileio.filename_control import validate_extension

from tqdm import tqdm

prog = "gsd2xtc"
desc = '''This program read and convert gsd file to xtc file.'''

def getargs_gsd2xtc(argv):

    # Command line argument parser

    parser = ArgumentParser(prog=prog, description=desc)
    
    parser.add_argument('-f', '--input', type=str, required=True, 
                    help="GSD file which is taken as input trajectory.")
        
    parser.add_argument('-o', '--output', type=str,
                        help="Output file path for trajectory. Allowed filetype: xtc")
    
    parser.add_argument('-ts', '--time-step', type=float, default = 0.1,
                        help="Time step of the gsd file in ns, default is 0.1 ns")
    
    args = parser.parse_args(argv)

    return args

def gsd2xtc(args):

    u = mda.Universe(args.input)

    if np.all(u.dimensions == 0):
        print("Warning: No PBC detected. Setting default PBC dimensions.")
        # You can define your box dimensions if necessary, e.g., cubic box
        u.dimensions = np.array([u.dimensions[0], u.dimensions[0], u.dimensions[0], 90.0, 90.0, 90.0])  # Box in nm

    output_filename = validate_extension(args.output, "xtc")

    factor = 10.0
    time_step = args.time_step * 1000

    print(f"## Will treat {len(u.trajectory)} frames in {args.input} with {u.trajectory.n_atoms} atoms.")

    with XTC.XTCWriter(output_filename, u.trajectory.n_atoms) as xtc_writer:

        for i, ts in tqdm(enumerate(u.trajectory)):
            # Multiply the coordinates by the factor
            ts.positions *= factor
            
            # Multiply the box vectors by the factor
            ts.dimensions = [ts.dimensions[0]*10.0, ts.dimensions[1]*10.0, ts.dimensions[2]*10.0,
                             ts.dimensions[3], ts.dimensions[4], ts.dimensions[5]]
            
            # Set the time for the frame
            ts.time = time_step + i * time_step  # Start at time_step and increment by time_step
            
            # Write the modified frame to the XTC file
            xtc_writer.write(u)
        print(f"## XTC file written to {output_filename}")

from dropps.share.command_class import single_command
gsd2xtc_commands = single_command("gsd2xtc", getargs_gsd2xtc, gsd2xtc, desc)