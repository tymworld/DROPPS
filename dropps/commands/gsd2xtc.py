# gsd2xtc tool in DROPPS package by Yiming Tang @ Fudan
# Development started on Nov 17 2025

from dropps.share.argument_parser import ArgumentParser

import MDAnalysis as mda
from MDAnalysis.coordinates import XTC
import numpy as np

from dropps.fileio.filename_control import validate_extension

from tqdm import tqdm

prog = "gsd2xtc"
desc = "Convert a GSD trajectory to XTC format."

def getargs_gsd2xtc(argv):

    # Command line argument parser

    parser = ArgumentParser(prog=prog, description=desc)
    
    parser.add_argument('-f', '--input', type=str, required=True, 
                    help="Input trajectory file (.gsd).")
        
    parser.add_argument('-o', '--output', type=str, required=True,
                        help="Output trajectory file (.xtc); the extension is added if omitted.")
    
    parser.add_argument('-ts', '--time-step', type=float, default = 0.1,
                        help="Time between consecutive GSD frames, in ns.")

    parser.add_argument('-box', '--box', type=float, nargs=3,
                        metavar=('X', 'Y', 'Z'),
                        help="Box lengths in nm; required when the GSD file has no periodic box.")
    
    args = parser.parse_args(argv)

    return args

def gsd2xtc(args):

    u = mda.Universe(args.input)

    box_override = None
    if args.box is not None:
        if any(length <= 0 for length in args.box):
            raise ValueError("All box lengths must be positive.")
        box_override = np.array([*args.box, 90.0, 90.0, 90.0])
    elif np.all(u.dimensions[:3] == 0):
        raise ValueError(
            "The GSD trajectory has no periodic box. Supply box lengths with --box X Y Z."
        )

    output_filename = validate_extension(args.output, "xtc")

    factor = 10.0
    time_step = args.time_step * 1000

    print(f"## Will treat {len(u.trajectory)} frames in {args.input} with {u.trajectory.n_atoms} atoms.")

    with XTC.XTCWriter(output_filename, u.trajectory.n_atoms) as xtc_writer:

        for i, ts in tqdm(enumerate(u.trajectory)):
            # Multiply the coordinates by the factor
            ts.positions *= factor

            if box_override is not None:
                ts.dimensions = box_override
            elif np.all(ts.dimensions[:3] == 0):
                raise ValueError(
                    f"GSD frame {i} has no periodic box. Supply box lengths with --box X Y Z."
                )
            
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
