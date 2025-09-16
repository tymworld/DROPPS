# extract tool in DROPPS package by Yiming Tang @ Fudan
# Development started on Sep 2 2025

from argparse import ArgumentParser
from dropps.share.trajectory import trajectory_class
from dropps.fileio.pdb_reader import write_pdb
from dropps.fileio.filename_control import validate_extension

from MDAnalysis.transformations import unwrap
prog = "extract"
desc = '''This program extract a single frame from a HPS trajectory.'''


def getargs_extract(argv):

    parser = ArgumentParser(prog=prog, description=desc)

    parser.add_argument('-s', '--run-input', type=str, required=True, 
                        help="TPR file containing all information for a simulation run.")

    parser.add_argument('-f', '--input', type=str, required=True, 
                    help="XTC file which is taken as input trajectory.")
    
    parser.add_argument('-o', '--output', type=str, required=True, 
                    help="PDB file which to write the single frame.")

    parser.add_argument('-n', '--index', type=str, required=False,
                        help="Index file containing non-default groups.")
    
    parser.add_argument('-sel', '--selection', type=int, nargs='+',
                        help="Groups of atoms to output")
    
    parser.add_argument('-t', '--time', type=int,
                        help="Time (ns) of the frame to extract.")
    
    parser.add_argument('-fm', '--frame', type=int,
                        help="Frame index to extract.")

    args = parser.parse_args(argv)

    return args

def extract(args):

    # load trajectory into memory

    try:
        trajectory = trajectory_class(args.run_input, args.index, args.input)
    except:
        print("## An exception occurred when trying to open trajectory file %s." % args.input)
        quit()
    
    if args.time is None and args.frame is None:
        print("ERROR: No frame is specified.")
        quit()
    
    if args.time is not None and args.frame is not None:
        print(f"ERROR: Only one of time and frame can be specified.")
        quit()
    
    if args.frame is not None:
        start_frame = args.frame
    else:
        start_frame, end_frame, interval_frame = trajectory.time2frame(args.time, args.time, 0)

    ts = trajectory.Universe.trajectory[start_frame]
    coordinates = trajectory.Universe.atoms.positions
    box_size = trajectory.Universe.dimensions[0:3] / 10.0

    x = coordinates[:,0]/ 10.0
    y = coordinates[:,1]/ 10.0
    z = coordinates[:,2]/ 10.0


    pdb_raw = trajectory.tpr.pdb_raw

    output_name = validate_extension(args.output, "pdb")

    write_pdb(output_name, box_size, pdb_raw.record_names, pdb_raw.serial_numbers,
              pdb_raw.atom_names, pdb_raw.residue_names, pdb_raw.chain_IDs,
              pdb_raw.residue_sequence_numbers, x, y, z, 
              pdb_raw.occupancys, pdb_raw.bfactors, pdb_raw.elements,
              pdb_raw.molecule_length_list)


from dropps.share.command_class import single_command
extract_commands = single_command("extract", getargs_extract, extract, desc)
