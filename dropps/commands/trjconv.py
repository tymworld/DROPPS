# trjconv tool in DROPPS package by Yiming Tang @ Fudan
# Development started on Nov 17 2025

from argparse import ArgumentParser
from dropps.share.trajectory import trajectory_class

prog = "trjconv"
desc = '''This program read and convert trajectory files in many ways.'''

def getargs_trjconv(argv):

    # Command line argument parser

    parser = ArgumentParser(prog=prog, description=desc)

    parser.add_argument('-s', '--run-input', type=str, required=True, 
                        help="TPR file containing all information for a simulation run.")
    
    parser.add_argument('-f', '--input', type=str, required=True, 
                    help="XTC file which is taken as input trajectory.")
    
    parser.add_argument('-n', '--index', type=str, required=False,
                        help="Index file containing non-default groups.")

    args = parser.parse_args(argv)

    return args

def trjconv(args):

    # load trajectory into memory

    try:
        trajectory = trajectory_class(args.run_input, args.index, args.input)
    except:
        print("## An exception occurred when trying to open trajectory file %s." % args.input)
        quit()

    print(trajectory.Universe.trajectory[0][0])
    print(trajectory.Universe.trajectory[0][1])



from dropps.share.command_class import single_command
trjconv_commands = single_command("trjconv", getargs_trjconv, trjconv, desc)