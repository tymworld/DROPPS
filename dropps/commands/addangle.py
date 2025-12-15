# addangle tool in CGPS.ng package by Yiming Tang @ Fudan
# Development started on June 11 2025

from argparse import ArgumentParser
from copy import deepcopy
from dropps.fileio.itp_reader import read_itp, write_itp, Angle
from openmm.unit import degree, kilojoule_per_mole, radian

def addangle(args):

    output_itp_prefix = args.output_topology[0:-4] if ".itp" in args.output_topology else args.output_topology
    output_itp_name = output_itp_prefix + ".itp"

    input_itp_name = args.input_topology
    if input_itp_name[-4:] != ".itp":
        print(f"ERROR: Input file {input_itp_name} is not an itp file.")
        quit()

    # We first read the itp file

    topology = read_itp(input_itp_name)
    print("## Input topology has been readed and loaded to memory.")

    # We now processed the angle list
    try:
        angle_lines = [line.strip().split() for line in open(args.angle_list, 'r') if len(line) > 1]
    except:
        print("## An exception occurred when trying to open angle file %s." % args.angle_list)
        quit()
    
    residue_NTD = topology.atoms[0].residueid
    residue_CTD = topology.atoms[-1].residueid

    angle_list = [[int(line[0]), float(line[1]), float(line[2])] for line in angle_lines
                if int(line[0]) not in [residue_NTD, residue_CTD]]
    

    if residue_NTD in [int(angle[0]) for angle in angle_lines]:
        print(f"## WARNING: Angle centering at 1 will be ignored for it being a terminal.")

    if residue_CTD in [int(angle[0]) for angle in angle_lines]:
        print(f"## WARNING: Angle centering at {len(topology.atoms)} will be ignored for it being a terminal.")

    print(f"## Frame contains {len(topology.atoms)} beads and angle list contains {len(angle_list)} entries.")

    if residue_NTD != 1:
        print(f"## IMPORTANT: Residue in itp file starts with {residue_NTD}.")

    # We now generate angles to the topology list

    for a2, theta_in_degree, force in angle_list:

        a1 = a2 - 1
        a3 = a2 + 1

        a1_id = a1 - residue_NTD
        a2_id = a2 - residue_NTD
        a3_id = a3 - residue_NTD

        print(f"## Adding angle to atoms {a1_id + 1}-{a2_id + 1}-{a3_id + 1} (resid: {a1}-{a2}-{a3}) at {theta_in_degree * degree} with {force * kilojoule_per_mole / radian ** 2}.")
        if topology.angles is None:
            topology.angles = list()
        topology.angles.append(Angle(a1_id, a2_id, a3_id, theta_in_degree * degree, force * kilojoule_per_mole / radian ** 2))

    # We now write output topology file
    write_itp(output_itp_name, topology)
    print(f"## Output topology file written to {output_itp_name}.")

prog = "addangle"
desc = '''This program add angle terms for an itp file.'''

def getargs_addangle(argv):

    parser = ArgumentParser(prog=prog, description=desc)

    parser.add_argument('-ip', '--input-topology', type=str, required=True, 
                        help="Input itp file.")

    parser.add_argument('-op', '--output-topology', type=str, required=True, 
                        help="Output itp file with secondary structure constrains added.")

    parser.add_argument('-al', '--angle-list', type=str, required=True,
                        help="ASCII File containing angle information. Format: \"ID-start-with-1 theta k\"")

    args = parser.parse_args(argv)

    return args

from dropps.share.command_class import single_command
addangle_commands = single_command("addangle", getargs_addangle, addangle, desc)

