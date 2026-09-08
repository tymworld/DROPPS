# check tool in DROPPS package by Yiming Tang @ Fudan
# Development started on July 17 2025

from dropps.share.argument_parser import ArgumentParser
from dropps.share.command_class import single_command
from dropps.share.trajectory import trajectory_class

prog = "check"
desc = "Inspect DROPPS run, trajectory, and index files before analysis."


def getargs_check(argv):
    parser = ArgumentParser(prog=prog, description=desc)

    parser.add_argument(
        "-s",
        "--run-input",
        type=str,
        required=True,
        help="Input DROPPS run file (.tpr) containing the system and simulation settings.",
    )
    parser.add_argument(
        "-f",
        "--trajectory",
        type=str,
        required=False,
        help="Optional input trajectory file (.xtc).",
    )
    parser.add_argument(
        "-n",
        "--index",
        type=str,
        required=False,
        help="Optional index file (.ndx) defining additional atom groups.",
    )

    args = parser.parse_args(argv)
    return args


def check(args):
    trajectory = trajectory_class(args.run_input, args.index, args.trajectory)
    print("\n#################### CHECKING ####################")

    print(
        f"## The system contains {trajectory.num_atoms()} atoms in {trajectory.num_chains()} chains."
    )
    print(f"## The system contains {trajectory.num_bonds()} bonds.")

    print(f"## The trajectory contains {trajectory.num_frames()} frames.")
    print(
        "## The time of the first, last frame is "
        + f"{trajectory.Universe.trajectory[0].time / 1000}, {trajectory.Universe.trajectory[-1].time / 1000} ns."
    )

    print("##################################################")
    print(
        f"## The initial system now contains {len(trajectory.index.index_groups)} index groups."
    )
    print("## Trying to select the first index group...")
    trajectory.getSelection("group 0")

    print("## Ready for data analysis.")
    print("#################### CHECKED  ####################")

    trajectory.index.print_all()


check_commands = single_command("check", getargs_check, check, desc)
