# make_ndx tool in DROPPS package by Yiming Tang @ Fudan
# Development started on July 15 2025

from dropps.share.argument_parser import ArgumentParser
from dropps.share.command_class import single_command
from dropps.share.indexing import indexGroups
from dropps.fileio.tpr_reader import read_tpr

prog = "make_ndx"
desc = "Create or extend an atom-group index file interactively."


def getargs_make_ndx(argv):
    parser = ArgumentParser(prog=prog, description=desc)

    parser.add_argument(
        "-s",
        "--run-input",
        type=str,
        required=True,
        help="Input DROPPS run file (.tpr) containing the system topology.",
    )
    parser.add_argument(
        "-n",
        "--index",
        type=str,
        required=False,
        help="Optional existing index file (.ndx) to extend.",
    )
    parser.add_argument(
        "-o", "--output", type=str, required=True, help="Output index file (.ndx)."
    )

    args = parser.parse_args(argv)
    return args


def make_ndx(args):
    try:
        tpr = read_tpr(args.run_input)
    except Exception as exc:
        print(f"ERROR: Cannot open tpr file {args.run_input}.")
        print(f"ERROR: Root cause: {exc}")
        quit()

    print(f"## Loading topology from {args.run_input}.")

    index = indexGroups(tpr.mdtopology, tpr.itp_list)

    try:
        if args.index is not None:
            index.load_ndx(args.index)
    except Exception as exc:
        print(f"ERROR: Cannot open index file {args.index}.")
        print(f"ERROR: Root cause: {exc}")
        quit()

    print(f"Loading index entries from file {args.index}.")

    index.command("p")

    while True:
        index.showhelp()
        user_input = input("\n> ")

        if user_input == "q":
            index.write_ndx(args.output)
            quit()
        elif user_input == "":
            index.command("p")
        else:
            index.command(user_input)


make_ndx_commands = single_command("make_ndx", getargs_make_ndx, make_ndx, desc)
