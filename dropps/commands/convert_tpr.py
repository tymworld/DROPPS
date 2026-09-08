"""Convert a legacy DROPPS pickle TPR into the portable TPR v2 format."""

from __future__ import annotations

import os

from dropps.fileio.tpr_reader import read_tpr, write_tpr
from dropps.share.argument_parser import ArgumentParser
from dropps.share.command_class import single_command


prog = "convert-tpr"
desc = "Convert a trusted legacy DROPPS TPR into the portable TPR v2 format."


def convert_tpr(args):
    source = os.path.abspath(args.run_input)
    destination = os.path.abspath(args.output)
    if source == destination:
        raise ValueError("Input and output TPR paths must be different.")

    content = read_tpr(source)
    manifest = write_tpr(destination, content)
    print(
        f"## Converted TPR format v{content.format_version} to portable "
        f"TPR v{manifest['format_version']}: {destination}."
    )
    print(f"## New TPR run ID: {manifest['run_id']}.")


def getargs_convert_tpr(argv):
    parser = ArgumentParser(prog=prog, description=desc)
    parser.add_argument(
        "-s",
        "--run-input",
        required=True,
        help="Trusted legacy or portable DROPPS TPR to read.",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Portable TPR v2 output file.",
    )
    return parser.parse_args(argv)


convert_tpr_commands = single_command(
    "convert-tpr", getargs_convert_tpr, convert_tpr, desc
)
