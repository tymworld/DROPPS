# mdrun tool in CGPS package by Yiming Tang @ Fudan
# Development started on June 6 2025

from pathlib import Path

from dropps.share.argument_parser import ArgumentParser
from dropps.share.command_class import single_command
from dropps.share.parameters import getparameter

from dropps.share.build_system import build_system

from dropps.fileio.pdb_reader import read_pdb, phrase_pdb_atoms
from dropps.fileio.tpr_reader import write_tpr


def _source_files(args):
    """Collect the primary inputs and ITP files referenced by the TOP file."""

    sources = {
        "structure": args.structure,
        "topology": args.topology,
        "parameters": args.parameter,
    }
    top_path = Path(args.topology).resolve()
    current_section = None
    itp_index = 0
    for raw_line in top_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split(";", 1)[0].split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            current_section = line[1:-1].strip().lower()
            continue
        if current_section != "itp files":
            continue
        itp_path = Path(line.split()[0])
        if not itp_path.is_absolute():
            itp_path = top_path.parent / itp_path
        sources[f"itp-{itp_index:04d}"] = itp_path
        itp_index += 1
    return sources


def grompp(args):
    if len(args.output) < 5 or args.output[-4:] != ".tpr":
        output_tpr = args.output + ".tpr"
    else:
        output_tpr = args.output

    parameters = getparameter(args.parameter)
    mdsystem, mdtopology, positions, ITP_Topology_list = build_system(
        args.structure, args.topology, parameters
    )

    atoms_pdb_raw, box_raw = read_pdb(args.structure)
    PDB_raw = phrase_pdb_atoms(atoms_pdb_raw, box_raw)

    runtime_files = {
        "parameters": parameters,
        "mdsystem": mdsystem,
        "mdtopology": mdtopology,
        "positions": positions,
        "pdb_raw": PDB_raw,
        "ITP_list": ITP_Topology_list,
    }

    manifest = write_tpr(
        output_tpr,
        runtime_files,
        sources=_source_files(args),
    )

    print(
        f"## Wrote portable TPR v{manifest['format_version']} to {output_tpr} "
        f"(run ID {manifest['run_id'][:12]})."
    )


prog = "grompp"
desc = (
    "Build a DROPPS run-input file from structure, topology, and simulation parameters."
)


def getargs_grompp(argv):
    parser = ArgumentParser(prog=prog, description=desc)

    parser.add_argument(
        "-f",
        "--structure",
        type=str,
        required=True,
        help="Input structure file (.pdb) containing the initial coordinates and box.",
    )
    parser.add_argument(
        "-p",
        "--topology",
        type=str,
        required=True,
        help="Input system topology file (.top) referencing the required ITP files.",
    )
    parser.add_argument(
        "-m",
        "--parameter",
        type=str,
        required=True,
        help="Input molecular-dynamics parameter file (.mdp).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Output DROPPS run-input file (.tpr); the extension is added if omitted.",
    )

    args = parser.parse_args(argv)
    return args


grompp_commands = single_command("grompp", getargs_grompp, grompp, desc)
