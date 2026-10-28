#!/usr/bin/env python3
"""Generate the complete command reference from the actual public CLI parsers."""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def render_documents():
    from dropps import __version__
    from dropps.share.all_commands import all_commands
    from dropps.share.argument_parser import ArgumentParser, _public_options
    from dropps.share.command_docs import COMMAND_NOTES
    from dropps.share.command_help import COMMAND_EXAMPLES, TASK_GROUPS

    names = all_commands.names()
    if set(COMMAND_NOTES) != set(names):
        raise RuntimeError("Command notes must cover every public command exactly once.")
    if set(COMMAND_EXAMPLES) != set(names) - {"help"}:
        raise RuntimeError("Examples must cover every functional command exactly once.")

    result = {}
    option_count = 0
    for name in names:
        captured = []
        original = ArgumentParser.parse_args

        def capture(parser, *args, **kwargs):
            captured.append(parser)
            return original(parser, *args, **kwargs)

        output = io.StringIO()
        with patch.object(ArgumentParser, "parse_args", capture):
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                try:
                    all_commands.getargs(name)(["--help"])
                except SystemExit as exc:
                    if exc.code != 0:
                        raise RuntimeError(f"Help failed for {name}") from exc
                else:
                    raise RuntimeError(f"Help did not exit for {name}")
        options = []
        for parser in captured:
            for action in parser._actions:
                if action.help == argparse.SUPPRESS:
                    continue
                aliases = _public_options(action)
                if aliases:
                    options.append(aliases)
        option_count += len(options)
        page = [
            f"# dps {name}", "",
            "[All commands](../command-reference.md) · [User guide](../user-guide.md)", "",
            all_commands.desc(name), "",
            f"DROPPS {__version__}. This page is generated from the executable CLI; "
            "defaults, choices, required inputs, and output options below match the installed command.", "",
            "## Usage and options", "", "```text", output.getvalue().rstrip(), "```", "",
        ]
        result[Path("docs/commands") / f"{name}.md"] = "\n".join(page)

    index = [
        "# Complete DROPPS command reference", "",
        f"DROPPS {__version__} exposes **{len(names)} public commands**. "
        f"The reference below includes every command and **{option_count} option entries** "
        "(including help/debug on each applicable command).", "",
        "Start with the [end-to-end tutorial](user-guide.md), "
        "[MDP settings](mdp-reference.md), or [Colab workflows](../examples/colab/README.md).", "",
        "## Conventions", "",
        "- Run `dps help COMMAND` or `dps COMMAND --help` for the same reference in the terminal.",
        "- Prefer long options in scripts. An omitted selection may prompt interactively; supply groups explicitly for batch jobs.",
        "- DROPPS group IDs, NDX atom indices, and selection `index`/`chainid` values start at **0**. PDB/ITP bead IDs and angle/elastic-file bead IDs start at **1**. `resid` follows the topology's residue numbers.",
        "- Distances normally use nm, energies kJ/mol, and temperatures K. Analysis times normally use ns; commands with `--time-unit` document alternatives. PDB coordinate fields use Angstrom and XTC timestamps use ps internally.",
        "- `.tpr` is DROPPS run input and `.edr` is DROPPS CSV energy data, not the binary GROMACS formats with the same extensions.",
        "- Output options choose which results are written. Read the per-command output descriptions; several analysis commands require at least one output even when individual flags are optional.",
        "- `--debug` shows runtime context and tracebacks. Normal output is plain text when redirected; use `NO_COLOR` to disable terminal colors.", "",
        "## Catalog", "",
        "- [help](commands/help.md): command catalog and detailed help.", "",
    ]
    for title, purpose, commands in TASK_GROUPS:
        index += [f"### {title}", "", purpose, "", "| Command | Purpose |", "| --- | --- |"]
        index += [f"| [{name}](commands/{name}.md) | {all_commands.desc(name)} |" for name in commands]
        index.append("")
    index += [
        "## Compatibility commands", "",
        "`intra-distance` / `idist` and `inter-distance` / `odist` remain accepted with their legacy parameters. "
        "Use `dps ALIAS --help` for those interfaces; new scripts should use [distance](commands/distance.md) "
        "with `--scope intra-chain` or `--scope inter-chain`.", "",
        "The former `extract` workflow is handled by [trjconv](commands/trjconv.md). "
        "The old `pdb2cgps` command is named [pdb2dps](commands/pdb2dps.md).", "",
        "## Keeping this reference current", "",
        "Run `python scripts/update_command_docs.py` after a CLI change and "
        "`python scripts/update_command_docs.py --check` in validation. "
        "The check covers the registry, practical notes, examples, and rendered help for every command.", "",
    ]
    result[Path("docs/command-reference.md")] = "\n".join(index)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    documents = render_documents()
    stale = []
    for relative, text in documents.items():
        path = ROOT / relative
        if args.check:
            if not path.is_file() or path.read_text(encoding="utf-8") != text:
                stale.append(str(relative))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
    if stale:
        print("Command documentation is stale: " + ", ".join(stale))
        return 1
    print(f"{'Checked' if args.check else 'Updated'} {len(documents)} command-reference files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
