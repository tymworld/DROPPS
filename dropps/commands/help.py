"""Top-level command catalog and detailed-help dispatcher."""

from __future__ import annotations

from textwrap import wrap

from dropps.share.command_class import (
    CommandUsageError,
    UnknownCommandError,
    single_command,
)
from dropps.share.command_help import COMMAND_CHOICES, START_HERE, TASK_GROUPS


_HELP_TEXT = """usage: dps help [COMMAND]

Show the task-oriented command catalog or detailed help for one command.

Arguments:
  COMMAND               Command to explain. Omit it, or use "commands", to
                        show the complete command catalog.

Examples:
  dps help
  dps help commands
  dps help mdrun
  dps mdrun -h
"""


def _print_rows(rows, label_width=None, text_width=96):
    rows = tuple(rows)
    if not rows:
        return
    if label_width is None:
        label_width = max(len(label) for label, _ in rows)

    content_width = max(30, text_width - label_width - 4)
    for label, description in rows:
        lines = wrap(description, width=content_width) or [""]
        print(f"  {label:<{label_width}}  {lines[0]}")
        continuation = " " * (label_width + 4)
        for line in lines[1:]:
            print(f"{continuation}{line}")


def getargs_help(argv):
    """Parse the lightweight ``dps help`` interface."""

    from dropps.share.all_commands import all_commands

    if any(argument in {"-h", "--help"} for argument in argv):
        print(_HELP_TEXT, end="")
        raise SystemExit(0)

    if len(argv) > 1:
        raise CommandUsageError("help accepts at most one COMMAND")

    if not argv:
        return ["commands"]

    command_name = argv[0]
    if command_name == "commands" or all_commands.exist(command_name):
        return [command_name]

    candidates = ["commands", *all_commands.command_names]
    raise UnknownCommandError(
        command_name,
        all_commands.suggestions(command_name, candidates),
    )


def _print_catalog():
    from dropps.share.all_commands import all_commands

    print("DROPPS command-line tools")
    print()
    print("Usage:")
    print("  dps <command> [options]")
    print("  dps help [command]")
    print("  dps <command> --help")
    print()
    print("Start here:")
    _print_rows(START_HERE)
    print()
    print("Commands by task:")

    for title, purpose, command_names in TASK_GROUPS:
        print()
        print(f"{title}")
        print(f"  {purpose}")
        rows = (
            (command_name, all_commands.desc(command_name))
            for command_name in command_names
        )
        _print_rows(rows, label_width=15)

    print()
    print("Choosing between similar commands:")
    _print_rows(COMMAND_CHOICES)
    print()
    print("Detailed help:")
    print("  dps help COMMAND")
    print("  dps COMMAND -h")


def help(args):
    """Render the catalog or delegate to a command's parser help."""

    from dropps.share.all_commands import all_commands

    command_name = args[0]
    if command_name == "commands":
        _print_catalog()
        return

    help_command = all_commands.getargs(command_name)
    help_command(["-h"])


help_commands = single_command(
    "help",
    getargs_help,
    help,
    "Show the command catalog or detailed help for one command.",
)
