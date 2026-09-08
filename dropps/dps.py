#!/usr/bin/env python3

from __future__ import annotations

import sys
import os

from dropps import __version__


def _print_unknown_command(error, stream):
    print(f"dps: error: unknown command '{error.name}'", file=stream)
    if error.suggestions:
        if len(error.suggestions) == 1:
            print(f"Did you mean '{error.suggestions[0]}'?", file=stream)
        else:
            suggestions = ", ".join(f"'{name}'" for name in error.suggestions)
            print(f"Did you mean one of: {suggestions}?", file=stream)
    print("Run 'dps help commands' to list available commands.", file=stream)


def _print_banner(argv, arguments):
    from dropps.hp.hello_world import print_hello

    entry = argv[0] if argv else "dps"
    print_hello(entry, " ".join(arguments))


def main(argv=None):
    argv = list(sys.argv if argv is None else argv)
    arguments = argv[1:]

    if arguments and arguments[0] == "--version":
        print(f"dps {__version__}")
        return 0

    from dropps.share.all_commands import all_commands
    from dropps.share.command_class import CommandUsageError, UnknownCommandError

    if not arguments or arguments[0] in {"-h", "--help"}:
        _print_banner(argv, arguments)
        all_commands.command("help")(["commands"])
        return 0

    command_name = arguments[0]
    try:
        getargs_command = all_commands.getargs(command_name)
        real_command = all_commands.command(command_name)
    except UnknownCommandError as error:
        _print_unknown_command(error, sys.stderr)
        return 2

    help_requested = any(argument in {"-h", "--help"} for argument in arguments[1:])
    if help_requested:
        _print_banner(argv, arguments)

    try:
        parsed_args = getargs_command(arguments[1:])
    except UnknownCommandError as error:
        _print_unknown_command(error, sys.stderr)
        return 2
    except CommandUsageError as error:
        print(f"dps {command_name}: error: {error}", file=sys.stderr)
        print(f"Run 'dps {command_name} -h' for usage.", file=sys.stderr)
        return 2
    except SystemExit as error:
        if error.code is None:
            return 1
        raise

    if not help_requested:
        _print_banner(argv, arguments)

    try:
        result = real_command(parsed_args)
    except KeyboardInterrupt:
        print(f"dps {command_name}: interrupted", file=sys.stderr)
        return 130
    except (OSError, RuntimeError, ValueError) as error:
        if os.environ.get("DROPPS_DEBUG"):
            raise
        print(f"dps {command_name}: error: {error}", file=sys.stderr)
        print(
            "Set DROPPS_DEBUG=1 to show a traceback for debugging.",
            file=sys.stderr,
        )
        return 1
    except SystemExit as error:
        # Legacy command implementations use bare quit() for fatal errors.
        # Python maps SystemExit(None) to shell status 0, so normalize only
        # that ambiguous case while preserving intentional help/error codes.
        if error.code is None:
            return 1
        raise
    return 0 if result is None else result


if __name__ == "__main__":
    raise SystemExit(main())
