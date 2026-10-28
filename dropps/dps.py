#!/usr/bin/env python3

from __future__ import annotations

import sys

from dropps import __version__
from dropps.share import console


def _extract_global_options(arguments):
    """Remove global flags that may appear before or after the command name."""

    debug = False
    remaining = []
    passthrough = False
    for argument in arguments:
        if argument == "--":
            passthrough = True
            remaining.append(argument)
        elif not passthrough and argument in {"-db", "--debug"}:
            debug = True
        else:
            remaining.append(argument)
    return remaining, debug


def _print_unknown_command(error, stream):
    console.emit("error", f"Unknown command '{error.name}'.", stream=stream)
    if error.suggestions:
        if len(error.suggestions) == 1:
            console.emit(
                "info",
                f"Did you mean '{error.suggestions[0]}'?",
                stream=stream,
            )
        else:
            suggestions = ", ".join(f"'{name}'" for name in error.suggestions)
            console.emit(
                "info",
                f"Did you mean one of: {suggestions}?",
                stream=stream,
            )
    console.emit(
        "info",
        "Run 'dps help commands' to list available commands.",
        stream=stream,
    )


def _print_banner(argv, arguments):
    from dropps.hp.hello_world import print_hello

    entry = argv[0] if argv else "dps"
    print_hello(entry, " ".join(arguments))


def main(argv=None):
    argv = list(sys.argv if argv is None else argv)
    arguments, debug = _extract_global_options(argv[1:])
    console.configure(debug=debug)
    command_name = arguments[0] if arguments else None
    console.emit_runtime_context([argv[0], *arguments], command_name)

    if arguments and arguments[0] in {"-v", "--version"}:
        print(f"dps {__version__}")
        console.emit_completion(0)
        return 0

    try:
        from dropps.share.all_commands import all_commands
        from dropps.share.command_class import CommandUsageError, UnknownCommandError

        if not arguments or arguments[0] in {"-h", "--help"}:
            _print_banner(argv, arguments)
            all_commands.command("help")(["commands"])
            console.emit_completion(0)
            return 0

        command_name = arguments[0]
        try:
            getargs_command = all_commands.getargs(command_name)
            real_command = all_commands.command(command_name)
        except UnknownCommandError as error:
            _print_unknown_command(error, sys.stderr)
            console.emit_completion(2)
            return 2

        help_requested = any(
            argument in {"-h", "--help"} for argument in arguments[1:]
        )
        if help_requested:
            _print_banner(argv, arguments)

        try:
            parsed_args = getargs_command(arguments[1:])
        except UnknownCommandError as error:
            _print_unknown_command(error, sys.stderr)
            console.emit_completion(2)
            return 2
        except CommandUsageError as error:
            console.emit("error", f"dps {command_name}: {error}", stream=sys.stderr)
            console.emit(
                "info",
                f"Run 'dps {command_name} -h' for usage.",
                stream=sys.stderr,
            )
            console.emit_completion(2)
            return 2
        except SystemExit as error:
            if error.code is None:
                console.emit_completion(1)
                return 1
            raise

        if hasattr(parsed_args, "__dict__"):
            setattr(parsed_args, "debug", debug)
        console.emit_parsed_arguments(parsed_args)
        if not help_requested:
            _print_banner(argv, arguments)

        lifecycle_output = command_name not in {"help", "report-methods"}
        if lifecycle_output:
            console.emit_command_start(command_name, parsed_args)

        with console.normalized_output(normalize_plain=lifecycle_output):
            try:
                result = real_command(parsed_args)
            except SystemExit as error:
                # Legacy command implementations use bare quit() for fatal errors.
                # Python maps SystemExit(None) to shell status 0, so normalize only
                # that ambiguous case while preserving intentional help/error codes.
                if error.code is None:
                    if lifecycle_output:
                        console.emit_command_result(command_name, 1)
                    console.emit_completion(1)
                    return 1
                raise
        status = 0 if result is None else result
        if lifecycle_output:
            console.emit_command_result(command_name, status)
        console.emit_completion(status)
        return status
    except Exception as error:
        console.emit_exception(error)
        console.emit_completion(1)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
