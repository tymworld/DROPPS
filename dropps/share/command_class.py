from __future__ import annotations

from dataclasses import dataclass
from difflib import get_close_matches


class UnknownCommandError(LookupError):
    """Raised when a command name is not registered."""

    def __init__(self, name, suggestions=()):
        self.name = name
        self.suggestions = tuple(suggestions)
        super().__init__(name)


class CommandUsageError(ValueError):
    """Raised when command-level arguments cannot be parsed."""


@dataclass(frozen=True)
class CommandSpec:
    """The parser, implementation, and summary for one CLI command."""

    name: str
    getargs: object
    main_func: object
    desc: str = ""


# Backwards-compatible name used by existing command modules.
single_command = CommandSpec


class CommandRegistry:
    """Validated lookup table for public commands and compatibility aliases."""

    def __init__(self, command_list, hidden_commands=()):
        public_commands = list(command_list)
        hidden_commands = list(hidden_commands)
        all_registered = [*public_commands, *hidden_commands]
        names = [command.name for command in all_registered]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(
                "Duplicate DROPPS command registration: " + ", ".join(duplicates)
            )

        self.command_dict = {command.name: command for command in all_registered}
        # Retain the historic misspelling for third-party callers in the 1.x API.
        self.commmand_dict = self.command_dict
        self.command_names = [command.name for command in public_commands]
        self.hidden_command_names = [command.name for command in hidden_commands]

    def names(self):
        return list(self.command_names)

    def exist(self, name):
        return name in self.command_dict

    def suggestions(self, name, candidates=None):
        choices = self.command_names if candidates is None else list(candidates)
        return get_close_matches(name, choices, n=3, cutoff=0.5)

    def _get(self, name):
        try:
            return self.command_dict[name]
        except KeyError as exc:
            raise UnknownCommandError(name, self.suggestions(name)) from exc

    def getargs(self, name):
        return self._get(name).getargs

    def command(self, name):
        return self._get(name).main_func
    
    def desc(self, name):
        return self._get(name).desc


# Backwards-compatible name used by the existing registry module.
all_commands_class = CommandRegistry
