from __future__ import annotations

from dataclasses import dataclass
from difflib import get_close_matches
from typing import Callable, Iterable


class UnknownCommandError(LookupError):
    """Raised when a command name is not registered."""

    def __init__(self, name, suggestions=()):
        self.name = name
        self.suggestions = tuple(suggestions)
        super().__init__(name)


class CommandUsageError(ValueError):
    """Raised when command-level arguments cannot be parsed."""


@dataclass(frozen=True)
class SingleCommand:
    """One registered command-line operation."""

    name: str
    getargs: Callable
    main_func: Callable
    desc: str = ""

    def __post_init__(self):
        if not self.name or self.name.strip() != self.name:
            raise ValueError("Command names must be non-empty and trimmed.")
        if any(character.isspace() for character in self.name):
            raise ValueError(f"Command names cannot contain whitespace: {self.name!r}.")
        if not callable(self.getargs) or not callable(self.main_func):
            raise TypeError(
                f"Command {self.name!r} requires callable parser and runner."
            )


# Backward-compatible constructor name used throughout the 0.5 series.
single_command = SingleCommand


class all_commands_class:
    """Validated, deterministic command registry."""

    def __init__(self, command_list: Iterable[SingleCommand]):
        commands = tuple(command_list)
        names = [command.name for command in commands]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(
                "Duplicate DROPPS command registration: " + ", ".join(duplicates)
            )
        self.command_dict = {command.name: command for command in commands}
        # Preserve the misspelled public attribute used by older extensions.
        self.commmand_dict = self.command_dict
        self.command_names = tuple(names)

    def names(self):
        return list(self.command_names)

    def exist(self, name):
        return name in self.command_names

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
