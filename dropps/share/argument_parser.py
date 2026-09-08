"""Shared command-line help formatting for DROPPS commands."""

from __future__ import annotations

import argparse
import re
from collections.abc import Sequence
from typing import Any

from dropps.share.command_help import format_examples


class HelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawDescriptionHelpFormatter,
):
    """Preserve descriptions and label defaults and required options."""

    _DOCUMENTED_DEFAULT = re.compile(
        r"\b(?:by\s+default|defaults?\s+to|default\s*:|default\s+is\b|the\s+default)",
        re.IGNORECASE,
    )

    def _get_help_string(self, action: argparse.Action) -> str:
        help_text = action.help or ""

        if getattr(action, "_dropps_required_group", False):
            return help_text + " (required: choose one option from this group)"

        if getattr(action, "required", False):
            return help_text + " (required)"

        if (
            "%(default)" not in help_text
            and self._DOCUMENTED_DEFAULT.search(help_text) is None
            and action.default is not argparse.SUPPRESS
        ):
            help_text += " (default: %(default)s)"
        return help_text


# Keep this import-compatible with modules that previously imported the class
# directly from argparse.
RawDescriptionHelpFormatter = HelpFormatter


class ArgumentParser(argparse.ArgumentParser):
    """Argument parser with consistent Input/Output/Parameters help sections.

    Optional arguments are classified from their destination name. A command
    can override the classification for an unusual option by passing
    ``section="input"``, ``section="output"``, or ``section="parameters"`` to
    :meth:`add_argument`.
    """

    _INPUT_DESTINATIONS = frozenset(
        {
            "angle_list",
            "checkpoint",
            "elastic_residues",
            "group_file",
            "index",
            "parameter",
            "sequence",
            "structure",
            "topology",
            "trajectory",
        }
    )
    _OUTPUT_DESTINATIONS = frozenset(
        {
            "asphericity",
            "cluster_number",
            "cluster_size",
            "cluster_size_distribution",
            "ellipticity",
            "molecule_fraction",
            "radius_gyration_largest",
        }
    )
    _VALID_SECTIONS = frozenset({"input", "output", "parameters"})

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        add_help = kwargs.pop("add_help", True)
        command_name = kwargs.get("prog")
        if isinstance(command_name, str) and not command_name.startswith("dps "):
            kwargs["prog"] = f"dps {command_name}"

        examples = (
            format_examples(command_name) if isinstance(command_name, str) else None
        )
        if examples is not None:
            existing_epilog = kwargs.get("epilog")
            kwargs["epilog"] = (
                f"{existing_epilog}\n\n{examples}" if existing_epilog else examples
            )

        kwargs.setdefault("formatter_class", HelpFormatter)
        super().__init__(*args, add_help=False, **kwargs)

        self._input_options = self.add_argument_group("Input")
        self._output_options = self.add_argument_group("Output")
        self._parameter_options = self.add_argument_group("Parameters")

        if add_help:
            self._parameter_options.add_argument(
                "-h",
                "--help",
                action="help",
                help="Show this help message and exit.",
            )

    @staticmethod
    def _get_destination(option_strings: Sequence[str], kwargs: dict[str, Any]) -> str:
        if "dest" in kwargs:
            return str(kwargs["dest"]).replace("-", "_")

        long_options = [option for option in option_strings if option.startswith("--")]
        option = long_options[0] if long_options else option_strings[0]
        return option.lstrip("-").replace("-", "_")

    @classmethod
    def _classify(cls, option_strings: Sequence[str], kwargs: dict[str, Any]) -> str:
        destination = cls._get_destination(option_strings, kwargs)
        destination_parts = destination.split("_")

        if "output" in destination_parts or destination in cls._OUTPUT_DESTINATIONS:
            return "output"

        if "input" in destination_parts or destination in cls._INPUT_DESTINATIONS:
            return "input"

        return "parameters"

    def add_argument(self, *args: Any, **kwargs: Any) -> argparse.Action:
        section = kwargs.pop("section", None)
        option_strings = [arg for arg in args if isinstance(arg, str)]

        # Positional arguments keep argparse's standard positional section.
        if not option_strings or not option_strings[0].startswith("-"):
            if section is not None:
                raise ValueError("section can only be used with optional arguments")
            return super().add_argument(*args, **kwargs)

        if section is None:
            section = self._classify(option_strings, kwargs)
        elif section not in self._VALID_SECTIONS:
            choices = ", ".join(sorted(self._VALID_SECTIONS))
            raise ValueError(f"unknown help section {section!r}; choose from {choices}")

        if "metavar" not in kwargs and "choices" not in kwargs:
            destination = self._get_destination(option_strings, kwargs)
            if section == "input":
                kwargs["metavar"] = "SEQUENCE" if destination == "sequence" else "FILE"
            elif section == "output":
                if destination.endswith("_prefix"):
                    kwargs["metavar"] = "PREFIX"
                elif destination == "output_name":
                    kwargs["metavar"] = "NAME"
                else:
                    kwargs["metavar"] = "FILE"

        group = {
            "input": self._input_options,
            "output": self._output_options,
            "parameters": self._parameter_options,
        }[section]
        return group.add_argument(*args, **kwargs)

    def add_mutually_exclusive_group(self, **kwargs: Any) -> Any:
        """Create a mutually exclusive group inside a named help section."""

        section = kwargs.pop("section", "parameters")
        if section not in self._VALID_SECTIONS:
            choices = ", ".join(sorted(self._VALID_SECTIONS))
            raise ValueError(f"unknown help section {section!r}; choose from {choices}")

        group = {
            "input": self._input_options,
            "output": self._output_options,
            "parameters": self._parameter_options,
        }[section]
        return group.add_mutually_exclusive_group(**kwargs)

    def format_help(self) -> str:
        """Mark required mutually exclusive choices before formatting help."""

        containers = [self, *self._action_groups]
        for container in containers:
            for group in container._mutually_exclusive_groups:
                required_group = bool(group.required)
                for action in group._group_actions:
                    action._dropps_required_group = required_group
        return super().format_help()
