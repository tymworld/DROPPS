"""Shared command-line help formatting for DROPPS commands."""

from __future__ import annotations

import argparse
import re
import string
import sys
from collections.abc import Sequence
from typing import Any

from dropps.share.command_help import format_examples
from dropps.share.command_docs import format_notes
from dropps.share import console


_CANONICAL_LONG_OPTION = re.compile(r"^--[a-z][a-z0-9]*(?:-[a-z0-9]+)*$")
_CANONICAL_SHORT_OPTION = re.compile(r"^-[a-z][a-z0-9]?$")
_COMMON_SHORT_OPTIONS = {
    "--append": "-a",
    "--average-axis": "-aa",
    "--axis": "-a",
    "--bin-width": "-bw",
    "--blocks": "-bl",
    "--center": "-c",
    "--center-axis": "-ca",
    "--center-mode": "-cm",
    "--checkpoint": "-cp",
    "--cutoff-scheme": "-cs",
    "--delta-time": "-dt",
    "--forcefield": "-ff",
    "--max-hours": "-mh",
    "--max-lag": "-ml",
    "--mode": "-m",
    "--normalization": "-nm",
    "--nsteps": "-n",
    "--pbc": "-p",
    "--platform": "-p",
    "--reference": "-r",
    "--reference-group": "-r",
    "--reference-time": "-r",
    "--select": "-se",
    "--selection": "-se",
    "--selection-calculate": "-sc",
    "--selection-group": "-sg",
    "--selection-groups": "-sg",
    "--surface": "-sf",
    "--temperature": "-tm",
    "--threads": "-t",
    "--threshold": "-t",
    "--time-unit": "-tu",
    "--treat-pbc": "-p",
    "--type": "-t",
}

# Input and output options are used on almost every invocation, so they always
# receive a visible lowercase alias of one or two letters.  Every output alias
# starts with ``o``.  Keep aliases semantic and stable across commands; the
# candidate order also provides a deterministic fallback on conflicts.
_ROLE_SHORT_OPTIONS = {
    # Inputs.
    ("input", "angle_list"): ("-a", "-al"),
    ("input", "center_select"): ("-cs",),
    ("input", "checkpoint"): ("-cp",),
    ("input", "elastic_residues"): ("-r", "-er"),
    ("input", "fit_select"): ("-fs",),
    ("input", "group_file"): ("-g", "-gf"),
    ("input", "index"): ("-n",),
    ("input", "input"): ("-f",),
    ("input", "input_structure"): ("-f", "-s"),
    ("input", "input_topology"): ("-p", "-it"),
    ("input", "map_input"): ("-m",),
    ("input", "parameter"): ("-m",),
    ("input", "run_input"): ("-s",),
    ("input", "sequence"): ("-s",),
    ("input", "selection_expressions"): ("-se",),
    ("input", "structure"): ("-f", "-s"),
    ("input", "topology"): ("-p", "-s"),
    ("input", "trajectory"): ("-f",),
    # Common and named outputs.
    ("output", "checkpoint_output"): ("-ok",),
    ("output", "energy_output"): ("-oe",),
    ("output", "final_structure"): ("-oc",),
    ("output", "internal_distance_output"): ("-oi",),
    ("output", "latex_output"): ("-ol",),
    ("output", "log_output"): ("-ol",),
    ("output", "map_output"): ("-om",),
    ("output", "molecule_output"): ("-om",),
    ("output", "number_output"): ("-on",),
    ("output", "occupancy_output"): ("-oo",),
    ("output", "output"): ("-o",),
    ("output", "output_all"): ("-o",),
    ("output", "output_average"): ("-oa",),
    ("output", "output_components"): ("-oc",),
    ("output", "output_conformation"): ("-o", "-oc"),
    ("output", "output_cumulative"): ("-oc",),
    ("output", "output_deviation"): ("-od",),
    ("output", "output_histogram"): ("-oh",),
    ("output", "output_mode"): ("-om",),
    ("output", "output_name"): ("-on",),
    ("output", "output_phase"): ("-op",),
    ("output", "output_prefix"): ("-o",),
    ("output", "output_reference"): ("-or",),
    ("output", "output_residue"): ("-or",),
    ("output", "output_residue_statistic"): ("-os",),
    ("output", "output_statistics"): ("-os",),
    ("output", "output_structure"): ("-o", "-oc"),
    ("output", "output_time"): ("-ot",),
    ("output", "output_topology"): ("-op",),
    ("output", "output_type"): ("-oy",),
    ("output", "output_verbose"): ("-ov",),
    ("output", "persistence_output"): ("-op",),
    ("output", "plot_prefix"): ("-og",),
    ("output", "strand_pair_output"): ("-op",),
    ("output", "strand_stat_output"): ("-os",),
    ("output", "stress_output"): ("-os",),
    ("output", "summary_output"): ("-ou",),
    ("output", "tangent_output"): ("-ot",),
    ("output", "trajectory_output"): ("-ox",),
    ("output", "xpm_output"): ("-ox",),
    # Contact-map and time-correlation output families.  Matching quantities
    # deliberately use matching aliases in contact, contact.ng, and timecorr.
    ("output", "output_inter_reference_selection"): ("-oa",),
    ("output", "output_inter_reference_reference"): ("-ob",),
    ("output", "output_inter_selection_selection"): ("-oc",),
    ("output", "output_intra_reference"): ("-oi",),
    ("output", "output_intra_selection"): ("-oj",),
    ("output", "output_time_inter_reference_selection"): ("-oe",),
    ("output", "output_time_inter_reference_reference"): ("-of",),
    ("output", "output_time_inter_selection_selection"): ("-og",),
    ("output", "output_time_intra_reference"): ("-oh",),
    ("output", "output_time_intra_selection"): ("-ol",),
    ("output", "output_corr_inter_reference_selection"): ("-oa",),
    ("output", "output_corr_inter_reference_reference"): ("-ob",),
    ("output", "output_corr_inter_selection_selection"): ("-oc",),
    ("output", "output_corr_intra_reference"): ("-oi",),
    ("output", "output_corr_intra_selection"): ("-oj",),
    ("output", "output_lifetime_inter_reference_selection"): ("-oe",),
    ("output", "output_lifetime_inter_reference_reference"): ("-of",),
    ("output", "output_lifetime_inter_selection_selection"): ("-og",),
    ("output", "output_lifetime_intra_reference"): ("-oh",),
    ("output", "output_lifetime_intra_selection"): ("-ol",),
    ("output", "average_output_type"): ("-oz",),
    # Assembly metrics.
    ("output", "asphericity"): ("-oa",),
    ("output", "cluster_number"): ("-on",),
    ("output", "cluster_size"): ("-oz",),
    ("output", "cluster_size_distribution"): ("-oq",),
    ("output", "ellipticity"): ("-oe",),
    ("output", "molecule_fraction"): ("-om",),
    ("output", "radius_gyration_largest"): ("-og",),
}

_ROLE_GENERIC_WORDS = frozenset(
    {"file", "files", "input", "name", "output", "path", "prefix"}
)


def _is_canonical_option(option: str) -> bool:
    return bool(
        _CANONICAL_SHORT_OPTION.fullmatch(option)
        or _CANONICAL_LONG_OPTION.fullmatch(option)
    )


def _public_options(action: argparse.Action) -> list[str]:
    """Return the normalized aliases intended for help and new scripts."""

    normalized = getattr(action, "_dropps_public_options", None)
    if normalized is not None:
        return list(normalized)
    return [
        option for option in action.option_strings if _is_canonical_option(option)
    ] or list(action.option_strings)


def _replace_option_token(text: str, option: str, replacement: str) -> str:
    return re.sub(
        rf"(?<![A-Za-z0-9_-]){re.escape(option)}(?=$|[\s/,;:.)\]])",
        replacement,
        text,
    )


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

    def _format_action_invocation(self, action: argparse.Action) -> str:
        if not action.option_strings:
            return super()._format_action_invocation(action)

        visible_options = _public_options(action)
        if action.nargs == 0:
            return ", ".join(visible_options)

        default = self._get_default_metavar_for_optional(action)
        args_string = self._format_args(action, default)
        return ", ".join(
            f"{option} {args_string}" for option in visible_options
        )


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
            format_examples(command_name)
            if isinstance(command_name, str)
            else None
        )
        if examples is not None:
            notes = format_notes(command_name)
            if notes:
                examples = notes + "\n\n" + examples
            existing_epilog = kwargs.get("epilog")
            kwargs["epilog"] = (
                f"{existing_epilog}\n\n{examples}"
                if existing_epilog
                else examples
            )

        kwargs.setdefault("formatter_class", HelpFormatter)
        kwargs.setdefault("allow_abbrev", False)
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
        self._parameter_options.add_argument(
            "-db",
            "--debug",
            action="store_true",
            default=False,
            help="Show runtime context, parsed arguments, timing, and tracebacks.",
        )

    @staticmethod
    def _preferred_option(action: argparse.Action) -> str | None:
        public = _public_options(action)
        short = [option for option in public if not option.startswith("--")]
        long = [option for option in public if option.startswith("--")]
        return (short or long or list(action.option_strings) or [None])[0]

    @staticmethod
    def _role_short_candidates(
        section: str,
        action: argparse.Action,
    ) -> list[str]:
        """Return stable lowercase aliases for an input/output action."""

        candidates = list(
            _ROLE_SHORT_OPTIONS.get((section, action.dest), ())
        )
        existing = [
            option
            for option in action.option_strings
            if _CANONICAL_SHORT_OPTION.fullmatch(option)
            and (section != "output" or option.startswith("-o"))
        ]
        candidates.extend(existing)
        meaningful_words = [
            word
            for word in action.dest.split("_")
            if word and word not in _ROLE_GENERIC_WORDS
        ]
        if section == "output":
            candidates.append("-o")
            candidates.extend(f"-o{letter}" for letter in string.ascii_lowercase)
        else:
            for word in meaningful_words:
                candidates.append(f"-{word[0].lower()}")
                if len(word) > 1:
                    candidates.append(f"-{word[:2].lower()}")
            candidates.append("-f")
            candidates.extend(f"-{letter}" for letter in string.ascii_lowercase)
        return list(dict.fromkeys(candidates))

    def _normalize_actions(self) -> None:
        """Validate names, add stable aliases, and normalize examples."""

        option_owners = self._option_string_actions
        action_sections = {
            id(action): section
            for section, group in (
                ("input", self._input_options),
                ("output", self._output_options),
            )
            for action in group._group_actions
        }
        selected_shorts: dict[int, str] = {}
        for action in self._actions:
            section = action_sections.get(id(action))
            if action.help == argparse.SUPPRESS or not action.option_strings:
                continue
            if section is not None:
                candidates = self._role_short_candidates(section, action)
            else:
                candidates = [
                    _COMMON_SHORT_OPTIONS[option]
                    for option in action.option_strings
                    if option in _COMMON_SHORT_OPTIONS
                ]
                candidates.extend(
                    option
                    for option in action.option_strings
                    if _CANONICAL_SHORT_OPTION.fullmatch(option)
                )

            for short_option in dict.fromkeys(candidates):
                if not _CANONICAL_SHORT_OPTION.fullmatch(short_option):
                    continue
                if section == "output" and not short_option.startswith("-o"):
                    continue
                owner = option_owners.get(short_option)
                if owner is not None and owner is not action:
                    continue
                if owner is None:
                    action.option_strings.insert(0, short_option)
                    option_owners[short_option] = action
                selected_shorts[id(action)] = short_option
                break

            if section is not None and id(action) not in selected_shorts:
                raise ValueError(
                    f"{self.prog} {section} option {action.option_strings!r} "
                    "has no available lowercase short alias"
                )

        replacements = {}
        for action in self._actions:
            if not action.option_strings:
                continue
            long_options = [
                option
                for option in action.option_strings
                if _CANONICAL_LONG_OPTION.fullmatch(option)
            ]
            if action.help != argparse.SUPPRESS:
                if not long_options:
                    raise ValueError(
                        f"{self.prog} option {action.option_strings!r} requires a "
                        "canonical lowercase --long-name alias"
                    )

            short = selected_shorts.get(id(action))
            public = [*([short] if short else []), *long_options]
            legacy = [
                option for option in action.option_strings if option not in public
            ]
            action._dropps_public_options = public
            action.option_strings[:] = [*public, *legacy]
            preferred = self._preferred_option(action)
            if preferred is not None:
                for option in legacy:
                    replacements[option] = preferred

        for legacy, preferred in sorted(
            replacements.items(), key=lambda item: len(item[0]), reverse=True
        ):
            if self.epilog:
                self.epilog = _replace_option_token(
                    self.epilog,
                    legacy,
                    preferred,
                )
            for action in self._actions:
                if isinstance(action.help, str):
                    action.help = _replace_option_token(
                        action.help,
                        legacy,
                        preferred,
                    )

    def parse_known_args(self, args=None, namespace=None):
        self._normalize_actions()
        raw_args = list(args) if args is not None else sys.argv[1:]
        legacy_options = {
            option: self._preferred_option(action)
            for action in self._actions
            for option in action.option_strings
            if option not in _public_options(action)
        }
        for token in raw_args:
            option = token.split("=", 1)[0]
            if option in legacy_options:
                console.debug(
                    f"Legacy option {option!r} accepted; prefer "
                    f"{legacy_options[option]!r}."
                )
        return super().parse_known_args(args, namespace)

    def error(self, message: str) -> None:
        self._normalize_actions()
        for action in self._actions:
            if not action.option_strings:
                continue
            complete = "/".join(action.option_strings)
            visible = "/".join(_public_options(action))
            if visible:
                message = message.replace(complete, visible)
        self.print_usage(sys.stderr)
        console.emit("error", f"{self.prog}: {message}", stream=sys.stderr)
        self.exit(2)

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

        if (
            "output" in destination_parts
            or destination in cls._OUTPUT_DESTINATIONS
        ):
            return "output"

        if (
            "input" in destination_parts
            or destination in cls._INPUT_DESTINATIONS
        ):
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
                kwargs["metavar"] = (
                    "SEQUENCE" if destination == "sequence" else "FILE"
                )
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

        self._normalize_actions()
        containers = [self, *self._action_groups]
        for container in containers:
            for group in container._mutually_exclusive_groups:
                required_group = bool(group.required)
                for action in group._group_actions:
                    action._dropps_required_group = required_group
        return super().format_help()
