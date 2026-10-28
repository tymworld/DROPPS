import unittest
import argparse
import ast
import inspect
import re
import shlex
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from dropps.share.all_commands import all_commands
from dropps.share.argument_parser import (
    ArgumentParser,
    _public_options,
)
from dropps.share.command_help import COMMAND_EXAMPLES, TASK_GROUPS
from scripts.update_readme_commands import _parse_all_commands, _render_commands_section


class CLIHelpConsistencyTests(unittest.TestCase):
    @staticmethod
    def _parse(command, argv):
        return all_commands.getargs(command)(argv)

    def _parse_quietly(self, command, argv):
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            return self._parse(command, argv)

    def test_every_registered_command_has_grouped_help(self):
        for command in all_commands.command_names:
            if command == "help":
                continue
            with self.subTest(command=command):
                stdout = StringIO()
                stderr = StringIO()
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    with self.assertRaises(SystemExit) as raised:
                        self._parse(command, ["-h"])

                self.assertEqual(raised.exception.code, 0)
                help_text = stdout.getvalue()
                self.assertIn(f"usage: dps {command}", help_text)
                self.assertIn("Input:", help_text)
                self.assertIn("Parameters:", help_text)
                self.assertIn("Examples:", help_text)
                self.assertIn(f"dps {command} ", help_text)
                self.assertNotIn("\noptions:", help_text)
                self.assertLess(help_text.index("Input:"), help_text.index("Parameters:"))
                if "Output:" in help_text:
                    self.assertLess(help_text.index("Input:"), help_text.index("Output:"))
                    self.assertLess(help_text.index("Output:"), help_text.index("Parameters:"))

    def test_every_visible_option_reports_required_or_default_status(self):
        for command in all_commands.command_names:
            if command == "help":
                continue
            with self.subTest(command=command):
                stdout = StringIO()
                captured_parsers = []
                original_parse_args = ArgumentParser.parse_args

                def capture_parser(parser, *args, **kwargs):
                    captured_parsers.append(parser)
                    return original_parse_args(parser, *args, **kwargs)

                with patch.object(ArgumentParser, "parse_args", capture_parser):
                    with redirect_stdout(stdout), redirect_stderr(StringIO()):
                        with self.assertRaises(SystemExit):
                            self._parse(command, ["-h"])

                self.assertEqual(len(captured_parsers), 1)
                parser = captured_parsers[0]
                parser.format_help()
                formatter = parser._get_formatter()

                for action in parser._actions:
                    if action.help is argparse.SUPPRESS:
                        continue
                    status_help = formatter._get_help_string(action)
                    if (
                        getattr(action, "required", False)
                        or getattr(action, "_dropps_required_group", False)
                    ):
                        self.assertIn("(required", status_help)
                    elif action.default is not argparse.SUPPRESS:
                        has_default = (
                            "%(default)" in status_help
                            or formatter._DOCUMENTED_DEFAULT.search(status_help)
                            is not None
                        )
                        self.assertTrue(
                            has_default,
                            f"{command} option {action.option_strings} omits its default",
                        )

    def test_help_only_displays_standard_option_names(self):
        for command in all_commands.command_names:
            if command == "help":
                continue
            captured_parsers = []
            original_parse_args = ArgumentParser.parse_args

            def capture_parser(parser, *args, **kwargs):
                captured_parsers.append(parser)
                return original_parse_args(parser, *args, **kwargs)

            with self.subTest(command=command):
                with patch.object(ArgumentParser, "parse_args", capture_parser):
                    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                        with self.assertRaises(SystemExit):
                            self._parse(command, ["-h"])

                parser = captured_parsers[0]
                parser.format_help()
                formatter = parser._get_formatter()
                for action in parser._actions:
                    if action.help is argparse.SUPPRESS or not action.option_strings:
                        continue
                    for option in action.option_strings:
                        self.assertEqual(
                            option,
                            option.lower(),
                            f"{command} accepts uppercase option {option}",
                        )
                    canonical = _public_options(action)
                    self.assertTrue(
                        any(option.startswith("--") for option in canonical),
                        f"{command}: {action.option_strings}",
                    )
                    for option in canonical:
                        self.assertEqual(
                            option,
                            option.lower(),
                            f"{command} exposes uppercase option {option}",
                        )
                    if "--forcefield" in canonical:
                        self.assertIn(
                            "-ff",
                            canonical,
                            f"{command} must use -ff for --forcefield",
                        )
                    invocation = formatter._format_action_invocation(action)
                    for option in action.option_strings:
                        if option not in canonical:
                            self.assertNotRegex(
                                invocation,
                                rf"(?<!\S){re.escape(option)}(?=\s|,|$)",
                            )

    def test_every_input_and_output_option_has_a_short_alias(self):
        for command in all_commands.command_names:
            if command == "help":
                continue
            captured_parsers = []

            def capture_parser(parser, *args, **kwargs):
                captured_parsers.append(parser)
                raise SystemExit(0)

            with self.subTest(command=command):
                with patch.object(ArgumentParser, "parse_args", capture_parser):
                    with self.assertRaises(SystemExit):
                        self._parse(command, [])

                parser = captured_parsers[0]
                parser._normalize_actions()
                for section, group in (
                    ("input", parser._input_options),
                    ("output", parser._output_options),
                ):
                    for action in group._group_actions:
                        if action.help is argparse.SUPPRESS:
                            continue
                        short_options = [
                            option
                            for option in _public_options(action)
                            if not option.startswith("--")
                        ]
                        self.assertTrue(
                            short_options,
                            f"{command} {section} option "
                            f"{action.option_strings} has no short alias",
                        )
                        for option in short_options:
                            self.assertRegex(option, r"^-[a-z][a-z0-9]?$")
                            if section == "output":
                                self.assertTrue(
                                    option.startswith("-o"),
                                    f"{command} output alias {option} must start with o",
                                )

    def test_parsed_namespace_covers_attributes_used_by_command_modules(self):
        for command in all_commands.command_names:
            if command == "help":
                continue
            captured_parsers = []

            def capture_parser(parser, *args, **kwargs):
                captured_parsers.append(parser)
                raise SystemExit(0)

            with self.subTest(command=command):
                with patch.object(ArgumentParser, "parse_args", capture_parser):
                    with self.assertRaises(SystemExit):
                        self._parse(command, [])

                parser = captured_parsers[0]
                parser._normalize_actions()
                destinations = {action.dest for action in parser._actions}
                module = inspect.getmodule(all_commands.command(command))
                source_path = Path(inspect.getsourcefile(module))
                tree = ast.parse(
                    source_path.read_text(encoding="utf-8"),
                    filename=str(source_path),
                )
                used_attributes = {
                    node.attr
                    for node in ast.walk(tree)
                    if isinstance(node, ast.Attribute)
                    and isinstance(node.value, ast.Name)
                    and node.value.id == "args"
                    and isinstance(node.ctx, ast.Load)
                }
                self.assertEqual(
                    used_attributes - destinations,
                    set(),
                    f"{command} reads fields absent from its parsed Namespace",
                )

    def test_command_catalog_has_consistent_usage_and_sections(self):
        stdout = StringIO()
        with redirect_stdout(stdout):
            all_commands.command("help")(["commands"])

        catalog = stdout.getvalue()
        self.assertIn("dps <command> [options]", catalog)
        self.assertIn("Start here:", catalog)
        self.assertIn("Commands by task:", catalog)
        self.assertIn("Build a simulation system", catalog)
        self.assertIn("Inspect and convert files", catalog)
        self.assertIn("Analyze structure and contacts", catalog)
        self.assertIn("Analyze phase behavior and transport", catalog)
        self.assertIn("Choosing between similar commands:", catalog)
        self.assertIn("-db, --debug", catalog)
        self.assertIn("-v, --version", catalog)
        self.assertNotIn("help           ", catalog)

    def test_task_catalog_lists_every_functional_command_once(self):
        grouped_commands = [
            command
            for _, _, commands in TASK_GROUPS
            for command in commands
        ]
        functional_commands = [
            command for command in all_commands.command_names if command != "help"
        ]

        self.assertCountEqual(grouped_commands, functional_commands)
        self.assertEqual(len(grouped_commands), len(set(grouped_commands)))

    def test_every_functional_command_has_a_parseable_example(self):
        functional_commands = [
            command for command in all_commands.command_names if command != "help"
        ]
        self.assertCountEqual(COMMAND_EXAMPLES, functional_commands)

        for command, examples in COMMAND_EXAMPLES.items():
            for example in examples:
                with self.subTest(command=command, example=example):
                    tokens = shlex.split(example)
                    self.assertEqual(tokens[:2], ["dps", command])
                    for token in tokens[2:]:
                        if token.startswith("-") and not token.startswith("--"):
                            self.assertRegex(
                                token,
                                r"^-[a-z][a-z0-9]?$",
                                f"{command} example uses nonstandard option {token}",
                            )
                    self._parse_quietly(command, tokens[2:])

    def test_readme_catalog_tracks_shared_and_exclusive_requirements(self):
        catalog = _render_commands_section(_parse_all_commands())

        self.assertIn(
            "`contact`: Calculate residue-level contact maps and contact-number "
            "time series.\n  Required args: `-s/--run-input`, `-f/--input`, "
            "`-cs/--cutoff-scheme`",
            catalog,
        )
        self.assertIn(
            "one of `-s/--run-input` or `-p/--topology`",
            catalog,
        )

    def test_default_outputs_are_not_marked_required(self):
        args = self._parse_quietly(
            "genmesh",
            ["-f", "molecule.pdb", "-p", "molecule.itp", "-n", "10"],
        )
        self.assertEqual(args.output_conformation, "system.pdb")
        self.assertEqual(args.output_topology, "system.top")

    def test_genmesh_output_short_aliases_are_available(self):
        args = self._parse_quietly(
            "genmesh",
            [
                "-f",
                "molecule.pdb",
                "-p",
                "molecule.itp",
                "-n",
                "10",
                "-o",
                "packed.pdb",
                "-op",
                "packed.top",
            ],
        )
        self.assertEqual(args.output_conformation, "packed.pdb")
        self.assertEqual(args.output_topology, "packed.top")

    def test_pdb2dps_lowercase_parameter_aliases_are_available(self):
        args = self._parse_quietly(
            "pdb2dps",
            [
                "-s",
                "AG",
                "-ff",
                "HPS",
                "-pt",
                "S1SMP",
                "-cn",
                "-cc",
                "-o",
                "chain.pdb",
                "-op",
                "chain.itp",
            ],
        )
        self.assertEqual(args.forcefield, "HPS")
        self.assertEqual(args.post_translational_modification, ["S1SMP"])
        self.assertTrue(args.charged_NTD)
        self.assertTrue(args.charged_CTD)
        self.assertEqual(args.output_conformation, "chain.pdb")
        self.assertEqual(args.output_topology, "chain.itp")

    def test_required_output_paths_are_enforced(self):
        for command, argv in (
            ("trjconv", ["-s", "run.tpr", "-f", "run.xtc"]),
            ("gsd2xtc", ["-f", "run.gsd"]),
            ("msd", ["-s", "run.tpr", "-f", "run.xtc"]),
        ):
            with self.subTest(command=command):
                with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                    with self.assertRaises(SystemExit) as raised:
                        self._parse(command, argv)
                self.assertEqual(raised.exception.code, 2)

    def test_trjconv_is_time_based_without_frame_or_skip_options(self):
        base = ["-s", "run.tpr", "-f", "run.xtc", "-o", "frame.pdb"]
        args = self._parse_quietly(
            "trjconv",
            [*base, "-b", "2.5", "-e", "2.5", "-tu", "ns"],
        )
        self.assertEqual(args.start_time, 2.5)
        self.assertEqual(args.end_time, 2.5)
        self.assertFalse(hasattr(args, "frame"))
        self.assertFalse(hasattr(args, "skip"))

    def test_trjconv_common_short_options_are_available(self):
        args = self._parse_quietly(
            "trjconv",
            [
                "-s",
                "run.tpr",
                "-f",
                "run.xtc",
                "-o",
                "centered.xtc",
                "-se",
                "group 0",
                "-c",
                "mass",
                "-ca",
                "z",
                "-p",
                "mol",
                "-dt",
                "2",
                "-tu",
                "ps",
            ],
        )
        self.assertEqual(args.selection, "group 0")
        self.assertEqual(args.center, "mass")
        self.assertEqual(args.center_axis, "z")
        self.assertEqual(args.pbc, "mol")
        self.assertEqual(args.delta_time, 2.0)
        self.assertEqual(args.time_unit, "ps")

    def test_exactly_one_topology_source_is_required(self):
        base = ["-f", "input.pdb", "-o", "output.pdb"]
        for sources in (
            [],
            ["--run-input", "run.tpr", "--topology", "system.top"],
        ):
            with self.subTest(sources=sources):
                with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                    with self.assertRaises(SystemExit) as raised:
                        self._parse("pdb2bond", [*base, *sources])
                self.assertEqual(raised.exception.code, 2)

    def test_corrected_options_keep_legacy_aliases(self):
        pdb2dps = self._parse_quietly(
            "pdb2dps",
            ["-s", "AG", "--charged-ntd", "--charged-ctd"],
        )
        self.assertTrue(pdb2dps.charged_NTD)
        self.assertTrue(pdb2dps.charged_CTD)

        assembly = self._parse_quietly(
            "assembly",
            ["-s", "run.tpr", "-f", "run.xtc", "--threashold", "7"],
        )
        self.assertEqual(assembly.threshold, 7)

        timecorr = self._parse_quietly(
            "timecorr",
            [
                "-s",
                "run.tpr",
                "-f",
                "run.xtc",
                "-cs",
                "global",
                "--otpye",
                "xlsx",
            ],
        )
        self.assertEqual(timecorr.average_output_type, "xlsx")

        pdb2bond = self._parse_quietly(
            "pdb2bond",
            [
                "-s",
                "run.tpr",
                "-f",
                "input.pdb",
                "-o",
                "output.pdb",
                "--not-obmit-atom-type",
            ],
        )
        self.assertTrue(pdb2bond.preserve_atom_types)


if __name__ == "__main__":
    unittest.main()
