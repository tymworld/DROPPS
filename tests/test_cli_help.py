import unittest
import argparse
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from unittest.mock import patch

from dropps.share.all_commands import all_commands
from dropps.share.argument_parser import ArgumentParser
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
                self.assertIn("usage:", help_text)
                self.assertIn("Input:", help_text)
                self.assertIn("Parameters:", help_text)
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

                help_text = stdout.getvalue()
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

    def test_command_catalog_has_consistent_usage_and_sections(self):
        stdout = StringIO()
        with redirect_stdout(stdout):
            all_commands.command("help")(["commands"])

        catalog = stdout.getvalue()
        self.assertIn("Usage: dps COMMAND [OPTIONS]", catalog)
        self.assertIn("System modeling:", catalog)
        self.assertIn("Simulation:", catalog)
        self.assertIn("Analysis:", catalog)
        self.assertNotIn("help           : help", catalog)

    def test_readme_catalog_tracks_shared_and_exclusive_requirements(self):
        catalog = _render_commands_section(_parse_all_commands())

        self.assertIn(
            "`contact`: Calculate residue-level contact maps and contact-number "
            "time series.\n  Required args: `-s/--run-input`, `-f/--input`, "
            "`-cs/--cutoff-scheme`",
            catalog,
        )
        self.assertIn(
            "one of `-t/--time` or `-fm/--frame`",
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

    def test_exactly_one_frame_selector_is_required(self):
        base = ["-s", "run.tpr", "-f", "run.xtc", "-o", "frame.pdb"]
        for selectors in ([], ["--time", "10", "--frame", "2"]):
            with self.subTest(selectors=selectors):
                with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                    with self.assertRaises(SystemExit) as raised:
                        self._parse("extract", [*base, *selectors])
                self.assertEqual(raised.exception.code, 2)

        args = self._parse_quietly("extract", [*base, "--frame", "2"])
        self.assertEqual(args.frame, 2)
        self.assertIsNone(args.time)

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
