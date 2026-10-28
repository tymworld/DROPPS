import unittest
from pathlib import Path
from contextlib import redirect_stderr
from io import StringIO

from dropps.share.argument_parser import ArgumentParser


class ArgumentParserHelpTests(unittest.TestCase):
    def setUp(self):
        self.parser = ArgumentParser(prog="example", description="Example command.")
        self.parser.add_argument("-s", "--run-input", required=True, help="Input TPR.")
        self.parser.add_argument("-o", "--output-prefix", required=True, help="Output prefix.")
        self.parser.add_argument("--axis", choices=["x", "y", "z"], default="z", help="Axis.")
        self.parser.add_argument("--enabled", action="store_true", default=False, help="Enable it.")
        self.parser.add_argument("--optional", help="Optional override.")

    def test_help_is_grouped_by_option_role(self):
        help_text = self.parser.format_help()

        input_start = help_text.index("Input:")
        output_start = help_text.index("Output:")
        parameters_start = help_text.index("Parameters:")

        self.assertLess(input_start, output_start)
        self.assertLess(output_start, parameters_start)
        self.assertIn("--run-input", help_text[input_start:output_start])
        self.assertIn("--output-prefix", help_text[output_start:parameters_start])
        self.assertIn("--axis", help_text[parameters_start:])
        self.assertNotIn("options:", help_text.lower())
        self.assertIn("-db, --debug", help_text[parameters_start:])

    def test_help_displays_explicit_default_values(self):
        help_text = self.parser.format_help()

        self.assertIn("(default: z)", help_text)
        self.assertIn("(default: False)", help_text)
        self.assertIn("Optional override. (default: None)", help_text)

    def test_help_marks_required_arguments(self):
        help_text = self.parser.format_help()

        self.assertIn("Input TPR. (required)", help_text)
        self.assertIn("Output prefix. (required)", help_text)
        self.assertNotIn("Input TPR. (default:", help_text)

    def test_file_and_prefix_metavars_are_consistent(self):
        help_text = self.parser.format_help()

        self.assertIn("--run-input FILE", help_text)
        self.assertIn("--output-prefix PREFIX", help_text)

    def test_explicit_section_overrides_automatic_classification(self):
        parser = ArgumentParser(prog="override")
        parser.add_argument("--destination", section="output", help="Output path.")

        help_text = parser.format_help()
        output_start = help_text.index("Output:")
        parameters_start = help_text.index("Parameters:")
        self.assertIn("--destination", help_text[output_start:parameters_start])

    def test_output_word_in_destination_is_classified_as_output(self):
        parser = ArgumentParser(prog="summary")
        parser.add_argument("--summary-output", help="Summary path.")
        parser.add_argument(
            "--format",
            dest="average_output_type",
            choices=["dat", "xlsx"],
            help="Output format.",
        )

        help_text = parser.format_help()
        output_start = help_text.index("Output:")
        parameters_start = help_text.index("Parameters:")
        output_help = help_text[output_start:parameters_start]
        self.assertIn("--summary-output", output_help)
        self.assertIn("--format", output_help)

    def test_mutually_exclusive_arguments_stay_in_requested_section(self):
        parser = ArgumentParser(prog="exclusive")
        sources = parser.add_mutually_exclusive_group(required=True, section="input")
        sources.add_argument("--run-input", metavar="FILE", help="Run input.")
        sources.add_argument("--topology", metavar="FILE", help="Topology.")

        help_text = parser.format_help()
        self.assertIn("(-s FILE | -p FILE)", help_text)
        self.assertIn("Input:", help_text)
        self.assertNotIn("options:", help_text.lower())
        self.assertEqual(
            help_text.count("(required: choose one option from this"),
            2,
        )

    def test_existing_default_description_is_not_duplicated(self):
        parser = ArgumentParser(prog="documented")
        parser.add_argument("--count", default=5, help="Count. Default: 5.")

        help_text = parser.format_help()
        self.assertEqual(help_text.count("Default: 5"), 1)
        self.assertNotIn("(default: 5)", help_text)

    def test_argument_parsing_is_unchanged(self):
        args = self.parser.parse_args(
            ["-s", "run.tpr", "-o", "run", "--axis", "x", "--enabled"]
        )

        self.assertEqual(args.run_input, "run.tpr")
        self.assertEqual(args.output_prefix, "run")
        self.assertEqual(args.axis, "x")
        self.assertTrue(args.enabled)

    def test_legacy_multi_character_short_option_is_hidden_but_accepted(self):
        parser = ArgumentParser(prog="legacy")
        parser.add_argument("-ref", "--reference", required=True)

        help_text = parser.format_help()
        self.assertIn("--reference", help_text)
        self.assertNotRegex(help_text, r"(?<!\S)-ref(?=\s|,|$)")
        self.assertEqual(parser.parse_args(["-ref", "model.pdb"]).reference, "model.pdb")

    def test_visible_option_requires_a_canonical_long_name(self):
        parser = ArgumentParser(prog="invalid")
        parser.add_argument("-legacy")

        with self.assertRaisesRegex(ValueError, "canonical lowercase --long-name"):
            parser.format_help()

    def test_required_error_omits_legacy_aliases(self):
        parser = ArgumentParser(prog="required")
        parser.add_argument("-out", "--output", required=True)
        stderr = StringIO()

        with redirect_stderr(stderr), self.assertRaises(SystemExit):
            parser.parse_args([])

        self.assertIn("arguments are required: -o/--output", stderr.getvalue())
        self.assertNotRegex(stderr.getvalue(), r"(?<!\S)-out(?=\s|/|,|$)")

    def test_common_short_options_are_added_when_unambiguous(self):
        parser = ArgumentParser(prog="shortcuts")
        parser.add_argument("--center", choices=("none", "mass"))
        parser.add_argument("--selection")
        parser.add_argument("--delta-time", type=float)

        args = parser.parse_args(
            ["-c", "mass", "-se", "group 0", "-dt", "2"]
        )
        self.assertEqual(args.center, "mass")
        self.assertEqual(args.selection, "group 0")
        self.assertEqual(args.delta_time, 2.0)
        help_text = parser.format_help()
        self.assertIn("-c {none,mass}, --center {none,mass}", help_text)
        self.assertIn("-se SELECTION, --selection SELECTION", help_text)
        self.assertIn("-dt DELTA_TIME, --delta-time DELTA_TIME", help_text)

    def test_input_and_output_options_receive_stable_short_aliases(self):
        parser = ArgumentParser(prog="file-aliases")
        parser.add_argument("--input-structure", section="input")
        parser.add_argument("--output-conformation", section="output")
        parser.add_argument("--output-topology", section="output")

        args = parser.parse_args(
            ["-f", "input.pdb", "-o", "packed.pdb", "-op", "system.top"]
        )

        self.assertEqual(args.input_structure, "input.pdb")
        self.assertEqual(args.output_conformation, "packed.pdb")
        self.assertEqual(args.output_topology, "system.top")
        help_text = parser.format_help()
        self.assertIn("-f FILE, --input-structure FILE", help_text)
        self.assertIn("-o FILE, --output-conformation FILE", help_text)
        self.assertIn("-op FILE, --output-topology FILE", help_text)

    def test_common_short_option_does_not_override_an_existing_alias(self):
        parser = ArgumentParser(prog="shortcut-conflict")
        parser.add_argument("-c", "--count", type=int)
        parser.add_argument("--center")

        args = parser.parse_args(["-c", "3", "--center", "mass"])
        self.assertEqual(args.count, 3)
        self.assertEqual(args.center, "mass")
        center = next(action for action in parser._actions if action.dest == "center")
        self.assertNotIn("-c", center.option_strings)

    def test_all_command_modules_use_the_shared_parser(self):
        package_root = Path(__file__).resolve().parents[1] / "dropps"
        command_modules = [
            *sorted((package_root / "commands").glob("*.py")),
            *sorted((package_root / "analysis").glob("*.py")),
        ]

        for module in command_modules:
            if module.name.startswith("."):
                continue
            source = module.read_text(encoding="utf-8")
            if "ArgumentParser(" not in source:
                continue
            with self.subTest(module=module.name):
                self.assertIn(
                    "from dropps.share.argument_parser import",
                    source,
                )


if __name__ == "__main__":
    unittest.main()
