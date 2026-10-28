import re
import unittest
from argparse import Namespace
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import dropps
from dropps import dps
from dropps.hp import hello_world
from dropps.hp.quotes import QUOTES


class DPSVersionTests(unittest.TestCase):
    def test_source_tree_version_matches_pyproject(self):
        pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
        match = re.search(
            r'^version\s*=\s*["\']([^"\']+)["\']',
            pyproject.read_text(encoding="utf-8"),
            re.MULTILINE,
        )

        self.assertIsNotNone(match)
        self.assertEqual(dropps.__version__, match.group(1))

    def test_version_option_prints_version_without_loading_commands(self):
        for option in ("-v", "--version"):
            stdout = StringIO()
            with patch.object(dps, "__version__", "1.2.3"), redirect_stdout(stdout):
                status = dps.main(["dps", option])

            self.assertEqual(stdout.getvalue(), "dps 1.2.3\n")
            self.assertEqual(status, 0)


class DPSTopLevelHelpTests(unittest.TestCase):
    def test_no_arguments_and_help_option_show_the_catalog_with_banner(self):
        outputs = []
        for argv in (["dps"], ["dps", "--help"]):
            stdout = StringIO()
            with redirect_stdout(stdout):
                status = dps.main(argv)
            self.assertEqual(status, 0)
            outputs.append(stdout.getvalue())

        for output in outputs:
            self.assertIn("DROPPS · DPS", output)
            self.assertIn("╭", output)
            self.assertIn("DROPPS command-line tools", output)
            self.assertIn("Choosing between similar commands:", output)

    def test_detailed_help_has_examples_and_banner(self):
        stdout = StringIO()
        with redirect_stdout(stdout):
            with self.assertRaises(SystemExit) as raised:
                dps.main(["dps", "mdrun", "-h"])

        self.assertEqual(raised.exception.code, 0)
        help_text = stdout.getvalue()
        self.assertIn("DROPPS · DPS", help_text)
        self.assertIn("usage: dps mdrun", help_text)
        self.assertIn("Examples:", help_text)


class DPSErrorHandlingTests(unittest.TestCase):
    def test_unknown_command_returns_two_and_suggests_a_match(self):
        stderr = StringIO()
        with redirect_stderr(stderr):
            status = dps.main(["dps", "mdru"])

        self.assertEqual(status, 2)
        error = stderr.getvalue()
        self.assertIn("Unknown command 'mdru'", error)
        self.assertIn("'mdrun'", error)
        self.assertIn("dps help commands", error)

    def test_unknown_help_target_returns_two(self):
        stderr = StringIO()
        with redirect_stderr(stderr):
            status = dps.main(["dps", "help", "contcat"])

        self.assertEqual(status, 2)
        self.assertIn("Unknown command 'contcat'", stderr.getvalue())

    def test_argument_errors_are_not_prefixed_by_the_run_banner(self):
        stdout = StringIO()
        stderr = StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as raised:
                dps.main(["dps", "mdrun"])

        self.assertEqual(raised.exception.code, 2)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("[ERROR] dps mdrun:", stderr.getvalue())
        self.assertNotIn("-deffnm", stderr.getvalue())

    def test_every_command_gets_a_standard_lifecycle_summary(self):
        from dropps.share.all_commands import all_commands

        def command(_args):
            print("## Processing frame 1.")
            print("WARNING: Example warning.")

        parsed = Namespace(input="run.xtc", output="result.xvg", threshold=3)
        stdout = StringIO()
        stderr = StringIO()
        with (
            patch.object(all_commands, "getargs", return_value=lambda _args: parsed),
            patch.object(all_commands, "command", return_value=command),
            patch.object(dps, "_print_banner"),
            redirect_stdout(stdout),
            redirect_stderr(stderr),
        ):
            status = dps.main(["dps", "example"])

        self.assertEqual(status, 0)
        self.assertIn("[INFO] Starting command 'example'.", stdout.getvalue())
        self.assertIn("[INFO] Inputs: input=run.xtc", stdout.getvalue())
        self.assertIn("[INFO] Outputs: output=result.xvg", stdout.getvalue())
        self.assertIn("[INFO] Parameters: threshold=3", stdout.getvalue())
        self.assertIn("[INFO] Processing frame 1.", stdout.getvalue())
        self.assertIn("[OK] Command 'example' completed successfully.", stdout.getvalue())
        self.assertIn("[WARNING] Example warning.", stderr.getvalue())

    def test_debug_option_works_before_or_after_the_command(self):
        from dropps.share.all_commands import all_commands

        for argv in (
            ["dps", "--debug", "example"],
            ["dps", "example", "--debug"],
            ["dps", "-db", "example"],
            ["dps", "example", "-db"],
        ):
            stderr = StringIO()
            parsed = Namespace(output="result.dat")
            with (
                patch.object(all_commands, "getargs", return_value=lambda args: parsed),
                patch.object(all_commands, "command", return_value=lambda args: None),
                patch.object(dps, "_print_banner"),
                redirect_stdout(StringIO()),
                redirect_stderr(stderr),
            ):
                status = dps.main(argv)

            self.assertEqual(status, 0)
            diagnostic = stderr.getvalue()
            self.assertIn("[DEBUG] Debug diagnostics enabled.", diagnostic)
            self.assertIn("[DEBUG] Parsed arguments:", diagnostic)
            self.assertIn("[DEBUG] Command completed: status=0", diagnostic)

    def test_debug_failure_includes_a_traceback(self):
        from dropps.share.all_commands import all_commands

        def command(_args):
            raise ValueError("invalid frame")

        stderr = StringIO()
        with (
            patch.object(
                all_commands,
                "getargs",
                return_value=lambda _args: Namespace(input="run.xtc"),
            ),
            patch.object(all_commands, "command", return_value=command),
            patch.object(dps, "_print_banner"),
            redirect_stdout(StringIO()),
            redirect_stderr(stderr),
        ):
            status = dps.main(["dps", "example", "--debug"])

        self.assertEqual(status, 1)
        diagnostic = stderr.getvalue()
        self.assertIn("[ERROR] ValueError: invalid frame", diagnostic)
        self.assertIn("Traceback (most recent call last):", diagnostic)
        self.assertIn("raise ValueError(\"invalid frame\")", diagnostic)

    def test_legacy_bare_quit_is_normalized_to_a_failure_status(self):
        from dropps.share.all_commands import all_commands

        def bare_quit(_args):
            raise SystemExit()

        def bare_quit_parser(_args):
            raise SystemExit()

        with (
            patch.object(all_commands, "getargs", return_value=lambda _args: object()),
            patch.object(all_commands, "command", return_value=bare_quit),
            patch.object(dps, "_print_banner"),
            redirect_stdout(StringIO()),
            redirect_stderr(StringIO()),
        ):
            status = dps.main(["dps", "mdrun"])

        self.assertEqual(status, 1)

        with patch.object(
            all_commands,
            "getargs",
            return_value=bare_quit_parser,
        ):
            status = dps.main(["dps", "mdrun"])

        self.assertEqual(status, 1)


class DPSBannerTests(unittest.TestCase):
    def test_banner_prints_a_random_quote_with_its_author(self):
        stdout = StringIO()
        selected_quote = ("Testing", "Test quote.", "Test Author")

        with (
            patch.object(
                hello_world,
                "choice",
                side_effect=("DROPPS quote", selected_quote),
            ),
            redirect_stdout(stdout),
        ):
            hello_world.print_hello("dps", "help")

        self.assertIn(
            'DROPPS quote [Testing]: "Test quote." — Test Author',
            stdout.getvalue(),
        )
        self.assertIn("dps help commands", stdout.getvalue())
        self.assertIn("List commands by task", stdout.getvalue())
        self.assertIn("Developed by Yiming Tang · Fudan University", stdout.getvalue())
        self.assertIn("ymtang@fudan.edu.cn", stdout.getvalue())

    def test_quote_catalog_is_large_unique_and_english(self):
        self.assertGreaterEqual(len(QUOTES), 100)
        quote_texts = [quote for _, quote, _ in QUOTES]
        self.assertEqual(len(quote_texts), len(set(quote_texts)))
        self.assertTrue(all(quote.isascii() for quote in quote_texts))


if __name__ == "__main__":
    unittest.main()
