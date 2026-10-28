import os
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from unittest.mock import patch

from dropps.share import console


class ConsoleOutputTests(unittest.TestCase):
    def setUp(self):
        console.configure(debug=False)

    def test_redirected_output_has_stable_plain_prefixes(self):
        output = StringIO()
        console.emit("info", "Ready.", stream=output)

        self.assertEqual(output.getvalue(), "[INFO] Ready.\n")
        self.assertNotIn("\033[", output.getvalue())

    def test_color_levels_are_visually_distinct_when_forced(self):
        colors = {}
        console.configure(debug=True)
        with patch.dict(os.environ, {"DROPPS_COLOR": "always"}, clear=True):
            for level in ("debug", "info", "success", "warning", "error"):
                output = StringIO()
                console.emit(level, "message", stream=output)
                colors[level] = output.getvalue().split("m", 1)[0]

        self.assertEqual(len(set(colors.values())), 5)
        self.assertTrue(all("\033[" in value for value in colors.values()))

    def test_no_color_overrides_forced_color(self):
        output = StringIO()
        with patch.dict(
            os.environ,
            {"DROPPS_COLOR": "always", "NO_COLOR": "1"},
            clear=True,
        ):
            console.emit("error", "Stopped.", stream=output)

        self.assertEqual(output.getvalue(), "[ERROR] Stopped.\n")

    def test_legacy_status_lines_are_normalized_and_routed(self):
        stdout = StringIO()
        stderr = StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            with console.normalized_output(normalize_plain=True):
                print("## Reading trajectory.")
                print("plain progress")
                print("WARNING: Check the input.")
                print("ERROR: Cannot continue.")

        self.assertEqual(
            stdout.getvalue(),
            "[INFO] Reading trajectory.\n[INFO] plain progress\n",
        )
        self.assertEqual(
            stderr.getvalue(),
            "[WARNING] Check the input.\n[ERROR] Cannot continue.\n",
        )

    def test_already_standardized_messages_are_not_double_prefixed(self):
        stdout = StringIO()
        with redirect_stdout(stdout):
            with console.normalized_output(normalize_plain=True):
                console.info("One prefix only.")

        self.assertEqual(stdout.getvalue(), "[INFO] One prefix only.\n")

    def test_plain_stderr_is_not_misclassified_as_a_warning(self):
        stderr = StringIO()
        with redirect_stderr(stderr):
            with console.normalized_output(normalize_plain=True):
                print("library diagnostic", file=os.sys.stderr)

        self.assertEqual(stderr.getvalue(), "[INFO] library diagnostic\n")

    def test_carriage_return_progress_is_left_untagged(self):
        stderr = StringIO()
        with redirect_stderr(stderr):
            with console.normalized_output(normalize_plain=True):
                os.sys.stderr.write("\rFrames: 50%|#####     | 5/10")

        progress = stderr.getvalue()
        self.assertEqual(progress, "\rFrames: 50%|#####     | 5/10")
        self.assertNotIn("[WARNING]", progress)


if __name__ == "__main__":
    unittest.main()
