from __future__ import annotations

import contextlib
import io
import unittest

from dropps import __version__
from dropps.dps import main
from dropps.share.all_commands import all_commands


DOCUMENTED_COMMANDS = {
    "addangle",
    "angle",
    "assembly",
    "check",
    "contact",
    "density",
    "editconf",
    "genelastic",
    "genmesh",
    "grompp",
    "gsd2xtc",
    "gyrate",
    "idist",
    "make_ndx",
    "mdrun",
    "modifyres",
    "msd",
    "odist",
    "pdb2dps",
    "trjconv",
}

UNDOCUMENTED_COMMANDS = {
    "convert-tpr",
    "cstat",
    "energy",
    "exchange",
    "help",
    "pdb2bond",
    "rerun",
    "rmsd",
}

REMOVED_COMMANDS = {
    "coexistence",
    "contact.ng",
    "pbcontact",
    "phase-msd",
    "surftension",
    "timecorr",
    "viscosity-gk",
}


class CommandContractTests(unittest.TestCase):
    def test_public_release_version(self):
        self.assertEqual(__version__, "1.0")

    def test_registry_matches_manuscript_audit(self):
        registered = set(all_commands.names())
        self.assertEqual(registered, DOCUMENTED_COMMANDS | UNDOCUMENTED_COMMANDS)
        self.assertEqual(len(registered), 28)
        self.assertTrue(registered.isdisjoint(REMOVED_COMMANDS))

    def test_removed_commands_are_rejected(self):
        for command in sorted(REMOVED_COMMANDS):
            with self.subTest(command=command):
                stderr = io.StringIO()
                with contextlib.redirect_stderr(stderr):
                    self.assertEqual(main(["dps", command]), 2)
                self.assertIn("unknown command", stderr.getvalue())

    def test_every_registered_command_has_working_help(self):
        for command in all_commands.names():
            with self.subTest(command=command):
                output = io.StringIO()
                with (
                    contextlib.redirect_stdout(output),
                    contextlib.redirect_stderr(output),
                ):
                    with self.assertRaises(SystemExit) as result:
                        all_commands.getargs(command)(["--help"])
                self.assertEqual(result.exception.code, 0)
                self.assertIn("usage:", output.getvalue())

    def test_version_and_unknown_command_exit_codes(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            self.assertEqual(main(["dps", "--version"]), 0)
        self.assertEqual(stdout.getvalue().strip(), f"dps {__version__}")

        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            self.assertEqual(main(["dps", "gromp"]), 2)
        self.assertIn("grompp", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
