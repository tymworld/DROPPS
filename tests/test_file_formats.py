from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from dropps.fileio.pdb_reader import atom_id_to_written_id, written_id_to_atom_id
from dropps.share.forcefield import forcefield_list, forcefields_files, getff
from dropps.share.indexing import parse_number_range, parse_string_range
from dropps.share.parameters import getparameter


ROOT = Path(__file__).resolve().parents[1]


class FileFormatTests(unittest.TestCase):
    def test_all_bundled_force_fields_parse(self):
        self.assertEqual(
            forcefield_list,
            ["CALVADOS2", "HPS", "HPSRNA", "HPST", "MPiPi", "MPiPi_PTM"],
        )
        with contextlib.redirect_stdout(io.StringIO()):
            parsed = [getff(path) for path in forcefields_files]
        self.assertTrue(all(forcefield.abbr for forcefield in parsed))
        self.assertTrue(
            all(not path.name.startswith("._") for path in forcefields_files)
        )

    def test_mdp_template_and_validation_errors(self):
        parameters = getparameter(ROOT / "tests" / "data" / "smoke.mdp")
        self.assertEqual(parameters["integrator"], "Langevin")
        self.assertEqual(parameters["nsteps"], 6)
        self.assertFalse(parameters["pcoulp"])

        with tempfile.TemporaryDirectory() as directory:
            bad = Path(directory) / "bad.mdp"
            bad.write_text("nsteps = 1\nnsteps = 2\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate MDP parameter"):
                getparameter(bad)

    def test_hybrid36_pdb_serial_round_trip(self):
        maximum = 100_000 + 2 * 26 * 36**4 - 1
        for serial in (0, 99_999, 100_000, 43_770_015, maximum):
            with self.subTest(serial=serial):
                self.assertEqual(
                    written_id_to_atom_id(atom_id_to_written_id(serial)),
                    serial,
                )

    def test_index_range_parsers_are_strict(self):
        self.assertEqual(parse_number_range("-2--1, 3, 5-7"), [-2, -1, 3, 5, 6, 7])
        self.assertEqual(parse_string_range("A, B,C"), ["A", "B", "C"])
        with self.assertRaisesRegex(ValueError, "Descending range"):
            parse_number_range("5-3")
        with self.assertRaisesRegex(ValueError, "Empty item"):
            parse_number_range("1,,2")


if __name__ == "__main__":
    unittest.main()
