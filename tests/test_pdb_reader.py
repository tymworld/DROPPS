import unittest

from dropps.fileio.pdb_reader import (
    atom_id_to_written_id,
    conect_lines_from_bond_dict,
    written_id_to_atom_id,
)


class PDBHybrid36Tests(unittest.TestCase):
    def test_serials_round_trip_across_decimal_and_hybrid36_ranges(self):
        upper_block = 26 * 36**4
        values = [
            0,
            99_999,
            100_000,
            100_001,
            131_072,
            100_000 + upper_block - 1,
            100_000 + upper_block,
            100_000 + 2 * upper_block - 1,
        ]

        for value in values:
            with self.subTest(value=value):
                encoded = atom_id_to_written_id(value)
                self.assertEqual(len(encoded), 5)
                self.assertEqual(written_id_to_atom_id(encoded), value)

        self.assertEqual(atom_id_to_written_id(100_000), "A0000")
        self.assertEqual(atom_id_to_written_id(100_000 + upper_block), "a0000")

    def test_serial_overflow_is_rejected_instead_of_truncated(self):
        maximum = 100_000 + 2 * 26 * 36**4 - 1
        with self.assertRaisesRegex(ValueError, "exceeds"):
            atom_id_to_written_id(maximum + 1)

    def test_conect_records_use_the_same_hybrid36_encoding(self):
        lines = conect_lines_from_bond_dict({99_999: [100_000]})
        self.assertEqual(lines, ["CONECTA0000A0001"])


if __name__ == "__main__":
    unittest.main()
