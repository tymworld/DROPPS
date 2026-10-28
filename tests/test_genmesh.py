import unittest
from types import SimpleNamespace

from dropps.commands.genmesh import (
    _chain_id_for_molecule,
    _forcefield_function_types,
)


class GenmeshValidationTests(unittest.TestCase):
    def test_chain_ids_cycle_without_a_molecule_count_limit(self):
        labels = [_chain_id_for_molecule(index) for index in range(1_000)]

        self.assertTrue(all(len(label) == 1 for label in labels))
        self.assertTrue(
            all(current != following for current, following in zip(labels, labels[1:]))
        )
        self.assertEqual(labels[702], "A")

    def test_coulomb_function_types_are_checked_independently(self):
        topologies = [
            SimpleNamespace(
                function_type_LJ="Ashbaugh-Hatch",
                function_type_Coulomb="Debye-Huckel",
            ),
            SimpleNamespace(
                function_type_LJ="Ashbaugh-Hatch",
                function_type_Coulomb="Reaction-Field",
            ),
        ]

        with self.assertRaisesRegex(ValueError, "Coulomb"):
            _forcefield_function_types(topologies)

    def test_common_forcefield_types_are_returned(self):
        topology = SimpleNamespace(
            function_type_LJ="Wang-Frenkel",
            function_type_Coulomb="Debye-Huckel",
        )

        self.assertEqual(
            _forcefield_function_types([topology, topology]),
            ("Wang-Frenkel", "Debye-Huckel"),
        )


if __name__ == "__main__":
    unittest.main()
