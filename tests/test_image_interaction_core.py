import unittest

import numpy as np

from dropps.analysis.image_interaction_core import (
    analyze_image_edges,
    build_unwrap_plan,
    canonicalize_image_contacts,
)


class ImageInteractionGraphTests(unittest.TestCase):
    def test_single_cross_boundary_edge_is_not_a_self_image_path(self):
        result = analyze_image_edges(2, [[0, 1, 1, 0, 0]])

        np.testing.assert_array_equal(result.affected, [False, False])
        self.assertEqual(result.wrapped_components, 0)

    def test_direct_self_image_contact_is_detected(self):
        result = analyze_image_edges(2, [[0, 0, 1, 0, 0]])

        np.testing.assert_array_equal(result.affected, [True, False])
        np.testing.assert_array_equal(result.direct, [True, False])
        self.assertEqual(result.wrapped_components, 1)

    def test_two_molecule_indirect_path_is_detected(self):
        edges = [
            [0, 1, 0, 0, 0],
            [0, 1, 1, 0, 0],
        ]
        result = analyze_image_edges(2, edges)

        np.testing.assert_array_equal(result.affected, [True, True])
        np.testing.assert_array_equal(result.direct, [False, False])
        self.assertEqual(result.wrapped_components, 1)

    def test_three_molecule_path_and_attached_molecule_are_detected(self):
        edges = [
            [0, 1, 0, 0, 0],
            [1, 2, 0, 0, 0],
            [2, 0, 1, 0, 0],
            [2, 3, 0, 0, 0],
        ]
        result = analyze_image_edges(4, edges)

        # Molecule 3 reaches the winding cycle, traverses it, then follows the
        # translated reverse branch and is therefore connected to its image too.
        np.testing.assert_array_equal(result.affected, [True, True, True, True])
        self.assertEqual(result.wrapped_components, 1)

    def test_zero_winding_cycle_is_not_an_image_interaction(self):
        edges = [
            [0, 1, 1, 0, 0],
            [1, 2, 0, 0, 0],
            [2, 0, -1, 0, 0],
        ]
        result = analyze_image_edges(3, edges)

        np.testing.assert_array_equal(result.affected, [False, False, False])

    def test_contact_threshold_is_applied_to_each_image_specific_edge(self):
        molecule_a = [1, 0, 0, 0, 0, 0]
        molecule_b = [0, 1, 1, 1, 1, 0]
        shifts = [
            [-1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],  # zero-shift self contact is discarded
        ]

        edges, counts = canonicalize_image_contacts(
            molecule_a, molecule_b, shifts, minimum_contacts=2
        )

        np.testing.assert_array_equal(
            edges,
            [
                [0, 1, 0, 0, 0],
                [0, 1, 1, 0, 0],
            ],
        )
        np.testing.assert_array_equal(counts, [2, 3])


class ImageInteractionUnwrapPlanTests(unittest.TestCase):
    def test_plan_uses_complete_molecules_for_partial_contact_selection(self):
        plan = build_unwrap_plan(
            molecule_atom_indices=([0, 1, 2], [3]),
            selected_atom_indices=[0, 2, 3],
            bonds=[[0, 1], [1, 2]],
        )

        np.testing.assert_array_equal(plan.atom_indices, [0, 1, 2, 3])
        np.testing.assert_array_equal(plan.selected_local_indices, [0, 2, 3])
        np.testing.assert_array_equal(plan.selected_molecule_ids, [0, 0, 1])
        self.assertEqual(len(plan.levels), 2)

    def test_disconnected_multibead_molecule_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "disconnected"):
            build_unwrap_plan(
                molecule_atom_indices=([0, 1],),
                selected_atom_indices=[0, 1],
                bonds=[],
            )


if __name__ == "__main__":
    unittest.main()
