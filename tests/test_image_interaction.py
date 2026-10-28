import unittest

import numpy as np

from dropps.analysis.image_interaction import _analyze_frame
from dropps.analysis.image_interaction_core import build_unwrap_plan


BOX = np.array([10.0, 10.0, 10.0, 90.0, 90.0, 90.0])


class ImageInteractionFrameTests(unittest.TestCase):
    def test_compact_molecule_crossing_box_boundary_is_not_a_false_positive(self):
        plan = build_unwrap_plan(([0, 1],), [0, 1], [[0, 1]])
        positions = np.array(
            [
                [9.5, 1.0, 1.0],
                [0.5, 1.0, 1.0],
            ]
        )

        result, edges, _ = _analyze_frame(
            positions, BOX, plan, cutoff_angstrom=1.5, minimum_contacts=1
        )

        np.testing.assert_array_equal(result.affected, [False])
        self.assertEqual(edges.shape[0], 0)

    def test_extended_molecule_contacting_own_image_is_detected(self):
        plan = build_unwrap_plan(
            ([0, 1, 2, 3],),
            [0, 1, 2, 3],
            [[0, 1], [1, 2], [2, 3]],
        )
        positions = np.array(
            [
                [1.0, 1.0, 1.0],
                [4.0, 1.0, 1.0],
                [7.0, 1.0, 1.0],
                [9.0, 1.0, 1.0],
            ]
        )

        result, _, _ = _analyze_frame(
            positions, BOX, plan, cutoff_angstrom=2.1, minimum_contacts=1
        )

        np.testing.assert_array_equal(result.affected, [True])
        np.testing.assert_array_equal(result.direct, [True])

    def test_three_molecule_winding_contact_path_is_detected(self):
        plan = build_unwrap_plan(([0], [1], [2]), [0, 1, 2], [])
        positions = np.array(
            [
                [0.0, 1.0, 1.0],
                [3.3, 1.0, 1.0],
                [6.6, 1.0, 1.0],
            ]
        )

        result, edges, _ = _analyze_frame(
            positions, BOX, plan, cutoff_angstrom=3.5, minimum_contacts=1
        )

        self.assertEqual(edges.shape[0], 3)
        np.testing.assert_array_equal(result.affected, [True, True, True])
        np.testing.assert_array_equal(result.direct, [False, False, False])

    def test_cutoff_requiring_ambiguous_minimum_images_is_rejected(self):
        plan = build_unwrap_plan(([0],), [0], [])

        with self.assertRaisesRegex(ValueError, "less than half"):
            _analyze_frame(
                np.array([[1.0, 1.0, 1.0]]),
                BOX,
                plan,
                cutoff_angstrom=5.0,
                minimum_contacts=1,
            )


if __name__ == "__main__":
    unittest.main()
