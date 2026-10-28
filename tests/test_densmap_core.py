import unittest

import numpy as np

from dropps.analysis.densmap_core import (
    density_grid_shape,
    histogram_density_map,
    periodic_plane_center,
    plane_axes,
    slice_mask_and_thickness,
)


class DensmapGeometryTests(unittest.TestCase):
    def test_average_axis_defines_the_retained_plane(self):
        self.assertEqual(plane_axes("z"), (0, 1))
        self.assertEqual(plane_axes("y"), (0, 2))
        self.assertEqual(plane_axes("x"), (1, 2))

    def test_grid_shape_uses_gromacs_style_nearest_rounding_and_overrides(self):
        self.assertEqual(density_grid_shape([4.0, 5.0], 0.6), (7, 8))
        self.assertEqual(
            density_grid_shape([4.0, 5.0], 0.6, bins_first=3),
            (3, 8),
        )

    def test_periodic_center_resolves_a_droplet_across_both_boundaries(self):
        fractions = np.asarray(
            [
                [0.98, 0.99],
                [0.02, 0.01],
                [0.99, 0.02],
                [0.01, 0.98],
            ]
        )

        center, strength = periodic_plane_center(fractions)

        np.testing.assert_allclose(
            np.minimum(center, 1.0 - center),
            [0.0, 0.0],
            atol=1.0e-12,
        )
        self.assertTrue(np.all(strength > 0.99))

    def test_slice_bounds_are_clipped_to_the_physical_box(self):
        mask, thickness = slice_mask_and_thickness(
            [0.0, 0.25, 0.5, 0.75],
            4.0,
            -1.0,
            2.1,
        )

        np.testing.assert_array_equal(mask, [True, True, True, False])
        self.assertAlmostEqual(thickness, 2.1)


class DensmapNormalizationTests(unittest.TestCase):
    def setUp(self):
        self.fractions = np.asarray(
            [
                [0.1, 0.1],
                [0.6, 0.1],
                [0.1, 0.6],
                [0.6, 0.6],
            ]
        )
        self.weights = np.asarray([1.0, 2.0, 3.0, 4.0])

    def test_volume_density_integrates_to_total_weight(self):
        density = histogram_density_map(
            self.fractions,
            self.weights,
            plane_lengths_nm=[2.0, 4.0],
            averaging_thickness_nm=5.0,
            bins=(2, 2),
            normalization="volume",
        )

        cell_volume = (2.0 / 2) * (4.0 / 2) * 5.0
        self.assertAlmostEqual(float(density.sum() * cell_volume), 10.0)

    def test_area_density_integrates_to_total_weight(self):
        density = histogram_density_map(
            self.fractions,
            self.weights,
            plane_lengths_nm=[2.0, 4.0],
            averaging_thickness_nm=5.0,
            bins=(2, 2),
            normalization="area",
        )

        cell_area = (2.0 / 2) * (4.0 / 2)
        self.assertAlmostEqual(float(density.sum() * cell_area), 10.0)

    def test_count_map_conserves_signed_charge(self):
        charge = histogram_density_map(
            self.fractions,
            [1.0, -2.0, 3.0, -4.0],
            plane_lengths_nm=[2.0, 4.0],
            averaging_thickness_nm=5.0,
            bins=(2, 2),
            normalization="count",
        )

        self.assertAlmostEqual(float(charge.sum()), -2.0)

    def test_particles_outside_the_primary_cell_are_wrapped(self):
        density = histogram_density_map(
            [[-0.1, 1.1]],
            [2.0],
            plane_lengths_nm=[2.0, 2.0],
            averaging_thickness_nm=1.0,
            bins=(2, 2),
            normalization="count",
        )

        self.assertEqual(density[1, 0], 2.0)
        self.assertEqual(np.count_nonzero(density), 1)


if __name__ == "__main__":
    unittest.main()
