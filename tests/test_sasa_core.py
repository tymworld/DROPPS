import unittest
from types import SimpleNamespace

import numpy as np

from dropps.analysis.sasa_core import (
    fibonacci_sphere_points,
    forcefield_radii_nm,
    orthorhombic_periodic_neighbor_vectors,
    shrake_rupley_areas,
    shrake_rupley_nonperiodic,
)


class SASACoreGeometryTests(unittest.TestCase):
    def test_fibonacci_points_are_unit_length_and_antipodal(self):
        points = fibonacci_sphere_points(960)
        self.assertEqual(points.shape, (960, 3))
        np.testing.assert_allclose(np.linalg.norm(points, axis=1), 1.0, atol=1e-12)
        np.testing.assert_allclose(points[:480], -points[480:], atol=1e-15)

    def test_single_sphere_matches_analytic_area(self):
        bead_radius = 0.30
        probe_radius = 0.14
        actual = shrake_rupley_nonperiodic(
            [[0.0, 0.0, 0.0]],
            [bead_radius],
            probe_radius=probe_radius,
            sphere_point_count=960,
        )
        expected = 4.0 * np.pi * (bead_radius + probe_radius) ** 2
        self.assertAlmostEqual(actual[0], expected, places=12)

    def test_two_overlapping_equal_spheres_converge_to_cap_area(self):
        radius = 0.44
        separation = 0.40
        actual = shrake_rupley_nonperiodic(
            [[0.0, 0.0, 0.0], [separation, 0.0, 0.0]],
            [radius, radius],
            probe_radius=0.0,
            sphere_point_count=20000,
        )
        expected_total = 4.0 * np.pi * radius**2 + 2.0 * np.pi * radius * separation
        self.assertAlmostEqual(actual[0], actual[1], places=12)
        self.assertAlmostEqual(actual.sum() / expected_total, 1.0, delta=0.003)

    def test_periodic_search_reduces_coordinates_many_boxes_away(self):
        radii = np.asarray([0.44, 0.44])
        points = fibonacci_sphere_points(960)
        near = np.asarray([[0.1, 0.0, 0.0], [0.5, 0.0, 0.0]])
        far = np.asarray([[1000.1, 0.0, 0.0], [-999.5, 0.0, 0.0]])

        near_neighbors = orthorhombic_periodic_neighbor_vectors(
            near, radii, [10.0, 10.0, 10.0]
        )
        far_neighbors = orthorhombic_periodic_neighbor_vectors(
            far, radii, [10.0, 10.0, 10.0]
        )
        near_areas = shrake_rupley_areas(radii, *near_neighbors, points)
        far_areas = shrake_rupley_areas(radii, *far_neighbors, points)
        np.testing.assert_allclose(far_areas, near_areas, atol=1e-12)
        self.assertAlmostEqual(far_areas[0], far_areas[1], places=12)

    def test_ambiguous_periodic_overlap_cutoff_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "half the shortest"):
            orthorhombic_periodic_neighbor_vectors(
                [[0.0, 0.0, 0.0]],
                [0.6],
                [2.0, 2.0, 2.0],
            )


class SASARadiusTests(unittest.TestCase):
    def test_resolves_ashbaugh_hatch_and_wang_frenkel_self_sigmas(self):
        ah = SimpleNamespace(
            type2sigma={"A": 0.50, "B": 0.70},
            atoms=[SimpleNamespace(abbr="A"), SimpleNamespace(abbr="B")],
        )
        wf = SimpleNamespace(
            types2sigma={"C-C": 0.80},
            atoms=[SimpleNamespace(abbr="C")],
        )
        radii = forcefield_radii_nm([ah, wf], convention="sigma-half")
        np.testing.assert_allclose(radii, [0.25, 0.35, 0.40])

        minimum_radii = forcefield_radii_nm(
            [ah],
            convention="lj-minimum-half",
            scale=1.2,
        )
        np.testing.assert_allclose(
            minimum_radii,
            np.asarray([0.50, 0.70]) * 0.5 * 2 ** (1 / 6) * 1.2,
        )

    def test_missing_self_sigma_is_reported(self):
        topology = SimpleNamespace(
            types2sigma={"A-B": 0.6},
            atoms=[SimpleNamespace(abbr="A")],
        )
        with self.assertRaisesRegex(ValueError, "no self sigma"):
            forcefield_radii_nm([topology])


if __name__ == "__main__":
    unittest.main()
