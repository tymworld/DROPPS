import unittest

import numpy as np

from dropps.analysis.rdf_core import (
    RdfAccumulator,
    RdfPositionSet,
    box_measure,
    build_radial_grid,
    frame_pair_histogram,
    frame_surface_histogram,
    minimum_image_vectors,
    safe_periodic_radius,
    topology_exclusion_keys,
)


def _positions(coordinates, atoms, chains):
    return RdfPositionSet(
        np.asarray(coordinates, dtype=float),
        np.asarray(atoms, dtype=int),
        tuple(frozenset((chain,)) for chain in chains),
    )


class RadialGridTests(unittest.TestCase):
    def test_gromacs_integer_centers_and_half_width_first_bin(self):
        grid = build_radial_grid(0.2, 1.0, "3d")

        np.testing.assert_allclose(grid.radii, [0.0, 0.2, 0.4, 0.6, 0.8])
        np.testing.assert_allclose(grid.edges, [0.0, 0.1, 0.3, 0.5, 0.7, 0.9])
        self.assertAlmostEqual(grid.actual_rmax, 0.9)
        self.assertAlmostEqual(
            grid.shell_measures.sum(),
            4.0 * np.pi * 0.9**3 / 3.0,
        )
        np.testing.assert_allclose(grid.cumulative_radii, grid.edges[1:])

    def test_xy_uses_annular_area(self):
        grid = build_radial_grid(0.1, 0.46, "xy")

        self.assertAlmostEqual(grid.actual_rmax, 0.45)
        self.assertAlmostEqual(grid.shell_measures.sum(), np.pi * 0.45**2)


class PeriodicGeometryTests(unittest.TestCase):
    def test_orthorhombic_minimum_image_and_safe_radius(self):
        dimensions = np.asarray([2.0, 3.0, 4.0, 90.0, 90.0, 90.0])
        vectors = np.asarray([[1.1, 0.0, 0.0], [0.0, 1.6, 2.1]])

        minimum = minimum_image_vectors(vectors, dimensions, "3d")

        np.testing.assert_allclose(minimum[0], [-0.9, 0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(minimum[1], [0.0, -1.4, -1.9], atol=1e-12)
        self.assertAlmostEqual(safe_periodic_radius(dimensions), 1.0)
        self.assertAlmostEqual(box_measure(dimensions), 24.0)

    def test_xy_ignores_z_and_wraps_skew_plane(self):
        dimensions = np.asarray([2.0, 2.0, 5.0, 90.0, 90.0, 60.0])
        vector = np.asarray([[1.5, np.sqrt(3.0), 20.0]])

        minimum = minimum_image_vectors(vector, dimensions, "xy")

        np.testing.assert_allclose(minimum, [[0.5, 0.0, 0.0]], atol=1e-12)
        self.assertAlmostEqual(box_measure(dimensions, "xy"), 2.0 * np.sqrt(3.0))


class PairHistogramTests(unittest.TestCase):
    def setUp(self):
        self.grid = build_radial_grid(0.2, 1.0)
        self.dimensions = np.asarray([2.0, 2.0, 2.0, 90.0, 90.0, 90.0])

    def test_self_overlap_is_removed_by_strict_zero_cut(self):
        positions = _positions(
            [[0.0, 0.0, 0.0], [0.35, 0.0, 0.0]],
            [0, 1],
            [0, 1],
        )

        histogram = frame_pair_histogram(
            positions,
            positions,
            self.grid,
            self.dimensions,
            pbc=True,
        )

        np.testing.assert_array_equal(histogram, [0, 0, 2, 0, 0])

    def test_topology_and_same_molecule_exclusions(self):
        reference = _positions(
            [[0.0, 0.0, 0.0], [0.6, 0.0, 0.0]],
            [0, 2],
            [0, 1],
        )
        target = _positions(
            [[0.2, 0.0, 0.0], [0.8, 0.0, 0.0]],
            [1, 3],
            [0, 1],
        )
        keys = topology_exclusion_keys([[0, 1], [2, 3]], 4)

        topology_histogram = frame_pair_histogram(
            reference,
            target,
            self.grid,
            self.dimensions,
            exclusion="topology",
            topology_keys=keys,
            atom_count=4,
        )
        molecule_histogram = frame_pair_histogram(
            reference,
            target,
            self.grid,
            self.dimensions,
            exclusion="same-molecule",
        )

        np.testing.assert_array_equal(topology_histogram, molecule_histogram)
        self.assertEqual(int(topology_histogram.sum()), 2)

    def test_xy_pair_search_uses_projected_periodic_lattice(self):
        dimensions = np.asarray([2.0, 2.0, 5.0, 90.0, 90.0, 60.0])
        reference = _positions([[0.0, 0.0, 0.0]], [0], [0])
        target = _positions([[1.55, np.sqrt(3.0), 20.0]], [1], [1])
        grid = build_radial_grid(0.2, 1.0, "xy")

        histogram = frame_pair_histogram(
            reference,
            target,
            grid,
            dimensions,
            geometry="xy",
        )

        self.assertEqual(int(histogram.sum()), 1)
        self.assertEqual(histogram[3], 1)  # projected minimum image is 0.55 nm

    def test_surface_uses_one_closest_atom_per_reference_entity(self):
        reference = _positions(
            [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.7, 0.0, 0.0]],
            [0, 1, 2],
            [0, 0, 1],
        )
        target = _positions([[0.25, 0.0, 0.0]], [3], [2])

        histogram = frame_surface_histogram(
            reference,
            [0, 0, 1],
            target,
            self.grid,
            self.dimensions,
        )

        self.assertEqual(int(histogram.sum()), 2)
        self.assertEqual(histogram[1], 1)  # 0.15 nm to surface group 0
        self.assertEqual(histogram[2], 1)  # 0.45 nm to surface group 1


class NormalizationTests(unittest.TestCase):
    def test_all_normalizations_and_cumulative_use_raw_counts(self):
        grid = build_radial_grid(0.2, 0.7)
        accumulator = RdfAccumulator(1, grid)
        histogram = np.asarray([[2.0, 4.0, 6.0, 8.0]])
        accumulator.add_frame(histogram, 2, [4], 8)
        accumulator.add_frame(histogram, 2, [4], 4)

        expected_none = histogram / 2.0 / 0.2
        expected_number_density = histogram / 2.0 / grid.shell_measures
        mean_density = 0.5 * (4.0 / 8.0 + 4.0 / 4.0)

        np.testing.assert_allclose(accumulator.normalized("none"), expected_none)
        np.testing.assert_allclose(
            accumulator.normalized("number-density"),
            expected_number_density,
        )
        np.testing.assert_allclose(
            accumulator.normalized("rdf"),
            expected_number_density / mean_density,
        )
        np.testing.assert_allclose(
            accumulator.cumulative(),
            np.cumsum(histogram, axis=1) / 2.0,
        )


if __name__ == "__main__":
    unittest.main()
