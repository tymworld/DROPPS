import unittest

import numpy as np

from dropps.analysis.polystat_core import (
    RunningMoments,
    calculate_molecular_shape,
    contour_distance_sums,
    internal_distance_sums,
    persistence_1e,
    tangent_correlation_sums,
)


class MolecularShapeTests(unittest.TestCase):
    def test_straight_chains_have_known_size_and_shape(self):
        coordinates = np.asarray(
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                [[5.0, 1.0, 0.0], [5.0, 2.0, 0.0], [5.0, 3.0, 0.0]],
            ]
        )

        result = calculate_molecular_shape(coordinates)

        np.testing.assert_allclose(result.end_to_end, 2.0)
        np.testing.assert_allclose(result.radius_gyration, np.sqrt(2.0 / 3.0))
        np.testing.assert_allclose(
            result.eigenvalues,
            [[2.0 / 3.0, 0.0, 0.0]] * 2,
            atol=1.0e-14,
        )
        np.testing.assert_allclose(result.relative_shape_anisotropy, 1.0)

    def test_independent_rotation_does_not_change_molecular_principal_values(self):
        molecule = np.asarray(
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        )
        rotation = np.asarray(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )
        coordinates = np.stack((molecule, molecule @ rotation))

        result = calculate_molecular_shape(coordinates)

        np.testing.assert_allclose(
            result.eigenvalues[0],
            result.eigenvalues[1],
            atol=1.0e-14,
        )
        self.assertAlmostEqual(
            result.relative_shape_anisotropy[0],
            result.relative_shape_anisotropy[1],
        )

    def test_mass_weighting_changes_the_gyration_tensor(self):
        coordinates = np.asarray(
            [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [4.0, 0.0, 0.0]]]
        )

        uniform = calculate_molecular_shape(coordinates)
        weighted = calculate_molecular_shape(coordinates, [100.0, 1.0, 1.0])

        self.assertLess(weighted.radius_gyration[0], uniform.radius_gyration[0])


class LagKernelTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(20260811)
        increments = rng.normal(size=(4, 31, 3))
        self.coordinates = np.concatenate(
            (
                np.zeros((4, 1, 3)),
                np.cumsum(increments, axis=1),
            ),
            axis=1,
        )

    def test_fft_internal_distances_match_direct_differences(self):
        lags = np.arange(1, self.coordinates.shape[1])

        direct_sums, direct_counts = internal_distance_sums(
            self.coordinates,
            lags,
            method="direct",
        )
        fft_sums, fft_counts = internal_distance_sums(
            self.coordinates,
            lags,
            method="fft",
        )

        np.testing.assert_array_equal(fft_counts, direct_counts)
        np.testing.assert_allclose(fft_sums, direct_sums, rtol=2.0e-13, atol=2.0e-12)

    def test_fft_tangent_correlations_match_direct_dot_products(self):
        lags = np.arange(self.coordinates.shape[1] - 1)

        direct_sums, direct_counts = tangent_correlation_sums(
            self.coordinates,
            lags,
            method="direct",
        )
        fft_sums, fft_counts = tangent_correlation_sums(
            self.coordinates,
            lags,
            method="fft",
        )

        np.testing.assert_array_equal(fft_counts, direct_counts)
        np.testing.assert_allclose(fft_sums, direct_sums, rtol=2.0e-13, atol=2.0e-12)

    def test_straight_chain_internal_and_contour_distances_are_exact(self):
        coordinates = np.asarray(
            [[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0], [1.5, 0.0, 0.0]]]
        )
        lags = np.asarray([1, 2, 3])

        squared_sums, counts = internal_distance_sums(coordinates, lags)
        contour_sums, contour_counts = contour_distance_sums(coordinates, lags)

        np.testing.assert_allclose(squared_sums / counts, (0.5 * lags) ** 2)
        np.testing.assert_allclose(contour_sums / contour_counts, 0.5 * lags)

    def test_straight_chain_tangent_midpoint_contours_follow_lag(self):
        coordinates = np.asarray(
            [[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0], [1.5, 0.0, 0.0]]]
        )
        lags = np.asarray([0, 1, 2])

        correlations, counts = tangent_correlation_sums(coordinates, lags)
        contour_sums, contour_counts = contour_distance_sums(
            coordinates,
            lags,
            tangent_midpoints=True,
        )

        np.testing.assert_allclose(correlations / counts, 1.0)
        np.testing.assert_allclose(contour_sums / contour_counts, 0.5 * lags)


class PersistenceTests(unittest.TestCase):
    def test_exponential_curve_has_exact_one_over_e_crossing(self):
        lags = np.asarray([0, 1, 2, 3, 4])
        correlations = np.exp(-lags / 2.5)
        contours = 0.4 * lags

        result = persistence_1e(correlations, lags, contours)

        self.assertEqual(result.status, "valid")
        self.assertAlmostEqual(result.steps, 2.5)
        self.assertAlmostEqual(result.contour_length, 1.0)

    def test_nonpositive_bracket_and_censored_curve_are_reported(self):
        nonpositive = persistence_1e([1.0, 0.7, -0.1], [0, 1, 2])
        censored = persistence_1e([1.0, 0.9, 0.8], [0, 1, 2])

        self.assertEqual(nonpositive.status, "nonpositive-bracket")
        self.assertTrue(np.isnan(nonpositive.steps))
        self.assertEqual(censored.status, "censored")
        self.assertTrue(np.isnan(censored.contour_length))


class RunningMomentsTests(unittest.TestCase):
    def test_block_merging_matches_one_pass_population_statistics(self):
        values = np.asarray([1.0, 2.0, 4.0, 8.0, 16.0])
        first = RunningMoments()
        first.update(values[:2])
        second = RunningMoments()
        second.update(values[2:])
        first.merge(second)

        summary = first.summary()

        self.assertEqual(summary.count, values.size)
        self.assertAlmostEqual(summary.mean, float(np.mean(values)))
        self.assertAlmostEqual(summary.std, float(np.std(values)))
        self.assertAlmostEqual(summary.rms, float(np.sqrt(np.mean(values**2))))


if __name__ == "__main__":
    unittest.main()
