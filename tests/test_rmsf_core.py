import unittest

import numpy as np

from dropps.analysis.rmsf_core import (
    CoordinateMoments,
    aggregate_site_values,
    b_factors_from_covariance,
    rmsf_from_covariance,
)


class CoordinateMomentsTests(unittest.TestCase):
    def test_streaming_population_covariance_matches_numpy(self):
        frames = np.asarray(
            [
                [[0.0, 0.0, 0.0], [2.0, 1.0, -1.0]],
                [[1.0, 2.0, 0.0], [4.0, 1.0, 1.0]],
                [[2.0, 1.0, 3.0], [3.0, 2.0, 0.0]],
                [[3.0, 3.0, 1.0], [5.0, 0.0, 2.0]],
            ]
        )
        moments = CoordinateMoments(2, reference=frames[0])

        for frame in frames:
            moments.update(frame)

        centered = frames - frames.mean(axis=0)
        expected_covariance = (
            np.einsum("fai,faj->aij", centered, centered) / frames.shape[0]
        )
        expected_reference_rmsd = np.sqrt(
            np.mean(np.sum(np.square(frames - frames[0]), axis=2), axis=0)
        )
        np.testing.assert_allclose(moments.mean, frames.mean(axis=0), atol=1e-14)
        np.testing.assert_allclose(
            moments.covariance,
            expected_covariance,
            atol=1e-14,
        )
        np.testing.assert_allclose(
            moments.rmsf,
            np.sqrt(np.trace(expected_covariance, axis1=1, axis2=2)),
            atol=1e-14,
        )
        np.testing.assert_allclose(
            moments.reference_rmsd,
            expected_reference_rmsd,
            atol=1e-14,
        )

    def test_rmsf_is_invariant_to_a_constant_coordinate_translation(self):
        frames = np.asarray(
            [
                [[0.0, 0.0, 0.0]],
                [[1.0, 2.0, 3.0]],
                [[2.0, 0.0, 1.0]],
            ]
        )
        shifted = frames + np.asarray([17.0, -4.0, 8.0])
        original_moments = CoordinateMoments(1)
        shifted_moments = CoordinateMoments(1)

        for original, translated in zip(frames, shifted):
            original_moments.update(original)
            shifted_moments.update(translated)

        np.testing.assert_allclose(
            original_moments.covariance,
            shifted_moments.covariance,
            atol=1e-14,
        )

    def test_one_frame_has_zero_population_rmsf(self):
        moments = CoordinateMoments(2)
        moments.update([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        np.testing.assert_array_equal(moments.rmsf, [0.0, 0.0])

    def test_reading_results_before_any_update_is_rejected(self):
        moments = CoordinateMoments(1)

        with self.assertRaisesRegex(ValueError, "At least one"):
            _ = moments.covariance


class RMSFAggregationTests(unittest.TestCase):
    def test_residue_covariance_is_mass_weighted_before_square_root(self):
        covariance = np.asarray(
            [
                np.diag([1.0, 0.0, 0.0]),
                np.diag([0.0, 9.0, 0.0]),
                np.diag([0.0, 0.0, 4.0]),
            ]
        )
        grouped = aggregate_site_values(
            covariance,
            groups=([0, 1], [2]),
            weights=np.asarray([3.0, 1.0, 2.0]),
        )

        np.testing.assert_allclose(grouped[0], np.diag([0.75, 2.25, 0.0]))
        np.testing.assert_allclose(rmsf_from_covariance(grouped), [np.sqrt(3.0), 2.0])

    def test_b_factor_conversion_uses_nm_squared_to_angstrom_squared(self):
        covariance = np.asarray([np.diag([0.01, 0.02, 0.03])])

        b_factor = b_factors_from_covariance(covariance)

        expected = (8.0 * np.pi**2 / 3.0) * 6.0
        np.testing.assert_allclose(b_factor, [expected])

    def test_zero_total_group_weight_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "positive weight"):
            aggregate_site_values(
                np.ones((2, 3, 3)),
                groups=([0, 1],),
                weights=np.zeros(2),
            )


if __name__ == "__main__":
    unittest.main()
