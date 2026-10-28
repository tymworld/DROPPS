import unittest

import numpy as np

from dropps.analysis.rmsd_core import (
    batched_kabsch_residuals,
    batched_kabsch_rmsd,
    batched_residual_rmsd,
    batched_unfitted_rmsd,
)


class BatchedKabschRMSDTests(unittest.TestCase):
    def test_independent_rigid_translation_and_rotation_are_removed(self):
        reference = np.asarray(
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.2, 1.1, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.4, 1.0]],
            ]
        )
        rotations = np.asarray(
            [
                [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
            ]
        )
        translations = np.asarray([[7.0, -3.0, 2.0], [-4.0, 8.0, 5.0]])
        mobile = np.einsum("mai,mij->maj", reference, rotations)
        mobile += translations[:, np.newaxis, :]

        rmsd = batched_kabsch_rmsd(mobile, reference)

        np.testing.assert_allclose(rmsd, 0.0, atol=1.0e-12)

    def test_internal_deformation_remains_after_fit(self):
        reference = np.asarray([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
        mobile = reference.copy()
        mobile[0, 2, 1] = 2.0

        rmsd = batched_kabsch_rmsd(mobile, reference)

        self.assertGreater(rmsd[0], 0.2)

    def test_unfitted_rmsd_retains_rigid_translation(self):
        reference = np.asarray([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
        mobile = reference + np.asarray([2.0, -1.0, 2.0])

        unfitted = batched_unfitted_rmsd(mobile, reference)
        fitted = batched_kabsch_rmsd(mobile, reference)

        np.testing.assert_allclose(unfitted, 3.0, atol=1.0e-12)
        np.testing.assert_allclose(fitted, 0.0, atol=1.0e-12)

    def test_kabsch_residuals_can_be_aggregated_at_another_scope(self):
        reference = np.asarray(
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
            ]
        )
        mobile = reference.copy()
        mobile[1, 2, 1] += 1.0

        residuals = batched_kabsch_residuals(mobile, reference)
        per_molecule = batched_residual_rmsd(residuals)
        pooled = batched_residual_rmsd(
            residuals.reshape(1, -1, 3),
        )[0]

        expected = np.sqrt(np.mean(per_molecule**2))
        self.assertAlmostEqual(pooled, expected)

    def test_mass_weighting_reduces_a_light_atom_deformation(self):
        reference = np.asarray([[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 1.0, 0.0]]])
        mobile = reference.copy()
        mobile[0, 2, 1] = 3.0

        uniform = batched_kabsch_rmsd(mobile, reference)
        weighted = batched_kabsch_rmsd(
            mobile,
            reference,
            weights=np.asarray([[100.0, 100.0, 0.01]]),
        )

        self.assertLess(weighted[0], uniform[0] / 10.0)

    def test_reflection_is_not_accepted_as_a_rotation(self):
        reference = np.asarray(
            [
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ]
        )
        mobile = reference.copy()
        mobile[:, :, 0] *= -1.0

        rmsd = batched_kabsch_rmsd(mobile, reference)

        self.assertGreater(rmsd[0], 0.1)

    def test_invalid_weights_are_rejected(self):
        coordinates = np.zeros((1, 3, 3))

        with self.assertRaisesRegex(ValueError, "positive total"):
            batched_kabsch_rmsd(coordinates, coordinates, weights=np.zeros(3))


if __name__ == "__main__":
    unittest.main()
