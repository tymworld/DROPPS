import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


# Load the NumPy-only core without importing the trajectory-dependent package.
CORE_PATH = (
    Path(__file__).resolve().parents[1]
    / "dropps"
    / "analysis"
    / "coexistence_core.py"
)
SPEC = importlib.util.spec_from_file_location("dropps_coexistence_core_test", CORE_PATH)
core = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = core
SPEC.loader.exec_module(core)


class CoexistenceCenteringTests(unittest.TestCase):
    def test_periodic_center_moves_boundary_spanning_slab_to_box_center(self):
        fractions = np.asarray([0.96, 0.98, 0.01, 0.03, 0.05])
        center, strength = core.periodic_center_fraction(fractions)
        shifted = core.center_periodic_fractions(fractions, center)

        shifted_center, shifted_strength = core.periodic_center_fraction(shifted)
        self.assertAlmostEqual(shifted_center, 0.5, places=12)
        self.assertAlmostEqual(strength, shifted_strength, places=12)
        self.assertGreater(strength, 0.9)

    def test_histogram_density_conserves_total_weight(self):
        fractions = np.asarray([0.05, 0.20, 0.55, 0.95])
        weights = np.asarray([1.0, 2.0, 3.0, 4.0])
        volume = 20.0
        density = core.histogram_density(
            fractions, weights, volume, bins=5, conversion=2.5
        )
        reconstructed_weight = density.sum() * (volume / 5.0) / 2.5
        self.assertAlmostEqual(reconstructed_weight, weights.sum())

    def test_frame_block_and_none_centering_schedules(self):
        centers = np.asarray([0.95, 0.05, 0.45, 0.55])
        strengths = np.ones(4)

        frame = core.centering_schedule(
            centers, strengths, mode="frame", blocks=2
        )
        np.testing.assert_allclose(frame["centers"], centers)
        self.assertEqual(frame["groups"], 4)

        block = core.centering_schedule(
            centers, strengths, mode="block", blocks=2
        )
        first_distance = np.abs(
            (block["centers"][:2] - 0.0 + 0.5) % 1.0 - 0.5
        )
        second_distance = np.abs(
            (block["centers"][2:] - 0.5 + 0.5) % 1.0 - 0.5
        )
        np.testing.assert_allclose(first_distance, 0.0, atol=1.0e-14)
        np.testing.assert_allclose(second_distance, 0.0, atol=1.0e-14)
        self.assertEqual(block["groups"], 2)
        self.assertTrue(np.all(block["order_values"] < 1.0))

        none = core.centering_schedule(
            centers, strengths, mode="none", blocks=2
        )
        self.assertTrue(np.all(np.isnan(none["centers"])))
        self.assertEqual(none["groups"], 0)


class CoexistenceFitTests(unittest.TestCase):
    def setUp(self):
        self.length = 24.0
        self.bins = 160
        self.coordinates = (
            (np.arange(self.bins, dtype=float) + 0.5) / self.bins - 0.5
        ) * self.length
        self.parameters = {
            "dilute_density": 4.5,
            "dense_density": 86.0,
            "half_width_nm": 4.8,
            "interface_parameter_nm": 0.55,
        }
        self.profile = core.double_tanh_profile(
            self.coordinates, **self.parameters
        )

    def test_double_tanh_fit_recovers_synthetic_profile(self):
        result = core.fit_slab_profile(self.coordinates, self.profile)

        self.assertAlmostEqual(result["dilute_density"], 4.5, places=3)
        self.assertAlmostEqual(result["dense_density"], 86.0, places=3)
        self.assertAlmostEqual(result["half_width_nm"], 4.8, places=3)
        self.assertAlmostEqual(result["interface_parameter_nm"], 0.55, places=3)
        self.assertGreater(result["r_squared"], 0.999999)

    def test_noisy_block_fit_reports_finite_standard_errors(self):
        rng = np.random.default_rng(20260722)
        frames = np.asarray(
            [
                np.clip(self.profile + rng.normal(scale=1.5, size=self.bins), 0.0, None)
                for _ in range(40)
            ]
        )
        result = core.fit_slab_profile_blocks(
            self.coordinates, frames, blocks=5
        )

        self.assertEqual(result["blocks"], 5)
        self.assertTrue(np.isfinite(result["sem"]["dense_density"]))
        self.assertTrue(np.isfinite(result["sem"]["interface_width_10_90_nm"]))
        self.assertGreater(result["fit"]["r_squared"], 0.99)

    def test_plateau_masks_exclude_interface_region(self):
        dense, dilute = core.plateau_masks(
            self.coordinates,
            self.parameters["half_width_nm"],
            self.parameters["interface_parameter_nm"],
            buffer_factor=2.0,
        )
        self.assertTrue(np.all(np.abs(self.coordinates[dense]) < 4.8))
        self.assertTrue(np.all(np.abs(self.coordinates[dilute]) > 4.8))
        self.assertGreater(np.count_nonzero(dense), 2)
        self.assertGreater(np.count_nonzero(dilute), 2)


class CoexistenceStatisticsTests(unittest.TestCase):
    def test_contiguous_block_sem_uses_block_means(self):
        values = np.asarray([[1.0], [1.0], [3.0], [3.0]])
        result = core.contiguous_block_statistics(values, blocks=2)

        np.testing.assert_allclose(result["mean"], [2.0])
        np.testing.assert_allclose(result["block_means"], [[1.0], [3.0]])
        np.testing.assert_allclose(result["sem"], [1.0])


if __name__ == "__main__":
    unittest.main()
