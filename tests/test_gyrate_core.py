import importlib.util
from pathlib import Path
import unittest

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "dropps" / "analysis" / "gyrate_core.py"
)
SPEC = importlib.util.spec_from_file_location("gyrate_core", MODULE_PATH)
gyrate_core = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gyrate_core)


class GyrateCoreTests(unittest.TestCase):
    def test_periodic_rg_reconstructs_a_boundary_crossing_molecule(self):
        positions = np.asarray([[9.8, 1.0, 1.0], [0.2, 1.0, 1.0]])
        rg = gyrate_core.periodic_radius_of_gyration(
            positions,
            np.ones(2),
            np.diag([10.0, 10.0, 10.0]),
        )
        self.assertAlmostEqual(rg, 0.2)

    def test_phase_statistics_do_not_pool_dense_and_dilute_values(self):
        rg_nm = np.asarray(
            [
                [0.2, 0.4, 0.8, 1.0],
                [0.3, 0.5, 0.9, 1.1],
            ]
        )
        states = np.asarray(
            [
                [0, 0, 2, 2],
                [0, 0, 2, 2],
            ]
        )
        groups = np.zeros(4, dtype=int)
        rows = gyrate_core.phase_rg_summary(
            rg_nm,
            states,
            groups,
            group_count=1,
            phases=(0, 2),
            blocks=2,
        )
        dense, dilute = rows
        self.assertAlmostEqual(dense["mean_rg_nm"], 0.35)
        self.assertAlmostEqual(dilute["mean_rg_nm"], 0.95)
        self.assertEqual(dense["observations"], 4)
        self.assertEqual(dilute["observations"], 4)
        self.assertAlmostEqual(dense["occupancy_fraction"], 0.5)
        self.assertAlmostEqual(dilute["occupancy_fraction"], 0.5)

    def test_frame_means_report_empty_phase_as_nan_and_zero_count(self):
        result = gyrate_core.phase_rg_time_series(
            np.asarray([[0.2, 0.4], [0.3, 0.5]]),
            np.zeros((2, 2), dtype=int),
            np.zeros(2, dtype=int),
            group_count=1,
            phases=(0, 2),
        )
        np.testing.assert_allclose(result["mean_rg_nm"][:, 0], [0.3, 0.4])
        self.assertTrue(np.all(np.isnan(result["mean_rg_nm"][:, 1])))
        np.testing.assert_array_equal(result["counts"][:, 1], [0, 0])

    def test_histograms_use_one_grid_and_each_phase_integrates_to_one(self):
        result = gyrate_core.phase_rg_histogram(
            np.asarray([[0.2, 0.4, 0.8, 1.0]]),
            np.asarray([[0, 0, 2, 2]]),
            np.zeros(4, dtype=int),
            group_count=1,
            phases=(0, 2),
            bin_width_nm=0.2,
        )
        widths = np.diff(result["edges_nm"])
        integrals = np.sum(result["density_per_nm"] * widths[:, None], axis=0)
        np.testing.assert_allclose(integrals, [1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
