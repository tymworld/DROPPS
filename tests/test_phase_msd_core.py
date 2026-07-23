import importlib.util
from pathlib import Path
import unittest

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "dropps"
    / "analysis"
    / "phase_msd_core.py"
)
SPEC = importlib.util.spec_from_file_location("phase_msd_core", MODULE_PATH)
phase_msd_core = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(phase_msd_core)


class PhaseMsdCoreTests(unittest.TestCase):
    def test_fractional_trajectory_unwraps_boundary_crossing(self):
        fractions = np.asarray(
            [
                [[0.90, 0.2, 0.3]],
                [[0.05, 0.2, 0.3]],
                [[0.20, 0.2, 0.3]],
            ]
        )
        lengths = np.tile([10.0, 8.0, 6.0], (3, 1))
        unwrapped = phase_msd_core.unwrap_fractional_trajectories(
            fractions, lengths
        )
        np.testing.assert_allclose(unwrapped[:, 0, 0], [9.0, 10.5, 12.0])

    def test_phase_translation_is_removed_from_intrinsic_msd(self):
        initial = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        positions = np.stack([initial + [frame, 0.0, 0.0] for frame in range(4)])
        states = np.zeros((4, 3), dtype=int)
        rows = phase_msd_core.phase_conditioned_msd(
            positions,
            states,
            np.ones(3),
            phase=0,
            dimensions=(0,),
            conditioning="continuous",
            max_lag_frames=2,
            min_particles=2,
            blocks=2,
        )
        self.assertAlmostEqual(rows[1]["raw_msd_nm2"], 1.0)
        self.assertAlmostEqual(rows[1]["translation_msd_nm2"], 1.0)
        self.assertAlmostEqual(rows[1]["intrinsic_msd_nm2"], 0.0)
        self.assertAlmostEqual(rows[2]["raw_msd_nm2"], 4.0)
        self.assertAlmostEqual(rows[2]["intrinsic_msd_nm2"], 0.0)

    def test_intrinsic_motion_remains_after_collective_translation(self):
        positions = np.asarray(
            [
                [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                [[2.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            ]
        )
        states = np.zeros((2, 2), dtype=int)
        row = phase_msd_core.phase_conditioned_msd(
            positions,
            states,
            np.ones(2),
            phase=0,
            dimensions=(0,),
            conditioning="continuous",
            max_lag_frames=1,
            min_particles=2,
            blocks=2,
        )[1]
        self.assertAlmostEqual(row["raw_msd_nm2"], 2.0)
        self.assertAlmostEqual(row["translation_msd_nm2"], 1.0)
        self.assertAlmostEqual(row["intrinsic_msd_nm2"], 1.0)

    def test_continuous_and_intermittent_conditioning_differ_on_return(self):
        positions = np.zeros((3, 2, 3), dtype=float)
        states = np.asarray([[0, 0], [1, 1], [0, 0]])
        continuous = phase_msd_core.phase_conditioned_msd(
            positions,
            states,
            np.ones(2),
            phase=0,
            conditioning="continuous",
            max_lag_frames=2,
            min_particles=2,
        )
        intermittent = phase_msd_core.phase_conditioned_msd(
            positions,
            states,
            np.ones(2),
            phase=0,
            conditioning="intermittent",
            max_lag_frames=2,
            min_particles=2,
        )
        self.assertEqual(continuous[2]["contributing_pairs"], 0)
        self.assertEqual(intermittent[2]["contributing_pairs"], 2)

    def test_raw_intrinsic_and_translation_terms_decompose(self):
        rng = np.random.default_rng(812)
        positions = np.cumsum(rng.normal(size=(8, 5, 3)), axis=0)
        states = np.zeros((8, 5), dtype=int)
        rows = phase_msd_core.phase_conditioned_msd(
            positions,
            states,
            np.linspace(1.0, 2.0, 5),
            phase=0,
            dimensions=(0, 2),
            conditioning="continuous",
            max_lag_frames=4,
            min_particles=2,
            blocks=3,
        )
        for row in rows:
            self.assertAlmostEqual(
                row["raw_msd_nm2"],
                row["intrinsic_msd_nm2"] + row["translation_msd_nm2"],
            )


if __name__ == "__main__":
    unittest.main()
