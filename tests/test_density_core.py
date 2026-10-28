import unittest
from types import SimpleNamespace

import numpy as np

from dropps.analysis.density import getargs_density, shift_density_center
from dropps.analysis.density_core import (
    DEFAULT_DENSITY_BIN_WIDTH_NM,
    box_geometry_nm,
    select_frame_indices,
    threshold_profile_center,
)


class _Trajectory:
    def __init__(self, times_ps):
        self._frames = [SimpleNamespace(time=time) for time in times_ps]

    def __iter__(self):
        return iter(self._frames)


class DensityFrameSelectionTests(unittest.TestCase):
    def test_actual_timestamps_define_the_selected_range(self):
        trajectory = _Trajectory([0.0, 1000.0, 3000.0, 4000.0])

        indices, median_interval = select_frame_indices(
            trajectory,
            0.0,
            3.5,
            None,
            minimum_frames=2,
        )

        self.assertEqual(indices, [0, 1, 2])
        self.assertEqual(median_interval, 1.5)

    def test_uniform_grid_requirement_rejects_irregular_dynamics(self):
        trajectory = _Trajectory([0.0, 1000.0, 3000.0, 4000.0])

        with self.assertRaisesRegex(ValueError, "uniformly spaced"):
            select_frame_indices(
                trajectory,
                None,
                None,
                None,
                minimum_frames=2,
                require_uniform=True,
            )


class DensityCenteringTests(unittest.TestCase):
    def test_boundary_spanning_dense_region_is_centered_without_bin_offset(self):
        profile = np.asarray(
            [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]
        )

        center, order = threshold_profile_center(profile, 0.5)
        shifted = shift_density_center([profile], 0.5, profile)[0]

        self.assertAlmostEqual(center, 0.0)
        self.assertAlmostEqual(order, 1.0)
        np.testing.assert_array_equal(np.flatnonzero(shifted), [3, 4, 5, 6])

    def test_non_cubic_box_keeps_axis_specific_lengths(self):
        timestep = SimpleNamespace(
            triclinic_dimensions=np.diag([40.0, 40.0, 200.0])
        )

        _, lengths, volume = box_geometry_nm(timestep)

        np.testing.assert_allclose(lengths, [4.0, 4.0, 20.0])
        self.assertAlmostEqual(volume, 320.0)


class DensityArgumentTests(unittest.TestCase):
    def test_time_arguments_are_fractional_and_delta_defaults_to_all_frames(self):
        args = getargs_density(
            [
                "-s",
                "run.tpr",
                "-f",
                "run.xtc",
                "-o",
                "density.xvg",
                "-b",
                "1.25",
                "-e",
                "2.75",
            ]
        )

        self.assertEqual(args.start_time, 1.25)
        self.assertEqual(args.end_time, 2.75)
        self.assertIsNone(args.delta_time)
        self.assertEqual(args.bin_width, DEFAULT_DENSITY_BIN_WIDTH_NM)


if __name__ == "__main__":
    unittest.main()
